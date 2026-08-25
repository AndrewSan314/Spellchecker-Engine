// ============================================================
// RecallReranker — recall-improvement plan Task 6.
// Immutable, leakage-safe inference for the recall-pairwise-v1 feature
// contract. The model artifact (src/data/recall-reranker.json) is loaded
// ONCE at engine construction — never per token. score() is a deterministic
// dot product + clamped sigmoid with no allocations proportional to any
// vocabulary size.
//
// Canonical rank definitions shared by trainer-side extraction and
// inference-time feature assembly (single source of truth so training and
// serving features can never drift):
//   poolRank  — 1-based position of a candidate surface inside
//               buildCorrectionCandidates().entries under the sort
//               (dist asc, freq desc, word asc);
//   cheapRank — 1-based position under (dist asc, freq desc, word asc)
//               counting only DISTINCT surfaces (the dist/freq core of the
//               production cheap order, without its inline attest probe).
// ============================================================
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { extractContextEvidence } from './context-evidence.mjs';

export const RECALL_RERANKER_SCHEMA = 'recall-reranker-v1';
export const RECALL_FEATURE_CONTRACT = 'recall-pairwise-v1';

/** Ordered exactly as trained — bias first, constant 1, never standardized. */
export const RECALL_FEATURE_ORDER = Object.freeze([
  'bias',
  'editDistance',
  'sameAccentKey',
  'candidateMinusOriginalLogFrequency',
  'leftBigramLogRatio',
  'rightBigramLogRatio',
  'centeredTrigramLogRatio',
  'forwardTrigramLogRatio',
  'backwardTrigramLogRatio',
  'candidateAttestedWindows',
  'originalAttestedWindows',
  'tokenLength',
  'originalIsDictionary',
  'candidateCheapRank',
  'candidatePoolRank',
]);
const NON_BIAS_FEATURES = RECALL_FEATURE_ORDER.slice(1);

/** Documented rank sentinel when a surface is absent from every ranked view. */
export const RECALL_RANK_SENTINEL = NON_BIAS_FEATURES.length + 2;

/**
 * Assemble the full recall-pairwise-v1 feature object for one
 * candidate-vs-original pair. All values finite; ranks default to a
 * documented sentinel when the candidate is absent from a ranked view.
 */
export function buildRecallPairwiseFeatures(
  { languageModel, services, words, idx, candidateWord, originalWord,
    cheapRank = 0, poolRank = 0 },
) {
  const ev = extractContextEvidence({
    languageModel,
    words: words.map((w) => (typeof w === 'string' ? w : w.normalized)),
    idx,
    candidateWord,
    originalWord,
  });
  const cf = services.lexicon.frequency(candidateWord) || 0;
  const of = services.lexicon.frequency(originalWord) || 0;
  const original = typeof originalWord === 'string'
    ? originalWord
    : originalWord.normalized;
  return {
    editDistance: ev.editDistance,
    sameAccentKey: ev.sameAccentKey ? 1 : 0,
    candidateMinusOriginalLogFrequency: Math.log(1 + cf) - Math.log(1 + of),
    leftBigramLogRatio: ev.leftBigramLogRatio,
    rightBigramLogRatio: ev.rightBigramLogRatio,
    centeredTrigramLogRatio: ev.centeredTrigramLogRatio,
    forwardTrigramLogRatio: ev.forwardTrigramLogRatio,
    backwardTrigramLogRatio: ev.backwardTrigramLogRatio,
    candidateAttestedWindows: ev.candidateAttestedWindows,
    originalAttestedWindows: ev.originalAttestedWindows,
    tokenLength: original.length,
    originalIsDictionary: services.lexicon.contains(original) ? 1 : 0,
    // documented sentinels: absent from every ranked view
    candidateCheapRank: cheapRank || RECALL_RANK_SENTINEL,
    candidatePoolRank: poolRank || RECALL_RANK_SENTINEL,
  };
}

/**
 * Canonical pool/cheap ranks for one built candidate pool.
 * @param {Array<{word:string,dist:number,freq:number}>} entries
 * @returns {Map<string,{poolRank:number,cheapRank:number}>} keyed by
 *   lowercased surface; distinct surfaces only (first occurrence wins).
 */
export function ranksForCandidates(entries) {
  const key = (c) => c.word.toLowerCase();
  const seen = new Set();
  const distinct = [];
  for (const c of entries) {
    const k = key(c);
    if (!seen.has(k)) {
      seen.add(k);
      distinct.push(c);
    }
  }
  const out = new Map();
  // poolRank: RAW generation order (SymSpell key ranking + sibling merges)
  distinct.forEach((c, i) => out.set(key(c), { poolRank: i + 1, cheapRank: 0 }));
  // cheapRank: canonical dist/freq core of the production cheap order
  [...distinct]
    .sort((a, b) => a.dist - b.dist
      || (b.freq ?? 0) - (a.freq ?? 0) || a.word.localeCompare(b.word))
    .forEach((c, i) => {
      out.get(key(c)).cheapRank = i + 1;
    });
  return out;
}

function sigmoidClamped(z) {
  // ±36 keeps the result STRICTLY inside (0,1) in float64 (exp(-36) ≈
  // 2.3e-16 > double epsilon), while still spanning the full useful
  // probability range.
  const zc = Math.max(-36, Math.min(36, z));
  return 1 / (1 + Math.exp(-zc));
}

/**
 * Immutable inference wrapper. Construct via RecallReranker.load() or
 * loadDefault(); never mutate the artifact.
 */
export class RecallReranker {
  constructor(model, sourcePath = null) {
    validateModel(model);
    this.model = model;
    this.sourcePath = sourcePath;
    this.enabled = true;
    this.weightsByName = model.weights;
    this.stdByName = model.standardization;
    Object.freeze(this);
  }

  /** Load from an explicit path; throws on present-but-malformed artifacts. */
  static load(filePath) {
    let raw;
    try {
      raw = readFileSync(filePath, 'utf8');
    } catch (err) {
      if (err && err.code === 'ENOENT') return disabled('artifact-absent');
      throw err;
    }
    return new RecallReranker(JSON.parse(raw), filePath);
  }

  /** Engine default: src/data/recall-reranker.json; missing → disabled. */
  static loadDefault() {
    const p = path.join(path.dirname(fileURLToPath(import.meta.url)),
      'data', 'recall-reranker.json');
    try {
      return RecallReranker.load(p);
    } catch (err) {
      // ENOENT already returned disabled; anything else is malformed-present
      // and MUST fail fast (plan Task 6 Step 5).
      throw err instanceof SyntaxError
        ? new Error(`malformed recall-reranker artifact at ${p}: ${err.message}`)
        : err;
    }
  }

  /**
   * Calibrated sigmoid probability that `features` is a true correction.
   * Throws on missing/non-finite features (fail fast, no silent zeros).
   */
  score(features) {
    let z = this.weightsByName.bias ?? 0;
    for (const name of NON_BIAS_FEATURES) {
      const v = features[name];
      if (v === undefined || v === null || !Number.isFinite(v)) {
        throw new Error(
          `recall-reranker: missing or non-finite feature "${name}"`);
      }
      const st = this.stdByName[name];
      z += this.weightsByName[name] * ((v - st.mean) / st.std);
    }
    return sigmoidClamped(z);
  }
}

export function disabled(reason) {
  return Object.freeze({ enabled: false, reason, score: () => null });
}

function validateModel(model) {
  const fail = (msg) => { throw new Error(`recall-reranker artifact: ${msg}`); };
  if (!model || typeof model !== 'object') fail('not an object');
  if (model.schema !== RECALL_RERANKER_SCHEMA) {
    fail(`schema ${JSON.stringify(model.schema)} !== ${RECALL_RERANKER_SCHEMA}`);
  }
  if (model.featureContract !== RECALL_FEATURE_CONTRACT) {
    fail(`featureContract ${JSON.stringify(model.featureContract)} !== `
      + RECALL_FEATURE_CONTRACT);
  }
  if (!Array.isArray(model.featureOrder)
    || model.featureOrder.length !== RECALL_FEATURE_ORDER.length
    || !RECALL_FEATURE_ORDER.every((f, i) => model.featureOrder[i] === f)) {
    fail(`feature order mismatch — expected ${RECALL_FEATURE_ORDER.join(',')}`);
  }
  const w = model.weights;
  if (!w || typeof w !== 'object') fail('weights missing');
  for (const name of RECALL_FEATURE_ORDER) {
    if (!Number.isFinite(w[name])) fail(`weight "${name}" not finite`);
  }
  const st = model.standardization;
  if (!st || typeof st !== 'object') fail('standardization missing');
  for (const name of NON_BIAS_FEATURES) {
    const s = st[name];
    if (!s || !Number.isFinite(s.mean) || !Number.isFinite(s.std) || s.std <= 0) {
      fail(`standardization for "${name}" invalid`);
    }
  }
}

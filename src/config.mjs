// ============================================================
// ValidationConfigService — plan §13 / §38
// Immutable snapshot built once at startup (and reloadable);
// RULES MUST NEVER QUERY DB/FILES PER TOKEN.
// ============================================================
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
/**
 * Review B9: config/spelling-tuning.json declared itself "frozen" but NOTHING
 * under src/ read it — the served thresholds came from DEFAULT_SNAPSHOT alone
 * and had already drifted from the calibrated file (attention keys). The file
 * is now the source of truth for every tunable it names; see
 * loadFrozenTuning() below.
 */
export const TUNING_FILE = process.env.SPELLING_TUNING_FILE
  ?? path.join(HERE, '..', 'config', 'spelling-tuning.json');

const DEFAULT_SNAPSHOT = Object.freeze({
  enabled: true,
  rules: {
    deterministicEnabled: true,
    missingDiacriticEnabled: true,   // demo enables; production starts OFF (§38)
    spellingEnabled: true,
    zeroWidthSeverity: 'WARNING',
    invalidCharacterSeverity: 'ERROR',
  },
  characters: {
    // ALLOW | INFO | ERROR — default ALLOW because business undecided (§11.10)
    emojiPolicy: 'ALLOW',
  },
  spelling: {
    attentionMode: 'OFF',                 // OFF | SHADOW | EXPERIMENTAL_ACTIVE
    // Attention gate thresholds. They used to exist ONLY in
    // config/spelling-tuning.json, which the runtime never read, so
    // linguistic-rules.mjs silently fell back to 0.80 instead of the
    // calibrated 0.5 whenever the lane was switched on (review B9).
    attentionMinProbability: 0.5,
    attentionMinCandidateWindows: 2,
    attentionMaxOriginalWindows: 1,
  },
  linguistic: {
    mode: 'ACTIVE',                       // OFF | SHADOW | ACTIVE (§39)
    attentionMode: 'OFF',                 // OFF | SHADOW | EXPERIMENTAL_ACTIVE
    attentionMinProbability: 0.5,
    attentionMinCandidateWindows: 2,
    attentionMaxOriginalWindows: 1,
    maxAccentCandidatesPerToken: 12,      // §16.3
    beamWidth: 5,                         // §18.2
    minTokenLength: 2,                    // §16.4
    missingDiacriticMinConfidence: 0.7,   // §19 tuned on POC bench (seed was 0.85)
    missingDiacriticMinMargin: 0.20,
    // Phase 1 (calibration plan) — temperature for the unified confidence
    // scale sigmoid((s1-s2)/T). Both linguistic rules share the semantics;
    // T=1 reproduces the legacy pre-calibration numbers exactly, so raising
    // T is the only lever that changes emitted confidence. Fitted offline by
    // tools/fit_confidence_temperature.mjs against VSEC dev.
    missingDiacriticConfidenceTemperature: 11.7759,
    spellingConfidenceTemperature: 18.9688,
    spellingMinConfidence: 0.70,
    spellingMinMargin: 0.15,
    spellingMinFrequency: 1000,
    spellingMinTokenLength: 2, // runtime verifier below keeps short-token emissions proof-gated
    maxSpellingSuggestions: 3,            // §22
    originalPriorBonusMissingAccent: 0.8, // §18.3 preserve-original bias (tuned on POC bench)
    originalPriorBonusSpelling: 0.8,      // token already failed dictionary gate
    // plain-form gate: a bare token that is itself a high-frequency dictionary
    // word ("ngay", "hoa") needs direct evidence before being flipped:
    // trigram collocation > bigram joint > strong conf+margin escape hatch.
    plainFormGateMinFrequency: 5000,
    plainFormStrongConfidence: 0.95,      // T3 escape hatch (thin context)
    plainFormStrongMargin: 0.90,
    // cheap-ranker lexical frequency term (plan Task 5 Step 4): log10-scaled
    // lexicon frequency added to PMD local candidate scores so nonsense twins
    // cannot outrank the correct word merely through lattice dilution
    lexicalFreqWeight: 0.6,
    // Real-word wrong-tone correction (VIWIKI-SP class: "quyết đinh",
    // "ảnh hưỏng", "cơ sơ"). Requires DECISIVE trigram evidence — an
    // attested collocation for the candidate AND zero for the original in
    // the same context window — otherwise stays silent (precision-first).
    // OFF by default: with bigram-era signal it produced real-word FPs
    // ("bảo hành chính hãng" -> "hàng"); re-enable only after Task 9/10
    // calibration proves a precision-safe win on VSEC dev.
    wrongToneEnabled: false,
    // Task 4 (recall-improvement plan) — explicit mode for the accented
    // same-key WRONG-DIACRITIC spelling lane. Supersedes the ambiguous
    // boolean above when present:
    //   OFF    (default) lane fully closed — legacy boolean governs nothing;
    //   SHADOW           compute + report wouldEmit, NEVER surface issues;
    //   ACTIVE           lane may emit (post-calibration only, Task 9).
    // Legacy mapping: absent key + wrongToneEnabled=true behaves ACTIVE.
    // ACTIVE since the wrong-diacritic breakthrough: ACCENTED_SAME_KEY is
    // 53% of real dev errors and this lane lifts its recall 0.145 -> 0.500.
    wrongDiacriticMode: 'ACTIVE',
    // Symmetric counterpart of wrongDiacriticMode for UNACCENTED_VALID
    // tokens ("quang" -> "quảng"). Same same-key lane, same trigram/bigram
    // proof; separate switch so the two can be measured independently.
    unaccentedRealWordMode: 'OFF',
    // Task 5 (recall-improvement plan) — DIFFERENT_KEY_REAL_WORD lane for
    // dictionary-valid real-word typos ("đế" -> "đến", sentence-initial
    // "Các" -> "Cách"). Tighter eligibility than the wrong-diacritic lane:
    // corpus-backed candidate + DIRECT bigram/trigram evidence + positive
    // pairwise margin. ACTIVE is reserved for the Task 9 calibration freeze
    // and currently behaves exactly like SHADOW.
    realWordTypoMode: 'ACTIVE',
    // Task 9 (recall-improvement plan) — calibrated REAL-WORD lane gates.
    // The lane emits ONLY when every provisional hard check passes AND the
    // trained pairwise probability clears the frozen threshold. Defaults are
    // the dev-frozen accepted values (see config/spelling-tuning.json);
    // realWordTypoMode governs whether the lane can emit at all.
    // 0.97 (was 0.95): with the wrong-diacritic lane ACTIVE the two lanes
    // compose, so this one is tightened to hold overall precision >= 0.72.
    realWordTypoMinProbability: 0.97,
    realWordTypoMinCandidateWindows: 1,
    realWordTypoMaxOriginalWindows: 3,
    realWordTypoMinMargin: 0.0,
    // Phase 1: this lane reports a TRAINED pairwise probability, so its
    // unified-scale margin is that probability's log-odds. Kept separate
    // from spellingConfidenceTemperature because the two axes have different
    // natural units (trained logits vs raw LM score gaps); falls back to the
    // spelling temperature when null.
    realWordTypoConfidenceTemperature: 6.0299,
    // Wrong-diacritic lane calibration keys (lane stays SHADOW pre-freeze;
    // recorded so Task 9 trials and the frozen file stay schema-complete).
    wrongDiacriticMinProbability: null,
    wrongDiacriticMinCandidateWindows: 1,
    wrongDiacriticMaxOriginalWindows: 0,
    wrongDiacriticMinMargin: 0.0,
    // Task 8 (recall-improvement plan) — bounded word-boundary lane
    // (split "cảmơn" -> "cảm ơn"; merge "như ng" -> "nhưng"). SHADOW
    // computes decisions without emitting; ACTIVE emits post-calibration.
    wordBoundaryCorrectionMode: 'SHADOW',
    // both/all parts of a boundary repair must be corpus-backed above this
    wordBoundaryMinFrequency: 2000,
    typoAlpha: 1.0, typoBeta: 0.3, typoGamma: 1.2, // §22 seed weights
    // Task 5 cascade sizing: wide cheap generation, narrow expensive rerank.
    spellingCandidateKeys: 20,     // stripped keys per lookup (was 6; dev
                                   // oracle shows correct keys cluster at
                                   // rank 14-29 among equal-distance rivals)
    spellingSurfacesPerKey: 4,     // retained surfaces per key (raw kept)
    spellingPoolMax: 80,           // hard bound on the cheap pool
    spellingCheapTopK: 3,          // candidates entering trigram reranking
    // Task 7 (recall-improvement plan): diversity-preserving shortlist size
    // — union of {best-distance, best-context-attested, best-same-key, cheap
    // fill}, capped so the expensive stage never sees more than 4 rivals.
    spellingShortlistSize: 4,
    // cheap-ranker weights (edit dist / log-freq / family / attestation)
    cheapDistWeight: 1.0,
    cheapFreqWeight: 0.5,
    cheapFamilyBonus: 0.75,
    cheapAttestBonus: 0.25,
    // Task 3/9 — candidate-scoring weights (see src/language.mjs SCORE_WEIGHTS)
    // and joint-smoothing pseudo-counts; tuned on dev only (Task 9).
    scoreWeights: { A: 0.2, B: 1.0, C: 1.0, D: 1.5, E: 1.0, F: 1.0 },
    rightJointSmoothing: 20,
    forwardJointSmoothing: 10,
    // Review B3/B8: the engine's "proven enough to surface" confidence floor.
    // It used to be the literal 0.96 hardcoded in BOTH src/engine.mjs and
    // src/rules/linguistic-rules.mjs — two modules coupled through a magic
    // number. It is a threshold on the RAW calibrated confidence; rounding
    // for display happens only in the server's serializeIssue.
    verifiedConfidenceFloor: 0.96,
    // Confidence attached to a direct error-channel proof (a pair observed in
    // the training pairs table). Deliberately at the floor: the emission is
    // justified by the proof, not by a model score.
    errorChannelDirectConfidence: 0.96,
    // Review B4: pairs seen EXACTLY this many times in VSEC train are vetoed
    // in the engine's verifier. The value was fitted on dev (48 TP / 9 FP,
    // docs/precision-breakthrough-log.md:95) and has no linguistic basis —
    // set to null to disable the veto and re-derive it out-of-fold before
    // trusting it in production.
    errorChannelPairCountVeto: 2,
    // Review §7: an unaccented SMS validated in ACCENTED mode yields one
    // warning per token. When at least this share of words (and at least
    // `unaccentedMinWords` words) look unaccented, the result carries a
    // message-level flag so the UI can show ONE notice. Issues themselves are
    // never suppressed — benchmarks and per-token metrics stay comparable.
    unaccentedRatioThreshold: 0.8,
    unaccentedMinWords: 5,
  },
});

/** keys the frozen tuning file is allowed to move (review B9) */
const TUNABLE_SECTIONS = Object.freeze({
  attentionMode: 'spelling',
  attentionMinProbability: 'spelling',
  attentionMinCandidateWindows: 'spelling',
  attentionMaxOriginalWindows: 'spelling',
});

/**
 * Reads the frozen calibration file and turns it into a config override.
 *
 * Contract (fail loud, never silently diverge):
 *   - a missing file is fine (defaults serve);
 *   - a malformed / non-frozen / unknown-key file THROWS at startup rather
 *     than letting the served config drift from the recorded one;
 *   - `servingOverrides` records keys deliberately NOT served with the frozen
 *     value (with a reason), so the divergence is explicit and reviewable
 *     instead of accidental.
 *
 * @returns {{overrides: object, source: object}|null}
 */
export function loadFrozenTuning(file = TUNING_FILE) {
  if (!file || !existsSync(file)) return null;
  let doc;
  try {
    doc = JSON.parse(readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`spelling tuning file is not valid JSON: ${file} — ${err.message}`);
  }
  const params = doc?.frozenParams;
  if (!params || typeof params !== 'object') {
    throw new Error(`spelling tuning file has no "frozenParams" object: ${file}`);
  }
  const overrides = { linguistic: {}, spelling: {} };
  /** writes one tuned key into every section that owns it, or throws */
  const assign = (key, value, what) => {
    const section = TUNABLE_SECTIONS[key]
      ?? (key in DEFAULT_SNAPSHOT.linguistic ? 'linguistic' : null);
    if (!section) {
      throw new Error(`spelling tuning file ${what} unknown key "${key}" (${file}); `
        + 'add it to DEFAULT_SNAPSHOT or remove it from the file');
    }
    overrides[section][key] = value;
    // a few attention keys are read from BOTH spelling.* and linguistic.*
    if (section === 'spelling' && key in DEFAULT_SNAPSHOT.linguistic) {
      overrides.linguistic[key] = value;
    }
  };

  const applied = {};
  for (const [key, value] of Object.entries(params)) {
    assign(key, value, 'sets');
    applied[key] = value;
  }
  // Confidence temperatures live in their own block in the file.
  for (const [key, value] of Object.entries(doc?.confidenceCalibration?.constants ?? {})) {
    assign(key, value, 'sets temperature');
    applied[key] = value;
  }
  const served = {};
  for (const [key, entry] of Object.entries(doc?.servingOverrides?.overrides ?? {})) {
    assign(key, entry.value, 'overrides');
    served[key] = entry;
  }
  return {
    overrides,
    source: {
      file: path.basename(file),
      version: doc.version ?? null,
      frozen: doc.frozen === true,
      createdAt: doc.createdAt ?? null,
      frozenParams: applied,
      servingOverrides: served,
    },
  };
}

export class ValidationConfigSnapshot {
  constructor(data) {
    this.data = deepFreeze(structuredClone(data));
    Object.freeze(this);
  }
  get(path_) {
    return path_.split('.').reduce((o, k) => o?.[k], this.data);
  }
}

function deepFreeze(obj) {
  if (obj && typeof obj === 'object') {
    for (const v of Object.values(obj)) deepFreeze(v);
    Object.freeze(obj);
  }
  return obj;
}

export class ValidationConfigService {
  /**
   * @param {ValidationConfigSnapshot} [snapshot] explicit snapshot (tests).
   *   When omitted the service builds DEFAULT_SNAPSHOT + the frozen tuning
   *   file, so "the calibrated config" and "the served config" are the same
   *   object by construction (review B9).
   */
  constructor(snapshot = null, { tuningFile = TUNING_FILE } = {}) {
    this.tuningSource = null;
    if (snapshot) {
      this._snapshot = snapshot;
      return;
    }
    const frozen = loadFrozenTuning(tuningFile);
    this.tuningSource = frozen?.source ?? null;
    this._snapshot = new ValidationConfigSnapshot(
      frozen ? mergeDeep(DEFAULT_SNAPSHOT, frozen.overrides) : DEFAULT_SNAPSHOT,
    );
  }
  snapshot() {
    return this._snapshot;
  }
  /** plan §13: reload without restart (admin endpoint / scheduled job) */
  reload(overrides = {}) {
    const frozen = loadFrozenTuning(this.tuningSource ? TUNING_FILE : null);
    this.tuningSource = frozen?.source ?? this.tuningSource;
    const base = frozen ? mergeDeep(DEFAULT_SNAPSHOT, frozen.overrides) : DEFAULT_SNAPSHOT;
    const merged = mergeDeep(base, overrides);
    this._snapshot = new ValidationConfigSnapshot(merged);
    return this._snapshot;
  }
}

export { DEFAULT_SNAPSHOT };

function mergeDeep(base, patch) {
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(patch ?? {})) {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      out[k] = mergeDeep(out[k] ?? {}, v);
    } else if (v !== undefined) {
      out[k] = v;
    }
  }
  return out;
}

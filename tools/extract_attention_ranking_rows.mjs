// ============================================================
// Task 3 (tiny-attention-spelling-reranker-FIXED plan) — leakage-safe
// ranking/message split extraction, candidate-oracle@K measurement and
// hard-negative / synthetic mining from ALLOWED TRAIN sources ONLY.
//
//   node tools/extract_attention_ranking_rows.mjs \
//     --vsec-train dataset_artifacts/vsec/vsec-train.jsonl \
//     --clean-train dataset_artifacts/clean-source/clean-train.txt \
//     --out-dir .tmp
//
// Guarantees enforced here:
//   - path guard rejects held-out markers BEFORE any read;
//   - candidates come only from production buildCorrectionCandidates +
//     selectDiverseShortlist; gold is NEVER injected (miss => row dropped);
//   - option index 0 is always KEEP_ORIGINAL; KEEP lanes label index 0;
//   - groupId = normalized sentence hash; synthetic rows inherit the source
//     sentence's groupId so original/corrupt variants cannot cross splits;
//   - deterministic split: sha256(groupId+salt) bucket 00..79 train,
//     80..89 calibration, 90..99 internal-test;
//   - K frozen from oracle@{4,6,8} on train+calibration groups BEFORE any
//     model work; internal-test never influences K.
// ============================================================
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { buildValidationDocument } from '../src/document-builder.mjs';
import {
  classifyToken, buildCorrectionCandidates, buildUnifiedAttentionEntries, selectDiverseShortlist,
} from '../src/rules/linguistic-rules.mjs';
import {
  ranksForCandidates, buildRecallPairwiseFeatures, RECALL_FEATURE_CONTRACT,
} from '../src/recall-reranker.mjs';
import { accentKey } from '../src/normalizer.mjs';
import {
  applyTelexHints, damerauOsaDistance, maxEditDistanceFor,
} from '../src/language.mjs';
import {
  CLASSICAL_BASELINE_OVERRIDES, verifyOverridesApplied,
} from './run_attention_baseline.mjs';
import { sha256File } from './run_spelling_eval.mjs';
import { createEvaluationHeader, sha256OfJson } from '../src/evaluation-provenance.mjs';
import { unitsFromDocument } from '../src/attention-tokenizer.mjs';
import {
  ATTENTION_RANKING_SCHEMA, ATTENTION_MESSAGES_SCHEMA,
  ATTENTION_SHORTLIST_CONFIG_SCHEMA,
  assertAllowedTrainingSources, assertRankingRowShape,
  calculateOracleMetrics,
  normalizedSentenceHash, oracleAtK, selectShortlistK,
  splitForGroup,
} from '../src/attention-ranking-schema.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
/** stop-condition gap: wide-pool oracle materially above oracle@8 */
const MATERIAL_SHORTLIST_GAP = 0.10;

function parseArgs(argv) {
  const flags = {
    vsecTrain: 'dataset_artifacts/vsec/vsec-train.jsonl',
    cleanTrain: 'dataset_artifacts/clean-source/clean-train.txt',
    outDir: '.tmp', maxVsec: 0, maxClean: 0,
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--vsec-train') flags.vsecTrain = argv[++i];
    else if (argv[i] === '--clean-train') flags.cleanTrain = argv[++i];
    else if (argv[i] === '--out-dir') flags.outDir = argv[++i];
    else if (argv[i] === '--max-vsec') flags.maxVsec = Number(argv[++i]) || 0;
    else if (argv[i] === '--max-clean') flags.maxClean = Number(argv[++i]) || 0;
  }
  return flags;
}

function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function applyCase(surface, originalForm) {
  return originalForm === originalForm.toUpperCase() && /\p{L}/u.test(originalForm)
    ? surface.toUpperCase()
    : surface;
}

function writeJsonlWithHeader(outPath, header, records) {
  const lines = [JSON.stringify(header)];
  let count = 0;
  for (const r of records) {
    if (r.recordType !== 'header') lines.push(JSON.stringify(r));
    count++;
  }
  writeFileSync(outPath, `${lines.join('\n')}\n`, 'utf8');
  return count;
}

function main() {
  const t0 = Date.now();
  const flags = parseArgs(process.argv.slice(2));
  const root = path.join(HERE, '..');
  const resolveInput = (p) => (path.isAbsolute(p) ? p : path.join(root, p));

  // ---- guard BEFORE any read ---------------------------------------------
  assertAllowedTrainingSources([flags.vsecTrain, flags.cleanTrain]);
  const vsecPath = resolveInput(flags.vsecTrain);
  const cleanPath = resolveInput(flags.cleanTrain);

  // ---- frozen classical runtime (Task 0 baseline) -------------------------
  const engine = createDefaultEngine();
  engine.configService.reload({ linguistic: { ...CLASSICAL_BASELINE_OVERRIDES } });
  const snap = engine.configService.snapshot();
  if (!verifyOverridesApplied(snap.data)) {
    throw new Error('STOP: could not reproduce the agreed classical baseline '
      + 'overrides on the live snapshot');
  }
  const services = engine.services;
  const lm = services.languageModel;
  const ctxOf = (text) => new ValidationContext(text, 'ACCENTED', 'TENDOO');

  const generatorConfigHash = sha256OfJson({
    overrides: CLASSICAL_BASELINE_OVERRIDES,
    shortlistSizeCap: snap.get('linguistic.spellingShortlistSize'),
    candidateKeys: snap.get('linguistic.spellingCandidateKeys'),
    poolMax: snap.get('linguistic.spellingPoolMax'),
    saltVersion: 'attention-v2-20260825',
  });

  // ---- shared production-candidate machinery ------------------------------
  function neighbourFams(words, idx) {
    const fam = (w) => {
      const set = new Set([w.normalized.toLowerCase()]);
      for (const c of services.accentIndex.candidates(accentKey(w.normalized))) {
        set.add(c.word.toLowerCase());
      }
      return [...set];
    };
    return {
      prevFam: idx > 0 ? fam(words[idx - 1]) : [],
      nextFam: idx + 1 < words.length ? fam(words[idx + 1]) : [],
    };
  }

  /** direct-context attestation per candidate word, production semantics */
  function attestMap(entries, words, idx) {
    const { prevFam, nextFam } = neighbourFams(words, idx);
    const side = nextFam.length >= prevFam.length
      ? { surfs: nextFam, side: 'right' } : { surfs: prevFam, side: 'left' };
    const m = new Map();
    for (const c of entries) {
      m.set(c.word.toLowerCase(), side.surfs.length > 0
        && lm.bestJointOverSurfaces(c.word.toLowerCase(), side.surfs, side.side) > 0);
    }
    return m;
  }

  /**
   * Wide pool via the PRODUCTION UNKNOWN_TYPO builder. Mirrors the
   * production cascade's shortlist INPUT: every generated entry (poolMax-
   * capped by the builder itself), deduplicated, WITHOUT an extra distance
   * cut — the diversity shortlist and downstream gates own that policy.
   */
  function buildWidePool(token, ctx = null, doc = null) {
    const cls = ctx && doc ? classifyToken(token, ctx, doc, services) : 'UNKNOWN';
    const built = buildUnifiedAttentionEntries({ token, cls, services, snap, ctx, doc });
    const seen = new Set([token.normalized.toLowerCase()]);
    const entries = [];
    for (const c of built.entries ?? []) {
      const lw = c.word.toLowerCase();
      if (seen.has(lw)) continue;
      seen.add(lw);
      entries.push(c);
    }
    entries.sort((a, b) => a.dist - b.dist
      || (b.freq ?? 0) - (a.freq ?? 0) || a.word.localeCompare(b.word));
    return { entries, ranks: ranksForCandidates(built.entries ?? []) };
  }
  /** production-style cheap ordering for the diversity shortlist fill */
  function cheapOrdered(entries, token, attestOf) {
    const cw = {
      dist: snap.get('linguistic.cheapDistWeight') ?? 1.0,
      freq: snap.get('linguistic.cheapFreqWeight') ?? 0.5,
      family: snap.get('linguistic.cheapFamilyBonus') ?? 0.75,
      attest: snap.get('linguistic.cheapAttestBonus') ?? 0.25,
    };
    const telexTok = applyTelexHints(token.normalized);
    const famKey = accentKey(token.normalized);
    const scoreOf = (c) => {
      let sc = -cw.dist
        * damerauOsaDistance(telexTok, applyTelexHints(c.stripped ?? accentKey(c.word)));
      sc += cw.freq * Math.log10(1 + (c.freq || 1));
      if (c.stripped === famKey) sc += cw.family;
      if (attestOf?.get(c.word.toLowerCase())) sc += cw.attest;
      return sc;
    };
    return [...entries].sort((a, b) => scoreOf(b) - scoreOf(a)
      || a.word.localeCompare(b.word));
  }

  /** ordered diverse shortlist (prefix slices give @4/@6/@8 consistently) */
  function shortlistPrefix(entries, ranks, token, words, idx, size = 8) {
    if (!entries.length) return [];
    const attestOf = attestMap(entries, words, idx);
    const famKey = accentKey(token.normalized);
    const order = cheapOrdered(entries, token, attestOf);
    const rr = services.recallReranker;
    const recallScoreOf = rr?.enabled ? (candidate) => {
      try {
        const rank = ranks.get(candidate.word.toLowerCase()) ?? {};
        return rr.score(buildRecallPairwiseFeatures({
          languageModel: lm,
          services,
          words,
          idx,
          candidateWord: candidate.word,
          originalWord: token.normalized,
          cheapRank: rank.cheapRank,
          poolRank: rank.poolRank,
        }));
      } catch {
        return null;
      }
    } : null;
    return selectDiverseShortlist(entries, {
      attestOf, famKey, size,
      cheapOrder: order,
      recallScoreOf,
    }).map((c) => ({
      surface: c.word,
      cheapRank: ranks.get(c.word.toLowerCase())?.cheapRank ?? 0,
      poolRank: ranks.get(c.word.toLowerCase())?.poolRank ?? 0,
    }));
  }

  /** lane classification honoring HARD GUARDS (never rows for guarded tokens) */
  function classifyLane(token, ctx, doc) {
    const cls = classifyToken(token, ctx, doc, services);
    if (cls === 'PROTECTED' || cls === 'WHITELISTED'
      || cls === 'CODE_LIKE' || cls === 'ABBREVIATION') {
      return { skip: true, reason: `hard-guard-${cls}` };
    }
    const norm = token.normalized;
    const famKey = accentKey(norm);
    const unaccented = norm === famKey;
    const siblings = services.accentIndex.candidates(famKey)
      .filter((c) => c.word.toLowerCase() !== norm);
    if (cls === 'UNKNOWN') return { lane: 'UNKNOWN_TYPO' };
    if (cls === 'UNACCENTED_VALID' || (unaccented && siblings.length > 0)) {
      return { lane: 'UNACCENTED_SAME_KEY' };
    }
    if (cls === 'DICTIONARY') {
      return { lane: 'DIFFERENT_KEY_REAL_WORD', siblings };
    }
    return { skip: true, reason: `classified-${cls}` };
  }

  function locateLabelToken(doc, words, value) {
    const v = normSurface(value);
    if (!v) return -1;
    let idx = words.findIndex((w) => normSurface(w.original) === v);
    if (idx >= 0) return idx;
    return words.findIndex((w) => w.normalized.toLowerCase() === v);
  }

  // -------------------------------------------------------------------------
  const stats = {
    vsecRowsRead: 0, vsecCases: 0, unlabeledNoSuggestion: 0,
    labelTokenNotFound: 0, hardGuardSkips: {}, candidateMiss: 0,
    candidateMissTrainCal: 0,
    shortlistMiss: { 4: 0, 6: 0, 8: 0 },
    cleanSentences: 0, hardNegativeRows: 0, cleanKeepRows: 0,
    syntheticAttempts: 0, syntheticCandidateMiss: 0, syntheticRows: 0,
    messagesBySource: {},
  };

  /** @type {Array<stored case>} */
  const cases = [];
  /** @type {Map<string,message record>} */
  const messages = new Map();

  function putMessage(groupId, source, text, clean, labels, extra = {}) {
    const key = `${groupId}|${text}`;
    if (!messages.has(key)) {
      messages.set(key, {
        recordType: 'message-row',
        id: `msg:${groupId.slice(0, 16)}:${messages.size}`,
        groupId, source, text, mode: 'ACCENTED', brand: 'TENDOO',
        clean, labels,
        ...extra,
      });
      stats.messagesBySource[source]
        = (stats.messagesBySource[source] ?? 0) + 1;
    }
  }

  function attachLabelOffsets(doc, labels) {
    const words = doc.tokens.filter((t) => t.type === 'WORD');
    const used = new Set();
    return (labels ?? []).map((label) => {
      const wanted = normSurface(label.value);
      const pos = words.findIndex((word, index) => !used.has(index)
        && (normSurface(word.original) === wanted
          || normSurface(word.normalized) === wanted));
      if (pos < 0) return { ...label };
      used.add(pos);
      return { ...label, start: words[pos].start, end: words[pos].end };
    });
  }

  function storeCase(base) {
    cases.push(base);
    return base;
  }

  function mineCorrectionCase({ source, externalId, groupId, text, ctx, doc,
    words, idx, targetSuggestions }) {
    const token = words[idx];
    const primary = normSurface(targetSuggestions[0]);
    if (!primary) {
      stats.unlabeledNoSuggestion++;
      return null;
    }
    const { entries, ranks } = buildWidePool(token, ctx, doc);
    const wideSet = new Set(entries.map((c) => c.word.toLowerCase()));
    if (!wideSet.has(primary)) {
      stats.candidateMiss++;
      // per-split tracking keeps the stop-condition comparison apples-to-
      // apples (wide-pool vs shortlist on the SAME train+calibration set)
      const sp = splitForGroup(groupId);
      if (sp !== 'internal-test') stats.candidateMissTrainCal++;
      return null;
    }
    const sl = shortlistPrefix(entries, ranks, token, words, idx, 8);
    const slWords = sl.map((c) => c.surface.toLowerCase());
    const hitAt = {};
    for (const k of [4, 6, 8]) {
      hitAt[k] = slWords.slice(0, k).includes(primary);
      if (!hitAt[k]) stats.shortlistMiss[k]++;
    }
    const units = unitsFromDocument(doc);
    const targetUnitIdx = units.findIndex((u) => u.kind === 'word'
      && u.start === token.start);
    stats.vsecCases += source === 'vsec-train' ? 1 : 0;
    return storeCase({
      kind: 'correction',
      id: externalId,
      groupId, source, split: splitForGroup(groupId),
      lane: classifyLane(token, ctx, doc).lane ?? 'UNKNOWN_TYPO',
      text,
      cleanFlag: false,
      labels: [{ value: normSurface(token.original), suggestions: targetSuggestions }],
      targetLower: primary,
      words, idx, tokenStart: token.start,
      orderedTop8: sl,
      hitAt,
      targetUnitIdx,
      unitsCount: units.length,
    });
  }

  // ---- VSEC train correction rows -----------------------------------------
  const vsecRaw = readFileSync(vsecPath, 'utf8').split(/\r?\n/).filter(Boolean);
  for (const line of (flags.maxVsec ? vsecRaw.slice(0, flags.maxVsec) : vsecRaw)) {
    const row = JSON.parse(line);
    stats.vsecRowsRead++;
    const text = String(row.text ?? '');
    if (!text.trim()) continue;
    const groupId = normalizedSentenceHash(text);
    const vsecLabels = (row.correction_pairs ?? []).map((p) => ({
        value: normSurface(p?.error), suggestions: [normSurface(p?.correction)],
      })).filter((l) => l.value && l.suggestions[0]);
    putMessage(groupId, 'vsec-train', text, false, vsecLabels);
    const ctx = ctxOf(text);
    const doc = buildValidationDocument(ctx);
    const storedVsec = messages.get(`${groupId}|${text}`);
    if (storedVsec) storedVsec.labels = attachLabelOffsets(doc, storedVsec.labels);
    const words = doc.tokens.filter((t) => t.type === 'WORD');
    (row.correction_pairs ?? []).forEach((pair, expIdx) => {
      const errValue = pair?.error;
      const corrections = [pair?.correction].filter(Boolean).map(normSurface);
      if (!corrections.length) {
        stats.unlabeledNoSuggestion++;
        return;
      }
      const idx = locateLabelToken(doc, words, errValue);
      if (idx < 0) {
        stats.labelTokenNotFound++;
        return;
      }
      const guard = classifyLane(words[idx], ctx, doc);
      if (guard.skip) {
        stats.hardGuardSkips[guard.reason]
          = (stats.hardGuardSkips[guard.reason] ?? 0) + 1;
        return;
      }
      mineCorrectionCase({
        source: 'vsec-train',
        externalId: `vsec:${row.id ?? stats.vsecRowsRead}:${expIdx}`,
        groupId, text, ctx, doc, words, idx,
        targetSuggestions: corrections,
      });
    });
  }

  // ---- clean-train: hard negatives + CLEAN_KEEP + synthetic ---------------
  const cleanLines = readFileSync(cleanPath, 'utf8').split(/\r?\n/)
    .map((l) => l.trim()).filter(Boolean);
  const cleanSubset = flags.maxClean ? cleanLines.slice(0, flags.maxClean) : cleanLines;

  function addKeepRow({ source, lane, groupId, text, ctx, doc, words, idx, cleanFlag }) {
    const token = words[idx];
    const { entries, ranks } = buildWidePool(token, ctx, doc);
    const sl = shortlistPrefix(entries, ranks, token, words, idx, 8);
    const units = unitsFromDocument(doc);
    const targetUnitIdx = units.findIndex((u) => u.kind === 'word'
      && u.start === token.start);
    storeCase({
      kind: 'keep',
      id: `${source}:${groupId.slice(0, 12)}:${idx}`,
      groupId, source, split: splitForGroup(groupId),
      lane, text, cleanFlag,
      labels: [],
      targetLower: null,
      words, idx, tokenStart: token.start,
      orderedTop8: sl,
      hitAt: null,
      targetUnitIdx,
      unitsCount: units.length,
    });
    return true;
  }

  cleanLines.forEach((text, lineIdx) => {
    const groupId = normalizedSentenceHash(text);
    const split = splitForGroup(groupId);
    putMessage(groupId, 'clean-train', text, true, []);
    if (flags.maxClean && lineIdx >= cleanSubset.length) return;
    const ctx = ctxOf(text);
    const doc = buildValidationDocument(ctx);
    const words = doc.tokens.filter((t) => t.type === 'WORD');
    stats.cleanSentences++;

    const result = engine.validate(ctx);
    const spellingIssues = result.issues.filter(
      (i) => i.ruleId === 'POSSIBLE_SPELLING_ERROR');

    // HARD_NEGATIVE_KEEP: the frozen classical engine proposes a correction
    // on a CLEAN sentence — teach KEEP with the SAME production pool.
    let hnAdded = 0;
    for (const issue of spellingIssues) {
      if (hnAdded >= 2) break;
      const idx = words.findIndex((w) => w.start === issue.start);
      if (idx < 0) continue;
      const guard = classifyLane(words[idx], ctx, doc);
      if (guard.skip) continue;
      addKeepRow({
        source: 'hard-negative-clean-train', lane: 'HARD_NEGATIVE_KEEP',
        groupId, text, ctx, doc, words, idx, cleanFlag: true,
      });
      stats.hardNegativeRows++;
      hnAdded++;
    }

    // If classical engine had no issues, mine challenging content words that have
    // multiple competing same-key siblings or edit-distance alternatives (hard KEEP)
    if (hnAdded === 0 && lineIdx % 2 === 0) {
      for (let wIdx = 0; wIdx < words.length; wIdx++) {
        const w = words[wIdx];
        if (w.normalized.length < 3) continue;
        const sibs = services.accentIndex.candidates(accentKey(w.normalized));
        if (sibs.length >= 2 && services.lexicon.realFrequency(w.normalized) > 0) {
          const guard = classifyLane(w, ctx, doc);
          if (!guard.skip) {
            addKeepRow({
              source: 'hard-negative-clean-train', lane: 'HARD_NEGATIVE_KEEP',
              groupId, text, ctx, doc, words, idx: wIdx, cleanFlag: true,
            });
            stats.hardNegativeRows++;
            hnAdded++;
            break;
          }
        }
      }
    }

    // CLEAN_KEEP: one corpus-backed content word per sentence (deterministic
    // rotation) so abstention is learned on ordinary vocabulary too.
    const eligible = words.map((w, i) => [w, i]).filter(([w]) =>
      w.normalized.length >= 3
      && /^[\p{L}]+$/u.test(w.original)
      && services.lexicon.realFrequency(w.normalized) > 0);
    if (eligible.length > 0) {
      const pick = eligible[(lineIdx * 7 + 3) % eligible.length][1];
      const guard = classifyLane(words[pick], ctx, doc);
      if (!guard.skip) {
        addKeepRow({
          source: 'clean-train', lane: 'CLEAN_KEEP',
          groupId, text, ctx, doc, words, idx: pick, cleanFlag: true,
        });
        stats.cleanKeepRows++;
      }
    }

    // SYNTHETIC corruption — train-bucket groups ONLY (internal-test stays
    // pristine; calibration may hold synthetic eval rows by hash chance).
    if (split !== 'train') return;
    const opSeed = Number.parseInt(groupId.slice(0, 6), 16);
    const op = opSeed % 5;
    const candTokens = words.map((w, i) => [w, i]).filter(([w]) =>
      w.normalized.length >= 3 && /^[\p{L}]+$/u.test(w.original)
      && !doc.inProtectedRange(w.start, w.end)
      && classifyLane(w, ctx, doc).skip !== true);
    if (!candTokens.length) return;
    const [tok, tokIdx] = candTokens[opSeed % candTokens.length];

    const corrupted = corruptToken(tok, op, opSeed);
    if (!corrupted || corrupted === tok.normalized) return;
    stats.syntheticAttempts++;

    const newText = text.slice(0, tok.start) + applyCase(corrupted, tok.original)
      + text.slice(tok.end);
    const ctx2 = ctxOf(newText);
    const doc2 = buildValidationDocument(ctx2);
    const words2 = doc2.tokens.filter((t) => t.type === 'WORD');
    if (words2.length !== words.length) return; // single-token swap invariant
    const tok2 = words2[tokIdx];
    const laneInfo = classifyLane(tok2, ctx2, doc2);
    if (laneInfo.skip) return;

    // the ANSWER is the ORIGINAL clean surface; production generation must
    // return it — never inject.
    const { entries, ranks } = buildWidePool(tok2, ctx2, doc2);
    const primary = tok.normalized;
    if (!entries.some((c) => c.word.toLowerCase() === primary)) {
      stats.syntheticCandidateMiss++;
      if (split !== 'internal-test') stats.candidateMissTrainCal++;
      return;
    }
    const sl = shortlistPrefix(entries, ranks, tok2, words2, tokIdx, 8);
    const slWords = sl.map((c) => c.surface.toLowerCase());
    const hitAt = {};
    for (const k of [4, 6, 8]) {
      hitAt[k] = slWords.slice(0, k).includes(primary);
      if (!hitAt[k]) stats.shortlistMiss[k]++;
    }
    const units = unitsFromDocument(doc2);
    storeCase({
      kind: 'correction',
      id: `synthetic:${groupId.slice(0, 12)}:${op}:${tok.start}`,
      groupId, // INHERITED — cannot cross splits
      source: 'synthetic-clean-train',
      split,
      lane: laneInfo.lane,
      text: newText,
      cleanFlag: false,
      labels: [{ value: normSurface(tok2.original), suggestions: [primary] }],
      targetLower: primary,
      words: words2, idx: tokIdx, tokenStart: tok2.start,
      orderedTop8: sl,
      hitAt,
      targetUnitIdx: units.findIndex((u) => u.kind === 'word' && u.start === tok2.start),
      unitsCount: units.length,
    });
    stats.syntheticRows++;
    const syntheticLabels = [{
      value: normSurface(tok2.original), suggestions: [primary],
      start: tok2.start, end: tok2.end,
    }];
    putMessage(groupId, 'synthetic-clean-train', newText, false, syntheticLabels);
  });

  function corruptToken(token, operator, seed) {
      const norm = token.normalized;
      const famKey = accentKey(norm);
      switch (operator) {
        case 0: return accentKey(norm);                       // strip diacritics
        case 1: {                                             // wrong tone, same key
          const sibs = services.accentIndex.candidates(famKey)
            .map((c) => c.word.toLowerCase()).filter((w) => w !== norm);
          return sibs.length ? sibs[seed % sibs.length] : null;
        }
        case 2: {                                             // delete middle char
          const pos = Math.floor(norm.length / 2);
          return norm.slice(0, pos) + norm.slice(pos + 1);
        }
        case 3: {                                             // transpose neighbours
          const pos = Math.floor(norm.length / 2) - 1;
          if (pos < 0) return null;
          return norm.slice(0, pos) + norm[pos + 1] + norm[pos] + norm.slice(pos + 2);
        }
        default: {                                            // limited real-word corruption
          if (seed % 3 !== 0) return null;
          const maxDist = maxEditDistanceFor(norm.length);
          const pool = buildWidePool(token).entries;
          const rw = pool.find((c) => c.dist <= Math.min(1, maxDist)
            && c.stripped !== famKey
            && services.lexicon.realFrequency(c.word.toLowerCase()) > 0);
          return rw ? rw.word.toLowerCase() : null;
        }
      }
  }

  function applyCase(surface, originalForm) {
    return originalForm === originalForm.toUpperCase() && /\p{L}/u.test(originalForm)
      ? surface.toUpperCase()
      : surface;
  }

  // ---- freeze K on train+calibration correction cases ---------------------
  const oracleRows = cases.filter((c) => c.kind === 'correction'
    && (c.split === 'train' || c.split === 'calibration'));
  const considered = oracleRows.length; // widePoolHits
  const allAttempts = considered + stats.candidateMissTrainCal;

  function countHits(k) {
    let hits = 0;
    for (const c of oracleRows) {
      if (c.targetLower && c.orderedTop8.slice(0, k).some((x) => x.surface.toLowerCase() === c.targetLower)) {
        hits++;
      }
    }
    return hits;
  }

  const shortlistHitsAt4 = countHits(4);
  const shortlistHitsAt6 = countHits(6);
  const shortlistHitsAt8 = countHits(8);

  const oracleStats = calculateOracleMetrics({
    allAttempts,
    widePoolHits: considered,
    shortlistHitsAt4,
    shortlistHitsAt6,
    shortlistHitsAt8,
  });

  const selection = selectShortlistK(oracleStats);
  const K = selection.k;

  const gap = oracleStats.widePoolOracle - oracleStats.absoluteOracle[8];
  if (gap > MATERIAL_SHORTLIST_GAP || oracleStats.shortlistRetention[8] < 0.90) {
    console.warn(JSON.stringify({
      WARNING: 'Shortlist retention below 90% or gap > 10%',
      oracleStats,
      gap,
    }, null, 2));
  }

  // ---- emit ranking rows ---------------------------------------------------
  mkdirSync(flags.outDir, { recursive: true });
  const outP = (name) => path.join(flags.outDir, name);
  const headerHashes = {
    vsecTrain: sha256File(vsecPath),
    cleanTrain: sha256File(cleanPath),
    generatorConfig: generatorConfigHash,
  };
  const headerFor = (splitName, schema) => createEvaluationHeader({
    schema, split: splitName, hashes: headerHashes,
    config: {
      configHash: sha256OfJson(snap.get('linguistic')),
      featureContract: RECALL_FEATURE_CONTRACT,
      frozenK: K,
    },
    createdBy: 'tools/extract_attention_ranking_rows.mjs',
  });

  const rowsBySplit = { train: [], calibration: [], 'internal-test': [] };
  const counts = {
    rowsBySource: {}, rowsByLane: {}, labelZero: 0, corrected: 0,
    droppedShortlistMissAtFrozenK: 0,
  };
  for (const c of cases) {
    if (!rowsBySplit[c.split]) continue;
    let candidates = c.orderedTop8.slice(0, K);
    let labelIndex = 0;
    if (c.kind === 'correction') {
      const pos = candidates.findIndex((x) => x.surface.toLowerCase() === c.targetLower);
      if (pos < 0) {
        counts.droppedShortlistMissAtFrozenK++;
        continue;
      }
      labelIndex = pos + 1;
      counts.corrected++;
    } else {
      counts.labelZero++;
    }
    // classical features per candidate, production contract + extras
    const classicalFeatures = candidates.map((cand) => ({
      ...buildRecallPairwiseFeatures({
        languageModel: lm, services, words: c.words, idx: c.idx,
        candidateWord: cand.surface, originalWord: c.words[c.idx].normalized,
        cheapRank: cand.cheapRank, poolRank: cand.poolRank,
      }),
      candidateIsDictionary: services.lexicon.contains(cand.surface.toLowerCase()) ? 1 : 0,
    }));

    const row = {
      recordType: 'ranking-row',
      id: c.id,
      groupId: c.groupId,
      source: c.source,
      context: {
        text: c.text,
        unitsCount: c.unitsCount,
        targetUnitIdx: c.targetUnitIdx,
        units: undefined, // filled below (kept last for readability)
      },
      original: c.words[c.idx].normalized,
      candidates,
      classicalFeatures,
      labelIndex,
      lane: c.lane,
      targetLower: c.targetLower,
    };
    row.context.units = unitsOfStored(c);
    assertRankingRowShape(row);
    rowsBySplit[c.split].push(row);
    counts.rowsBySource[c.source] = (counts.rowsBySource[c.source] ?? 0) + 1;
    counts.rowsByLane[c.lane] = (counts.rowsByLane[c.lane] ?? 0) + 1;
  }

  function unitsOfStored(c) {
    // rebuild units once more at emission (cheap; keeps mining memory low)
    return unitsFromDocument(buildValidationDocument(ctxOf(c.text)));
  }

  const splitRecords = {};
  for (const splitName of Object.keys(rowsBySplit)) {
    const rows = rowsBySplit[splitName];
    splitRecords[splitName] = rows.length;
    writeJsonlWithHeader(
      outP(`attention-ranking-${splitName}.jsonl`),
      headerFor(splitName, ATTENTION_RANKING_SCHEMA), rows);
    // message-level files carry full-engine evaluation material
    const msgs = [...messages.values()].filter((m) => splitForGroup(m.groupId) === splitName);
    writeJsonlWithHeader(
      outP(`attention-messages-${splitName}.jsonl`),
      headerFor(splitName, ATTENTION_MESSAGES_SCHEMA), msgs);
  }

  // ---- deny list (pretraining exclusions) ----------------------------------
  const denyList = {
    calibration: [...new Set(rowsBySplit.calibration.map((r) => r.groupId)
      .concat([...messages.values()]
        .filter((m) => splitForGroup(m.groupId) === 'calibration')
        .map((m) => m.groupId)))],
    internalTest: [...new Set(rowsBySplit['internal-test'].map((r) => r.groupId)
      .concat([...messages.values()]
        .filter((m) => splitForGroup(m.groupId) === 'internal-test')
        .map((m) => m.groupId)))],
  };

  const manifest = {
    schema: 'attention-ranking-split-manifest-v1',
    createdAt: new Date().toISOString(),
    plan: 'docs/plans/2026-08-25-tiny-attention-spelling-reranker-FIXED.md#task-3',
    devOpened: false,
    hashes: {
      ...headerHashes,
      extractor: sha256File(path.join(HERE, 'extract_attention_ranking_rows.mjs')),
      engine: sha256File(path.join(root, 'src', 'engine.mjs')),
      linguisticRules: sha256File(path.join(root, 'src', 'rules', 'linguistic-rules.mjs')),
      config: sha256File(path.join(root, 'src', 'config.mjs')),
    },
    counts: {
      ...counts,
      splits: splitRecords,
      stats,
    },
    oracle: {
      allAttempts: oracleStats.allAttempts,
      widePoolHits: oracleStats.widePoolHits,
      shortlistHits: oracleStats.shortlistHits,
      widePoolOracle: oracleStats.widePoolOracle,
      shortlistRetention: oracleStats.shortlistRetention,
      absoluteOracle: oracleStats.absoluteOracle,
      numerators: oracleStats.numerators,
      denominators: oracleStats.denominators,
    },
    frozenK: K,
    selectionReason: selection.reason,
    denyList,
  };
  const manifestPath = outP('attention-ranking-split-manifest.json');
  manifest.hashes.generated = {};
  for (const splitName of Object.keys(rowsBySplit)) {
    manifest.hashes.generated[`ranking${splitName}`] = sha256File(
      outP(`attention-ranking-${splitName}.jsonl`));
    manifest.hashes.generated[`messages${splitName}`] = sha256File(
      outP(`attention-messages-${splitName}.jsonl`));
  }
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');

  const shortlistConfig = {
    schema: ATTENTION_SHORTLIST_CONFIG_SCHEMA,
    createdAt: manifest.createdAt,
    k: K,
    choices: [4, 6, 8],
    oracle: manifest.oracle,
    selectionReason: selection.reason,
    hashes: {
      generatorConfig: generatorConfigHash,
      extractor: sha256File(path.join(HERE, 'extract_attention_ranking_rows.mjs')),
      inputs: headerHashes,
      manifest: sha256File(manifestPath),
    },
    note: 'All later tasks MUST load K from this file; never hard-code 4.',
  };
  writeFileSync(outP('attention-shortlist-config.json'),
    `${JSON.stringify(shortlistConfig, null, 2)}\n`, 'utf8');

  console.log(JSON.stringify({
    ok: true,
    elapsedMs: Date.now() - t0,
    frozenK: K,
    oracle: manifest.oracle,
    counts,
    shortlistMiss: stats.shortlistMiss,
    candidateMiss: stats.candidateMiss,
    synthetic: { attempts: stats.syntheticAttempts, mined: stats.syntheticRows, missed: stats.syntheticCandidateMiss },
    hardNegatives: stats.hardNegativeRows,
    cleanKeeps: stats.cleanKeepRows,
    messages: stats.messagesBySource,
  }, null, 2));
}

main();

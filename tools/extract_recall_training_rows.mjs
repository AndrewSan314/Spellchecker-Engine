// Recall-improvement plan Task 6 Step 2 — leakage-safe TRAIN-row extraction.
//
// Builds candidate-vs-original feature rows for the recall-pairwise-v1
// trainer using THE PRODUCTION helpers (buildCorrectionCandidates,
// extractContextEvidence via buildRecallPairwiseFeatures, canonical ranks):
//   positives : VSEC train labels whose target exists in the production
//               oracle pool (candidate = labeled target);
//   negatives : top-frequency same-distance rival, best context-attested
//               wrong rival, cheapest-ranked rival, and rivals of unchanged
//               dictionary tokens from the clean-train split.
//
// LEAKAGE GUARD: refuses any input path containing held-out markers BEFORE
// reading, and writes a header declaring the exact sources. Never reads
// vsec-dev / vsec-test / viwiki / benchmark artifacts.
import { readFileSync, writeFileSync, renameSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import {
  buildCorrectionCandidates,
} from '../src/rules/linguistic-rules.mjs';
import { classifyCorrectionRelation } from '../src/correction-taxonomy.mjs';
import {
  buildRecallPairwiseFeatures, ranksForCandidates,
} from '../src/recall-reranker.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');

const FORBIDDEN_MARKS = ['vsec-dev', 'vsec-test', 'viwiki', 'external-test',
  'benchmark'];
const LANE_FOR_RELATION = {
  ACCENTED_SAME_KEY: 'ACCENTED_SAME_KEY',
  DIFFERENT_KEY_SINGLE_TOKEN: 'DIFFERENT_KEY_REAL_WORD',
};
const NEGATIVES_PER_POSITIVE = 3;
const ELIGIBLE_TOKENS_PER_CLEAN_LINE = 3;

function parseArgs(argv) {
  const flags = {
    vsecTrain: 'dataset_artifacts/vsec/vsec-train.jsonl',
    cleanTrain: 'dataset_artifacts/clean-source/clean-train.txt',
    out: '.tmp/recall-training-rows.jsonl',
    cleanCap: 20000,
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--vsec-train') flags.vsecTrain = argv[++i];
    else if (argv[i] === '--clean-train') flags.cleanTrain = argv[++i];
    else if (argv[i] === '--out') flags.out = argv[++i];
    else if (argv[i] === '--clean-cap') flags.cleanCap = Number(argv[++i]);
  }
  return flags;
}

function assertAllowed(p) {
  const lowered = path.resolve(p).replace(/\\/g, '/').toLowerCase();
  if (FORBIDDEN_MARKS.some((m) => lowered.includes(m))) {
    throw new Error(`REFUSED forbidden input path ${p}`);
  }
}

/** benchmark-compatible surface normalization (same policy as the evaluator) */
function normSurfaceOf(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

/** one candidate-vs-original row */
function makeRow(id, source, label, feats) {
  return { id, source, label, features: feats };
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  const vsecPath = path.resolve(ROOT, flags.vsecTrain);
  const cleanPath = path.resolve(ROOT, flags.cleanTrain);
  assertAllowed(vsecPath);
  assertAllowed(cleanPath);

  const engine = createDefaultEngine();
  const services = engine.services;
  const snap = engine.configService.snapshot();

  const rows = [];
  const counts = {
    labelsSeen: 0, positives: 0, negatives: 0, oracleMiss: 0,
    unsupportedRelation: 0, tokenNotFound: 0, cleanRows: 0,
  };

  // ---- positives + in-pool hard negatives from VSEC train --------------
  const raw = readFileSync(vsecPath, 'utf8');
  raw.split(/\r?\n/).forEach((line, rowIdx) => {
    if (!line.trim()) return;
    const rec = JSON.parse(line);
    const ctx = new ValidationContext(rec.text, 'ACCENTED', 'TENDOO');
    const doc = engine.documentBuilder.build(ctx);
    const words = doc.tokens.filter((t) => t.type === 'WORD');

    for (const [pairIdx, pair] of (rec.correction_pairs ?? []).entries()) {
      counts.labelsSeen++;
      const errRaw = String(pair.error ?? '');
      const targetRaw = String(pair.correction ?? '');
      if (!errRaw || !targetRaw || /\s/.test(targetRaw.trim())) {
        counts.unsupportedRelation++;
        continue;
      }
      const rel = classifyCorrectionRelation(errRaw, targetRaw);
      const lane = LANE_FOR_RELATION[rel];
      if (!lane) {
        counts.unsupportedRelation++;
        continue;
      }
      const errLower = normSurfaceOf(errRaw).toLowerCase();
      let tokIdx = words.findIndex(
        (w) => w.normalized.toLowerCase() === errLower);
      if (tokIdx < 0) {
        tokIdx = words.findIndex(
          (w) => w.normalized.toLowerCase().includes(errLower));
      }
      if (tokIdx < 0) {
        counts.tokenNotFound++;
        continue;
      }
      const t = words[tokIdx];
      const built = buildCorrectionCandidates({
        token: t, lane, services, snap, ctx, doc,
      });
      const pool = built.entries ?? [];
      if (!built.eligible || pool.length === 0) continue;

      const rankMap = ranksForCandidates(pool);
      const byLower = new Map();
      for (const c of pool) {
        const k = c.word.toLowerCase();
        if (!byLower.has(k)) byLower.set(k, c);
      }
      const targetKey = normSurfaceOf(targetRaw).toLowerCase();
      const targetEntry = byLower.get(targetKey);
      if (!targetEntry) {
        counts.oracleMiss++;
        continue;
      }

      const featFor = (candWord) => {
        const r = rankMap.get(candWord.toLowerCase());
        return buildRecallPairwiseFeatures({
          languageModel: services.languageModel,
          services,
          words,
          idx: tokIdx,
          candidateWord: candWord,
          originalWord: t.normalized,
          cheapRank: r?.cheapRank ?? 0,
          poolRank: r?.poolRank ?? 0,
        });
      };

      rows.push(makeRow(`t${rowIdx}-p${pairIdx}`, 'vsec-train', 1,
        featFor(targetEntry.word)));
      counts.positives++;

      // hard negatives inside the same pool
      const nextW = tokIdx + 1 < words.length
        ? words[tokIdx + 1].normalized.toLowerCase() : null;
      const prevW = tokIdx > 0
        ? words[tokIdx - 1].normalized.toLowerCase() : null;
      const jointCount = (w) => {
        let c = 0;
        if (nextW) c += services.languageModel.bigram.get(`${w} ${nextW}`) ?? 0;
        if (prevW) c += services.languageModel.bigram.get(`${prevW} ${w}`) ?? 0;
        return c;
      };
      const rivals = [...byLower.values()].filter(
        (c) => c.word.toLowerCase() !== targetKey);
      const picks = [];
      const take = (entry) => {
        if (!entry) return;
        const k = entry.word.toLowerCase();
        if (picks.some((p2) => p2.word.toLowerCase() === k)) return;
        picks.push(entry);
      };
      // n1: highest-frequency rival at the target's edit distance
      take([...rivals]
        .filter((c) => c.dist === targetEntry.dist)
        .sort((a, b) => (b.freq ?? 0) - (a.freq ?? 0)
          || a.word.localeCompare(b.word))[0]);
      // n2: best context-attested wrong rival (direct bigram evidence only)
      take([...rivals]
        .sort((a, b) => jointCount(b.word.toLowerCase())
          - jointCount(a.word.toLowerCase())
          || (b.freq ?? 0) - (a.freq ?? 0))[0]);
      // n3: cheapest-ranked remaining rival
      take([...rivals].sort((a, b) => {
        const ra = rankMap.get(a.word.toLowerCase());
        const rb = rankMap.get(b.word.toLowerCase());
        return (ra?.cheapRank ?? 999) - (rb?.cheapRank ?? 999);
      })[0]);

      for (const [negIdx, r] of picks.slice(0, NEGATIVES_PER_POSITIVE)
        .entries()) {
        rows.push(makeRow(`t${rowIdx}-p${pairIdx}-n${negIdx}`, 'vsec-train', 0,
          featFor(r.word)));
        counts.negatives++;
      }
    }
  });

  // ---- clean-train negatives: rivals of unchanged dictionary tokens -----
  const cleanLines = readFileSync(cleanPath, 'utf8').split(/\r?\n/)
    .filter((l) => l.trim());
  cleanLoop:
  for (const [lineIdx, text] of cleanLines.entries()) {
    if (counts.cleanRows >= flags.cleanCap) break cleanLoop;
    const ctx = new ValidationContext(text, 'ACCENTED', 'TENDOO');
    const doc = engine.documentBuilder.build(ctx);
    const words = doc.tokens.filter((t) => t.type === 'WORD');
    let used = 0;
    for (let idx = 0; idx < words.length && used < ELIGIBLE_TOKENS_PER_CLEAN_LINE
      && counts.cleanRows < flags.cleanCap; idx++) {
      const t = words[idx];
      if (t.normalized.length < 3) continue;
      if (!services.lexicon.contains(t.normalized)) continue;
      const built = buildCorrectionCandidates({
        token: t, lane: 'DIFFERENT_KEY_REAL_WORD', services, snap, ctx, doc,
      });
      const rankMap = ranksForCandidates(built.entries ?? []);
      const rival = [...new Map((built.entries ?? []).map(
        (c) => [c.word.toLowerCase(), c])).values()]
        .filter((c) => c.word.toLowerCase() !== t.normalized)
        .sort((a, b) => (b.freq ?? 0) - (a.freq ?? 0)
          || a.word.localeCompare(b.word))[0];
      if (!rival) continue;
      const r = rankMap.get(rival.word.toLowerCase());
      rows.push(makeRow(`c${lineIdx}-${idx}`, 'clean-train', 0,
        buildRecallPairwiseFeatures({
          languageModel: services.languageModel,
          services,
          words,
          idx,
          candidateWord: rival.word,
          originalWord: t.normalized,
          cheapRank: r?.cheapRank ?? 0,
          poolRank: r?.poolRank ?? 0,
        })));
      counts.cleanRows++;
      counts.negatives++;
      used++;
    }
  }

  // stable ordering: positives block already interleaved — sort by id for a
  // byte-stable file regardless of map iteration order
  rows.sort((a, b) => a.id.localeCompare(b.id));

  const header = {
    header: true,
    schema: 'recall-training-rows-v1',
    sources: [
      path.relative(ROOT, vsecPath).replace(/\\/g, '/'),
      path.relative(ROOT, cleanPath).replace(/\\/g, '/'),
    ],
    contract: 'recall-pairwise-v1',
  };
  mkdirSync(path.dirname(path.resolve(flags.out)), { recursive: true });
  const body = [
    JSON.stringify(header),
    ...rows.map((r) => JSON.stringify(r)),
  ].join('\n') + '\n';
  const tmp = `${flags.out}.tmp-${process.pid}`;
  writeFileSync(tmp, body, 'utf8');
  renameSync(tmp, flags.out);

  console.error(JSON.stringify({
    out: flags.out,
    sha256: createHash('sha256').update(body).digest('hex'),
    ...counts,
  }));
}

try {
  main();
} catch (err) {
  console.error(err.message ?? err);
  process.exit(1);
}

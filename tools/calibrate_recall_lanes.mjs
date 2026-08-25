// Recall-improvement plan Task 9 — dev-only calibration of the recall lanes.
//
//   node tools/calibrate_recall_lanes.mjs \
//     --decisions .tmp/task6-shadow-decisions.jsonl \
//     --out dataset_artifacts/evaluation/recall-calibration.json
//
// Flow: REPLAY the shadow-decision records over a small threshold grid per
// lane (instant, no engine), keep only grid points whose lane-local
// precision clears the hard floor, then VERIFY the survivors with real
// in-process dev runs (semantic views, red-team clean corpus, synthetic
// missing-diacritic recall, SMS-length latency percentiles). Selection is a
// PURE deterministic function (exported) over the verified trials.
// Held-out splits are never touched.
import { readFileSync, writeFileSync, renameSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { adaptVsecSplit } from './spelling_benchmark_adapter_vsec.mjs';
import { linguisticCorrectionMatches } from '../src/correction-taxonomy.mjs';
import {
  readEvaluationJsonl, validateEvaluationHeader,
} from '../src/evaluation-provenance.mjs';
import { RECALL_FEATURE_CONTRACT } from '../src/recall-reranker.mjs';
import { sha256File } from './run_spelling_eval.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');

export const HARD_CONSTRAINTS = {
  lanePrecisionMin: 0.70,        // plan: never activate below this
  semanticPrecisionMaxDrop: 0.02, // global semantic precision may not regress
  // plan global gate: cross-lane dev recall must improve by >=10 absolute
  // points over the 0.1677 baseline BEFORE any lane activates
  semanticRecallMinAbsolute: 0.2677,
  mdRecallDropMaxPp: 0.5,         // synthetic missing-diacritic recall floor
  sms160P95MaxMs: 15,
  cleanCorpusNewIssuesMax: 0,     // no NEW unexpected issue on red-team corpus
};

export const BASELINE_LANE_OVERRIDES = Object.freeze({
  wrongDiacriticMode: 'OFF',
  realWordTypoMode: 'OFF',
  wordBoundaryCorrectionMode: 'SHADOW',
});

function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

/** atomic JSON write + sha256 manifest sidecar */
function writeArtifact(outPath, payload) {
  mkdirSync(path.dirname(outPath), { recursive: true });
  const body = Buffer.from(JSON.stringify(payload, null, 2), 'utf8');
  const tmp = `${outPath}.tmp-${process.pid}`;
  writeFileSync(tmp, body);
  renameSync(tmp, outPath);
  const sha256 = createHash('sha256').update(body).digest('hex');
  writeFileSync(`${outPath}.manifest.json`, JSON.stringify({
    artifact: path.basename(outPath), bytes: body.length, sha256,
  }, null, 2));
  return sha256;
}

/**
 * PURE: replay shadow-decision records for one lane under a threshold set.
 * Hit accounting mirrors the evaluator (benchmark surface normalization).
 */
export function replayLaneThresholds(records, {
  minProbability, minCandidateWindows = 1, maxOriginalWindows = 3,
  minMargin = 0,
}) {
  let wouldEmit = 0; let tp = 0; let fp = 0;
  for (const r of records) {
    if (!r || r.wouldEmit !== true) continue;
    const f = r.features ?? {};
    if ((r.modelProbability ?? 0) < minProbability) continue;
    if ((f.candidateAttestedWindows ?? 0) < minCandidateWindows) continue;
    if ((f.originalAttestedWindows ?? 99) > maxOriginalWindows) continue;
    if ((r.evidenceScore ?? 0) < minMargin) continue;
    wouldEmit++;
    const hit = normSurface(r.candidate) === normSurface(r.target)
      && normSurface(r.value) !== '';
    if (hit) tp++; else fp++;
  }
  return {
    wouldEmit, tp, fp,
    precision: wouldEmit ? Math.round((tp / wouldEmit) * 10000) / 10000 : null,
  };
}

/**
 * PURE deterministic selection: reject trials violating ANY hard constraint,
 * then maximize semanticF05; tie-break higher semanticPrecision, lower
 * latencyMs, then lexical id order. Repeatable byte-identical output.
 */
export function selectCalibrationWinner(trials, constraints = HARD_CONSTRAINTS) {
  const feasible = [];
  const rejected = [];
  for (const t of [...trials].sort((a, b) => a.id.localeCompare(b.id))) {
    const m = t.metrics ?? {};
    const violations = [];
    if (m.lanePrecision != null && m.lanePrecision < constraints.lanePrecisionMin) {
      violations.push(`lanePrecision ${m.lanePrecision} < ${constraints.lanePrecisionMin}`);
    }
    if (m.semanticPrecision != null && t.baseline?.semanticPrecision != null
      && m.semanticPrecision
        < t.baseline.semanticPrecision - constraints.semanticPrecisionMaxDrop) {
      violations.push(`semanticPrecision drop ${(t.baseline.semanticPrecision - m.semanticPrecision).toFixed(4)}`);
    }
    if (m.semanticRecall != null
      && constraints.semanticRecallMinAbsolute != null
      && m.semanticRecall < constraints.semanticRecallMinAbsolute) {
      violations.push(`semanticRecall ${m.semanticRecall} < `
        + `${constraints.semanticRecallMinAbsolute} (plan +10pp bar before activation)`);
    }
    if (m.mdRecallSynthetic != null && t.baseline?.mdRecallSynthetic != null
      && (t.baseline.mdRecallSynthetic - m.mdRecallSynthetic) * 100
        > constraints.mdRecallDropMaxPp + 1e-9) {
      violations.push('mdRecallSynthetic drop > '
        + `${constraints.mdRecallDropMaxPp}pp`);
    }
    if (m.sms160P95Ms != null && m.sms160P95Ms > constraints.sms160P95MaxMs) {
      violations.push(`sms160P95Ms ${m.sms160P95Ms} > ${constraints.sms160P95MaxMs}`);
    }
    if (m.cleanCorpusUnexpectedIssues != null
      && t.baseline?.cleanCorpusUnexpectedIssues != null
      && m.cleanCorpusUnexpectedIssues - t.baseline.cleanCorpusUnexpectedIssues
        > constraints.cleanCorpusNewIssuesMax) {
      violations.push('new unexpected red-team clean issues');
    }
    if (violations.length > 0) rejected.push({ id: t.id, violations });
    else feasible.push(t);
  }
  feasible.sort((a, b) => (b.metrics.semanticF05 ?? 0)
    - (a.metrics.semanticF05 ?? 0)
    || (b.metrics.semanticPrecision ?? 0) - (a.metrics.semanticPrecision ?? 0)
    || (a.latencyMs ?? 0) - (b.latencyMs ?? 0)
    || a.id.localeCompare(b.id));
  return {
    winner: feasible[0] ?? null,
    feasible,
    rejected,
    selection: 'max semanticF05; ties: precision desc, latency asc, id lex',
    constraints,
  };
}

/**
 * Pick replay survivors for expensive in-process verification. By default
 * verify every survivor: the grid is deliberately small and truncating it
 * in generation order can skip the highest-precision thresholds. An
 * explicit cap remains available for quick diagnostic runs, ordered by
 * replay precision, then true positives, then stable lane/config keys.
 */
export function selectCandidatesForVerification(candidates, maxVerified = null) {
  const ranked = [...candidates].sort((a, b) =>
    (b.replay?.precision ?? -1) - (a.replay?.precision ?? -1)
    || (b.replay?.tp ?? 0) - (a.replay?.tp ?? 0)
    || String(a.lane).localeCompare(String(b.lane))
    || (b.replay?.minProbability ?? 0) - (a.replay?.minProbability ?? 0)
    || (a.replay?.maxOriginalWindows ?? 0)
      - (b.replay?.maxOriginalWindows ?? 0));
  if (maxVerified == null) return ranked;
  return ranked.slice(0, Math.max(0, Math.floor(maxVerified)));
}

/** percentile of an unsorted array of ms samples */
function p95(samples) {
  if (!samples.length) return null;
  const s = [...samples].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1);
  return Math.round(s[idx] * 100) / 100;
}

function parseArgs(argv) {
  const flags = {
    decisions: '.tmp/task6-shadow-decisions.jsonl',
    out: 'dataset_artifacts/evaluation/recall-calibration.json',
    maxVerified: null,
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--decisions') flags.decisions = argv[++i];
    else if (argv[i] === '--out') flags.out = argv[++i];
    else if (argv[i] === '--max-verified') flags.maxVerified = Number(argv[++i]);
  }
  return flags;
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  // Task 1 (attention plan): provenance validation. Stale decision dumps —
  // header-less files, or files whose source/artifact hashes no longer match
  // this repository — are REFUSED instead of silently replayed.
  const { header: dumpHeader, records } = readEvaluationJsonl(flags.decisions);
  const currentHashes = {
    linguisticRulesDotMjs: sha256File(path.join(ROOT, 'src', 'rules', 'linguistic-rules.mjs')),
    configDotMjs: sha256File(path.join(ROOT, 'src', 'config.mjs')),
    lmNgramsTsv: sha256File(path.join(ROOT, 'src', 'data', 'lm-ngrams.tsv')),
    recallRerankerModelJson: sha256File(
      path.join(ROOT, 'src', 'data', 'recall-reranker.json')),
  };
  validateEvaluationHeader(dumpHeader, {
    schema: 'shadow-decisions-v2',
    hashes: currentHashes,
    config: { featureContract: RECALL_FEATURE_CONTRACT },
  });

  // ---- Stage A: replay grids (no engine) --------------------------------
  const LANES = [
    { key: 'ACCENTED_SAME_KEY', label: 'wrong-diacritic' },
    { key: 'DIFFERENT_KEY_REAL_WORD', label: 'real-word' },
  ];
  const GRID_PROB = [0.80, 0.85, 0.90, 0.95];
  const GRID_ORIG = [0, 1, 3];
  const replay = {};
  for (const lane of LANES) {
    const rs = records.filter((r) => r.lane === lane.key);
    replay[lane.key] = [];
    for (const p of GRID_PROB) {
      for (const o of GRID_ORIG) {
        const r = replayLaneThresholds(rs, {
          minProbability: p, minCandidateWindows: 1, maxOriginalWindows: o,
        });
        replay[lane.key].push({ minProbability: p, maxOriginalWindows: o, ...r });
      }
    }
  }

  // ---- Stage B: verify feasible replay points with real dev runs --------
  const engine = createDefaultEngine();
  const svc = engine.services;
  const snapOf = () => svc.configService.snapshot();
  const devRows = adaptVsecSplit('dev').rows.filter((r) => r.expect.length > 0);
  const synRows = JSON.parse(readFileSync(
    path.join(ROOT, 'benchmark', 'corpus-synthetic-diacritics.json'), 'utf8')).rows;
  const cleanRows = JSON.parse(readFileSync(
    path.join(ROOT, 'benchmark', 'corpus.json'), 'utf8')).rows;
  const ctxOf = (row) => new ValidationContext(
    row.text, row.mode ?? 'ACCENTED', row.brand ?? row.brandname ?? 'TENDOO');

  const f05Of = (p, r) => (p + r) ? 1.25 * p * r / (0.25 * p + r || 1) : 0;

  function measureDev() {
    let tpStrict = 0; let fpStrict = 0; let tpSem = 0; let fpSem = 0;
    let labels = 0; const lat = [];
    for (const row of devRows) {
      const t0 = process.hrtime.bigint();
      const result = engine.validate(ctxOf(row));
      // the plan's latency gate is defined on SMS-length messages
      // (<=160 chars) — sample only those, mirroring profile:engine
      if ((row.text ?? '').length <= 160) {
        lat.push(Number(process.hrtime.bigint() - t0) / 1e6);
      }
      const spell = result.issues.filter((i) =>
        i.ruleId === 'POSSIBLE_SPELLING_ERROR');
      const ling = result.issues.filter((i) =>
        i.ruleId === 'POSSIBLE_SPELLING_ERROR'
        || i.ruleId === 'POSSIBLE_MISSING_DIACRITIC');
      const labelValues = new Set(row.expect.map((e) => normSurface(e.value)));
      for (const iss of spell) {
        if (!labelValues.has(normSurface(iss.value))) fpStrict++;
      }
      for (const iss of ling) {
        const semHit = row.expect.some((e) => linguisticCorrectionMatches(iss, e));
        if (!semHit) fpSem++;
      }
      for (const exp of row.expect) {
        labels++;
        const sMatch = (iss) => (iss.suggestions ?? []).some((s) =>
          normSurface(s) === normSurface(String(exp.suggestions?.[0] ?? '')));
        if (spell.some((i) => normSurface(i.value) === normSurface(exp.value)
          && sMatch(i))) tpStrict++;
        if (ling.some((i) => linguisticCorrectionMatches(i, exp))) tpSem++;
      }
    }
    const pS = tpSem / (tpSem + fpSem || 1);
    const rS = tpSem / (labels || 1);
    return {
      labels, tpStrict, fpStrict,
      strictPrecision: tpStrict / (tpStrict + fpStrict || 1),
      tpSem,
      semanticPrecision: Math.round(pS * 10000) / 10000,
      semanticRecall: Math.round(rS * 10000) / 10000,
      semanticF05: Math.round(f05Of(pS, rS) * 10000) / 10000,
      sms160P95Ms: p95(lat),
    };
  }

  function measureSyntheticMd() {
    let caught = 0; let expected = 0;
    for (const row of synRows) {
      const res = engine.validate(ctxOf(row));
      const pm = res.issues.filter((i) =>
        i.ruleId === 'POSSIBLE_MISSING_DIACRITIC');
      for (const exp of row.expect ?? []) {
        if (exp.ruleId !== 'POSSIBLE_MISSING_DIACRITIC') continue;
        expected++;
        const hit = pm.some((i) => normSurface(i.value) === normSurface(exp.value)
          && (exp.suggestion === undefined || (i.suggestions ?? []).some((s) =>
            normSurface(s) === normSurface(exp.suggestion))));
        if (hit) caught++;
      }
    }
    return expected ? Math.round((caught / expected) * 10000) / 10000 : null;
  }

  function measureCleanCorpus() {
    let unexpected = 0;
    for (const row of cleanRows) {
      const res = engine.validate(ctxOf(row));
      const expectedIds = new Set((row.expect ?? []).map((e) => e.ruleId));
      for (const iss of res.issues) {
        if (!expectedIds.has(iss.ruleId)) unexpected++;
      }
    }
    return unexpected;
  }

  // baseline under current (OFF-lane) defaults
  svc.configService.reload({ linguistic: BASELINE_LANE_OVERRIDES });
  const baseDev = measureDev();
  const baseMd = measureSyntheticMd();
  const baseClean = measureCleanCorpus();

  // candidate configs per lane (one lane at a time — plan Step 2/4 order):
  // wrong-diacritic FIRST, then real-word.
  const candidates = [];
  for (const cfg of replay.ACCENTED_SAME_KEY) {
    if (cfg.tp < 5) continue;
    if (cfg.precision == null || cfg.precision < HARD_CONSTRAINTS.lanePrecisionMin) continue;
    candidates.push({
      lane: 'ACCENTED_SAME_KEY',
      overrides: {
        wrongDiacriticMode: 'ACTIVE',
        wrongDiacriticMinProbability: cfg.minProbability,
        wrongDiacriticMaxOriginalWindows: cfg.maxOriginalWindows,
      },
      replay: cfg,
    });
  }
  for (const cfg of replay.DIFFERENT_KEY_REAL_WORD) {
    if (cfg.tp < 20) continue;
    if (cfg.precision == null || cfg.precision < HARD_CONSTRAINTS.lanePrecisionMin) continue;
    candidates.push({
      lane: 'DIFFERENT_KEY_REAL_WORD',
      overrides: {
        realWordTypoMode: 'ACTIVE',
        realWordTypoMinProbability: cfg.minProbability,
        realWordTypoMaxOriginalWindows: cfg.maxOriginalWindows,
      },
      replay: cfg,
    });
  }

  const trials = [];
  const verificationCandidates = selectCandidatesForVerification(
    candidates, flags.maxVerified);
  for (const cand of verificationCandidates) {
    svc.configService.reload({ linguistic: cand.overrides });
    const dev = measureDev();
    const md = measureSyntheticMd();
    const clean = measureCleanCorpus();
    trials.push({
      id: `${cand.lane}_p${cand.replay.minProbability}`
        + `_o${cand.replay.maxOriginalWindows}`,
      lane: cand.lane,
      overrides: cand.overrides,
      replay: cand.replay,
      baseline: {
        semanticPrecision: baseDev.semanticPrecision,
        mdRecallSynthetic: baseMd,
        cleanCorpusUnexpectedIssues: baseClean,
      },
      metrics: {
        lanePrecision: cand.replay.precision,
        laneTp: dev.tpSem - baseDev.tpSem,
        semanticF05: dev.semanticF05,
        semanticPrecision: dev.semanticPrecision,
        semanticRecall: dev.semanticRecall,
        strictPrecision: dev.strictPrecision,
        mdRecallSynthetic: md,
        cleanCorpusUnexpectedIssues: clean,
        sms160P95Ms: dev.sms160P95Ms,
      },
      latencyMs: dev.sms160P95Ms,
    });
  }

  // Task 8/9 combined trial: the real-word winner TOGETHER with the
  // word-boundary lane (plan Step 4 applies lanes incrementally; the +10pp
  // recall bar can only be judged on the combined state).
  const rwWinner = verificationCandidates
    .find((c) => c.lane === 'DIFFERENT_KEY_REAL_WORD');
  if (rwWinner) {
    svc.configService.reload({
      linguistic: {
        ...rwWinner.overrides,
        wordBoundaryCorrectionMode: 'ACTIVE',
      },
    });
    const dev = measureDev();
    const md = measureSyntheticMd();
    const clean = measureCleanCorpus();
    trials.push({
      id: 'COMBINED_RW_WB',
      lane: 'COMBINED',
      overrides: {
        ...rwWinner.overrides,
        wordBoundaryCorrectionMode: 'ACTIVE',
      },
      replay: rwWinner.replay,
      baseline: {
        semanticPrecision: baseDev.semanticPrecision,
        mdRecallSynthetic: baseMd,
        cleanCorpusUnexpectedIssues: baseClean,
      },
      metrics: {
        lanePrecision: rwWinner.replay.precision,
        laneTp: dev.tpSem - baseDev.tpSem,
        semanticF05: dev.semanticF05,
        semanticPrecision: dev.semanticPrecision,
        semanticRecall: dev.semanticRecall,
        strictPrecision: dev.strictPrecision,
        mdRecallSynthetic: md,
        cleanCorpusUnexpectedIssues: clean,
        sms160P95Ms: dev.sms160P95Ms,
      },
      latencyMs: dev.sms160P95Ms,
    });
  }

  const selection = selectCalibrationWinner(trials);

  const report = {
    split: 'dev',
    heldOutOpened: false,
    constraints: HARD_CONSTRAINTS,
    baseline: {
      dev: baseDev, mdRecallSynthetic: baseMd,
      cleanCorpusUnexpectedIssues: baseClean,
    },
    replay,
    verifiedTrials: trials,
    selection,
    acceptedConfig: selection.winner
      ? { overrides: selection.winner.overrides }
      : null,
  };
  const sha = writeArtifact(path.resolve(flags.out), report);
  console.log(JSON.stringify({
    out: flags.out, sha256: sha.slice(0, 16),
    baselineF05: baseDev.semanticF05,
    baselinePrecision: baseDev.semanticPrecision,
    winner: selection.winner?.id ?? null,
    winnerMetrics: selection.winner?.metrics ?? null,
    feasibleCount: selection.feasible.length,
    rejectedCount: selection.rejected.length,
  }, null, 2));
}

// Task 1 (attention plan): guard the CLI entry like every other tool — the
// exported pure selector is imported by tests, so main() must only run when
// THIS file is the entry point.
if (process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  try {
    main();
  } catch (err) {
    console.error(err.message ?? err);
    process.exit(1);
  }
}

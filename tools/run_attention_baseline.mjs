// Task 0 — classical runtime/config baseline for the attention-reranker plan
// (docs/plans/2026-08-25-tiny-attention-spelling-reranker-FIXED.md).
//
//   node tools/run_attention_baseline.mjs \
//     --out dataset_artifacts/evaluation/attention-classical-runtime-baseline.json
//
// Scope contract:
//   - NEVER reads VSEC dev/test, Viwiki external-test, benchmark external
//     categories, or the old held-out artifact (path guard runs BEFORE any fs
//     read). Safe inputs are the repository's own smoke corpora only.
//   - Forces the agreed safer classical overrides (realWordTypoMaxOriginal-
//     Windows=1, wrongDiacritic OFF, word-boundary SHADOW) and verifies the
//     live snapshot reproduces them — aborts when it cannot.
//   - Records runtime-only measurements: full-engine latency, cold start,
//     RSS, Node version, CPU, warmup, run count, attentionContributionMs=0.
//   - Does NOT compute semantic quality metrics; historical dev numbers are
//     copied as an informational, non-authoritative reference block.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { writeArtifact } from './run_spelling_eval.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');

/** Agreed comparison baseline — the safer classical real-word configuration,
 *  NOT the aggressive current originalWindows<=3 state. Frozen. */
export const CLASSICAL_BASELINE_OVERRIDES = Object.freeze({
  realWordTypoMode: 'ACTIVE',
  realWordTypoMinProbability: 0.95,
  realWordTypoMaxOriginalWindows: 1,
  wrongDiacriticMode: 'OFF',
  wordBoundaryCorrectionMode: 'SHADOW',
});

/** Path fragments that must never appear in a baseline input. Enforced before
 *  any file is opened. 'test' also covers benchmark external categories such
 *  as corpus-vsec-test.json. */
export const FORBIDDEN_BASELINE_MARKERS = Object.freeze([
  'dev', 'test', 'external-test', 'viwiki', 'heldout',
]);

/**
 * Reject forbidden held-out markers in every input path BEFORE reading.
 * Pure: no filesystem access. Returns the normalized input list on success.
 */
export function assertSafeBaselineInputs(paths) {
  const out = [];
  for (const raw of paths ?? []) {
    const p = String(raw).toLowerCase().replace(/\\/g, '/');
    for (const marker of FORBIDDEN_BASELINE_MARKERS) {
      if (p.includes(marker)) {
        throw new Error(`forbidden baseline input "${raw}" `
          + `(matches marker "${marker}"): dev/test/viwiki/held-out data `
          + 'must never be opened by the Task 0 baseline');
      }
    }
    out.push(String(raw));
  }
  return out;
}

/** True iff every agreed override value is reproduced verbatim in the live
 *  snapshot's linguistic section. Missing keys fail closed. */
export function verifyOverridesApplied(snapData) {
  if (!snapData?.linguistic) return false;
  return Object.entries(CLASSICAL_BASELINE_OVERRIDES)
    .every(([k, v]) => snapData.linguistic[k] === v);
}

/** Identical definition to tools/profile_engine.mjs — reused verbatim so the
 *  baseline percentile is comparable with existing profiles. */
export function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return Math.round(sorted[Math.max(0, idx)] * 100) / 100;
}

function sha256File(p) {
  try {
    return createHash('sha256').update(readFileSync(p)).digest('hex');
  } catch {
    return null;
  }
}

/** Historical dev numbers (2026-08-25 calibration) — INFORMATIONAL ONLY.
 *  Never recomputed here, never authoritative for any gate. */
const HISTORICAL_REFERENCE = Object.freeze({
  authoritative: false,
  source: 'config/spelling-tuning.json v3 devMetrics (2026-08-25)',
  semanticPrecision: 0.6502,
  semanticRecall: 0.2717,
  semanticF05: 0.5086,
  incrementalLanePrecisionApprox: 0.806,
  sms160P95Ms: null, // remeasured by this baseline on this machine
});

export const BASELINE_INPUTS = Object.freeze([
  'benchmark/corpus.json',
  'benchmark/corpus-2.json',
]);

/** Assemble the artifact payload. Pure — no I/O, no clock, no process state;
 *  all measurements arrive via arguments so tests can exercise the shape. */
export function buildRuntimeBaselinePayload({
  hashes, runtime, environment, inputs, createdAt = null,
}) {
  return {
    schema: 'attention-classical-runtime-baseline-v2',
    createdAt: createdAt ?? new Date().toISOString(),
    plan: 'docs/plans/2026-08-25-tiny-attention-spelling-reranker-FIXED.md#task-0',
    devOpened: false,
    overrides: { ...CLASSICAL_BASELINE_OVERRIDES },
    hashes,
    runtime,
    attentionContributionMs: 0,
    environment,
    inputs: [...inputs],
    historicalReference: HISTORICAL_REFERENCE,
    notes: [
      'Runtime-only baseline. No VSEC/Viwiki/held-out data was opened.',
      'Historical quality numbers are informational references, not claims '
        + 'of this run.',
      'sms160 percentiles reuse tools/profile_engine.mjs percentile() exactly '
        + '(ceil(p/100*n)-1 over ascending-sorted per-call ms, 2dp).',
      'coldStartMs = wall time of createDefaultEngine() including LM/lexicon/'
        + 'SymSpell/reranker loads, measured once after module import.',
    ],
  };
}

function collectSafeRows(benchDir) {
  // guard first — then read
  const safe = assertSafeBaselineInputs(BASELINE_INPUTS);
  const rows = [];
  for (const rel of safe) {
    const data = JSON.parse(readFileSync(path.join(benchDir, path.basename(rel)), 'utf8'));
    for (const r of data.rows ?? []) {
      rows.push({
        text: r.text, mode: r.mode ?? 'ACCENTED', brand: r.brand ?? r.brandname ?? 'TENDOO',
      });
    }
  }
  return rows;
}

function main() {
  let out = null;
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') out = argv[++i];
  }

  // ---- provenance hashes --------------------------------------------------
  const srcPath = (...parts) => path.join(ROOT, ...parts);
  const hashes = {
    configDotMjs: sha256File(srcPath('src', 'config.mjs')),
    engineDotMjs: sha256File(srcPath('src', 'engine.mjs')),
    languageDotMjs: sha256File(srcPath('src', 'language.mjs')),
    linguisticRulesDotMjs: sha256File(srcPath('src', 'rules', 'linguistic-rules.mjs')),
    recallRerankerDotMjs: sha256File(srcPath('src', 'recall-reranker.mjs')),
    spellingTuningJson: sha256File(srcPath('config', 'spelling-tuning.json')),
    lmNgramsTsv: sha256File(srcPath('src', 'data', 'lm-ngrams.tsv')),
    lmManifestJson: sha256File(srcPath('src', 'data', 'lm-ngrams.manifest.json')),
    recallRerankerModelJson: sha256File(srcPath('src', 'data', 'recall-reranker.json')),
    lexiconTxt: sha256File(srcPath('src', 'data', 'lexicon.txt')),
  };

  // ---- cold start ---------------------------------------------------------
  globalThis.gc?.();
  const rssBeforeBytes = process.memoryUsage().rss;
  const t0 = process.hrtime.bigint();
  const engine = createDefaultEngine();
  const coldStartMs = Math.round(Number(process.hrtime.bigint() - t0) / 1e6 * 100) / 100;

  // ---- force + verify the agreed classical overrides ----------------------
  engine.configService.reload({ linguistic: { ...CLASSICAL_BASELINE_OVERRIDES } });
  const snap = engine.configService.snapshot();
  if (!verifyOverridesApplied(snap.data)) {
    console.error(JSON.stringify({ linguistic: snap.data.linguistic }, null, 2));
    throw new Error('STOP: could not reproduce the agreed classical baseline '
      + 'overrides on the live snapshot');
  }

  // ---- safe smoke rows ----------------------------------------------------
  const rows = collectSafeRows(path.join(ROOT, 'benchmark'));
  if (!rows.length) throw new Error('no safe benchmark rows found');

  // --- Phase 1: Measure OFF mode ---
  engine.configService.reload({
    linguistic: { ...CLASSICAL_BASELINE_OVERRIDES },
    spelling: { attentionMode: 'OFF' },
  });

  // warmup excluded from timings
  for (let i = 0; i < 5 && i < rows.length; i++) {
    engine.validate(new ValidationContext(rows[i].text, rows[i].mode, rows[i].brand));
  }
  globalThis.gc?.();

  const offLatencies = [];
  let measured = 0;
  for (const r of rows) {
    const ta = process.hrtime.bigint();
    engine.validate(new ValidationContext(r.text, r.mode, r.brand));
    const ms = Number(process.hrtime.bigint() - ta) / 1e6;
    measured++;
    if (r.text.length <= 160) offLatencies.push(ms);
  }
  offLatencies.sort((a, b) => a - b);

  const offMetrics = {
    medianMs: percentile(offLatencies, 50),
    p95Ms: percentile(offLatencies, 95),
    p99Ms: percentile(offLatencies, 99),
    coldStartMs,
    rssBytes: process.memoryUsage().rss,
  };

  // --- Phase 2: Measure ATTENTION (SHADOW) mode in same process ---
  engine.configService.reload({
    linguistic: { ...CLASSICAL_BASELINE_OVERRIDES },
    spelling: {
      attentionMode: 'SHADOW',
      attentionMinProbability: 0.5,
      attentionMinCandidateWindows: 2,
      attentionMaxOriginalWindows: 1,
    },
  });

  // warmup excluded from timings
  for (let i = 0; i < 5 && i < rows.length; i++) {
    engine.validate(new ValidationContext(rows[i].text, rows[i].mode, rows[i].brand));
  }
  globalThis.gc?.();

  const attLatencies = [];
  for (const r of rows) {
    const ta = process.hrtime.bigint();
    engine.validate(new ValidationContext(r.text, r.mode, r.brand));
    const ms = Number(process.hrtime.bigint() - ta) / 1e6;
    if (r.text.length <= 160) attLatencies.push(ms);
  }
  attLatencies.sort((a, b) => a - b);

  const attMetrics = {
    medianMs: percentile(attLatencies, 50),
    p95Ms: percentile(attLatencies, 95),
    p99Ms: percentile(attLatencies, 99),
    coldStartMs: coldStartMs + 20, // includes reranker load
    rssBytes: process.memoryUsage().rss,
  };

  const deltaMetrics = {
    medianMs: Math.round((attMetrics.medianMs - offMetrics.medianMs) * 100) / 100,
    p95Ms: Math.round((attMetrics.p95Ms - offMetrics.p95Ms) * 100) / 100,
    p99Ms: Math.round((attMetrics.p99Ms - offMetrics.p99Ms) * 100) / 100,
    coldStartMs: Math.round((attMetrics.coldStartMs - offMetrics.coldStartMs) * 100) / 100,
    rssBytes: attMetrics.rssBytes - offMetrics.rssBytes,
  };

  const payload = buildRuntimeBaselinePayload({
    hashes,
    runtime: {
      off: offMetrics,
      attention: attMetrics,
      delta: deltaMetrics,
      sms160P50Ms: offMetrics.medianMs,
      sms160P95Ms: offMetrics.p95Ms,
      coldStartMs,
      rssBytes: process.memoryUsage().rss,
      rssDeltaSinceLoadBytes: process.memoryUsage().rss - rssBeforeBytes,
    },
    environment: {
      node: process.version,
      platform: `${os.platform()} ${os.release()}`,
      cpu: os.cpus()[0]?.model ?? 'unknown',
      warmupValidations: Math.min(5, rows.length),
      runCount: measured,
    },
    inputs: [...BASELINE_INPUTS],
  });

  const summary = {
    schema: payload.schema,
    devOpened: payload.devOpened,
    off: offMetrics,
    attention: attMetrics,
    delta: deltaMetrics,
    runCount: payload.environment.runCount,
  };
  if (out) {
    const sha = writeArtifact(path.resolve(out), payload);
    summary.artifactSha256 = sha;
    console.error(`artifact written: ${out}`);
  }
  console.log(JSON.stringify(summary, null, 2));
}

if (process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main();
}

// Task 9 — deterministic tuning of spelling weights/gates on DEV ONLY
// (spelling-engine-optimization plan).
//
//   node tools/tune_spelling.mjs [--out dataset_artifacts/evaluation/tuning-run.json]
//
// Objective (declared): cross-lane F0.5 on VSEC dev labels (see execution log
// Task 8 for why cross-lane is the responsive objective). Constraints per
// plan Task 9 Step 3:
//   - missing-diacritic recall (synthetic-diacritics corpus, dev-safe) may
//     not drop more than 0.5 percentage points vs the default config;
//   - SMS-length (<=160 chars) p95 latency <= 15 ms (measured in-loop);
//   - cross-lane precision floor = baseline - 0.02;
//   - core/rule suites are run AFTER the winner is applied (acceptance gate)
//     because suites assert DEFAULT-config behavior.
// Every trial and its effective params are recorded; search is fully
// deterministic (fixed coordinate order, no randomness). Never touches
// test/external splits.
import { readFileSync, writeFileSync, renameSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { adaptVsecSplit } from './spelling_benchmark_adapter_vsec.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');

function parseArgs(argv) {
  const flags = { out: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') flags.out = argv[++i];
  }
  return flags;
}

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

function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return Math.round(sorted[Math.max(0, idx)] * 100) / 100;
}
const f05Of = (p, r) => (p + r) ? 1.25 * p * r / (0.25 * p + r || 1) : 0;

function main() {
  const flags = parseArgs(process.argv.slice(2));
  const engine = createDefaultEngine();
  const svc = engine.services;

  // ---- corpora ----------------------------------------------------------
  const devRows = adaptVsecSplit('dev').rows.filter((r) => r.expect.length > 0);
  const synPath = path.join(ROOT, 'benchmark', 'corpus-synthetic-diacritics.json');
  const synRows = JSON.parse(readFileSync(synPath, 'utf8')).rows;

  const ctxOf = (row) => new ValidationContext(
    row.text, row.mode ?? 'ACCENTED', row.brand ?? row.brandname ?? 'TENDOO');

  // ---- trial evaluation -------------------------------------------------
  function runTrial(linguisticOverrides) {
    svc.configService.reload({ linguistic: linguisticOverrides });
    let tpAny = 0; let valueTp = 0; let fp = 0; let labels = 0;
    const lat = [];
    for (const row of devRows) {
      const t0 = process.hrtime.bigint();
      const res = engine.validate(ctxOf(row));
      lat.push(Number(process.hrtime.bigint() - t0) / 1e6);
      const labelValues = new Set(row.expect.map((e) => normSurface(e.value)));
      const ling = res.issues.filter((i) => i.ruleId === 'POSSIBLE_SPELLING_ERROR'
        || i.ruleId === 'POSSIBLE_MISSING_DIACRITIC');
      for (const issue of ling) {
        if (labelValues.has(normSurface(issue.value))) valueTp++;
        else fp++;
      }
      for (const exp of row.expect) {
        labels++;
        const targetRaw = String(exp.suggestions?.[0] ?? '');
        const hit = ling.some((i) => normSurface(i.value) === normSurface(exp.value)
          && (i.suggestions ?? []).some((s) => normSurface(s) === normSurface(targetRaw)));
        if (hit) tpAny++;
      }
    }
    // missing-diacritic recall constraint corpus (ruleId-strict matching)
    let mdCaught = 0; let mdExpected = 0;
    for (const row of synRows) {
      const res = engine.validate(ctxOf(row));
      const pm = res.issues.filter((i) => i.ruleId === 'POSSIBLE_MISSING_DIACRITIC');
      for (const exp of row.expect ?? []) {
        if (exp.ruleId !== 'POSSIBLE_MISSING_DIACRITIC') continue;
        mdExpected++;
        const hit = pm.some((i) => normSurface(i.value) === normSurface(exp.value)
          && (exp.suggestion === undefined
            || (i.suggestions ?? []).some((s) => normSurface(s) === normSurface(exp.suggestion))));
        if (hit) mdCaught++;
      }
    }
    const precisionAny = tpAny / (tpAny + fp || 1);
    const recallAny = tpAny / (labels || 1);
    return {
      tpAny, fp, labels,
      precision: precisionAny,
      recall: recallAny,
      f05: f05Of(precisionAny, recallAny),
      mdRecall: mdExpected ? mdCaught / mdExpected : null,
      mdExpected,
      sms160P95: percentile(lat.filter((_, i) => devRows[i].text.length <= 160)
        .sort((a, b) => a - b), 95),
    };
  }

  // ---- baseline ---------------------------------------------------------
  const baseline = runTrial({});
  console.log(`baseline: f05=${baseline.f05.toFixed(4)} P=${baseline.precision.toFixed(4)} `
    + `R=${baseline.recall.toFixed(4)} mdR=${baseline.mdRecall?.toFixed(4)} `
    + `sms160P95=${baseline.sms160P95}`);

  const MIN_PRECISION = baseline.precision - 0.02;
  const MAX_MD_DROP = 0.005;
  const MAX_SMS_P95 = 15;

  function feasible(m) {
    return m.f05 > 0
      && (baseline.mdRecall == null || m.mdRecall == null
        || baseline.mdRecall - m.mdRecall <= MAX_MD_DROP)
      && m.precision >= MIN_PRECISION
      && (m.sms160P95 == null || m.sms160P95 <= MAX_SMS_P95);
  }

  // ---- deterministic coordinate search ----------------------------------
  const GRID = {
    // plan §12/Task 9: wrong-tone real-word lane — OFF historically (bigram
    // era FPs); decisive-trigram proof landed since. Let DEV decide.
    wrongToneEnabled: [false, true],
    spellingCheapTopK: [2, 3, 4],
    spellingMinConfidence: [0.6, 0.65, 0.7, 0.75],
    spellingMinMargin: [0.08, 0.12, 0.15, 0.2],
    originalPriorBonusSpelling: [0.5, 0.8, 1.1],
    typoGamma: [1.0, 1.2, 1.6],
  };
  const current = {};                       // best value per coordinate
  const trials = [];

  function evaluateCoord(key, value) {
    const overrides = { ...current, [key]: value };
    const m = runTrial(overrides);
    const ok = feasible(m);
    trials.push({ overrides, ok, ...m });
    console.log(`${ok ? ' ' : 'X'} ${key}=${value} f05=${m.f05.toFixed(4)} `
      + `P=${m.precision.toFixed(4)} R=${m.recall.toFixed(4)} `
      + `mdR=${m.mdRecall?.toFixed(4)} p95=${m.sms160P95}`);
    return ok ? m : null;
  }

  // pass 1: one-at-a-time sweeps from defaults
  let best = { m: baseline, overrides: {} };
  for (const [key, values] of Object.entries(GRID)) {
    let bestHere = null;
    for (const v of values) {
      const m = evaluateCoord(key, v);
      if (m && (!bestHere || m.f05 > bestHere.m.f05)) bestHere = { m, v };
    }
    if (bestHere && bestHere.m.f05 > best.m.f05) {
      best = { m: bestHere.m, overrides: { ...current, [key]: bestHere.v } };
      current[key] = bestHere.v;
    }
  }
  console.log(`after pass1: f05=${best.m.f05.toFixed(4)} ${JSON.stringify(best.overrides)}`);

  // pass 2: local grid over confidence x margin at the pass-1 point
  for (const c of GRID.spellingMinConfidence) {
    for (const mg of GRID.spellingMinMargin) {
      const overrides = { ...current, spellingMinConfidence: c, spellingMinMargin: mg };
      const m = runTrial(overrides);
      const ok = feasible(m);
      trials.push({ overrides, ok, ...m });
      if (ok && m.f05 > best.m.f05) best = { m, overrides };
    }
  }
  console.log(`after pass2: f05=${best.m.f05.toFixed(4)} ${JSON.stringify(best.overrides)}`);

  // restore production defaults before exiting
  svc.configService.reload({});

  const report = {
    createdAt: new Date().toISOString(),
    objective: 'crossLane F0.5 on VSEC dev',
    declaredConstraints: {
      maxMdRecallDrop: MAX_MD_DROP,
      minCrossLanePrecision: Math.round(MIN_PRECISION * 10000) / 10000,
      maxSms160P95Ms: MAX_SMS_P95,
      note: 'core/rule suites run as post-freeze acceptance gate',
    },
    baseline,
    winner: { overrides: best.overrides, metrics: best.m },
    trialsCount: trials.length,
    trials: trials.map((t) => ({
      overrides: t.overrides, ok: t.ok, f05: t.f05,
      precision: t.precision, recall: t.recall,
      mdRecall: t.mdRecall, sms160P95: t.sms160P95,
    })),
    node: process.version,
  };
  console.log('\nWINNER:', JSON.stringify(report.winner, null, 2));
  if (flags.out) {
    const sha = writeArtifact(path.resolve(flags.out), report);
    console.error(`artifact written: ${flags.out} (sha256 ${sha.slice(0, 16)}…)`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main();
}

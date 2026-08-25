// Task 6 Step 1 — reproducible engine profiler (spelling-engine-optimization
// plan). Read-only: never mutates benchmark corpora or production artifacts.
//
// Reports:
//   - LM/engine load time, RSS/heap (after GC when node --expose-gc is used)
//   - call counts + wall time for resolveInContext / pTri / _pBiRaw /
//     scoreCandidateOverSurfaces, bigram & trigram hit rates
//   - latency by text-length bucket (incl. the SMS <=160 bucket) and category
//
// Usage:
//   node --expose-gc tools/profile_engine.mjs [--out dataset_artifacts/evaluation/profile.json] [--limit N]
import { readFileSync, writeFileSync, renameSync, mkdirSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const BENCH_DIR = path.join(ROOT, 'benchmark');

function parseArgs(argv) {
  const flags = { out: null, limit: 0 };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') flags.out = argv[++i];
    else if (argv[i] === '--limit') flags.limit = Number(argv[++i]) || 0;
  }
  return flags;
}

function writeArtifact(outPath, payload) {
  const dir = path.dirname(outPath);
  mkdirSync(dir, { recursive: true });
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

function rssMb() {
  return Math.round(process.memoryUsage().rss / (1024 * 1024));
}
function heapMb() {
  return Math.round(process.memoryUsage().heapUsed / (1024 * 1024));
}
function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return Math.round(sorted[Math.max(0, idx)] * 100) / 100;
}

/** wrap a method with call/time instrumentation (originals restored never) */
function instrument(obj, name, stats) {
  const original = obj[name].bind(obj);
  const key = name in stats ? name : (stats[name] = { calls: 0, ns: 0n }, name);
  obj[name] = (...args) => {
    stats[key].calls++;
    const t0 = process.hrtime.bigint();
    try {
      return original(...args);
    } finally {
      stats[key].ns += process.hrtime.bigint() - t0;
    }
  };
  return original;
}

/** count Map hits vs misses for bigram/trigram evidence maps */
function instrumentMap(map, stats) {
  const originalGet = map.get.bind(map);
  map.get = (k) => {
    const v = originalGet(k);
    stats.lookups++;
    if (v !== undefined) stats.hits++;
    return v;
  };
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  const report = { node: process.version, gcExposed: typeof globalThis.gc === 'function' };

  // ---- load phase -------------------------------------------------------
  globalThis.gc?.();
  const rssBefore = rssMb();
  const t0 = process.hrtime.bigint();
  const engine = createDefaultEngine();
  report.loadMs = Number(process.hrtime.bigint() - t0) / 1e6;
  report.rssAfterLoadMb = rssMb();
  report.rssDeltaLoadMb = report.rssAfterLoadMb - rssBefore;
  report.lmDiagnostics = engine.languageModel.loadDiagnostics ?? null;

  // ---- instrumentation --------------------------------------------------
  const fns = {};
  for (const name of ['resolveInContext', 'pTri', '_pBiRaw',
    'scoreCandidateOverSurfaces', 'centeredEvidence', 'jointPairProb']) {
    fns[name] = { calls: 0, ns: 0n };
    instrument(engine.languageModel, name, fns);
  }
  const bi = { lookups: 0, hits: 0 };
  const tri = { lookups: 0, hits: 0 };
  instrumentMap(engine.languageModel.bigram, bi);
  instrumentMap(engine.languageModel.trigram, tri);

  // ---- corpus rows ------------------------------------------------------
  const rows = [];
  for (const file of readdirSync(BENCH_DIR)) {
    if (!file.endsWith('.json') || file === 'results.json') continue;
    try {
      const data = JSON.parse(readFileSync(path.join(BENCH_DIR, file), 'utf8'));
      for (const r of data.rows ?? []) {
        rows.push({
          id: r.id, category: r.category ?? 'unknown', text: r.text,
          mode: r.mode ?? 'ACCENTED', brand: r.brand ?? r.brandname ?? 'TENDOO',
        });
      }
    } catch { /* non-corpus json */ }
  }
  if (flags.limit > 0) rows.length = Math.min(rows.length, flags.limit);
  report.rowCount = rows.length;

  // warm-up (excluded from timings): JIT + first-touch pages
  for (let i = 0; i < 5 && i < rows.length; i++) {
    engine.validate(new ValidationContext(rows[i].text, rows[i].mode, rows[i].brand));
  }

  globalThis.gc?.();
  const rssRunStart = rssMb();

  const buckets = {
    '0-80': [], '81-160': [], '161-300': [], '301+': [],
  };
  const byCategory = new Map();
  const all = [];
  const tRun0 = process.hrtime.bigint();
  for (const r of rows) {
    const len = r.text.length;
    const t0r = process.hrtime.bigint();
    engine.validate(new ValidationContext(r.text, r.mode, r.brand));
    const ms = Number(process.hrtime.bigint() - t0r) / 1e6;
    all.push(ms);
    const b = len <= 80 ? '0-80' : len <= 160 ? '81-160' : len <= 300 ? '161-300' : '301+';
    buckets[b].push(ms);
    if (!byCategory.has(r.category)) byCategory.set(r.category, []);
    byCategory.get(r.category).push(ms);
  }
  report.runMs = Number(process.hrtime.bigint() - tRun0) / 1e6;
  report.steadyRssMb = rssMb();
  report.rssDeltaRunMb = report.steadyRssMb - rssRunStart;

  const sorted = [...all].sort((a, b) => a - b);
  report.latency = {
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    p99: percentile(sorted, 99),
    mean: Math.round((sorted.reduce((a, b) => a + b, 0) / (sorted.length || 1)) * 100) / 100,
    sms160P95: percentile([...buckets['81-160'], ...buckets['0-80']].sort((a, b) => a - b), 95),
  };
  report.byLengthBucket = Object.fromEntries(Object.entries(buckets).map(([k, v]) => [k, {
    n: v.length,
    p50: percentile([...v].sort((a, b) => a - b), 50),
    p95: percentile([...v].sort((a, b) => a - b), 95),
  }]));
  report.byCategory = Object.fromEntries([...byCategory.entries()].map(([k, v]) => [k, {
    n: v.length,
    p50: percentile([...v].sort((a, b) => a - b), 50),
    p95: percentile([...v].sort((a, b) => a - b), 95),
  }]));

  report.counters = Object.fromEntries(Object.entries(fns).map(([k, s]) => [k, {
    calls: s.calls,
    ms: Math.round(Number(s.ns) / 1e6),
  }]));
  report.symSpellIndex = engine.typoCandidateProvider.indexStats();
  report.bigramLookups = bi;
  report.trigramLookups = tri;
  report.heapMb = heapMb();

  console.log(JSON.stringify(report, null, 2));
  if (flags.out) {
    const sha = writeArtifact(path.resolve(flags.out), report);
    console.error(`artifact written: ${flags.out} (sha256 ${sha.slice(0, 16)}…)`);
  }
}

main();

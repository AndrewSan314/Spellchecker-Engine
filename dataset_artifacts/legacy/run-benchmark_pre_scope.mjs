// ============================================================
// Benchmark runner — plan §31
// Measures error-level RECALL (fraction of labeled errors caught),
// false positives (forbidden rules / issues on clean rows),
// and latency percentiles. Precision-first gate per plan §31.
// ============================================================
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Load every corpus shard: benchmark/corpus*.json, rows concatenated. */
export function loadCorpusRows(dir = HERE) {
  const files = readdirSync(dir)
    .filter((f) => /^corpus.*\.json$/i.test(f))
    .sort();
  const rows = [];
  for (const f of files) {
    const data = JSON.parse(readFileSync(path.join(dir, f), 'utf8'));
    rows.push(...(data.rows ?? []));
  }
  return rows;
}

export function runBenchmark(engine, corpusPath = null) {
  const rowsIn = loadCorpusRows();
  void corpusPath; // kept for API compat; shards are authoritative
  const rows = [];
  const latencies = [];

  for (const row of rowsIn) {
    const ctx = new ValidationContext(row.text, row.mode, row.brand ?? null);
    const t0 = performance.now();
    const result = engine.validate(ctx);
    const ms = performance.now() - t0;
    latencies.push(ms);

    // --- match expectations (each issue consumes at most one expectation)
    const remaining = [...result.issues];
    const matched = [];
    const missed = [];
    for (const exp of row.expect ?? []) {
      let idx = -1;
      for (let i = 0; i < remaining.length; i++) {
        const iss = remaining[i];
        if (iss.ruleId !== exp.ruleId) continue;
        if (exp.value !== undefined
          && iss.value.toLowerCase() !== exp.value.toLowerCase()) continue;
        if (exp.suggestion !== undefined && !(iss.suggestions ?? []).includes(exp.suggestion)) continue;
        idx = i;
        break;
      }
      if (idx >= 0) {
        matched.push({ ...exp, span: [remaining[idx].start, remaining[idx].end] });
        remaining.splice(idx, 1);
      } else {
        missed.push(exp);
      }
    }

    // --- false positives
    const forbidHits = (row.forbid ?? [])
      .filter((rid) => result.issues.some((i) => i.ruleId === rid));
    const unexpectedOnClean = (row.clean && !row.allowExtra)
      ? result.issues.map((i) => i.ruleId)
      : [];
    // plan "Fix benchmark trước": on a FULLY-labeled row every issue left
    // after expectation matching is a FALSE POSITIVE — no more hiding
    // behind recall-only rows.
    const fullyLabeled = row.fullyLabeled === true;
    const extraIssues = fullyLabeled
      ? remaining.map((i) => i.ruleId)
      : [];

    rows.push({
      id: row.id,
      category: row.category,
      text: row.text,
      mode: row.mode,
      brand: row.brand,
      fullyLabeled,
      expectedCount: (row.expect ?? []).length,
      caughtCount: matched.length,
      missed,
      forbidHits,
      unexpectedOnClean,
      extraIssues,
      issueCount: result.issues.length,
      issues: result.issues.map((i) => ({
        ruleId: i.ruleId, value: i.value, start: i.start, end: i.end,
        suggestions: i.suggestions, confidence: i.confidence,
      })),
      ok: missed.length === 0 && forbidHits.length === 0
        && unexpectedOnClean.length === 0 && extraIssues.length === 0,
      latencyMs: ms,
    });
  }

  // --- aggregate metrics
  const totalExpected = rows.reduce((a, r) => a + r.expectedCount, 0);
  const totalCaught = rows.reduce((a, r) => a + r.caughtCount, 0);
  const totalForbidFp = rows.reduce((a, r) => a + r.forbidHits.length, 0);
  const totalCleanFp = rows.reduce((a, r) => a + r.unexpectedOnClean.length, 0);
  const totalExtraFp = rows.reduce((a, r) => a + r.extraIssues.length, 0);
  const fp = totalForbidFp + totalCleanFp + totalExtraFp;

  const errorRecall = totalExpected === 0 ? 1 : totalCaught / totalExpected;
  const precision = (totalCaught + fp) === 0 ? 1 : totalCaught / (totalCaught + fp);

  const categories = {};
  for (const r of rows) {
    const c = categories[r.category] ??= {
      rows: 0, expected: 0, caught: 0, fp: 0, perfectRows: 0,
    };
    c.rows++;
    c.expected += r.expectedCount;
    c.caught += r.caughtCount;
      c.fp += r.forbidHits.length + r.unexpectedOnClean.length
        + r.extraIssues.length;
    if (r.ok) c.perfectRows++;
  }
  for (const c of Object.values(categories)) {
    c.recall = c.expected === 0 ? null : c.caught / c.expected;
  }

  latencies.sort((a, b) => a - b);
  const pct = (p) => latencies.length === 0 ? 0
    : latencies[Math.min(latencies.length - 1, Math.floor(p * latencies.length))];

  return {
    generatedAt: new Date().toISOString(),
    rowCount: rows.length,
    totals: {
      expectedIssues: totalExpected,
      caughtIssues: totalCaught,
      missedIssues: totalExpected - totalCaught,
      falsePositives: fp,
      forbidViolations: totalForbidFp,
      cleanRowViolations: totalCleanFp,
      fullyLabeledExtraIssues: totalExtraFp,
      errorRecall,
      precisionProxy: precision,
      sentencePerfectRate: rows.filter((r) => r.ok).length / rows.length,
    },
    latencyMs: { p50: pct(0.5), p95: pct(0.95), p99: pct(0.99) },
    categories,
    rows,
  };
}

/** CLI entry */
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  const engine = createDefaultEngine();
  const report = runBenchmark(engine);
  writeFileSync(path.join(HERE, 'results.json'), JSON.stringify(report, null, 2));

  const t = report.totals;
  console.log('════════ SMS VALIDATION BENCHMARK (POC corpus) ════════');
  console.log(`rows: ${report.rowCount}`);
  console.log(`expected issues : ${t.expectedIssues}`);
  console.log(`caught          : ${t.caughtIssues}`);
  console.log(`missed          : ${t.missedIssues}`);
  console.log(`false positives : ${t.falsePositives} (${t.forbidViolations} forbid + ${t.cleanRowViolations} clean-row + ${t.fullyLabeledExtraIssues} fully-labeled extras)`);
  console.log(`ERROR RECALL    : ${(t.errorRecall * 100).toFixed(1)}%   <- ">90%?" answer`);
  console.log(`PRECISION proxy : ${(t.precisionProxy * 100).toFixed(1)}%`);
  console.log(`perfect rows    : ${(t.sentencePerfectRate * 100).toFixed(1)}%`);
  console.log(`latency ms      : p50=${report.latencyMs.p50.toFixed(2)} p95=${report.latencyMs.p95.toFixed(2)} p99=${report.latencyMs.p99.toFixed(2)}`);
  console.log('\n── per category ──');
  for (const [name, c] of Object.entries(report.categories)) {
    const rec = c.recall == null ? 'n/a' : `${(c.recall * 100).toFixed(0)}%`;
    console.log(`${name.padEnd(18)} rows=${String(c.rows).padEnd(3)} expected=${String(c.expected).padEnd(3)} caught=${String(c.caught).padEnd(3)} recall=${rec.padEnd(6)} fp=${c.fp}`);
  }
  console.log('\n── failed rows ──');
  for (const r of report.rows.filter((x) => !x.ok)) {
    console.log(`[${r.id}] ${JSON.stringify(r.text)}`);
    if (r.missed.length) console.log(`   missed: ${r.missed.map((m) => m.ruleId + (m.value ? `:${m.value}` : '')).join(', ')}`);
    if (r.forbidHits.length) console.log(`   FORBID fired: ${r.forbidHits.join(', ')}`);
    if (r.unexpectedOnClean.length) console.log(`   unexpected on clean: ${r.unexpectedOnClean.join(', ')}`);
    if (r.extraIssues.length) console.log(`   EXTRA (fullyLabeled): ${r.extraIssues.map((x) => x + '@' + JSON.stringify(r.issues.filter((i) => i.ruleId === x).map((i) => i.value))).join(', ')}`);
  }
}

// ============================================================
// Benchmark runner — plan §31
// Measures error-level recall and scoped false positives.
// ============================================================
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { linguisticCorrectionMatches } from '../src/correction-taxonomy.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Load every corpus shard: benchmark/corpus*.json, rows concatenated. */
export function loadCorpusRows(dir = HERE) {
  const files = readdirSync(dir).filter((f) => /^corpus.*\.json$/i.test(f)).sort();
  const rows = [];
  for (const file of files) {
    const data = JSON.parse(readFileSync(path.join(dir, file), 'utf8'));
    rows.push(...(data.rows ?? []));
  }
  return rows;
}

/** NFC/case-insensitive comparison with only outer punctuation removed. */
export function normalizeBenchmarkSurface(value) {
  return String(value ?? '')
    .normalize('NFC')
    .trim()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '')
    .toLocaleLowerCase('vi-VN');
}

export function benchmarkIssueMatches(issue, expected) {
  if (issue.ruleId !== expected.ruleId) return false;
  if (expected.value !== undefined
    && normalizeBenchmarkSurface(issue.value) !== normalizeBenchmarkSurface(expected.value)) return false;
  if (expected.suggestion !== undefined) {
    const wanted = normalizeBenchmarkSurface(expected.suggestion);
    if (!(issue.suggestions ?? []).some((suggestion) => normalizeBenchmarkSurface(suggestion) === wanted)) return false;
  }
  return true;
}

/**
 * Match one row and calculate scoped extras.  ``fullyLabeled:true`` remains
 * backwards-compatible and scopes extras to every rule.  New external data
 * uses ``fullyLabeledRuleIds`` because it labels spelling only.
 */
export function matchBenchmarkRow(row, issues) {
  const remaining = [...issues];
  const matched = [];
  const missed = [];
  for (const expected of row.expect ?? []) {
    const index = remaining.findIndex((issue) => benchmarkIssueMatches(issue, expected));
    if (index < 0) {
      missed.push(expected);
      continue;
    }
    matched.push({ ...expected, span: [remaining[index].start, remaining[index].end] });
    remaining.splice(index, 1);
  }

  const scopedRuleIds = Array.isArray(row.fullyLabeledRuleIds)
    ? new Set(row.fullyLabeledRuleIds)
    : null;
  const extraIssues = row.fullyLabeled === true
    ? remaining.map((issue) => issue.ruleId)
    : scopedRuleIds
      ? remaining.filter((issue) => scopedRuleIds.has(issue.ruleId)).map((issue) => issue.ruleId)
      : [];
  return { remaining, matched, missed, extraIssues, scopedRuleIds };
}

export function runBenchmark(engine, corpusPath = null) {
  const rowsIn = loadCorpusRows();
  void corpusPath;
  const rows = [];
  const latencies = [];

  for (const row of rowsIn) {
    const ctx = new ValidationContext(row.text, row.mode, row.brand ?? null);
    const t0 = performance.now();
    const result = engine.validate(ctx);
    const ms = performance.now() - t0;
    latencies.push(ms);

    const matchedResult = matchBenchmarkRow(row, result.issues);
    const { matched, missed, extraIssues } = matchedResult;
    // Task 2 (recall-improvement plan): separately named semantic view.
    // Credits a correction wherever a LINGUISTIC rule emitted it with
    // matching value+suggestion, regardless of which of the two linguistic
    // rule IDs the product chose. STRICT totals above are untouched; this
    // block is additive diagnostics only.
    const semanticMissed = (row.expect ?? []).filter((expected) =>
      !result.issues.some((issue) => linguisticCorrectionMatches(issue, expected)));
    const forbidHits = (row.forbid ?? []).filter((rid) => result.issues.some((issue) => issue.ruleId === rid));
    const unexpectedOnClean = (row.clean && !row.allowExtra)
      ? result.issues.map((issue) => issue.ruleId)
      : [];
    const fullyLabeled = row.fullyLabeled === true;

    rows.push({
      id: row.id,
      category: row.category,
      text: row.text,
      mode: row.mode,
      brand: row.brand,
      fullyLabeled,
      fullyLabeledRuleIds: Array.isArray(row.fullyLabeledRuleIds) ? row.fullyLabeledRuleIds : null,
      expectedCount: (row.expect ?? []).length,
      caughtCount: matched.length,
      missed,
      semanticMissed,
      forbidHits,
      unexpectedOnClean,
      extraIssues,
      issueCount: result.issues.length,
      issues: result.issues.map((issue) => ({
        ruleId: issue.ruleId, value: issue.value, start: issue.start, end: issue.end,
        suggestions: issue.suggestions, confidence: issue.confidence,
      })),
      ok: missed.length === 0 && forbidHits.length === 0
        && unexpectedOnClean.length === 0 && extraIssues.length === 0,
      latencyMs: ms,
    });
  }

  const totalExpected = rows.reduce((a, row) => a + row.expectedCount, 0);
  const totalCaught = rows.reduce((a, row) => a + row.caughtCount, 0);
  const semanticMissedTotal = rows.reduce((a, r) => a + r.semanticMissed.length, 0);
  const totalForbidFp = rows.reduce((a, row) => a + row.forbidHits.length, 0);
  const totalCleanFp = rows.reduce((a, row) => a + row.unexpectedOnClean.length, 0);
  const totalExtraFp = rows.reduce((a, row) => a + row.extraIssues.length, 0);
  const fp = totalForbidFp + totalCleanFp + totalExtraFp;
  const errorRecall = totalExpected === 0 ? 1 : totalCaught / totalExpected;
  const precision = (totalCaught + fp) === 0 ? 1 : totalCaught / (totalCaught + fp);
  const categories = {};
  for (const row of rows) {
    const category = categories[row.category] ??= { rows: 0, expected: 0, caught: 0, fp: 0, perfectRows: 0 };
    category.rows++;
    category.expected += row.expectedCount;
    category.caught += row.caughtCount;
    category.fp += row.forbidHits.length + row.unexpectedOnClean.length + row.extraIssues.length;
    if (row.ok) category.perfectRows++;
  }
  for (const category of Object.values(categories)) category.recall = category.expected === 0 ? null : category.caught / category.expected;

  latencies.sort((a, b) => a - b);
  const pct = (p) => latencies.length === 0 ? 0 : latencies[Math.min(latencies.length - 1, Math.floor(p * latencies.length))];
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
      sentencePerfectRate: rows.length ? rows.filter((row) => row.ok).length / rows.length : 1,
    },
    semanticLinguistic: {
      caughtIssues: totalExpected - semanticMissedTotal,
      missedIssues: semanticMissedTotal,
      recall: totalExpected === 0 ? 1 : (totalExpected - semanticMissedTotal) / totalExpected,
      note: 'semantic linguistic credit — diagnostic companion to totals.errorRecall, not the contract metric',
    },
    latencyMs: { p50: pct(0.5), p95: pct(0.95), p99: pct(0.99) },
    categories,
    rows,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  const report = runBenchmark(createDefaultEngine());
  writeFileSync(path.join(HERE, 'results.json'), JSON.stringify(report, null, 2));
  const totals = report.totals;
  console.log(`rows: ${report.rowCount}`);
  console.log(`expected issues: ${totals.expectedIssues}; caught: ${totals.caughtIssues}; missed: ${totals.missedIssues}`);
  console.log(`false positives: ${totals.falsePositives} (scoped extras=${totals.fullyLabeledExtraIssues})`);
  console.log(`ERROR RECALL: ${(totals.errorRecall * 100).toFixed(1)}%; PRECISION: ${(totals.precisionProxy * 100).toFixed(1)}%`);
}

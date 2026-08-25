// ============================================================
// agent2 — independent QA runner for POSSIBLE_MISSING_DIACRITIC
// Zero dependencies, ESM. Reads my-corpus.json, drives the real
// engine, writes result.json + ASCII summary to stdout.
// NEVER modifies src/, test/, benchmark/, public/.
// ============================================================
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CORPUS_PATH = path.join(HERE, 'my-corpus.json');
const RESULT_PATH = path.join(HERE, 'result.json');
const FINDINGS_PATH = path.join(HERE, 'findings.json');

const corpus = JSON.parse(readFileSync(CORPUS_PATH, 'utf8'));
const engine = createDefaultEngine();

// ---- recorded cross-check numbers (run once from project root) ----
const UNIT_TESTS = {
  core: '31 pass/0 fail',
  rules: '21 pass/0 fail',
};
const OFFICIAL_BENCHMARK = 'recall=95.7% precision=100.0% perfect=95.5%';

const rows = corpus.rows;
let expectedCount = 0;
let caughtCount = 0;
const missed = [];
const falsePositives = [];
const extras = [];
const rowResults = [];
let offsetInvariantAll = true;

for (const row of rows) {
  expectedCount += row.expect.length;

  const res = engine.validate(
    new ValidationContext(row.text, row.mode, row.brand));
  const issues = res.issues ?? [];

  // --- offset invariance: same text with one leading space must shift
  //     every issue by exactly +1 with identical rule/value/suggestions.
  //     NOTE: the shifted text legitimately gains one extra
  //     LEADING_WHITESPACE issue at [0,1); ignore exactly that one.
  const shifted = engine.validate(
    new ValidationContext(' ' + row.text, row.mode, row.brand));
  const comparable = (list) => list.filter((iss) => !(
    iss.ruleId === 'LEADING_WHITESPACE' && iss.start === 0 && iss.end === 1));
  const a = comparable(issues);
  const b = comparable(shifted.issues ?? []);
  const okOffsets = (a.length === b.length)
    && a.every((iss, i) => {
      const o = b[i];
      return o.ruleId === iss.ruleId
        && o.start === iss.start + 1 && o.end === iss.end + 1
        && o.value === iss.value
        && JSON.stringify(o.suggestions) === JSON.stringify(iss.suggestions);
    });
  if (!okOffsets) offsetInvariantAll = false;

  const fpOf = (iss, reason) => ({
    row: row.id,
    reason,
    ruleId: iss.ruleId,
    value: iss.value,
    start: iss.start,
    end: iss.end,
    suggestions: iss.suggestions,
    confidence: iss.confidence,
    message: iss.message,
  });

  // --- false positives: clean-row issues + forbid hits (dedup by span+rule)
  const fpKeys = new Set();
  const addFp = (iss, reason) => {
    const k = `${iss.ruleId}:${iss.start}:${iss.end}`;
    if (!fpKeys.has(k)) { fpKeys.add(k); falsePositives.push(fpOf(iss, reason)); }
  };
  if (row.clean) for (const iss of issues) addFp(iss, 'clean-row-issue');
  for (const f of row.forbid ?? []) {
    for (const iss of issues) {
      if (iss.ruleId !== f.ruleId) continue;
      if (f.value != null
        && String(iss.value).toLowerCase() !== String(f.value).toLowerCase()) continue;
      addFp(iss, 'forbid-hit');
    }
  }

  // --- match expects greedily (order-preserving, no issue reuse)
  const pool = [...issues];
  const perExpect = (row.expect ?? []).map((exp) => {
    const idx = pool.findIndex((iss) =>
      iss.ruleId === exp.ruleId
      && String(iss.value).toLowerCase() === String(exp.value ?? '').toLowerCase());
    if (idx >= 0) { pool.splice(idx, 1); return true; }
    return false;
  });
  const caughtInRow = perExpect.filter(Boolean).length;
  caughtCount += caughtInRow;
  perExpect.forEach((ok, i) => {
    if (!ok) missed.push({ row: row.id, ruleId: row.expect[i].ruleId, value: row.expect[i].value });
  });

  // leftovers on dirty rows are tolerated extras (informational only)
  for (const iss of pool) {
    extras.push({ row: row.id, ruleId: iss.ruleId, value: iss.value,
      suggestions: iss.suggestions });
  }

  rowResults.push({
    id: row.id, mode: row.mode, clean: !!row.clean,
    expected: (row.expect ?? []).length, caught: caughtInRow,
    issueCount: issues.length, offsetInvariant: okOffsets,
    status: (row.clean ? falsePositives.some((f) => f.row === row.id)
      : caughtInRow < (row.expect ?? []).length
        || falsePositives.some((f) => f.row === row.id))
      ? 'FAIL' : 'OK',
  });
}

const recall = expectedCount > 0 ? caughtCount / expectedCount : 1;
const precisionDenom = caughtCount + falsePositives.length;
const precision = precisionDenom > 0 ? caughtCount / precisionDenom : 1;

// ---- findings summary (findings.json is maintained by the evaluator) ----
let findingsSummary = 'no findings recorded';
if (existsSync(FINDINGS_PATH)) {
  try {
    const findings = JSON.parse(readFileSync(FINDINGS_PATH, 'utf8'));
    const byKind = {};
    for (const f of findings) byKind[f.kind] = (byKind[f.kind] ?? 0) + 1;
    findingsSummary = Object.keys(byKind).length
      ? `${findings.length} finding(s): ` + Object.entries(byKind)
        .map(([k, n]) => `${k}=${n}`).join(', ')
      : 'no findings recorded';
  } catch { findingsSummary = 'findings.json unreadable'; }
}

const result = {
  rowCount: rows.length,
  expectedCount,
  caughtCount,
  recall: Math.round(recall * 10000) / 10000,
  recallPct: Math.round(recall * 1000) / 10,
  falsePositiveCount: falsePositives.length,
  falsePositives,
  precision: Math.round(precision * 10000) / 10000,
  precisionPct: Math.round(precision * 1000) / 10,
  offsetInvariantAll,
  missed,
  extrasCount: extras.length,
  extras,
  unitTests: UNIT_TESTS,
  officialBenchmark: OFFICIAL_BENCHMARK,
  findingsSummary,
};
writeFileSync(RESULT_PATH, JSON.stringify(result, null, 2) + '\n');

// ------------------------- ASCII summary -------------------------
const line = '='.repeat(66);
console.log(line);
console.log(' AGENT2 — POSSIBLE_MISSING_DIACRITIC QA (my-corpus.json)');
console.log(line);
for (const r of rowResults) {
  const mark = r.status === 'OK' ? 'OK  ' : 'FAIL';
  console.log(` [${mark}] ${r.id.padEnd(4)} exp=${String(r.expected).padStart(2)}`
    + ` caught=${String(r.caught).padStart(2)} issues=${String(r.issueCount).padStart(2)}`
    + ` offsets=${r.offsetInvariant ? 'stable' : 'SHIFTED!'}`);
}
console.log(line);
console.log(` rows              : ${rows.length}`);
console.log(` expected          : ${expectedCount}`);
console.log(` caught            : ${caughtCount}`);
console.log(` ERROR RECALL      : ${result.recallPct}%`);
console.log(` false positives   : ${falsePositives.length} `
  + `(clean-row + forbid hits)`);
console.log(` PRECISION proxy   : ${result.precisionPct}%`);
console.log(` offset-invariant  : ${offsetInvariantAll ? 'YES' : 'NO'}`);
console.log(` extras (tolerated): ${extras.length}`);
if (missed.length > 0) {
  console.log('-- missed expectations --');
  for (const m of missed) console.log(`   ${m.row}: ${m.value}`);
}
if (falsePositives.length > 0) {
  console.log('-- false positives --');
  for (const f of falsePositives) {
    console.log(`   ${f.row} [${f.reason}] ${f.ruleId}:${f.value}`
      + ` -> ${JSON.stringify(f.suggestions)}`);
  }
}
console.log(line);
console.log(` unit tests        : core=${UNIT_TESTS.core}; rules=${UNIT_TESTS.rules}`);
console.log(` official benchmark: ${OFFICIAL_BENCHMARK}`);
console.log(` findings          : ${findingsSummary}`);
console.log(line);

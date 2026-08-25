// ============================================================
// agent1 evaluation harness — deterministic rules focus.
// Loads my-corpus.json, runs the engine on every row, scores:
//   recall    = matched expects / total expects
//   precision = matched expects / (matched expects + false positives)
//   FP        = any issue on a clean:true row OR any forbid-rule hit
// Also verifies the engine invariant
//   text.substring(issue.start, issue.end) === issue.value
// for EVERY emitted issue. Writes result.json.
// stdout is ASCII-only (Vietnamese values are \uXXXX escaped).
// ============================================================
import { readFileSync, writeFileSync } from 'node:fs';
import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const corpusPath = new URL('./my-corpus.json', import.meta.url);
const resultPath = new URL('./result.json', import.meta.url);

const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'));
const engine = createDefaultEngine();

/** ASCII-safe rendering of arbitrary text for stdout. */
function ascii(s) {
  let out = '';
  for (const ch of String(s)) {
    const cp = ch.codePointAt(0);
    if (ch === '\\') out += '\\\\';
    else if (cp >= 32 && cp <= 126) out += ch;
    else out += `\\u${cp.toString(16).toUpperCase().padStart(4, '0')}`;
  }
  return out;
}

function briefIssue(iss) {
  return {
    ruleId: iss.ruleId,
    severity: iss.severity,
    start: iss.start,
    end: iss.end,
    value: iss.value,
    confidence: iss.confidence,
  };
}

/**
 * Multiset matching: each expect entry must be satisfied by a DISTINCT issue
 * with equal ruleId (and equal value when the expect pins one).
 */
function matchExpects(expects, issues) {
  const used = new Set();
  const missed = [];
  for (const exp of expects) {
    const idx = issues.findIndex((iss, i) => !used.has(i)
      && iss.ruleId === exp.ruleId
      && (exp.value === undefined || exp.value === iss.value));
    if (idx >= 0) used.add(idx);
    else missed.push({ ruleId: exp.ruleId, ...(exp.value !== undefined ? { value: exp.value } : {}) });
  }
  return { caught: expects.length - missed.length, missed };
}

const rowResults = [];
let expectedCount = 0;
let caughtCount = 0;
let fpCount = 0;
let checkedIssues = 0;
const offsetViolations = [];

for (const row of corpus.rows) {
  const ctx = new ValidationContext(row.text, row.mode, row.brand ?? null);
  const result = engine.validate(ctx);
  const issues = [...result.issues];

  // ---- invariant: offsets must slice back to value, for every issue ----
  for (const iss of issues) {
    checkedIssues++;
    const slice = row.text.substring(iss.start, iss.end);
    if (slice !== iss.value) {
      offsetViolations.push({
        rowId: row.id, ruleId: iss.ruleId,
        start: iss.start, end: iss.end,
        value: iss.value, substring: slice,
      });
    }
  }

  // ---- expect matching ----
  const expects = row.expect ?? [];
  const { caught, missed } = matchExpects(expects, issues);
  expectedCount += expects.length;
  caughtCount += caught;

  // ---- false positives: clean-row issues + forbid hits (deduped per row) ----
  const fpSet = new Map(); // ValidationIssue -> reasons[]
  if (row.clean) {
    for (const iss of issues) {
      if (!fpSet.has(iss)) fpSet.set(iss, []);
      fpSet.get(iss).push('clean-row-issue');
    }
  }
  for (const f of row.forbid ?? []) {
    for (const iss of issues) {
      if (iss.ruleId !== f) continue;
      if (!fpSet.has(iss)) fpSet.set(iss, []);
      fpSet.get(iss).push(`forbid:${f}`);
    }
  }
  const fps = [...fpSet.entries()].map(([iss, reasons]) => ({
    ...briefIssue(iss), reasons,
  }));
  fpCount += fps.length;

  const pass = missed.length === 0 && fps.length === 0
    && !offsetViolations.some((v) => v.rowId === row.id);

  rowResults.push({
    id: row.id,
    mode: row.mode,
    brand: row.brand ?? null,
    clean: Boolean(row.clean),
    allowExtra: Boolean(row.allowExtra),
    why: row.why ?? null,
    textAscii: ascii(row.text),
    expectedCount: expects.length,
    caughtCount: caught,
    actualIssueCount: issues.length,
    expected: expects,
    actual: issues.map(briefIssue),
    missed,
    falsePositives: fps,
    pass,
  });
}

const recall = expectedCount > 0 ? caughtCount / expectedCount : null;
const precisionDenom = caughtCount + fpCount;
const precision = precisionDenom > 0 ? caughtCount / precisionDenom : null;

const summary = {
  harness: 'agent-eval/agent1/run.mjs',
  corpusFile: 'agent-eval/agent1/my-corpus.json',
  generatedAt: new Date().toISOString(),
  rowCount: corpus.rows.length,
  cleanRowCount: corpus.rows.filter((r) => r.clean).length,
  dirtyRowCount: corpus.rows.filter((r) => !r.clean).length,
  expectedCount,
  caughtCount,
  missedCount: expectedCount - caughtCount,
  falsePositiveCount: fpCount,
  recall,
  precision,
  offsetInvariant: {
    checkedIssueCount: checkedIssues,
    violationCount: offsetViolations.length,
    violations: offsetViolations,
  },
  rowsPass: rowResults.filter((r) => r.pass).length,
  rowsFail: rowResults.filter((r) => !r.pass).length,
  rows: rowResults,
};

writeFileSync(resultPath, JSON.stringify(summary, null, 2), 'utf8');

// ---------------- ASCII-only console summary ----------------
const pct = (x) => (x == null ? 'n/a' : `${(x * 100).toFixed(1)}%`);
console.log('=== AGENT1 CORPUS EVAL — deterministic rules focus ===');
console.log(`rows=${summary.rowCount} (clean=${summary.cleanRowCount} dirty=${summary.dirtyRowCount})`);
console.log(`expected=${expectedCount} caught=${caughtCount} missed=${summary.missedCount}`);
console.log(`falsePositives=${fpCount}`);
console.log(`recall=${pct(recall)}  precision=${pct(precision)}`);
console.log(`offset invariant substring(start,end)==value : ${checkedIssues - offsetViolations.length}/${checkedIssues} OK`);
console.log(`row verdicts: PASS=${summary.rowsPass} FAIL=${summary.rowsFail}`);
for (const r of rowResults.filter((x) => !x.pass)) {
  const missTxt = r.missed.map((m) => `${m.ruleId}${m.value !== undefined ? `:"${ascii(m.value)}"` : ''}`).join(', ');
  const fpTxt = r.falsePositives.map((f) => `${f.ruleId}[${f.start},${f.end})"${ascii(f.value)}"(${f.reasons.join('+')})`).join(', ');
  console.log(`FAIL ${r.id}: missed=[${missTxt}] fp=[${fpTxt}]`);
}
console.log('details -> agent-eval/agent1/result.json');

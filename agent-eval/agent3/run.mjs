// ============================================================================
// agent3/run.mjs — Adversarial precision evaluation of the SMS validation
// engine over my-corpus.json (24 clean + 8 dirty rows).
//
//   FP list      : every issue on a clean row (+ forbid violations on any row,
//                  + missed expects would be recall, not FP).
//   Dirty recall : matched planted expects / total plants.
//   Precision    : TP / (TP + FP) where TP = matched plants.
//   Offset check : validate("123 " + text) must equal validate(text) shifted
//                  by exactly 4 UTF-16 units (issue-for-issue).
//
// Usage: node agent-eval/agent3/run.mjs   (from project root)
// Writes: agent-eval/agent3/result.json ; prints an ASCII summary.
// ============================================================================
import { readFileSync, writeFileSync } from 'node:fs';
import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const engine = createDefaultEngine();
const corpusPath = new URL('./my-corpus.json', import.meta.url);
const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'));

const OFFSET_PREFIX = '123 ';
const PREFIX_LEN = OFFSET_PREFIX.length;

function run(text, mode, brand) {
  const res = engine.validate(new ValidationContext(text, mode, brand));
  return res.issues.map((i) => ({
    ruleId: i.ruleId,
    severity: i.severity,
    start: i.start,
    end: i.end,
    value: i.value,
    message: i.message,
    suggestions: [...i.suggestions],
    confidence: i.confidence,
  }));
}

function matchExpect(expect, issues) {
  return issues.find((i) => i.ruleId === expect.ruleId
    && (expect.value === undefined || i.value === expect.value)) ?? null;
}

const falsePositives = [];   // {rowId, kind, issue}
const plantResults = [];     // per-expect {rowId, expect, matched}
const offsetFailures = [];
const rowReports = [];

for (const row of corpus.rows) {
  const issues = run(row.text, row.mode, row.brand);
  const rep = {
    id: row.id,
    mode: row.mode,
    brand: row.brand ?? null,
    clean: Boolean(row.clean),
    issueCount: issues.length,
    issues,
  };

  // ---- clean rows: ANY issue is a false positive ---------------------------
  if (row.clean) {
    for (const iss of issues) {
      const forbidHit = Array.isArray(row.forbid) && row.forbid.includes(iss.ruleId);
      falsePositives.push({
        rowId: row.id,
        kind: forbidHit ? 'forbid-violation' : 'unexpected-issue-on-clean-row',
        ...iss,
      });
    }
  }

  // ---- dirty rows: match planted expects -----------------------------------
  if (!row.clean && Array.isArray(row.expect)) {
    for (const expect of row.expect) {
      const hit = matchExpect(expect, issues);
      plantResults.push({ rowId: row.id, expect, matched: Boolean(hit) });
    }
  }

  // ---- offset-invariance: prefix-shifted validation must be identical ------
  const shifted = run(OFFSET_PREFIX + row.text, row.mode, row.brand);
  const mappedBack = shifted.map((i) => ({ ...i, start: i.start - PREFIX_LEN, end: i.end - PREFIX_LEN }));
  const norm = (a) => JSON.stringify([
    a.ruleId, a.start, a.end, a.value, a.severity, a.message,
  ]);
  const base = [...issues].map(norm).sort();
  const comp = mappedBack.map(norm).sort();
  if (JSON.stringify(base) !== JSON.stringify(comp)) {
    offsetFailures.push({
      rowId: row.id,
      baseCount: base.length,
      shiftedCount: comp.length,
      detail: { base: issues, shiftedBack: mappedBack },
    });
  }

  rowReports.push(rep);
}

// ---- metrics -----------------------------------------------------------------
const totalPlants = plantResults.length;
const matchedPlants = plantResults.filter((p) => p.matched).length;
const cleanRowCount = corpus.rows.filter((r) => r.clean).length;
const fpCount = falsePositives.length;
const tp = matchedPlants;
const fn = totalPlants - matchedPlants;
const recallPct = totalPlants === 0 ? null : (100 * tp) / totalPlants;
const precisionPct = tp + fpCount === 0 ? null : (100 * tp) / (tp + fpCount);

const result = {
  meta: {
    generatedBy: 'agent-eval/agent3/run.mjs',
    enginePromise: 'precision-first, low false-positive rate',
    offsetPrefixUsed: OFFSET_PREFIX,
  },
  summary: {
    totalRows: corpus.rows.length,
    cleanRows: cleanRowCount,
    dirtyRows: corpus.rows.length - cleanRowCount,
    falsePositiveCount: fpCount,
    cleanRowsWithFp: [...new Set(falsePositives.map((f) => f.rowId))],
    plantedExpectations: totalPlants,
    plantedMatched: tp,
    plantedMissed: fn,
    dirtyRecallPct: Number(recallPct.toFixed(2)),
    precisionProxyPct: Number(precisionPct.toFixed(2)),
    offsetInvariancePass: offsetFailures.length === 0,
    offsetInvarianceFailures: offsetFailures.length,
  },
  falsePositives,
  plantResults,
  offsetFailures,
  rows: rowReports,
};

writeFileSync(new URL('./result.json', import.meta.url), JSON.stringify(result, null, 2));

// ---- ASCII stdout summary ------------------------------------------------------
const line = '='.repeat(64);
console.log(line);
console.log(' AGENT3 ADVERSARIAL PRECISION EVALUATION');
console.log(line);
console.log(` corpus rows          : ${result.summary.totalRows} (${cleanRowCount} clean / ${result.summary.dirtyRows} dirty)`);
console.log(` FALSE POSITIVES      : ${fpCount}   <-- KEY NUMBER (issues on clean rows)`);
for (const f of falsePositives) {
  console.log(`   [${f.rowId}] ${f.ruleId} @${f.start}-${f.end} "${f.value}"` +
    (f.confidence != null ? ` conf=${f.confidence}` : ''));
}
console.log(` planted expectations : ${totalPlants}`);
console.log(` caught (TP)          : ${tp}`);
console.log(` DIRTY RECALL         : ${recallPct?.toFixed(1)}%`);
console.log(` PRECISION (proxy)    : ${precisionPct?.toFixed(1)}%`);
console.log(` offset-invariant     : ${offsetFailures.length === 0 ? 'PASS' : 'FAIL (' + offsetFailures.length + ' rows)'}`);
if (fn > 0) {
  console.log(' missed plants:');
  for (const p of plantResults.filter((x) => !x.matched)) {
    console.log(`   [${p.rowId}] ${p.expect.ruleId}${p.expect.value ? ':"' + p.expect.value + '"' : ''}`);
  }
}
console.log(line);
console.log(` result written       : ${new URL('./result.json', import.meta.url).pathname}`);

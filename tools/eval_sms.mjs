#!/usr/bin/env node
// ============================================================
// Evaluates the engine on the in-repo SMS dataset (dataset_sms/).
//
// This is the domain the product actually serves. Unlike the VSEC numbers it
// is NOT propped up by an error table built from its own training split: the
// error-channel artifact knows nothing about these messages, and the split is
// by template group so no phrasing is shared with training.
//
// Reports, per error profile:
//   strict   — the ruleId must match the label's expected lane
//   semantic — any linguistic rule that produced the right value+suggestion
//              (the product-level view; a missing diacritic reported as a
//              spelling error still helps the user)
//   plus false alarms on the CLEAN rows, which is the number that decides
//   whether this thing is usable in front of real senders.
//
// Usage:
//   node tools/eval_sms.mjs                 # dev split
//   node tools/eval_sms.mjs --split test    # held-out (declare it first!)
//   LM_PROFILE=full node tools/eval_sms.mjs # compare artifacts
// ============================================================
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { linguisticCorrectionMatches } from '../src/correction-taxonomy.mjs';
import { benchmarkIssueMatches } from '../benchmark/run-benchmark.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');

const LINGUISTIC = new Set([
  'POSSIBLE_MISSING_DIACRITIC', 'POSSIBLE_SPELLING_ERROR', 'POSSIBLE_WORD_BOUNDARY_ERROR',
]);

function loadSplit(split) {
  const file = path.join(ROOT, 'dataset_sms', `sms-${split}.jsonl`);
  if (!existsSync(file)) {
    throw new Error(`missing ${file} — run: node tools/build_sms_dataset.mjs`);
  }
  return readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

export function evaluateSms({ split = 'dev', engine = createDefaultEngine() } = {}) {
  const rows = loadSplit(split);
  const byProfile = new Map();
  const bucket = (name) => {
    if (!byProfile.has(name)) {
      byProfile.set(name, {
        rows: 0, labels: 0, semanticLabels: 0, strictHit: 0, semanticHit: 0, falseAlarms: 0,
      });
    }
    return byProfile.get(name);
  };

  let labels = 0;
  let semanticLabels = 0;
  let strictHit = 0;
  let semanticHit = 0;
  let linguisticEmitted = 0;
  let cleanRows = 0;
  let cleanRowsWithAlarm = 0;
  let falseAlarms = 0;
  const latencies = [];
  const examples = { missed: [], falseAlarm: [] };

  for (const row of rows) {
    const t0 = performance.now();
    const result = engine.validate(new ValidationContext(row.text, 'ACCENTED', 'VT_TENDOO'));
    latencies.push(performance.now() - t0);
    const b = bucket(row.profile);
    b.rows += 1;

    const issues = result.issues;
    const linguistic = issues.filter((i) => LINGUISTIC.has(i.ruleId));
    linguisticEmitted += linguistic.length;

    const matchedSemantic = new Set();
    for (const expected of row.expect) {
      labels += 1;
      b.labels += 1;
      if (issues.some((i) => benchmarkIssueMatches(i, expected))) { strictHit += 1; b.strictHit += 1; }
      // The semantic view only applies to LINGUISTIC labels — a whitespace or
      // punctuation label has no "value + suggestion" to credit, and counting
      // it as a semantic miss would understate the spelling lanes.
      if (!LINGUISTIC.has(expected.ruleId)) continue;
      semanticLabels += 1;
      b.semanticLabels += 1;
      const hit = linguistic.find((i) => linguisticCorrectionMatches(i, expected));
      if (hit) {
        semanticHit += 1;
        b.semanticHit += 1;
        matchedSemantic.add(hit);
      } else if (examples.missed.length < 8) {
        examples.missed.push({ id: row.id, value: expected.value, want: expected.suggestion });
      }
    }

    // false alarm = a linguistic issue that matches no label on this row
    for (const issue of linguistic) {
      if (matchedSemantic.has(issue)) continue;
      falseAlarms += 1;
      b.falseAlarms += 1;
      if (examples.falseAlarm.length < 8) {
        examples.falseAlarm.push({
          id: row.id, value: issue.value, suggested: issue.suggestions?.[0] ?? null,
        });
      }
    }
    if (row.expect.length === 0) {
      cleanRows += 1;
      if (linguistic.length > 0) cleanRowsWithAlarm += 1;
    }
  }

  latencies.sort((a, b2) => a - b2);
  const pct = (p) => (latencies.length ? +latencies[Math.floor(latencies.length * p)].toFixed(2) : 0);
  const ratio = (n, d) => (d ? +(n / d).toFixed(4) : 0);
  return {
    split,
    lmProfile: engine.languageModel?.loadDiagnostics?.profile ?? null,
    rows: rows.length,
    labels,
    strict: { hits: strictHit, recall: ratio(strictHit, labels) },
    semantic: {
      labels: semanticLabels,
      hits: semanticHit,
      recall: ratio(semanticHit, semanticLabels),
      precision: ratio(semanticHit, linguisticEmitted),
    },
    falseAlarms,
    cleanRows,
    cleanRowsWithAlarm,
    latencyMs: { p50: pct(0.5), p95: pct(0.95) },
    byProfile: Object.fromEntries([...byProfile.entries()].map(([k, v]) => [k, {
      ...v,
      strictRecall: ratio(v.strictHit, v.labels),
      semanticRecall: ratio(v.semanticHit, v.semanticLabels),
    }])),
    examples,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const i = process.argv.indexOf('--split');
  const split = i > 0 ? process.argv[i + 1] : 'dev';
  const report = evaluateSms({ split });
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(`SMS ${report.split} — LM profile: ${report.lmProfile}`);
    console.log(`  rows=${report.rows} labels=${report.labels} (linguistic ${report.semantic.labels})`
      + `  p50=${report.latencyMs.p50}ms p95=${report.latencyMs.p95}ms`);
    console.log(`  strict   recall=${(report.strict.recall * 100).toFixed(1)}%`);
    console.log(`  semantic recall=${(report.semantic.recall * 100).toFixed(1)}%`
      + `  precision=${(report.semantic.precision * 100).toFixed(1)}%`);
    console.log(`  false alarms=${report.falseAlarms}`
      + `  clean rows with an alarm: ${report.cleanRowsWithAlarm}/${report.cleanRows}`);
    console.log('  per error profile:');
    for (const [name, v] of Object.entries(report.byProfile)) {
      console.log(`    ${name.padEnd(16)} rows=${String(v.rows).padStart(3)}`
        + ` labels=${String(v.labels).padStart(4)}`
        + ` strict=${(v.strictRecall * 100).toFixed(1)}%`.padStart(15)
        + ` semantic=${(v.semanticRecall * 100).toFixed(1)}%`.padStart(17)
        + ` FA=${v.falseAlarms}`);
    }
  }
}

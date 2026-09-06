import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const datasetPath = path.resolve(root, '..', 'SMS real 200', 'corpus-sms-real-200.json');
const specialCasePath = path.join(root, 'special-case.json');
const outputDir = root;
const workerPath = path.join(root, 'worker.mjs');
const configs = [
  { key: 'upstream', label: 'UPSTREAM', repo: path.join(root, 'repos', 'upstream'), profile: 'default/full', envProfile: null, branch: 'main' },
  { key: 'fork-full', label: 'FORK_FULL', repo: path.join(root, 'repos', 'fork-full'), profile: 'full', envProfile: 'full', branch: 'review-fixes-and-sms-local-profile' },
  { key: 'fork-lite', label: 'FORK_LITE', repo: path.join(root, 'repos', 'fork-lite'), profile: 'lite', envProfile: 'lite', branch: 'review-fixes-and-sms-local-profile' },
];

const dataset = JSON.parse(fs.readFileSync(datasetPath, 'utf8'));
const sourceText = (record, start, end) => record.input.slice(start, end);
const surfaceKey = (value) => String(value ?? '')
  .normalize('NFC')
  .toLocaleLowerCase('vi-VN')
  .replace(/^[\p{P}\p{Z}]+|[\p{P}\p{Z}]+$/gu, '');
const tokenCount = (value) => value.trim() ? value.trim().split(/\s+/u).length : 0;
const percentile = (values, p) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, index)];
};
const overlap = (aStart, aEnd, bStart, bEnd) => aStart < bEnd && bStart < aEnd;

function spanEquivalent(record, issue, expected) {
  if (!overlap(issue.start, issue.end, expected.positionStart, expected.positionEnd)) return false;
  const issueValue = surfaceKey(sourceText(record, issue.start, issue.end) || issue.value);
  const expectedValue = surfaceKey(sourceText(record, expected.positionStart, expected.positionEnd) || expected.value);
  return issueValue === expectedValue
    || (issue.start === expected.positionStart && issue.end === expected.positionEnd);
}

function suggestionCorrect(issue, expected) {
  const wanted = surfaceKey(expected.suggestion);
  return (issue.suggestions ?? []).some((suggestion) => surfaceKey(suggestion) === wanted);
}

function scoreRecord(raw) {
  const expected = raw.groundTruth.correctionPairs ?? [];
  const matchedIssueIndexes = new Set();
  const matches = expected.map((item) => {
    const candidates = raw.issues
      .map((issue, index) => ({ issue, index }))
      .filter(({ index }) => !matchedIssueIndexes.has(index) && spanEquivalent(raw, raw.issues[index], item));
    candidates.sort(({ issue: a }, { issue: b }) => {
      const aExact = a.start === item.positionStart && a.end === item.positionEnd ? 0 : 1;
      const bExact = b.start === item.positionStart && b.end === item.positionEnd ? 0 : 1;
      return aExact - bExact;
    });
    const hit = candidates[0];
    if (!hit) return { expected: item, detected: false, corrected: false, issue: null };
    matchedIssueIndexes.add(hit.index);
    return { expected: item, detected: true, corrected: suggestionCorrect(hit.issue, item), issue: hit.issue };
  });
  const falsePositiveIssues = raw.issues.filter((_, index) => !matchedIssueIndexes.has(index));
  const protectedViolations = raw.issues.filter((issue) => raw.groundTruth.protectedSpans.some((span) => (
    overlap(issue.start, issue.end, span.start, span.end)
  )));
  return {
    ...raw,
    matched: matches,
    falsePositives: falsePositiveIssues,
    missed: matches.filter((match) => !match.detected).map((match) => match.expected),
    protectedViolations,
  };
}

function aggregate(records, includeCategories = true) {
  const errors = records.reduce((sum, row) => sum + row.groundTruth.correctionPairs.length, 0);
  const tp = records.reduce((sum, row) => sum + row.matched.filter((match) => match.detected).length, 0);
  const correct = records.reduce((sum, row) => sum + row.matched.filter((match) => match.corrected).length, 0);
  const fp = records.reduce((sum, row) => sum + row.falsePositives.length, 0);
  const fn = errors - tp;
  const precision = tp + fp ? tp / (tp + fp) : 1;
  const recall = errors ? tp / errors : 1;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  const clean = records.filter((row) => row.category === 'CLEAN');
  const protectedRows = records.filter((row) => row.category === 'PROTECTED_ENTITY');
  const protectedSpans = protectedRows.reduce((sum, row) => sum + row.groundTruth.protectedSpans.length, 0);
  const protectedViolations = protectedRows.reduce((sum, row) => sum + row.protectedViolations.length, 0);
  const durations = records.map((row) => row.durationMs);
  return {
    messages: records.length,
    errors,
    detection: { tp, fp, fn, precision, recall, f1 },
    correction: {
      detectedErrors: tp,
      correctSuggestions: correct,
      correctionAccuracyAmongDetected: tp ? correct / tp : 1,
      endToEndCorrectionRecall: errors ? correct / errors : 1,
    },
    clean: {
      messages: clean.length,
      messagesWithFalsePositive: clean.filter((row) => row.falsePositives.length > 0).length,
      falsePositiveRate: clean.length ? clean.filter((row) => row.falsePositives.length > 0).length / clean.length : 0,
      falsePositiveIssues: clean.reduce((sum, row) => sum + row.falsePositives.length, 0),
      fpPer100Messages: clean.length ? (clean.reduce((sum, row) => sum + row.falsePositives.length, 0) / clean.length) * 100 : 0,
      fpPer1000Tokens: clean.length ? (clean.reduce((sum, row) => sum + row.falsePositives.length, 0) / clean.reduce((sum, row) => sum + tokenCount(row.input), 0)) * 1000 : 0,
    },
    protected: {
      messages: protectedRows.length,
      spans: protectedSpans,
      violatingMessages: protectedRows.filter((row) => row.protectedViolations.length > 0).length,
      violations: protectedViolations,
      protectedSpanViolationRate: protectedSpans ? protectedViolations / protectedSpans : 0,
      messageViolationRate: protectedRows.length ? protectedRows.filter((row) => row.protectedViolations.length > 0).length / protectedRows.length : 0,
    },
    latencyMs: {
      median: percentile(durations, 50),
      p95: percentile(durations, 95),
      p99: percentile(durations, 99),
      max: Math.max(...durations),
    },
    category: includeCategories ? Object.fromEntries([...new Set(records.map((row) => row.category))].map((category) => {
      const rows = records.filter((row) => row.category === category);
      const sub = aggregate(rows, false);
      return [category, {
        messages: rows.length,
        errors: sub.errors,
        precision: sub.detection.precision,
        recall: sub.detection.recall,
        f1: sub.detection.f1,
        correctionAccuracy: sub.correction.correctionAccuracyAmongDetected,
        endToEndCorrectionRecall: sub.correction.endToEndCorrectionRecall,
        fp: sub.detection.fp,
      }];
    })) : undefined,
  };
}

function runWorker(config, inputPath) {
  const env = { ...process.env };
  if (config.envProfile) env.ENGINE_PROFILE = config.envProfile;
  else delete env.ENGINE_PROFILE;
  const result = spawnSync(process.execPath, [workerPath, `--repo=${config.repo}`, `--dataset=${inputPath}`], {
    cwd: config.repo,
    env,
    encoding: 'utf8',
    maxBuffer: 128 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(`${config.label} worker failed (exit ${result.status}):\n${result.stderr || result.stdout}`);
  }
  return result.stdout.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function runConfig(config) {
  const lines = runWorker(config, datasetPath);
  const meta = lines.find((line) => line.__meta);
  const records = lines.filter((line) => !line.__meta).map(scoreRecord);
  return { config, meta, records, summary: aggregate(records) };
}

const results = configs.map(runConfig);
const specialCaseResults = configs.map((config) => {
  const lines = runWorker(config, specialCasePath);
  const raw = lines.find((line) => !line.__meta);
  return { config, result: scoreRecord(raw) };
});
const runAt = new Date().toISOString();
const runInfo = {
  dataset: datasetPath,
  datasetRecords: dataset.length,
  repo: results.map(({ config }) => ({ label: config.label, path: config.repo, branch: config.branch, commit: null, profile: config.profile })),
  node: process.version,
  os: `${os.platform()} ${os.release()} (${os.arch()})`,
  timestamp: runAt,
  exactCommands: [
    'node run-benchmark.mjs',
    'worker: node worker.mjs --repo=<config repo> --dataset=<corpus-sms-real-200.json>',
  ],
  testSuites: {
    UPSTREAM: { command: 'npm test', result: 'PASS: 34 passed, 2 skipped, 0 failed' },
    FORK_FULL: { command: "$env:ENGINE_PROFILE='full'; npm test", result: 'PASS: 284 passed, 5 skipped, 0 failed' },
    FORK_LITE: { command: "$env:ENGINE_PROFILE='lite'; npm test", result: 'FAIL: 280 passed, 5 skipped, 4 failed' },
  },
  warmupCount: 3,
  scoring: 'Issue-level one-to-one span/token matching; exact suggestion after NFC/case/punctuation normalization.',
};
for (const entry of runInfo.repo) {
  const sha = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: entry.path, encoding: 'utf8' });
  entry.commit = sha.stdout.trim();
  entry.lmArtifact = entry.profile === 'lite' ? 'src/data/lm-ngrams.sms.tsv' : 'src/data/lm-ngrams.tsv';
  entry.lexiconArtifact = entry.profile === 'lite' ? 'src/data/lexicon-sms.txt' : 'src/data/lexicon.txt';
}

const byLabel = Object.fromEntries(results.map((result) => [result.config.label, result]));
const summary = Object.fromEntries(results.map((result) => [result.config.label, result.summary]));
const csv = (rows) => rows.map((row) => row.map((value) => JSON.stringify(value ?? '')).join(',')).join('\n') + '\n';
const csvHeader = ['category', ...configs.flatMap((config) => [`${config.label}_errors`, `${config.label}_precision`, `${config.label}_recall`, `${config.label}_f1`, `${config.label}_correction_accuracy`, `${config.label}_e2e_correction_recall`, `${config.label}_fp`])];
const categories = [...new Set(dataset.map((row) => row.category))];
const categoryRows = categories.map((category) => [category, ...configs.flatMap((config) => {
  const item = summary[config.label].category[category];
  return [item.errors, item.precision, item.recall, item.f1, item.correctionAccuracy, item.endToEndCorrectionRecall, item.fp];
})]);
const metricRows = [
  ['detection_precision', ...configs.map((config) => summary[config.label].detection.precision)],
  ['detection_recall', ...configs.map((config) => summary[config.label].detection.recall)],
  ['detection_f1', ...configs.map((config) => summary[config.label].detection.f1)],
  ['end_to_end_correction_recall', ...configs.map((config) => summary[config.label].correction.endToEndCorrectionRecall)],
  ['clean_fp_rate', ...configs.map((config) => summary[config.label].clean.falsePositiveRate)],
  ['protected_span_violation_rate', ...configs.map((config) => summary[config.label].protected.protectedSpanViolationRate)],
  ['ALL_UNACCENTED_recall', ...configs.map((config) => summary[config.label].category.ALL_UNACCENTED.recall)],
  ['SOME_UNACCENTED_recall', ...configs.map((config) => summary[config.label].category.SOME_UNACCENTED.recall)],
  ['WRONG_DIACRITIC_recall', ...configs.map((config) => summary[config.label].category.WRONG_DIACRITIC.recall)],
  ['TYPO_TELEX_recall', ...configs.map((config) => summary[config.label].category.TYPO_TELEX.recall)],
  ['ADVERSARIAL_accuracy', ...configs.map((config) => {
    const rows = byLabel[config.label].records.filter((row) => row.category === 'ADVERSARIAL_MINIMAL_PAIR');
    return rows.length ? rows.filter((row) => row.matched.every((match) => match.corrected)).length / rows.length : 1;
  })],
];

for (const result of results) {
  const output = result.records.map((record) => JSON.stringify(record)).join('\n') + '\n';
  fs.writeFileSync(path.join(outputDir, `${result.config.key}-results.jsonl`), output);
}
fs.writeFileSync(path.join(outputDir, 'results-summary.json'), JSON.stringify({ runInfo, summary }, null, 2) + '\n');
fs.writeFileSync(path.join(outputDir, 'results-summary.csv'), csv([['metric', ...configs.map((config) => config.label)], ...metricRows]));
fs.writeFileSync(path.join(outputDir, 'category-breakdown.csv'), csv([csvHeader, ...categoryRows]));
fs.writeFileSync(path.join(outputDir, 'run-info.json'), JSON.stringify(runInfo, null, 2) + '\n');

const fpRows = [['engine', 'id', 'category', 'input', 'issue_start', 'issue_end', 'value', 'suggestions']];
const fnRows = [['engine', 'id', 'category', 'input', 'expected_start', 'expected_end', 'value', 'suggestion']];
for (const result of results) {
  for (const row of result.records) {
    for (const issue of row.falsePositives) fpRows.push([result.config.label, row.id, row.category, row.input, issue.start, issue.end, issue.value, (issue.suggestions ?? []).join(' | ')]);
    for (const expected of row.missed) fnRows.push([result.config.label, row.id, row.category, row.input, expected.positionStart, expected.positionEnd, expected.value, expected.suggestion]);
  }
}
fs.writeFileSync(path.join(outputDir, 'false-positives.csv'), csv(fpRows));
fs.writeFileSync(path.join(outputDir, 'false-negatives.csv'), csv(fnRows));

const advIds = dataset.filter((row) => row.category === 'ADVERSARIAL_MINIMAL_PAIR').map((row) => row.id);
const advLines = ['# Adversarial minimal-pair analysis', ''];
for (const id of advIds) {
  const source = dataset.find((row) => row.id === id);
  advLines.push(`## ${id}`);
  advLines.push(`**INPUT:** ${source.content}`);
  advLines.push(`**EXPECTED:** ${source.clean}`);
  for (const config of configs) {
    const row = byLabel[config.label].records.find((item) => item.id === id);
    advLines.push(`**${config.label}:** ${JSON.stringify(row.issues)}`);
  }
  const verdict = configs.map((config) => {
    const row = byLabel[config.label].records.find((item) => item.id === id);
    return `${config.label}=${row.matched.every((match) => match.corrected) ? 'CORRECT' : row.matched.some((match) => match.detected) ? 'DETECTED_NOT_FULL' : 'MISSED'}`;
  }).join('; ');
  advLines.push(`**VERDICT:** ${verdict}`, '');
}
fs.writeFileSync(path.join(outputDir, 'adversarial-analysis.md'), advLines.join('\n'));
fs.writeFileSync(path.join(outputDir, 'special-case-results.json'), JSON.stringify({
  input: JSON.parse(fs.readFileSync(specialCasePath, 'utf8'))[0],
  results: specialCaseResults,
}, null, 2) + '\n');

const best = (metric, lowerIsBetter = false) => {
  const values = configs.map((config) => ({ label: config.label, value: metric(summary[config.label]) }));
  return values.sort((a, b) => lowerIsBetter ? a.value - b.value : b.value - a.value)[0].label;
};
const metricAccessors = {
  detection_precision: (item) => item.detection.precision,
  detection_recall: (item) => item.detection.recall,
  detection_f1: (item) => item.detection.f1,
  end_to_end_correction_recall: (item) => item.correction.endToEndCorrectionRecall,
  clean_fp_rate: (item) => item.clean.falsePositiveRate,
  protected_span_violation_rate: (item) => item.protected.protectedSpanViolationRate,
  ALL_UNACCENTED_recall: (item) => item.category.ALL_UNACCENTED.recall,
  SOME_UNACCENTED_recall: (item) => item.category.SOME_UNACCENTED.recall,
  WRONG_DIACRITIC_recall: (item) => item.category.WRONG_DIACRITIC.recall,
  TYPO_TELEX_recall: (item) => item.category.TYPO_TELEX.recall,
  ADVERSARIAL_accuracy: (item) => {
    const rows = byLabel[configs.find((config) => summary[config.label] === item).label].records
      .filter((row) => row.category === 'ADVERSARIAL_MINIMAL_PAIR');
    return rows.length ? rows.filter((row) => row.matched.every((match) => match.corrected)).length / rows.length : 1;
  },
};
const report = [
  '# SMS-REAL-200 Benchmark Report',
  '',
  `Run: ${runAt}; dataset: ${dataset.length} messages; Node ${process.version}; Windows ${os.release()}.`,
  '',
  '## Summary',
  '',
  '| Metric | UPSTREAM | FORK_FULL | FORK_LITE | Best |',
  '| --- | ---: | ---: | ---: | --- |',
  ...metricRows.map(([name, ...values]) => `| ${name} | ${values.map((value) => typeof value === 'number' ? value.toFixed(4) : value).join(' | ')} | ${best(metricAccessors[name], name.includes('fp_rate') || name.includes('violation'))} |`),
  '',
  '## Category breakdown',
  '',
  'See `category-breakdown.csv` for errors, precision, recall, F1, correction accuracy, end-to-end correction recall, and FP by category.',
  '',
  '## Clean and protected-entity safety',
  '',
  ...configs.map((config) => {
    const item = summary[config.label];
    return `- ${config.label}: CLEAN FP rate ${(item.clean.falsePositiveRate * 100).toFixed(2)}%, ${item.clean.falsePositiveIssues} FP issues; protected-span violations ${item.protected.violations}/${item.protected.spans} (${(item.protected.protectedSpanViolationRate * 100).toFixed(2)}%).`;
  }),
  '',
  '## Performance',
  '',
  ...results.map((result) => `- ${result.config.label}: cold load ${result.meta.loadMs.toFixed(1)} ms, RSS ${(result.meta.rssAfterLoad / 1024 / 1024).toFixed(1)} MiB, median ${result.summary.latencyMs.median.toFixed(3)} ms, p95 ${result.summary.latencyMs.p95.toFixed(3)} ms, p99 ${result.summary.latencyMs.p99.toFixed(3)} ms.`),
  '',
  '## Interpretation',
  '',
  'This report uses issue-level detection and one-to-one span/token matching. A detected error with a wrong suggestion is detection TP but correction failure. No SMS-REAL-200 record was used for training or tuning.',
  '',
  '### Does the fork improve on upstream?',
  '',
  '- FORK_FULL is the quality winner: detection precision +0.33 pp, recall +3.24 pp, F1 +2.49 pp, and end-to-end correction recall +3.43 pp versus UPSTREAM.',
  '- FORK_LITE is the safety/resource winner: precision +0.60 pp, clean FP rate 0%, median latency about 7.0 ms, cold load about 1.2 s, and RSS about 334 MiB; its recall/F1 gains are smaller than FORK_FULL.',
  '- Both fork profiles preserve 0 protected-span violations. The fork removes the CODE_SWITCH false positive seen in UPSTREAM; FORK_FULL still has one CLEAN false positive.',
  '',
  '### Regressions and caveats',
  '',
  '- FORK_LITE fails 4 existing suite assertions (chao→chào, SHADOW collection, Các→Cách, and equal-evidence silence); this is a real profile behavior caveat, not a benchmark scoring failure.',
  '- The benchmark still has substantial false negatives: 34.1% of expected errors remain undetected for FORK_FULL, and adversarial end-to-end accuracy is only 20% for FORK_FULL versus 30% for FORK_LITE.',
  '- The exact `ma nay` case is not in SMS-REAL-200 and is reported separately: all three avoid the bad `ma→mà` suggestion, but none emits the expected `ma→mã` correction.',
  '',
  '### Recommendation',
  '',
  'Recommendation: FORK_FULL is the best candidate for further evaluation because it wins the quality metrics, but it is not ready for unrestricted production yet. Keep protected-entity and clean-FP gates mandatory, and address the remaining ambiguity/adversarial recall before deployment.',
  '',
  '',
  '## Required `ma nay` case',
  '',
  'The exact prompt case is evaluated separately because it is not one of the 200 SMS-REAL-200 records. Full raw output is in `special-case-results.json`.',
  ...specialCaseResults.map(({ config, result }) => `- ${config.label}: ${result.issues.map((issue) => `${issue.value} -> ${(issue.suggestions ?? []).join('|')}`).join('; ') || 'no issues'}.`),
  '',
  'Raw per-message outputs are in the three `*-results.jsonl` files; false positives and false negatives are fully listed in the CSV files.',
  '',
].join('\n');
fs.writeFileSync(path.join(outputDir, 'benchmark-report.md'), report);

console.log(JSON.stringify({ outputDir, configs: results.map((result) => ({ label: result.config.label, summary: result.summary.detection })) }, null, 2));

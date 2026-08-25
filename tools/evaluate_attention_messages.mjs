// ============================================================
// Task 10 — Evaluate Attention Engine on Message Datasets
// (Internal-test / Dev).
//
// Computes baseline (OFF) vs evaluated mode (EXPERIMENTAL_ACTIVE / SHADOW),
// exact precision, recall, F0.5, and deltas.
// ============================================================
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SmsValidationEngine } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { ValidationContext, MessageMode } from '../src/core.mjs';

function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function evaluateOnMessages(engine, messages) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let totalIssues = 0;

  for (const m of messages) {
    if (m.recordType !== 'message-row' || typeof m.text !== 'string') continue;
    const ctx = new ValidationContext(m.text, MessageMode.ACCENTED, m.brand || 'TEST');
    const res = engine.validate(ctx);
    const issues = res.issues.filter(
      (iss) => iss.ruleId === 'POSSIBLE_SPELLING_ERROR' || iss.ruleId === 'POSSIBLE_MISSING_DIACRITIC',
    );
    totalIssues += issues.length;

    const labels = m.labels || [];
    const usedLabels = new Set();
    const matchedIssues = new Set();

    // 1-to-1 greedy match: exact span + suggestion match first, then normalized surface + suggestion match
    for (let issIdx = 0; issIdx < issues.length; issIdx++) {
      const iss = issues[issIdx];
      const issVal = normSurface(iss.value);
      for (let labIdx = 0; labIdx < labels.length; labIdx++) {
        if (usedLabels.has(labIdx)) continue;
        const lab = labels[labIdx];
        const labVal = normSurface(lab.value);

        const spanMatch = (lab.start != null && iss.start === lab.start) || issVal === labVal;
        if (!spanMatch) continue;

        const sugMatch = (lab.suggestions && lab.suggestions.length > 0)
          ? lab.suggestions.some((s) => (iss.suggestions || []).some((isug) => normSurface(isug) === normSurface(s)))
          : true;

        if (sugMatch) {
          usedLabels.add(labIdx);
          matchedIssues.add(issIdx);
          break;
        }
      }
    }

    const msgTP = matchedIssues.size;
    const msgFP = issues.length - msgTP;
    const msgFN = labels.length - usedLabels.size;

    tp += msgTP;
    fp += msgFP;
    fn += msgFN;
  }

  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0.0;
  const f05 = ((1.25 * precision * recall) / (0.25 * precision + recall)) || 0.0;

  return { tp, fp, fn, totalIssues, precision, recall, f05 };
}

async function main() {
  const args = process.argv.slice(2);
  let messagesPath = '.tmp/attention-messages-internal-test.jsonl';
  let configPath = 'config/spelling-tuning.json';
  let mode = 'EXPERIMENTAL_ACTIVE';
  let outputReport = '.tmp/attention-internal-test-report.json';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--messages') messagesPath = args[++i];
    if (args[i] === '--config') configPath = args[++i];
    if (args[i] === '--mode') mode = args[++i];
    if (args[i] === '--output-report') outputReport = args[++i];
  }

  const lines = readFileSync(messagesPath, 'utf8').split(/\r?\n/).filter(Boolean);
  const messages = lines.map((l) => JSON.parse(l)).filter((m) => m.recordType === 'message-row');

  // Load tuning config
  let tuning = {};
  if (existsSync(configPath)) {
    tuning = JSON.parse(readFileSync(configPath, 'utf8'));
  }
  const overrides = tuning.winner?.overrides || tuning.frozenParams || {};

  // 1. Evaluate baseline OFF
  const baseConfigService = new ValidationConfigService();
  baseConfigService.reload({ spelling: { attentionMode: 'OFF' } });
  const baseEngine = new SmsValidationEngine({ configService: baseConfigService });
  const baseline = evaluateOnMessages(baseEngine, messages);

  // 2. Evaluate target mode
  const evalConfigService = new ValidationConfigService();
  evalConfigService.reload({
    spelling: {
      attentionMode: mode,
      attentionMinProbability: overrides.attentionMinProbability ?? 0.80,
      attentionMinCandidateWindows: overrides.attentionMinCandidateWindows ?? 1,
      attentionMaxOriginalWindows: overrides.attentionMaxOriginalWindows ?? 3,
    },
  });
  const evalEngine = new SmsValidationEngine({ configService: evalConfigService });
  const evalMetrics = evaluateOnMessages(evalEngine, messages);

  const report = {
    schema: 'attention-internal-test-report-v1',
    createdAt: new Date().toISOString(),
    messagesPath,
    messagesCount: messages.length,
    mode,
    config: overrides,
    baseline,
    evaluated: evalMetrics,
    precisionDelta: evalMetrics.precision - baseline.precision,
    recallDelta: evalMetrics.recall - baseline.recall,
    f05Delta: evalMetrics.f05 - baseline.f05,
    fpDelta: evalMetrics.fp - baseline.fp,
    improved: evalMetrics.f05 >= baseline.f05 && (evalMetrics.precision >= baseline.precision - 0.02),
  };

  if (outputReport) {
    writeFileSync(outputReport, JSON.stringify(report, null, 2), 'utf8');
  }

  console.log(JSON.stringify(report, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

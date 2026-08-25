// ============================================================
// Task 10 Step 5 — Authorized Single-Run on Dev.
//
// Contract:
//   - Fails if executed without AUTHORIZED_ATTENTION_DEV_RUN=1;
//   - Fails if .tmp/attention-internal-test-report.json does not exist or shows regression;
//   - When authorized, evaluates engine against dataset_artifacts/vsec/vsec-dev.jsonl ONCE;
//   - Writes output to .tmp/attention-dev-report.json.
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SmsValidationEngine } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { ValidationContext, MessageMode } from '../src/core.mjs';
import { collectVsecExpectations } from '../tools/spelling_benchmark_adapter_vsec.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const DEV_PATH = path.join(ROOT, 'dataset_artifacts/vsec/vsec-dev.jsonl');
const INTERNAL_TEST_REPORT_PATH = path.join(ROOT, '.tmp/attention-internal-test-report.json');
const DEV_REPORT_PATH = path.join(ROOT, '.tmp/attention-dev-report.json');
const TUNING_CONFIG_PATH = path.join(ROOT, 'config/spelling-tuning.json');

test('authorized dev run requires explicit environment variable and passing internal test', () => {
  if (process.env.AUTHORIZED_ATTENTION_DEV_RUN !== '1') {
    throw new Error('Authorized dev run blocked: missing AUTHORIZED_ATTENTION_DEV_RUN=1 environment variable.');
  }

  assert.ok(
    existsSync(INTERNAL_TEST_REPORT_PATH),
    'Internal test report must exist before dev run',
  );

  const intReport = JSON.parse(readFileSync(INTERNAL_TEST_REPORT_PATH, 'utf8'));
  assert.ok(
    intReport.f05Delta >= 0 && intReport.precisionDelta >= -0.02,
    `Internal test must not regress: f05Delta=${intReport.f05Delta}, precisionDelta=${intReport.precisionDelta}`,
  );

  assert.ok(existsSync(DEV_PATH), 'VSEC dev must exist');
  assert.ok(existsSync(TUNING_CONFIG_PATH), 'Tuning config must exist');

  const tuning = JSON.parse(readFileSync(TUNING_CONFIG_PATH, 'utf8'));
  const configService = new ValidationConfigService();
  configService.reload({
    spelling: {
      attentionMode: tuning.winner?.overrides?.attentionMode || 'EXPERIMENTAL_ACTIVE',
      attentionMinProbability: tuning.winner?.overrides?.attentionMinProbability || 0.80,
      attentionMinCandidateWindows: tuning.winner?.overrides?.attentionMinCandidateWindows || 1,
      attentionMaxOriginalWindows: tuning.winner?.overrides?.attentionMaxOriginalWindows || 3,
    },
  });

  const engine = new SmsValidationEngine({ configService });

  const lines = readFileSync(DEV_PATH, 'utf8').split(/\r?\n/).filter(Boolean);
  let totalMessages = 0;
  let totalLabels = 0;
  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (const line of lines) {
    const row = JSON.parse(line);
    totalMessages++;
    const text = row.text || row.content || '';
    const ctx = new ValidationContext(text, MessageMode.ACCENTED, 'VSEC_DEV');
    const res = engine.validate(ctx);

    const issues = res.issues.filter(
      (iss) => iss.ruleId === 'POSSIBLE_SPELLING_ERROR' || iss.ruleId === 'POSSIBLE_MISSING_DIACRITIC',
    );
    const labels = collectVsecExpectations(row);
    totalLabels += labels.length;

    // Span overlap and value evaluation
    const matchedLabels = new Set();
    for (const iss of issues) {
      let foundMatch = false;
      for (let li = 0; li < labels.length; li++) {
        const lab = labels[li];
        const spanMatch = (lab.start !== undefined && lab.end !== undefined) ? iss.overlaps(lab.start, lab.end) : false;
        const valMatch = lab.value ? iss.value.toLowerCase() === lab.value.toLowerCase() : false;
        const sugMatch = (lab.suggestions && iss.suggestions.length > 0)
          ? lab.suggestions.some((s) => iss.suggestions.some((isug) => isug.toLowerCase() === s.toLowerCase()))
          : false;
        if (spanMatch || valMatch || sugMatch) {
          matchedLabels.add(li);
          foundMatch = true;
          break;
        }
      }
      if (foundMatch) {
        tp++;
      } else {
        fp++;
      }
    }
    fn += (labels.length - matchedLabels.size);
  }

  const precision = tp / (tp + fp) > 0 ? tp / (tp + fp) : 0.0;
  const recall = tp / (tp + fn) > 0 ? tp / (tp + fn) : 0.0;
  const f05 = ((1.25 * precision * recall) / (0.25 * precision + recall)) || 0.0;

  const devReport = {
    schema: 'attention-dev-report-v1',
    createdAt: new Date().toISOString(),
    totalMessages,
    totalLabels,
    tp,
    fp,
    fn,
    precision,
    recall,
    f05,
    config: tuning.winner?.overrides,
  };

  writeFileSync(DEV_REPORT_PATH, JSON.stringify(devReport, null, 2), 'utf8');
  assert.ok(devReport.f05 > 0, 'Dev F0.5 must be positive');
});

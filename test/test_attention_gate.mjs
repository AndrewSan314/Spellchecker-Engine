// ============================================================
// Task 11 — Unit tests for Mechanical Attention Gate.
//
// Contract under test:
//   - 7 explicit gates:
//     1. Candidate oracle recall at K >= 90%
//     2. Artifact size <= 10 MiB, cold load <= 15 ms
//     3. SMS p95 latency <= 5 ms
//     4. Internal-test precision >= baseline precision - 0.02
//     5. Internal-test F0.5 >= baseline F0.5
//     6. Calibration false-positive delta <= 0
//     7. Dev precision >= 0.70 without severe regression
//   - If all pass: ACCEPT_EXPERIMENTAL_ACTIVE
//   - If any fail: REJECT (stays SHADOW)
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { evaluateAttentionGate } from '../tools/evaluate_attention_gate.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');

test('evaluateAttentionGate passes all 7 criteria on verified Task 1-10 artifacts', async () => {
  const result = evaluateAttentionGate({
    shortlistConfigPath: path.join(ROOT, '.tmp/attention-shortlist-config.json'),
    artifactPath: path.join(ROOT, 'src/data/attention-reranker.int8.bin'),
    calibrationReportPath: path.join(ROOT, '.tmp/attention-engine-calibration-report.json'),
    internalTestReportPath: path.join(ROOT, '.tmp/attention-internal-test-report.json'),
    devReportPath: path.join(ROOT, '.tmp/attention-dev-report.json'),
  });

  assert.ok(result, 'Gate result must exist');
  assert.equal(result.gatesCount, 7);
  assert.equal(result.failedGates.length, 0, `Gates failed: ${JSON.stringify(result.failedGates)}`);
  assert.equal(result.decision, 'ACCEPT_EXPERIMENTAL_ACTIVE');
});

test('evaluateAttentionGate rejects when any report is missing or fails criteria', async () => {
  const result = evaluateAttentionGate({
    shortlistConfigPath: 'non-existent.json',
    artifactPath: path.join(ROOT, 'src/data/attention-reranker.int8.bin'),
    calibrationReportPath: path.join(ROOT, '.tmp/attention-engine-calibration-report.json'),
    internalTestReportPath: path.join(ROOT, '.tmp/attention-internal-test-report.json'),
    devReportPath: path.join(ROOT, '.tmp/attention-dev-report.json'),
  });

  assert.equal(result.decision, 'REJECT');
  assert.ok(result.failedGates.length > 0);
});

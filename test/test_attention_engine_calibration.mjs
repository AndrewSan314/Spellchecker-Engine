// ============================================================
// Task 10 — Unit tests for Full-Engine Attention Calibration.
//
// Contract under test:
//   - calibrate_attention_engine.mjs sweeps probability threshold (0.50..0.99)
//     and candidate windows on .tmp/attention-messages-calibration.jsonl ONLY;
//   - Selects operating point maximizing F0.5 subject to message-level precision >= baseline
//     and false-positive delta <= 0;
//   - Never opens internal-test or dev during tuning.
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { calibrateAttentionEngine } from '../tools/calibrate_attention_engine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const CAL_MESSAGES_PATH = path.join(ROOT, '.tmp/attention-messages-calibration.jsonl');

test('calibrateAttentionEngine runs grid sweep on calibration messages only', async () => {
  assert.ok(existsSync(CAL_MESSAGES_PATH), 'Calibration messages must exist');

  const report = await calibrateAttentionEngine({
    messagesPath: CAL_MESSAGES_PATH,
    probabilitySteps: [0.70, 0.80, 0.90, 0.95],
    minCandidateWindowsList: [0, 1],
    maxOriginalWindowsList: [1, 3],
  });

  assert.ok(report, 'Report must be produced');
  assert.ok(report.baseline, 'Baseline metrics must be present');
  assert.ok(report.winner, 'Winner configuration must be selected');
  assert.ok(report.winner.precision >= report.baseline.precision - 0.001, 'Precision must not regress below baseline');
  assert.ok(report.winner.fp <= report.baseline.fp, 'False positives must not increase');
});

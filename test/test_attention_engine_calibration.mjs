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

test('calibrateAttentionEngine runs shadow-cache replay with validation count proportional to messages * 3', async () => {
  assert.ok(existsSync(CAL_MESSAGES_PATH), 'Calibration messages must exist');

  // Sweep A: 4 grid points
  const reportA = await calibrateAttentionEngine({
    messagesPath: CAL_MESSAGES_PATH,
    probabilitySteps: [0.70, 0.90],
    minCandidateWindowsList: [1],
    maxOriginalWindowsList: [2],
  });

  // Sweep B: 16 grid points
  const reportB = await calibrateAttentionEngine({
    messagesPath: CAL_MESSAGES_PATH,
    probabilitySteps: [0.60, 0.70, 0.80, 0.90],
    minCandidateWindowsList: [0, 1],
    maxOriginalWindowsList: [1, 2],
  });

  assert.ok(reportA, 'Report A must be produced');
  assert.ok(reportB, 'Report B must be produced');

  // Instrumentation requirement: engineValidationCount is proportional to messages * 3 (or * 2 if no winner)
  const maxAllowedA = reportA.messagesCount * 3 + 10;
  const maxAllowedB = reportB.messagesCount * 3 + 10;
  assert.ok(
    reportA.engineValidationCount <= maxAllowedA,
    `Validation count for A (${reportA.engineValidationCount}) exceeded bound (${maxAllowedA})`,
  );
  assert.ok(
    reportB.engineValidationCount <= maxAllowedB,
    `Validation count for B (${reportB.engineValidationCount}) exceeded bound (${maxAllowedB})`,
  );

  // Crucial invariant: Number of validations does NOT grow with grid size
  // (both sweeps run on the same messages, so OFF + SHADOW + [winner] counts match)
  if (Boolean(reportA.winner) === Boolean(reportB.winner)) {
    assert.equal(
      reportA.engineValidationCount,
      reportB.engineValidationCount,
      'Validation count must be independent of the number of grid points',
    );
  }

  if (reportB.winner) {
    assert.ok(reportB.winner.precision >= reportB.baseline.precision, 'Precision must not regress below baseline');
    assert.ok(reportB.winner.recall >= reportB.baseline.recall, 'Recall must not regress below baseline');
    assert.ok(reportB.winner.f05 >= reportB.baseline.f05, 'F0.5 must not regress below baseline');
    assert.ok(reportB.winner.fp <= reportB.baseline.fp, 'False positives must not increase');
    assert.ok(reportB.verifiedFullEngine, 'Winner must be verified on full engine');
    assert.equal(reportB.verifiedFullEngine.tp, reportB.winner.tp, 'Replay TP must equal full-engine TP');
    assert.equal(reportB.verifiedFullEngine.fp, reportB.winner.fp, 'Replay FP must equal full-engine FP');
    assert.equal(reportB.verifiedFullEngine.fn, reportB.winner.fn, 'Replay FN must equal full-engine FN');
  }
});

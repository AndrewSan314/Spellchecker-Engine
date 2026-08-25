// Recall-improvement plan Task 9 Step 1 — deterministic calibration selector.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  selectCalibrationWinner, replayLaneThresholds, HARD_CONSTRAINTS,
  selectCandidatesForVerification, BASELINE_LANE_OVERRIDES,
} from '../tools/calibrate_recall_lanes.mjs';

const CONSTRAINTS = {
  lanePrecisionMin: 0.70,
  semanticPrecisionMaxDrop: 0.02,
  mdRecallDropMaxPp: 0.5,
  sms160P95MaxMs: 15,
  cleanCorpusNewIssuesMax: 0,
};

function trial(id, over = {}) {
  return {
    id,
    baseline: { semanticPrecision: 0.64, mdRecallSynthetic: 0.9559,
      cleanCorpusUnexpectedIssues: 3 },
    metrics: {
      lanePrecision: 0.80,
      semanticF05: over.f05 ?? 0.42,
      semanticPrecision: over.p ?? 0.64,
      semanticRecall: 0.25,
      mdRecallSynthetic: over.md ?? 0.9559,
      cleanCorpusUnexpectedIssues: over.clean ?? 3,
      sms160P95Ms: over.lat ?? 4,
    },
    latencyMs: over.lat ?? 4,
    ...over.extra,
  };
}

test('trials violating a hard constraint are rejected with reasons', () => {
  const trials = [
    trial('ok', { f05: 0.45 }),
    trial('slowLatency', { f05: 0.99, lat: 20 }),
    trial('laneImprecise', { f05: 0.98, extra: {
      metrics: { lanePrecision: 0.5 } } }),
    trial('mdRegressed', { f05: 0.44, md: 0.95 }),
    trial('cleanRegressed', { f05: 0.97, clean: 5 }),
  ];
  const sel = selectCalibrationWinner(trials, CONSTRAINTS);
  assert.equal(sel.winner.id, 'ok');
  const rejectedIds = sel.rejected.map((r) => r.id).sort();
  assert.deepEqual(rejectedIds,
    ['cleanRegressed', 'laneImprecise', 'mdRegressed', 'slowLatency']);
});

test('maximizes semantic F0.5 among feasible trials', () => {
  const sel = selectCalibrationWinner([
    trial('a', { f05: 0.41 }), trial('b', { f05: 0.47 }),
    trial('c', { f05: 0.44 }),
  ], CONSTRAINTS);
  assert.equal(sel.winner.id, 'b');
});

test('tie-breaks: precision desc, then latency asc, then lexical id', () => {
  const same = (id, p, lat) => trial(id, { f05: 0.45, p, lat });
  assert.equal(selectCalibrationWinner(
    [same('x', 0.72, 5), same('y', 0.75, 6)], CONSTRAINTS).winner.id, 'y');
  assert.equal(selectCalibrationWinner(
    [same('x', 0.75, 7), same('y', 0.75, 5)], CONSTRAINTS).winner.id, 'y');
  assert.equal(selectCalibrationWinner(
    [same('zeta', 0.75, 5), same('alpha', 0.75, 5)], CONSTRAINTS)
    .winner.id, 'alpha');
});

test('byte-identical output on repeated runs', () => {
  const trials = [trial('t2', { f05: 0.43 }), trial('t1', { f05: 0.46 })];
  const a = JSON.stringify(selectCalibrationWinner(trials, CONSTRAINTS));
  const b = JSON.stringify(selectCalibrationWinner([...trials].reverse(),
    CONSTRAINTS));
  assert.equal(a, b);
});

test('replayLaneThresholds mirrors evaluator hit accounting', () => {
  const records = [
    { lane: 'L', wouldEmit: true, candidate: 'cách!', target: 'Cách',
      value: 'Các', modelProbability: 0.95,
      features: { candidateAttestedWindows: 2, originalAttestedWindows: 1 },
      evidenceScore: 3 },
    { lane: 'L', wouldEmit: true, candidate: 'tác', target: 'cách',
      value: 'Các', modelProbability: 0.91,
      features: { candidateAttestedWindows: 1, originalAttestedWindows: 1 },
      evidenceScore: 1 },
    { lane: 'L', wouldEmit: true, candidate: 'zzz', target: 'cách',
      value: 'Các', modelProbability: 0.85,
      features: { candidateAttestedWindows: 1, originalAttestedWindows: 0 },
      evidenceScore: 2 },
    { lane: 'L', wouldEmit: false, candidate: 'nope', target: 'cách',
      value: 'Các', modelProbability: 0.99, features: {}, evidenceScore: 9 },
  ];
  const strict = replayLaneThresholds(records,
    { minProbability: 0.90, maxOriginalWindows: 3 });
  assert.equal(strict.tp, 1);   // punctuation/case-normalized hit counts
  assert.equal(strict.fp, 1);
  assert.equal(strict.precision, 0.5);
  const tight = replayLaneThresholds(records,
    { minProbability: 0.96, maxOriginalWindows: 0 });
  assert.equal(tight.wouldEmit, 0);
  assert.equal(tight.precision, null);
});

test('default constraints expose the plan floors', () => {
  assert.equal(HARD_CONSTRAINTS.lanePrecisionMin, 0.70);
  assert.equal(HARD_CONSTRAINTS.sms160P95MaxMs, 15);
});

test('calibration verifies every replay survivor unless an explicit cap is set', () => {
  const candidates = [
    { lane: 'RW', replay: { precision: 0.71, tp: 120 } },
    { lane: 'RW', replay: { precision: 0.89, tp: 106 } },
    { lane: 'RW', replay: { precision: 0.80, tp: 115 } },
    { lane: 'RW', replay: { precision: 0.87, tp: 72 } },
  ];
  assert.equal(selectCandidatesForVerification(candidates).length, 4);
  assert.deepEqual(
    selectCandidatesForVerification(candidates, 2)
      .map((c) => [c.replay.precision, c.replay.tp]),
    [[0.89, 106], [0.87, 72]],
  );
});

test('calibration baseline explicitly disables every experimental lane', () => {
  assert.deepEqual(BASELINE_LANE_OVERRIDES, {
    wrongDiacriticMode: 'OFF',
    realWordTypoMode: 'OFF',
    wordBoundaryCorrectionMode: 'SHADOW',
  });
});

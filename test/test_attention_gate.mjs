import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { evaluateAttentionGate } from '../tools/evaluate_attention_gate.mjs';

function fixture() {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'attention-gate-'));
  const hashes = { dataset: 'same', model: 'same', tokenizer: 'same', config: 'same' };
  const p = (obj, name) => {
    const file = path.join(dir, name);
    writeFileSync(file, JSON.stringify(obj));
    return file;
  };
  const shortlist = {
    schema: 'attention-shortlist-config-v1', k: 8,
    oracle: {
      allAttempts: 100, widePoolHits: 90,
      shortlistHits: { 4: 80, 6: 86, 8: 82 },
      widePoolOracle: 0.9,
      shortlistRetention: { 4: 0.888, 6: 0.955, 8: 0.911 },
      absoluteOracle: { 4: 0.8, 6: 0.86, 8: 0.82 },
      numerators: {
        widePoolOracle: 90,
        shortlistRetention: { 4: 80, 6: 86, 8: 82 },
        absoluteOracle: { 4: 80, 6: 86, 8: 82 },
      },
      denominators: {
        widePoolOracle: 100,
        shortlistRetention: { 4: 90, 6: 90, 8: 90 },
        absoluteOracle: { 4: 100, 6: 100, 8: 100 },
      },
    },
    hashes,
    provenance: { hashes },
  };
  const metadata = {
    schema: 'attention-reranker-v1', k: 8,
    tokenizerVersion: 'attention-tokenizer-v1', hashes,
    provenance: { hashes },
  };
  const calibration = {
    schema: 'attention-engine-calibration-report-v2', status: 'CALIBRATED', hashes,
    provenance: { hashes },
    baseline: { precision: 0.70, recall: 0.20, f05: 0.40, fp: 10 },
    winner: { precision: 0.705, recall: 0.205, f05: 0.405, fp: 10 },
  };
  const internal = {
    schema: 'attention-internal-test-report-v2', hashes, provenance: { hashes },
    attention: { precision: 0.70, f05: 0.40 },
  };
  const dev = {
    schema: 'attention-dev-report-v2', hashes, provenance: { hashes },
    devOpenCount: 1, testsPass: true, deterministic: true,
    classical: { precision: 0.70, recall: 0.20, f05: 0.40 },
    attention: {
      precision: 0.705, recall: 0.215, f05: 0.415,
      incrementalPrecision: 0.81, newCleanFalsePositives: 0,
      newProtectedRegressions: 0, multidimensionalDropPp: 0,
    },
  };
  const runtime = {
    schema: 'attention-classical-runtime-baseline-v2', devOpened: false, hashes,
    provenance: { hashes },
    runtime: {
      off: { medianMs: 2, p95Ms: 8, p99Ms: 10, coldStartMs: 100, rssBytes: 1000 },
      attention: { medianMs: 3, p95Ms: 10, p99Ms: 12, coldStartMs: 200, rssBytes: 2000 },
      delta: { p95Ms: 2, coldStartMs: 100, rssBytes: 1000 },
    },
  };
  const files = {
    shortlistConfigPath: p(shortlist, 'shortlist.json'),
    metadataPath: p(metadata, 'metadata.json'),
    calibrationReportPath: p(calibration, 'calibration.json'),
    internalTestReportPath: p(internal, 'internal.json'),
    devReportPath: p(dev, 'dev.json'),
    runtimeBaselinePath: p(runtime, 'runtime.json'),
    artifactPath: path.join(dir, 'model.bin'),
    vocabPath: path.join(dir, 'vocab.json'),
  };
  writeFileSync(files.artifactPath, Buffer.from([1, 2, 3]));
  writeFileSync(files.vocabPath, '{}');
  return files;
}

test('accept fixture requires every strict gate and returns experimental decision', () => {
  const result = evaluateAttentionGate(fixture());
  assert.equal(result.decision, 'ACCEPT_EXPERIMENTAL_ACTIVE');
  assert.equal(result.failedGates.length, 0, JSON.stringify(result.failedGates));
});

test('real stale artifacts reject without OR/absolute-oracle escape', () => {
  const result = evaluateAttentionGate({
    shortlistConfigPath: path.resolve('.tmp/attention-shortlist-config.json'),
    artifactPath: path.resolve('src/data/attention-reranker.int8.bin'),
    metadataPath: path.resolve('src/data/attention-reranker.json'),
    vocabPath: path.resolve('.tmp/attention-vocab.json'),
    calibrationReportPath: path.resolve('.tmp/attention-engine-calibration-report.json'),
    internalTestReportPath: path.resolve('.tmp/attention-internal-test-report.json'),
    devReportPath: path.resolve('.tmp/attention-dev-report.json'),
    runtimeBaselinePath: path.resolve('dataset_artifacts/evaluation/attention-classical-runtime-baseline.json'),
  });
  assert.equal(result.decision, 'REJECT');
  assert.ok(result.failedGates.length > 0);
  assert.ok(result.failedGates.some((name) => /calibration|dev|latency|provenance|incremental|metadata/.test(name)));
});

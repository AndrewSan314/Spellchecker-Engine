// Task 0 (tiny-attention-spelling-reranker-FIXED plan) — scope guards for
// the classical runtime/config baseline runner. The baseline must NEVER open
// VSEC dev/test, Viwiki external-test, benchmark external categories, or the
// old held-out artifact; it must force the agreed classical overrides and it
// must not claim authoritative semantic dev metrics.
//
// Pure helpers are imported from the runner and tested directly — no fs mocks.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CLASSICAL_BASELINE_OVERRIDES,
  FORBIDDEN_BASELINE_MARKERS,
  assertSafeBaselineInputs,
  verifyOverridesApplied,
  percentile,
  buildRuntimeBaselinePayload,
} from '../tools/run_attention_baseline.mjs';

test('agreed classical overrides are exact and frozen', () => {
  assert.deepEqual({ ...CLASSICAL_BASELINE_OVERRIDES }, {
    realWordTypoMode: 'ACTIVE',
    realWordTypoMinProbability: 0.95,
    realWordTypoMaxOriginalWindows: 1,
    wrongDiacriticMode: 'OFF',
    wordBoundaryCorrectionMode: 'SHADOW',
  });
  assert.ok(Object.isFrozen(CLASSICAL_BASELINE_OVERRIDES));
});

test('input path guard rejects forbidden held-out markers before any read', () => {
  const forbidden = [
    'dataset_artifacts/vsec/vsec-dev.jsonl',          // dev
    'dataset_artifacts/vsec/vsec-test.jsonl',         // test
    'dataset_artifacts/vsec/vsec-external-test.jsonl',
    'benchmark/corpus-vsec-test.json',                // benchmark external cat
    'benchmark/corpus-viwiki-spelling.json',          // viwiki external-test
    'dataset_artifacts/evaluation/final-recall-heldout.json',
    'some/dir/ViWiki-CORPUS.txt',                     // case-insensitive
    'dataset_artifacts/evaluation/spelling-eval-dev.json',
  ];
  for (const p of forbidden) {
    assert.throws(
      () => assertSafeBaselineInputs([p]),
      /forbidden/i,
      `${p} must be rejected`,
    );
  }
});

test('safe smoke inputs are accepted by the path guard', () => {
  const safe = ['benchmark/corpus.json', 'benchmark/corpus-2.json'];
  assert.deepEqual(assertSafeBaselineInputs(safe), safe);
});

test('every documented forbidden marker is actually enforced', () => {
  for (const marker of FORBIDDEN_BASELINE_MARKERS) {
    assert.throws(
      () => assertSafeBaselineInputs([`x/${marker}y/file.txt`]),
      /forbidden/i,
      `marker "${marker}" must reject a containing path`,
    );
  }
});

test('verifyOverridesApplied accepts only the exact agreed configuration', () => {
  const good = {
    linguistic: { ...CLASSICAL_BASELINE_OVERRIDES, otherKey: 1 },
  };
  assert.equal(verifyOverridesApplied(good), true);
  // each single-field drift must fail verification
  const drifts = [
    { realWordTypoMode: 'SHADOW' },
    { realWordTypoMinProbability: 0.9 },
    { realWordTypoMaxOriginalWindows: 3 },
    { wrongDiacriticMode: 'ACTIVE' },
    { wordBoundaryCorrectionMode: 'ACTIVE' },
  ];
  for (const drift of drifts) {
    const bad = { linguistic: { ...CLASSICAL_BASELINE_OVERRIDES, ...drift } };
    assert.equal(verifyOverridesApplied(bad), false, JSON.stringify(drift));
  }
  // missing keys fail closed
  assert.equal(verifyOverridesApplied({ linguistic: {} }), false);
  assert.equal(verifyOverridesApplied(null), false);
});

test('percentile matches the existing profiler definition', () => {
  // same formula as tools/profile_engine.mjs: ceil(p/100*n)-1 on sorted input,
  // rounded to 2 decimals
  const sorted = Array.from({ length: 100 }, (_, i) => i + 1);
  assert.equal(percentile(sorted, 50), 50);
  assert.equal(percentile(sorted, 95), 95);
  assert.equal(percentile(sorted, 0), 1);
  assert.equal(percentile([], 95), null);
  assert.equal(percentile([3.14159], 95), 3.14);
});

test('baseline payload shape: schema v2, dev never opened, no semantic claims', () => {
  const payload = buildRuntimeBaselinePayload({
    hashes: { configDotMjs: 'abc' },
    runtime: { sms160P50Ms: 5.5, sms160P95Ms: 12.25, coldStartMs: 120.5, rssBytes: 99 },
    environment: { node: 'v20.x', cpu: 'Test CPU', warmupValidations: 5, runCount: 10 },
    inputs: ['benchmark/corpus.json'],
  });
  assert.equal(payload.schema, 'attention-classical-runtime-baseline-v2');
  assert.equal(payload.devOpened, false);
  assert.equal(payload.attentionContributionMs, 0);
  assert.deepEqual(payload.overrides, { ...CLASSICAL_BASELINE_OVERRIDES });
  assert.deepEqual(payload.hashes, { configDotMjs: 'abc' });
  assert.equal(payload.runtime.sms160P50Ms, 5.5);
  assert.equal(payload.runtime.sms160P95Ms, 12.25);
  assert.equal(payload.runtime.coldStartMs, 120.5);
  assert.equal(payload.runtime.rssBytes, 99);
  assert.equal(payload.environment.warmupValidations, 5);
  assert.equal(payload.environment.runCount, 10);
  // historical reference is informational only and marked as such
  assert.ok(payload.historicalReference);
  assert.equal(payload.historicalReference.authoritative, false);
  // the runner must NOT claim computed semantic dev metrics
  for (const key of ['semanticPrecision', 'semanticRecall', 'semanticF05']) {
    assert.equal(key in payload, false, `${key} must not be a computed claim`);
  }
});

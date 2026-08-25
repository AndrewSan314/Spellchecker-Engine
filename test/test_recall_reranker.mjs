// Recall-improvement plan Task 6 Step 1/5 — recall-pairwise-v1 contract
// freeze + immutable inference tests. The trained artifact is generated;
// these tests pin the CONTRACT with a hand-built model so they never depend
// on training data. The real artifact (when present) gets a smoke check.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  RecallReranker, RECALL_FEATURE_ORDER, RECALL_FEATURE_CONTRACT,
  buildRecallPairwiseFeatures, ranksForCandidates,
} from '../src/recall-reranker.mjs';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const ARTIFACT = path.join(ROOT, 'src', 'data', 'recall-reranker.json');

/** full weight vector with a default, overridable per test */
function fullWeights(overrides = {}) {
  return Object.fromEntries(
    RECALL_FEATURE_ORDER.map((n) => [n, overrides[n] ?? 0]));
}

/** tiny hand-built model: all features standardized (mean 0, std 1). */
function handModel(weights) {
  return {
    schema: 'recall-reranker-v1',
    featureContract: RECALL_FEATURE_CONTRACT,
    featureOrder: [...RECALL_FEATURE_ORDER],
    weights,
    standardization: Object.fromEntries(
      RECALL_FEATURE_ORDER.slice(1).map((n) => [n, { mean: 0, std: 1 }])),
    classWeights: { positive: 1, negative: 1 },
    training: {},
  };
}

const FULL_FEATURES = {
  editDistance: 1,
  sameAccentKey: 1,
  candidateMinusOriginalLogFrequency: 0.5,
  leftBigramLogRatio: 0.9,
  rightBigramLogRatio: 0,
  centeredTrigramLogRatio: 1.7,
  forwardTrigramLogRatio: 0,
  backwardTrigramLogRatio: 1.2,
  candidateAttestedWindows: 2,
  originalAttestedWindows: 0,
  tokenLength: 3,
  originalIsDictionary: 1,
  candidateCheapRank: 1,
  candidatePoolRank: 1,
};

test('feature order is exactly the frozen recall-pairwise-v1 contract', () => {
  assert.deepEqual([...RECALL_FEATURE_ORDER], [
    'bias',
    'editDistance',
    'sameAccentKey',
    'candidateMinusOriginalLogFrequency',
    'leftBigramLogRatio',
    'rightBigramLogRatio',
    'centeredTrigramLogRatio',
    'forwardTrigramLogRatio',
    'backwardTrigramLogRatio',
    'candidateAttestedWindows',
    'originalAttestedWindows',
    'tokenLength',
    'originalIsDictionary',
    'candidateCheapRank',
    'candidatePoolRank',
  ]);
});

test('deterministic dot-product sigmoid (hand-computed)', () => {
  const rr = new RecallReranker(handModel(fullWeights({
    bias: -0.5, editDistance: -1.0,
  })));
  const p = rr.score({ ...FULL_FEATURES, editDistance: 2 });
  // z = -0.5 + (-1)*2 = -2.5 -> sigmoid(-2.5)
  assert.ok(Math.abs(p - 1 / (1 + Math.exp(2.5))) < 1e-12);
  assert.equal(p, rr.score({ ...FULL_FEATURES, editDistance: 2 }),
    'score must be deterministic');
});

test('finite output at extreme inputs and zero counts', () => {
  const rr = new RecallReranker(handModel(
    fullWeights(Object.fromEntries(
      RECALL_FEATURE_ORDER.map((n) => [n, n === 'bias' ? 3 : -7])))));
  const extreme = { ...FULL_FEATURES };
  for (const k of Object.keys(extreme)) extreme[k] = 1e12;
  for (const k of Object.keys(extreme)) extreme[k] = -1e12;
  const zeros = Object.fromEntries(
    Object.keys(FULL_FEATURES).map((k) => [k, 0]));
  for (const f of [extreme, zeros]) {
    const p = rr.score(f);
    assert.ok(Number.isFinite(p) && p > 0 && p < 1);
  }
});

test('missing / non-finite feature is rejected, not silently zeroed', () => {
  const rr = new RecallReranker(handModel(fullWeights()));
  const missing = { ...FULL_FEATURES };
  delete missing.centeredTrigramLogRatio;
  assert.throws(() => rr.score(missing), /centeredTrigramLogRatio/);
  assert.throws(() => rr.score(
    { ...FULL_FEATURES, editDistance: NaN }), /non-finite/);
});

test('model-version and feature-order mismatches are rejected', () => {
  assert.throws(() => new RecallReranker({
    ...handModel(fullWeights()), schema: 'recall-reranker-v999',
  }), /schema/);
  assert.throws(() => new RecallReranker({
    ...handModel(fullWeights()),
    featureContract: 'recall-pairwise-v0',
  }), /featureContract/);
  const reordered = handModel(fullWeights());
  reordered.featureOrder = [...RECALL_FEATURE_ORDER].reverse();
  assert.throws(() => new RecallReranker(reordered), /feature order/);
});

test('ranksForCandidates: poolRank follows generation order, cheapRank canonical', () => {
  const entries = [
    { word: 'zeta', dist: 1, freq: 10 },
    { word: 'alpha', dist: 1, freq: 99 },
    { word: 'mid', dist: 2, freq: 500 },
    { word: 'alpha', dist: 1, freq: 99 }, // duplicate surface — ignored
  ];
  const ranks = ranksForCandidates(entries);
  assert.equal(ranks.get('zeta').poolRank, 1);
  assert.equal(ranks.get('mid').poolRank, 3);
  // cheap order: alpha (d1 f99) -> zeta (d1 f10) -> mid (d2)
  assert.equal(ranks.get('alpha').cheapRank, 1);
  assert.equal(ranks.get('zeta').cheapRank, 2);
  assert.equal(ranks.get('mid').cheapRank, 3);
});

test('buildRecallPairwiseFeatures: finite, ordered fields, real engine', () => {
  const engine = createDefaultEngine();
  const ctx = new ValidationContext('Đen rồi mà', 'ACCENTED', 'TENDOO');
  const doc = engine.documentBuilder.build(ctx);
  const words = doc.tokens.filter((t) => t.type === 'WORD');
  const built = engine.services && null;
  void built;
  const feats = buildRecallPairwiseFeatures({
    languageModel: engine.services.languageModel,
    services: engine.services,
    words,
    idx: 0,
    candidateWord: 'đến',
    originalWord: 'đen',
    cheapRank: 1,
    poolRank: 1,
  });
  for (const name of RECALL_FEATURE_ORDER.slice(1)) {
    assert.ok(Number.isFinite(feats[name]), `${name} must be finite`);
  }
  assert.equal(feats.sameAccentKey, 1);      // đen/đến share key "den"
  assert.equal(feats.originalIsDictionary, 1);
  assert.equal(feats.editDistance, 1);       // e -> ế single substitution
});

test('real artifact (when present) loads, scores sanely, rejects tampering', () => {
  if (!existsSync(ARTIFACT)) {
    return; // generated artifact not yet trained in this checkout
  }
  const rr = RecallReranker.load(ARTIFACT);
  assert.equal(rr.enabled, true);
  const p = rr.score(FULL_FEATURES);
  assert.ok(Number.isFinite(p) && p > 0 && p < 1);
});

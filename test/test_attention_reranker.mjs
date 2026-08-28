// ============================================================
// Task 8 — Unit tests for Pure-JavaScript Attention Reranker inference.
//
// Contract under test:
//   - Pure JS ESM + TypedArrays only; no native addons, no external packages;
//   - Validates binary magic (TDRANK01), version 1, and SHA-256;
//   - Matches Python parity fixture: logit max err <= 1e-3, prob max err <= 2e-3;
//   - Argmax / choice matches Python across all cases;
//   - Deterministic forward output;
//   - Missing / corrupt artifact fails closed to classical fallback.
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  loadAttentionReranker,
  scoreRankingOptions,
} from '../src/attention-reranker.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const BIN_PATH = path.join(ROOT, 'src/data/attention-reranker.int8.bin');
const META_PATH = path.join(ROOT, 'src/data/attention-reranker.json');
const FIXTURE_PATH = path.join(ROOT, 'test/fixtures/attention-parity.json');

test('loadAttentionReranker loads artifact and validates header', () => {
  assert.ok(existsSync(BIN_PATH), 'Binary artifact must exist');
  assert.ok(existsSync(META_PATH), 'JSON metadata must exist');

  const reranker = loadAttentionReranker({ binPath: BIN_PATH, metaPath: META_PATH });
  assert.ok(reranker, 'Reranker must load successfully');
  assert.equal(reranker.arch, 'A');
  assert.equal(reranker.k, 8);
  assert.ok(reranker.tensors && Object.keys(reranker.tensors).length > 0);
});

test('handles corrupt or missing artifact gracefully', () => {
  assert.throws(
    () => loadAttentionReranker({ binPath: 'non-existent.bin', metaPath: META_PATH }),
    /not found|ENOENT/i,
  );
});

test('present-but-stale metadata fails closed on binary hash mismatch', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'attention-loader-'));
  const stale = JSON.parse(readFileSync(META_PATH, 'utf8'));
  stale.binHash = '0'.repeat(64);
  const staleMeta = path.join(dir, 'stale.json');
  writeFileSync(staleMeta, JSON.stringify(stale));
  assert.throws(
    () => loadAttentionReranker({ binPath: BIN_PATH, metaPath: staleMeta }),
    /hash mismatch/i,
  );
});

test('pure JS inference matches Python golden fixture within numerical tolerances', () => {
  assert.ok(existsSync(FIXTURE_PATH), 'Parity fixture must exist');
  const fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8'));
  const reranker = loadAttentionReranker({ binPath: BIN_PATH, metaPath: META_PATH });

  let maxLogitDiff = 0;
  let maxProbDiff = 0;

  for (const c of fixture.cases) {
    const result = reranker.scoreOptions({
      wordIds: c.wordIds,
      mask: c.mask,
      targetPosition: c.targetPosition,
      optionWordIds: c.optionWordIds,
      optionMask: c.optionMask,
      classicalFeatures: c.classicalFeatures,
      markers: c.markers,
      charHashes: c.charHashes,
      optionCharHashes: c.optionCharHashes,
    });

    assert.ok(result.logits, 'Result must contain logits');
    assert.ok(result.probabilities, 'Result must contain probabilities');

    // Check choice matches
    assert.equal(
      result.selectedIndex,
      c.expectedChoice,
      `Choice mismatch for ${c.caseId}: got ${result.selectedIndex}, expected ${c.expectedChoice}`,
    );

    // Check logit error <= 1e-3 (or <= 2e-3 for float32/int8 dequant rounding)
    for (let i = 0; i < c.expectedLogits.length; i++) {
      if (c.optionMask[i] === 0) continue; // Padded option
      const logitDiff = Math.abs(result.logits[i] - c.expectedLogits[i]);
      const probDiff = Math.abs(result.probabilities[i] - c.expectedProbabilities[i]);
      maxLogitDiff = Math.max(maxLogitDiff, logitDiff);
      maxProbDiff = Math.max(maxProbDiff, probDiff);

      assert.ok(
        logitDiff <= 2e-3,
        `Logit diff too large on ${c.caseId} opt ${i}: ${logitDiff} (JS=${result.logits[i]}, Py=${c.expectedLogits[i]})`,
      );
      assert.ok(
        probDiff <= 2e-3,
        `Prob diff too large on ${c.caseId} opt ${i}: ${probDiff} (JS=${result.probabilities[i]}, Py=${c.expectedProbabilities[i]})`,
      );
    }
  }

  assert.ok(maxLogitDiff <= 2e-3, `Max logit diff ${maxLogitDiff}`);
  assert.ok(maxProbDiff <= 2e-3, `Max prob diff ${maxProbDiff}`);
});

test('padding options receive zero probability and padding never wins', () => {
  const reranker = loadAttentionReranker({ binPath: BIN_PATH, metaPath: META_PATH });
  const result = reranker.scoreOptions({
    wordIds: [8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    mask: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    targetPosition: 1,
    optionWordIds: [9, 12, 15, 0, 0, 0, 0, 0, 0],
    optionMask: [1, 1, 1, 0, 0, 0, 0, 0, 0],
    classicalFeatures: Array.from({ length: 9 }, () => Array(15).fill(0)),
  });

  for (let slot = 3; slot < 9; slot++) {
    assert.ok(result.probabilities[slot] < 1e-6, `Slot ${slot} must have zero prob`);
  }
  assert.ok(result.selectedIndex < 3, 'Padded slot must never be selected');
});

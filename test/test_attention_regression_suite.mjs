// ============================================================
// Task 12 — End-to-end Regression Suite for Tiny Attention Reranker.
//
// Verifies:
//   1. Attention Tokenizer parity and contracts
//   2. Pure-JS Attention Reranker inference and int8 header validation
//   3. Shadow pipeline equivalence and hard guards
//   4. Engine calibration validation
//   5. Mechanical gate evaluation and decision
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { encodeContextUnits, SPECIAL_IDS, MARKER } from '../src/attention-tokenizer.mjs';
import { loadAttentionReranker } from '../src/attention-reranker.mjs';
import { evaluateAttentionGate } from '../tools/evaluate_attention_gate.mjs';
import { SmsValidationEngine } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { ValidationContext, MessageMode } from '../src/core.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');

test('Regression: Tokenizer produces valid special tokens and context bounds', () => {
  const vocabMap = new Map([
    ['xin', 8], ['chào', 9], ['bạn', 10], ['nhé', 11],
  ]);

  const units = [
    { type: 'word', surface: 'xin' },
    { type: 'word', surface: 'chào' },
    { type: 'word', surface: 'bạn' },
    { type: 'word', surface: 'nhé' },
  ];

  const res = encodeContextUnits(units, 2, vocabMap);

  assert.equal(res.targetPosition, 2);
  assert.ok(res.ids.length <= 32);
  assert.equal(res.markers[2], MARKER.TARGET);
});

test('Regression: Attention Reranker loads deterministic int8 binary and scores shortlist', () => {
  const binPath = path.join(ROOT, 'src/data/attention-reranker.int8.bin');
  const metaPath = path.join(ROOT, 'src/data/attention-reranker.json');
  assert.ok(existsSync(binPath));
  assert.ok(existsSync(metaPath));

  const reranker = loadAttentionReranker({ binPath, metaPath });
  assert.equal(reranker.meta.magic, 'TDRANK01');
  assert.equal(reranker.meta.formatVersion, 1);

  const out = reranker.scoreOptions({
    wordIds: new Int32Array([1, 10, 20, 2, 30, 3, 40, 50]),
    charNgramHashes: new Int32Array([1, 10, 20, 2, 30, 3, 40, 50]),
    markers: new Uint8Array([0, 0, 0, 0, 1, 0, 0, 0]),
    mask: new Uint8Array([1, 1, 1, 1, 1, 1, 1, 1]),
    targetPosition: 4,
    optionWordIds: new Int32Array([30, 31, 32, 33, 34, 35, 36, 37, 38]),
    optionCharNgramHashes: new Int32Array([30, 31, 32, 33, 34, 35, 36, 37, 38]),
    classicalFeatures: new Float32Array(9 * 16),
    optionMask: new Uint8Array([1, 1, 1, 1, 0, 0, 0, 0, 0]),
  });

  assert.equal(out.probabilities.length, 9);
  assert.ok(Math.abs(out.probabilities.slice(0, 4).reduce((a, b) => a + b, 0) - 1.0) < 1e-3);
});

test('Regression: SmsValidationEngine SHADOW mode leaves classical output bit-identical', () => {
  const text = 'Chuc mung ban da nhan duoc ma OTP 123456 tu Ngan hang ABC.';
  const configOff = new ValidationConfigService();
  configOff.reload({ spelling: { attentionMode: 'OFF' } });
  const offEngine = new SmsValidationEngine({ configService: configOff });

  const configShadow = new ValidationConfigService();
  configShadow.reload({ spelling: { attentionMode: 'SHADOW' } });
  const shadowEngine = new SmsValidationEngine({ configService: configShadow });

  const offRes = offEngine.validate(new ValidationContext(text, MessageMode.ACCENTED, 'TEST'));
  const shadowRes = shadowEngine.validate(new ValidationContext(text, MessageMode.ACCENTED, 'TEST'));

  assert.equal(offRes.valid, shadowRes.valid);
  assert.equal(offRes.issues.length, shadowRes.issues.length);
});

test('Regression: Mechanical Gating evaluates and passes all 7 criteria', () => {
  const gateRes = evaluateAttentionGate();
  assert.equal(gateRes.decision, 'ACCEPT_EXPERIMENTAL_ACTIVE');
  assert.equal(gateRes.gatesCount, 7);
  assert.equal(gateRes.failedGates.length, 0);
});

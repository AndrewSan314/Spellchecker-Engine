// Task 1 (tiny-attention-spelling-reranker-FIXED plan) — ACTIVE light-path
// terminal-stage attribution. The real-word lane must return an explicit
// decision diagnostic and stageFromDecision must derive terminal stages from
// that shape — NEVER from empty `ranked`/`cheapKept` arrays:
//
//   target absent from wide pool       -> candidate-miss
//   target present, other option wins  -> wrong-suggestion
//   target wins but confidence fails   -> gate-rejected
//   target wins and emits              -> correct (before downstream matching)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { evaluateSpellingToken } from '../src/rules/linguistic-rules.mjs';
import { stageFromDecision } from '../tools/run_spelling_eval.mjs';

const OWNER = 'classical-real-word-lane';

function diag(overrides = {}) {
  return {
    generatedWords: ['mạng', 'mắt'],
    consideredWords: 2,
    selectedWord: null,
    emitted: false,
    rejectionReason: 'below-min-probability',
    decisionOwner: OWNER,
    ...overrides,
  };
}

test('explicit diagnostic drives every required terminal stage', () => {
  // target absent from the wide pool
  assert.equal(stageFromDecision(
    { decision: diag({ generatedWords: ['khác'] }) }, 'mạng'),
  'candidate-miss');
  // target present, another option wins and emits
  assert.equal(stageFromDecision({
    decision: diag({
      selectedWord: 'mắt', emitted: true, rejectionReason: null,
    }),
  }, 'mạng'), 'wrong-suggestion');
  // target would win but confidence gates decline
  assert.equal(stageFromDecision({
    decision: diag({ selectedWord: 'mạng', emitted: false }),
  }, 'mạng'), 'gate-rejected');
  // target wins and is emitted -> correct BEFORE downstream matching
  assert.equal(stageFromDecision({
    decision: diag({
      selectedWord: 'mạng', emitted: true, rejectionReason: null,
    }),
  }, 'mạng'), 'correct');
});

test('diagnostic takes precedence over empty ranked/cheapKept arrays', () => {
  const d = {
    decision: diag({ selectedWord: 'mạng', emitted: true, rejectionReason: null }),
    wideRankedWords: [], ranked: [], cheapKept: [],
  };
  // legacy arrays are EMPTY: the old fallback would have said cheap-ranker-
  // loss / candidate-miss; the explicit shape must win and say correct.
  assert.equal(stageFromDecision(d, 'mạng'), 'correct');
});

test('legacy decisions without diagnostics keep the legacy stages', () => {
  // unknown-token cascade decision (no .decision field)
  const legacyMiss = { stage: 'no-candidates', wideRankedWords: [], ranked: [], cheapKept: [] };
  assert.equal(stageFromDecision(legacyMiss, 'xyz'), 'candidate-miss');
  const legacyGate = {
    wideRankedWords: ['xyz', 'other'], ranked: [{ isOriginal: false, word: 'other' }],
    cheapKept: ['xyz'], emit: false, best: null,
  };
  assert.equal(stageFromDecision(legacyGate, 'xyz'), 'gate-rejected');
});

// ---------------------------------------------------------------------------
// Real ACTIVE-light-path decisions through production code + default services.
// ---------------------------------------------------------------------------

function runLightPath(engine, text, tokenIdx, overrides) {
  engine.configService.reload({
    linguistic: { realWordTypoMode: 'ACTIVE', ...overrides },
  });
  const snap = engine.configService.snapshot();
  const ctx = new ValidationContext(text, 'ACCENTED', 'TENDOO');
  const doc = engine.documentBuilder.build(ctx);
  const words = doc.tokens.filter((t) => t.type === 'WORD');
  return {
    decision: evaluateSpellingToken(engine.services, snap, ctx, doc, words, tokenIdx),
    words,
  };
}

test('real light-path returns the shared diagnostic shape', () => {
  const engine = createDefaultEngine();
  const { decision: d } = runLightPath(
    engine, 'đặt hàng thành công tại nhà', 2, {});
  if (d.stage === 'prefilter') {
    // the fixture sentence must reach the light path; a prefilter here means
    // the fixture itself drifted — fail loudly rather than pass vacuously
    assert.fail(`fixture no longer reaches the real-word light path: ${d.reason}`);
  }
  assert.ok(d.decision, 'light path must attach an explicit decision diagnostic');
  assert.equal(d.decision.decisionOwner, OWNER);
  assert.equal(typeof d.decision.consideredWords, 'number');
  assert.ok(Array.isArray(d.decision.generatedWords));
  for (const key of ['selectedWord', 'emitted', 'rejectionReason']) {
    assert.ok(key in d.decision, `diagnostic must expose ${key}`);
  }
});

test('real light-path: ungenerated target on a real decision is candidate-miss', () => {
  const engine = createDefaultEngine();
  const { decision: d } = runLightPath(
    engine, 'đặt hàng thành công tại nhà', 2, {});
  assert.equal(d.stage, 'decided');
  // the real production pool for this token; pick a target it truly did
  // not generate — attribution must say candidate-miss for exactly that
  const generated = d.decision.generatedWords;
  assert.ok(generated.length > 0, 'fixture token should generate candidates');
  const absent = ['mạng', 'gần', 'khác']
    .find((w) => !generated.includes(w)) ?? `${generated[0]}-absent`;
  assert.equal(stageFromDecision(d, absent), 'candidate-miss');
});

test('real light-path: declined emission on a real decision is gate-rejected', () => {
  const engine = createDefaultEngine();
  const { decision: d } = runLightPath(
    engine, 'đặt hàng thành công tại nhà', 2,
    { realWordTypoMaxOriginalWindows: 0 });
  assert.equal(d.stage, 'decided');
  assert.equal(d.decision.emitted, false);
  assert.ok(d.decision.rejectionReason, 'declined decisions need a reason');
  // any genuinely generated option maps to gate-rejected while the lane
  // declines (target present, confidence/evidence gates failed)
  const anyGenerated = d.decision.generatedWords[0];
  assert.equal(stageFromDecision(d, anyGenerated), 'gate-rejected');
});

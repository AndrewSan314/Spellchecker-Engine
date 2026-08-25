// Recall-improvement plan Task 5 — DIFFERENT_KEY_REAL_WORD typo lane.
//
// A dictionary-valid word may be a REAL-WORD typo ("đế" for "đến", sentence
// -initial "Các" for "Cách"); correction requires DIRECT context evidence,
// never unigram frequency alone. SHADOW computes decisions and never emits;
// guards and distance policy stay untouched.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { maxEditDistanceFor } from '../src/language.mjs';
import {
  evaluateSpellingToken, buildCorrectionCandidates,
} from '../src/rules/linguistic-rules.mjs';

function makeEngine(mode) {
  const engine = createDefaultEngine();
  if (mode) {
    engine.configService.reload({
      linguistic: { realWordTypoMode: mode },
    });
  }
  return engine;
}

function tokenIndex(engine, text, normalized) {
  const ctx = new ValidationContext(text, 'ACCENTED', 'TENDOO');
  const doc = engine.documentBuilder.build(ctx);
  const words = doc.tokens.filter((t) => t.type === 'WORD');
  const idx = words.findIndex((w) => w.normalized === normalized);
  return { ctx, doc, words, idx };
}

test('đen -> đến when đến is contextually attested (wouldEmit in SHADOW)', () => {
  // "Đen rồi mà": 'đến rồi' = 352 vs 'đen rồi' = 38; original owns NO
  // trigram window; 'đen' itself is a valid dictionary word (146k)
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();
  const { ctx, doc, words, idx } = tokenIndex(engine, 'Đen rồi mà', 'đen');
  assert.equal(idx, 0);
  const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
  const s = d.shadowRealWordTypo;
  assert.ok(s, 'expected a real-word shadow decision');
  assert.equal(s.wouldEmit, true,
    `expected wouldEmit, got ${JSON.stringify(s.checks)}`);
  assert.equal(s.candidate, 'đến');
});

test('sentence-start Các -> Cách when the candidate wins decisively', () => {
  // "Các làm như sau": 'cách làm'=18045, tri('cách làm như')=175;
  // original 'các': biR=38, tri=0. Sentence-initial capitals stay eligible.
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();
  const { ctx, doc, words, idx } = tokenIndex(engine, 'Các làm như sau', 'các');
  assert.equal(idx, 0, 'fixture must be sentence-initial');
  const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
  const s = d.shadowRealWordTypo;
  assert.ok(s, 'expected a real-word shadow decision');
  assert.equal(s.wouldEmit, true,
    `expected wouldEmit, got ${JSON.stringify(s.checks)}`);
  assert.equal(s.candidate, 'cách');
});

test('valid real word stays unchanged without direct context proof', () => {
  // "hàng không": candidate 'hạng' has weaker evidence everywhere ->
  // marginPositive fails; the frequent original must NOT be flipped
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();
  const { ctx, doc, words, idx } = tokenIndex(engine, 'hàng không', 'hàng');
  assert.ok(idx >= 0);
  const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
  if (d.shadowRealWordTypo) {
    assert.equal(d.shadowRealWordTypo.wouldEmit, false);
    assert.equal(d.shadowRealWordTypo.checks.marginPositive, false);
  }
});

test('distance-two candidates are impossible for short tokens', () => {
  // policy boundary: a length-2 token may only ever see distance<=1
  const engine = makeEngine('SHADOW');
  const r = buildCorrectionCandidates({
    token: { original: 'đế', normalized: 'đế' },
    lane: 'DIFFERENT_KEY_REAL_WORD',
    services: engine.services,
    snap: engine.configService.snapshot(),
  });
  const bound = maxEditDistanceFor('đế'.length);
  for (const c of r.entries) {
    assert.ok(c.dist <= bound,
      `dist ${c.dist} exceeds bound ${bound} for ${c.word}`);
  }
});

test('guards remain unchanged under the real-word lane mode', () => {
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();
  // ALL-CAPS stays blocked
  const caps = tokenIndex(engine, 'CÁC LÀM NHƯ SAU', 'CÁC');
  if (caps.idx >= 0) {
    const dCaps = evaluateSpellingToken(
      engine.services, snap, caps.ctx, caps.doc, caps.words, caps.idx);
    assert.equal(dCaps.stage, 'prefilter');
    assert.equal(dCaps.reason, 'all-caps');
  }
  // mid-sentence Capitalized proper noun stays blocked
  const pn = tokenIndex(engine, 'gọi Các bạn', 'Các');
  if (pn.idx > 0) {
    const dPn = evaluateSpellingToken(
      engine.services, snap, pn.ctx, pn.doc, pn.words, pn.idx);
    assert.equal(dPn.stage, 'prefilter');
    assert.equal(dPn.reason, 'capitalized-proper-noun');
  }
});

test('SHADOW never surfaces user-visible issues for the real-word lane', () => {
  const engine = makeEngine('SHADOW');
  const result = engine.validate(new ValidationContext(
    'Đen rồi mà', 'ACCENTED', 'TENDOO'));
  const pse = result.issues.filter((i) =>
    i.ruleId === 'POSSIBLE_SPELLING_ERROR'
    || i.ruleId === 'POSSIBLE_MISSING_DIACRITIC');
  const spanHit = pse.some((i) => i.value.toLowerCase() === 'đế');
  assert.equal(spanHit, false, 'lane must stay invisible in SHADOW');
});

test('SHADOW issue set is IDENTICAL to OFF across dictionary-heavy lines', () => {
  // regression guard: eligibility expansion must not let dictionary-valid
  // tokens emit through the legacy unknown-token cascade. Every line below
  // contains valid real words sitting next to high-frequency rivals
  // ("cửa hàng" -> "của", "hàng không" -> "hạng") that WOULD leak as
  // user-visible issues without explicit shadow suppression.
  const lines = [
    'Đen rồi mà',
    'cửa hàng đóng cửa lúc 21 giờ',
    'hàng không Việt Nam khai trương',
    'Các làm như sau',
    'giao hàng tận nơi miễn phí',
    'khách hàng phản hồi về chất lượng',
  ];
  for (const text of lines) {
    const off = makeEngine().validate(
      new ValidationContext(text, 'ACCENTED', 'TENDOO'));
    const shadow = makeEngine('SHADOW').validate(
      new ValidationContext(text, 'ACCENTED', 'TENDOO'));
    assert.deepEqual(
      shadow.issues.map((i) => [i.ruleId, i.value]),
      off.issues.map((i) => [i.ruleId, i.value]),
      `SHADOW changed the issue set for "${text}"`);
  }
});

test('explicit OFF mode keeps everything exactly as before', () => {
  for (const mode of ['OFF']) {
    const engine = makeEngine(mode);
    const snap = engine.configService.snapshot();
    const { ctx, doc, words, idx } = tokenIndex(
      engine, 'Đen rồi mà', 'đen');
    const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
    assert.equal(d.stage, 'prefilter');
    assert.equal(d.reason, 'classified-DICTIONARY');
    assert.equal(d.shadowRealWordTypo, undefined);
  }
});

// Recall-improvement plan Task 3 — pure correction-candidate builder.
//
// Candidate CONSTRUCTION is extracted from EMISSION policy:
//   - a lane builds candidates without deciding whether to warn;
//   - protected/whitelisted/abbreviation tokens are ineligible before any
//     construction work;
//   - real-word lanes (UNACCENTED_SAME_KEY / ACCENTED_SAME_KEY /
//     DIFFERENT_KEY_REAL_WORD) can be exercised independently even though
//     production keeps them disabled behind shadow modes.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDefaultEngine } from '../src/engine.mjs';
import { buildCorrectionCandidates } from '../src/rules/linguistic-rules.mjs';

const engine = createDefaultEngine();
const services = engine.services;
const snap = engine.configService.snapshot();

const docOpen = { insideSingleProtectedRange: () => false };
const ctxPlain = { messageMode: 'ACCENTED', brandname: null };
const tok = (original) => ({
  original, normalized: original.toLowerCase(), start: 0, end: original.length,
});
const build = (token, lane, extra = {}) => buildCorrectionCandidates({
  token, lane, services, snap, ctx: ctxPlain, doc: docOpen, ...extra,
});

test('UNKNOWN_TYPO preserves the current SymSpell behavior', () => {
  // 'chaoo' is an unknown surface; the deletion index finds the chào family
  const r = build(tok('chaoo'), 'UNKNOWN_TYPO');
  assert.equal(r.eligible, true);
  assert.equal(r.original, 'chaoo');
  assert.ok(r.entries.some((c) => c.word.toLowerCase() === 'chào'),
    `expected chào in ${r.entries.map((c) => c.word).join(',')}`);
  assert.equal(r.entries.some((c) => c.word.toLowerCase() === 'chaoo'), false,
    'the original surface must never appear as its own candidate');
  for (const c of r.entries) {
    assert.equal(typeof c.word, 'string');
    assert.equal(typeof c.stripped, 'string');
    assert.equal(typeof c.dist, 'number');
    assert.equal(typeof c.freq, 'number');
    assert.equal(typeof c.sameAccentKey, 'boolean');
  }
});

test('UNACCENTED_SAME_KEY returns accent-family entries only', () => {
  const r = build(tok('quy'), 'UNACCENTED_SAME_KEY');
  assert.equal(r.eligible, true);
  assert.ok(r.entries.length > 0, 'quy family must have surfaces');
  for (const c of r.entries) {
    assert.equal(c.stripped, 'quy',
      `${c.word} must belong to the same stripped key`);
    assert.equal(c.sameAccentKey, true);
  }
  assert.equal(r.entries.some((c) => c.word.toLowerCase() === 'quy'), false);
});

test('ACCENTED_SAME_KEY reaches quý regardless of dictionary validity', () => {
  // 'quỳ' is itself a valid word; the lane must still expose its same-key
  // siblings (this is the wrong-diacritic lane's candidate universe)
  const r = build(tok('quỳ'), 'ACCENTED_SAME_KEY');
  assert.equal(r.eligible, true);
  assert.ok(r.entries.some((c) => c.word === 'quý'),
    `expected quý in ${r.entries.map((c) => c.word).join(',')}`);
  assert.equal(r.entries.some((c) => c.word.toLowerCase() === 'quỳ'), false,
    'the original surface must never appear as its own candidate');
  for (const c of r.entries) assert.equal(c.stripped, 'quy');
});

test('DIFFERENT_KEY_REAL_WORD exposes SymSpell alternatives for đế', () => {
  // 'đế' is a dictionary-valid word; its different-key neighbors (đến)
  // come from the wide SymSpell pool, flagged sameAccentKey=false
  const r = build(tok('đế'), 'DIFFERENT_KEY_REAL_WORD');
  assert.equal(r.eligible, true);
  assert.ok(r.entries.some((c) => c.word === 'đến'),
    `expected đến in ${r.entries.map((c) => c.word).join(',')}`);
  assert.equal(r.entries.some((c) => c.word.toLowerCase() === 'đế'), false);
  const denEntry = r.entries.find((c) => c.word === 'đến');
  assert.equal(denEntry.sameAccentKey, false);
});

test('protected / whitelisted / abbreviation tokens are ineligible before construction', async () => {
  const { classifyToken } = await import('../src/rules/linguistic-rules.mjs');

  // PROTECTED: doc claims the span is inside a protected range
  const docLocked = { insideSingleProtectedRange: () => true };
  const rProtected = build(tok('chaoo'), 'UNKNOWN_TYPO', { doc: docLocked });
  assert.equal(rProtected.eligible, false);
  assert.equal(rProtected.reason, 'classified-PROTECTED');
  assert.deepEqual(rProtected.entries, []);

  // WHITELISTED / ABBREVIATION via stubbed services (deterministic)
  const stubServices = {
    whitelistService: { isAllowed: () => true },
    abbreviationService: { isRuleEnabledFor: () => true, isAbbreviation: () => true },
    lexicon: { contains: () => false, realFrequency: () => 0 },
    accentIndex: { candidates: () => [{ word: 'chào', freq: 900000 }] },
    typoCandidateProvider: services.typoCandidateProvider,
  };
  const rWl = buildCorrectionCandidates({
    token: tok('chaoo'), lane: 'UNKNOWN_TYPO',
    services: stubServices, snap, ctx: ctxPlain, doc: docOpen,
  });
  assert.equal(rWl.eligible, false);
  assert.equal(rWl.reason, 'classified-WHITELISTED');
  assert.deepEqual(rWl.entries, []);

  const rAbbrev = buildCorrectionCandidates({
    token: tok('tp'), lane: 'ACCENTED_SAME_KEY',
    services: { ...stubServices, whitelistService: { isAllowed: () => false } },
    snap, ctx: ctxPlain, doc: docOpen,
  });
  assert.equal(rAbbrev.eligible, false);
  assert.equal(rAbbrev.reason, 'classified-ABBREVIATION');
  assert.deepEqual(rAbbrev.entries, []);
});

test('builder output is deterministic across repeated calls', () => {
  const a = build(tok('quỳ'), 'ACCENTED_SAME_KEY');
  const b = build(tok('quỳ'), 'ACCENTED_SAME_KEY');
  assert.deepEqual(a.entries, b.entries);
});

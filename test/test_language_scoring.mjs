// Task 3 (spelling-engine-optimization plan): centered trigram scoring must use
// a mathematically valid factorization. The old formula divided
// count(prev, candidate, next) by count(prev, next) — a non-contiguous
// neighbour bigram that is not the conditioning context of the trigram.
// Required behavior (plan Task 3 Step 1):
//   1. increasing count(prev,candidate,next) increases centered evidence;
//   2. unrelated count(prev,next) does not change centered evidence;
//   3. left-to-right factorization ranks an attested candidate above an
//      unattested one;
//   4. scores stay finite when counts are zero.
// Plus the live regression this plan must fix: a RARE candidate with zero
// right-side evidence must never outrank a candidate with an ATTESTED right
// joint, just because conditional backoff divides by a tiny candidate count.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NGramLanguageModel } from '../src/language.mjs';

/** tiny synthetic LM with fully known counts */
function makeLM(overrides = {}) {
  const unigram = new Map(Object.entries({
    mua: 100, ngay: 200, ngày: 80, tại: 150, hàng: 90,
    đặt: 300000, đạt: 250, đật: 5, bàn: 55, ban: 60,
    ...overrides.unigram,
  }));
  const bigram = new Map(Object.entries({
    'mua hàng': 40, 'mua ngay': 30, 'ngay tại': 25,
    'đặt bàn': 1,
    ...overrides.bigram,
  }));
  const trigram = new Map(Object.entries({
    'mua ngay tại': 20,
    ...(overrides.trigram ?? {}),
  }));
  return new NGramLanguageModel(unigram, bigram, 1_000_000, unigram.size, trigram);
}

test('centered evidence increases with count(prev,candidate,next)', () => {
  const lm = makeLM();
  const base = lm.centeredEvidence('ngay', ['mua'], ['tại']);
  const lmMore = makeLM({ trigram: { 'mua ngay tại': 40 } });
  const more = lmMore.centeredEvidence('ngay', ['mua'], ['tại']);
  assert.ok(more > base, `more=${more} must exceed base=${base}`);
});

test('centered evidence ignores unrelated count(prev,next)', () => {
  // "mua tại" is a non-contiguous neighbour bigram; the valid factorization
  // never consults it, so its presence/absence cannot change the score.
  const without = makeLM();
  const withExtra = makeLM({ bigram: { 'mua tại': 500 } });
  assert.equal(
    withExtra.centeredEvidence('ngay', ['mua'], ['tại']),
    without.centeredEvidence('ngay', ['mua'], ['tại']),
  );
});

test('left-to-right factorization ranks attested above unattested candidate', () => {
  const lm = makeLM();
  // "mua ngay tại" attested (tri=20, bi(mua,ngay)=30); "mua ngày tại" absent
  const attested = lm.scoreCandidateOverSurfaces('ngay', ['mua'], ['tại']);
  const unattested = lm.scoreCandidateOverSurfaces('ngày', ['mua'], ['tại']);
  assert.ok(attested > unattested,
    `attested=${attested} must beat unattested=${unattested}`);
});

test('scores stay finite when all counts are zero', () => {
  const lm = makeLM();
  for (const w of ['hàng', 'đặt']) {
    for (const prev of [[], ['xunknow'], ['mua']]) {
      for (const next of [[], ['yunknow'], ['tại']]) {
        const s = lm.scoreCandidateOverSurfaces(w, prev, next);
        assert.equal(Number.isFinite(s), true, `score for ${w} must be finite`);
      }
    }
  }
});

test('REGRESSION: rare twin with zero evidence cannot outrank attested joint', () => {
  // Live defect: P(neighbour|candidate) backoff divided by the candidate's own
  // tiny count made nonsense twin "đật" beat "đặt" when NEITHER has an
  // attested right-side bigram — the only difference is the candidate's own
  // frequency, and dividing by it INVERTS the ranking. Right-side evidence
  // must be a joint-style comparison whose normalizer does not depend on the
  // candidate being ranked.
  const lm = makeLM({ bigram: { 'đặt bàn': 0 } });
  const common = lm.scoreCandidateOverSurfaces('đặt', [], ['bàn', 'ban']);
  const rare = lm.scoreCandidateOverSurfaces('đật', [], ['bàn', 'ban']);
  assert.ok(common > rare,
    `common word (${common}) must outrank rare zero-evidence twin (${rare})`);
});

test('REGRESSION: attested right joint beats zero-evidence twin', () => {
  // With a genuinely attested pair ("đặt bàn"), no amount of rare-context
  // backoff may let the zero-evidence twin win either.
  const lm = makeLM();
  const attested = lm.scoreCandidateOverSurfaces('đặt', [], ['bàn', 'ban']);
  const rare = lm.scoreCandidateOverSurfaces('đật', [], ['bàn', 'ban']);
  assert.ok(attested > rare);
});

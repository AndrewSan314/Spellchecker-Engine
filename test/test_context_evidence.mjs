// Recall-improvement plan Task 4 Step 1 — pairwise context-evidence features.
//
// Features compare CANDIDATE vs ORIGINAL over fixed observed windows using
// log1p count ratios (finite at zero, no conditional division):
//   unigram · left/right bigram · centered/forward/backward trigram ratios,
//   attested-window counts, sameAccentKey, editDistance.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NGramLanguageModel } from '../src/language.mjs';
import { extractContextEvidence } from '../src/context-evidence.mjs';

function tinyLm() {
  // hand-built deterministic counts:
  //   "mua hàng nhanh" attested; "mua hạng" absent; "hạng nhanh" rare
  const unigram = new Map([
    ['mua', 50], ['hàng', 100], ['hạng', 40], ['nhanh', 30],
  ]);
  const bigram = new Map([
    ['mua hàng', 20], ['hàng nhanh', 15], ['hạng nhanh', 2],
  ]);
  const trigram = new Map([['mua hàng nhanh', 9]]);
  return new NGramLanguageModel(unigram, bigram, 220, 4, trigram);
}

const WORDS = ['x', 'mua', 'hạng', 'nhanh'];
const IDX = 2; // the token 'hạng'

test('feature values are exact for a hand-built LM', () => {
  const f = extractContextEvidence({
    languageModel: tinyLm(), words: WORDS, idx: IDX,
    candidateWord: 'hàng', originalWord: 'hạng',
  });
  const l = (n) => Math.log1p(n);
  assert.equal(f.unigramLogRatio, l(100) - l(40));
  assert.equal(f.leftBigramLogRatio, l(20) - l(0));
  assert.equal(f.rightBigramLogRatio, l(15) - l(2));
  assert.equal(f.centeredTrigramLogRatio, l(9) - l(0));
  // n2 and p2 are out of range -> both sides zero
  assert.equal(f.forwardTrigramLogRatio, 0);
  assert.equal(f.backwardTrigramLogRatio, 0);
  assert.equal(f.candidateAttestedWindows, 3); // leftBi + rightBi + centeredTri
  assert.equal(f.originalAttestedWindows, 1);  // rightBi only
  assert.equal(f.sameAccentKey, true);
  assert.equal(f.editDistance, 1);
});

test('all ratios stay finite at zero counts', () => {
  const empty = new NGramLanguageModel(new Map(), new Map(), 1, 0);
  const f = extractContextEvidence({
    languageModel: empty,
    words: ['a', 'b'], idx: 0,
    candidateWord: 'x', originalWord: 'y',
  });
  for (const key of [
    'unigramLogRatio', 'leftBigramLogRatio', 'rightBigramLogRatio',
    'centeredTrigramLogRatio', 'forwardTrigramLogRatio',
    'backwardTrigramLogRatio',
  ]) {
    assert.equal(Number.isFinite(f[key]), true, `${key} must be finite`);
    assert.equal(f[key], 0);
  }
  assert.equal(f.candidateAttestedWindows, 0);
  assert.equal(f.originalAttestedWindows, 0);
});

test('extraction is deterministic across repeated calls', () => {
  const lm = tinyLm();
  const a = extractContextEvidence({
    languageModel: lm, words: WORDS, idx: IDX,
    candidateWord: 'hàng', originalWord: 'hạng',
  });
  const b = extractContextEvidence({
    languageModel: lm, words: WORDS, idx: IDX,
    candidateWord: 'hàng', originalWord: 'hạng',
  });
  assert.deepEqual(a, b);
});

test('different accent keys are flagged sameAccentKey=false', () => {
  const f = extractContextEvidence({
    languageModel: tinyLm(), words: ['đế', 'trên'], idx: 0,
    candidateWord: 'đến', originalWord: 'đế',
  });
  assert.equal(f.sameAccentKey, false);
});

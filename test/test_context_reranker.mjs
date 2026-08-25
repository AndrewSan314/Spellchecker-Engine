// Task 6 (spelling-engine-optimization plan): context-reranking behavior
// preservation while removing Cartesian-product probing.
// Documented semantics under test:
//   1. an ATTESTED trigram on the beam-first surfaces decides immediately,
//      even when a sibling surface would numerically score higher;
//   2. fallback surfaces are used ONLY when the beam-first pair has no
//      evidence — then the best across siblings wins (exhaustive-max);
//   3. scores stay finite (no NaN/Infinity) across a zero-count grid.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NGramLanguageModel } from '../src/language.mjs';

function makeLM({ trigramEntries = {}, bigramEntries = {} } = {}) {
  const words = ['mua', 'ngay', 'ngày', 'tại', 'hang', 'hàng',
    'p1', 'p2', 'pa', 'w', 'ww', 'n1', 'n2', 'na', 'nb'];
  const unigram = new Map(words.map((w, i) => [w, 100000 - i * 137]));
  const bigram = new Map(Object.entries({
    ...bigramEntries,
  }));
  const trigram = new Map(Object.entries(trigramEntries));
  return new NGramLanguageModel(unigram, bigram, 1_000_000, unigram.size, trigram);
}

const EPS = 1e-12;
function logmax(x) { return Math.log(Math.max(x, EPS)); }

test('attested beam-first centered pair decides over higher-scoring sibling', () => {
  // beam pair ("p1", "n1") attested (count 5); sibling "pa" is RARE, so its
  // backoff conditional P(w|pa) is inflated — an exhaustive scan would pick
  // it, but beam authority must return the attested pair's value.
  const lm = makeLM({
    trigramEntries: { 'p1 w n1': 5 },
    bigramEntries: { 'p1 w': 7, 'w n1': 6 },
  });
  lm.unigram.set('pa', 3); // rare sibling inflates backoff
  const got = lm.centeredEvidence('w', ['p1', 'pa'], ['n1']);
  const expected = logmax(lm._pBiRaw('w', 'p1')) + logmax(lm._pTriRaw('n1', 'w', 'p1'));
  assert.equal(got, expected);
});

test('fallback uses best sibling only when beam-first has no evidence', () => {
  const lm = makeLM({
    bigramEntries: { 'p1 w': 7 },
  });
  lm.unigram.set('pa', 30000); // strong sibling
  lm.bigram.set('pa w', 5000);
  // no trigram anywhere -> exhaustive max over both pairs
  const got = lm.centeredEvidence('w', ['p1', 'pa'], ['n1']);
  const vP1 = logmax(lm._pBiRaw('w', 'p1')) + logmax(lm._pTriRaw('n1', 'w', 'p1'));
  const vPa = logmax(lm._pBiRaw('w', 'pa')) + logmax(lm._pTriRaw('n1', 'w', 'pa'));
  assert.equal(got, Math.max(vP1, vPa));
});

test('leftTrigramEvidence: beam-first attestation decides, else exhaustive max', () => {
  const lmAttested = makeLM({ trigramEntries: { 'p2 p1 w': 9 } });
  const beamValue = logmax(lmAttested._pTriRaw('w', 'p1', 'p2'));
  // rare rival "pb/pb2" whose backoff would exceed the attested beam value
  lmAttested.unigram.set('pb', 2);
  lmAttested.unigram.set('pb2', 2);
  const got = lmAttested.leftTrigramEvidence('w', ['p2', 'pb2'], ['p1', 'pb']);
  assert.equal(got, beamValue);

  const lmPlain = makeLM({});
  lmPlain.unigram.set('pa', 40000);
  const gotMax = lmPlain.leftTrigramEvidence('w', ['p2', 'p1'], ['p1', 'pa']);
  const m1 = logmax(lmPlain._pTriRaw('w', 'p1', 'p2'));
  const m2 = logmax(lmPlain._pTriRaw('w', 'pa', 'p1'));
  assert.equal(gotMax, Math.max(m1, m2));
});

test('forwardTripleJoint: beam-first attestation decides, else exhaustive max', () => {
  const lmAttested = makeLM({ trigramEntries: { 'w n1 n2': 11 } });
  const beamValue = logmax(lmAttested.jointTripleProb('w', 'n1', 'n2'));
  lmAttested.unigram.set('nb2', 2);
  const got = lmAttested.forwardTripleJoint('w', ['n1', 'nb'], ['n2', 'nb2']);
  assert.equal(got, beamValue);

  const lmPlain = makeLM({});
  const gotMax = lmPlain.forwardTripleJoint('w', ['n1', 'na'], ['n2', 'nb']);
  const f1 = logmax(lmPlain.jointTripleProb('w', 'n1', 'n2'));
  const f2 = logmax(lmPlain.jointTripleProb('w', 'na', 'nb'));
  assert.equal(gotMax, Math.max(f1, f2));
});

test('no NaN or Infinity across a zero-count grid', () => {
  const lm = makeLM();
  for (const w of ['w', 'ww', 'zzz']) {
    for (const pl of [[], ['p1'], ['p1', 'pa']]) {
      for (const nl of [[], ['n1'], ['n1', 'nb']]) {
        for (const pl2 of [[], ['p2']]) {
          for (const nl2 of [[], ['n2']]) {
            const s = lm.scoreCandidateOverSurfaces(w, pl, nl, pl2, nl2);
            assert.ok(Number.isFinite(s), `score must be finite: ${s}`);
          }
        }
      }
    }
  }
});

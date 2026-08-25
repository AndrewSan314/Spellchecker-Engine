// Task 7 (spelling-engine-optimization plan): the SymSpell deletion index
// must be LENGTH-AWARE — words are indexed with delete depth
// maxEditDistanceFor(key.length) instead of always 2, cutting index size
// WITHOUT changing lookup results under the same distance policy.
// Correctness argument spot-checks + exhaustive fixture equivalence.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SymSpellCandidateProvider,
  maxEditDistanceFor,
  applyTelexHints,
} from '../src/language.mjs';
import { accentKey } from '../src/normalizer.mjs';

// mixed-length fixture lexicon (4..10 letters, accented families, telex-prone)
const WORDS = [
  ['kính', 50000], ['chào', 900000], ['quý', 70000], ['khách', 200000],
  ['đặt', 571011], ['đất', 1633266], ['bàn', 800000], ['bán', 21020322],
  ['trong', 4000000], ['truong', 300], ['chuong', 100],
  ['chương', 1193740], ['vib', 9000], ['kiểm', 130000], ['tra', 1804779],
  ['khuyến', 250000], ['mãi', 480000], ['cod', 12000], ['hoàn', 190000],
  ['tiền', 3000000], ['giao', 2200000], ['hang', 150000], ['loại', 340000],
];

function makeProvider({ lengthAware }) {
  return new SymSpellCandidateProvider(
    { allWords: () => WORDS.map(([word, freq]) => ({ word, freq })) },
    2,
    { lengthAware },
  );
}

// every vocabulary word plus systematic corruptions = lookup workload
function* inputs() {
  for (const [w] of WORDS) {
    const lw = w.toLowerCase();
    yield lw;
    yield lw + lw.slice(-1);          // doubling
    if (lw.length >= 4) {
      yield lw.slice(1);              // deletion
      yield lw.slice(0, 2) + lw.slice(3); // inner deletion
      yield lw.slice(0, 2) + 'x' + lw.slice(3); // substitution
      yield lw.slice(0, 2) + lw.slice(2).split('').reverse().join(''); // scramble tail
    }
  }
}

test('equivalence: length-aware index returns identical candidates', () => {
  const legacy = makeProvider({ lengthAware: false });
  const aware = makeProvider({ lengthAware: true });
  let checked = 0;
  for (const input of inputs()) {
    const d = maxEditDistanceFor(accentKey(input).length);
    const a = JSON.stringify(legacy.candidates(input, d));
    const b = JSON.stringify(aware.candidates(input, d));
    assert.equal(b, a, `candidates diverge for input "${input}"`);
    checked++;
  }
  assert.ok(checked >= WORDS.length * 4,
    `expected a substantial workload, got ${checked} lookups`);
});

test('telex-decoded lookups stay equivalent too', () => {
  const legacy = makeProvider({ lengthAware: false });
  const aware = makeProvider({ lengthAware: true });
  for (const input of ['dawnf', 'ddat', 'ngooj', 'kieemtra', 'khuyenmai']) {
    const d = maxEditDistanceFor(accentKey(input).length);
    assert.deepEqual(
      aware.candidates(input, d),
      legacy.candidates(input, d),
      `divergence for telex input ${input}`,
    );
  }
});

test('index diagnostics expose read-only counts', () => {
  const aware = makeProvider({ lengthAware: true });
  const s = aware.indexStats();
  assert.equal(s.keys, new Set(WORDS.map(([w]) => accentKey(w))).size);
  assert.ok(s.variants > 0 && s.edges > 0);
  assert.ok(s.maxBucket >= 1 && Number.isInteger(s.maxBucket));
  assert.equal(typeof s.buildMs, 'number');
  void applyTelexHints;
});

test('length-aware indexing strictly reduces delete edges', () => {
  const legacy = makeProvider({ lengthAware: false });
  const aware = makeProvider({ lengthAware: true });
  const l = legacy.indexStats();
  const a = aware.indexStats();
  assert.ok(a.edges < l.edges,
    `aware edges ${a.edges} must be fewer than legacy ${l.edges}`);
});

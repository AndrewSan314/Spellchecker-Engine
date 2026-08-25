// Recall-improvement plan Task 7 — diversity-preserving shortlist.
// The selector must rescue the only context-attested candidate even when it
// sits below the cheap-frequency cut, WITHOUT growing the expensive stage
// beyond `size` non-original candidates, and WITHOUT any access to labels.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectDiverseShortlist } from '../src/rules/linguistic-rules.mjs';

const E = (word, dist, freq, opts = {}) => ({ word, dist, freq, ...opts });

/** pool where the correct candidate ("cách") is cheap-rank 4 by frequency
 *  but the ONLY one with direct context evidence */
const CONTEXT_POOL = [
  E('cục', 1, 9000),
  E('cực', 1, 8000),
  E('tác', 1, 7000),
  E('cách', 1, 6000),
  E('cac', 1, 100),
];

test('context-attested candidate below the cheap cut survives', () => {
  const out = selectDiverseShortlist(CONTEXT_POOL, {
    attestOf: new Map([['cách', true]]),
    famKey: 'cac',
    size: 4,
  });
  assert.ok(out.some((c) => c.word === 'cách'),
    'the only attested candidate must survive the shortlist');
});

test('bound never exceeds size (four non-original candidates)', () => {
  const pool = Array.from({ length: 30 }, (_, i) => E(`w${i}`, 1, 1000 - i));
  const out = selectDiverseShortlist(pool, {
    attestOf: new Map([['w29', true]]),
    famKey: '',
    size: 4,
  });
  assert.ok(out.length <= 4, `expected <=4, got ${out.length}`);
});

test('union covers all four heads: distance, attested, family, cheap', () => {
  const pool = [
    E('far', 2, 50000),   // cheap-top AND freq giant
    E('twin', 1, 40),     // best distance
    E('fam', 1, 30, { stripped: 'cac' }), // same accent family
    E('ctx', 1, 20),      // context-attested
    E('filler', 1, 10),
  ];
  // production passes the pre-computed cheap-ranked wide pool (head 4)
  const cheapOrder = [pool[0], pool[1], pool[2], pool[3], pool[4]];
  const out = selectDiverseShortlist(pool, {
    attestOf: new Map([['ctx', true]]),
    famKey: 'cac',
    size: 4,
    cheapOrder,
  });
  const words = out.map((c) => c.word);
  for (const w of ['twin', 'ctx', 'far']) {
    assert.ok(words.includes(w), `${w} head missing from ${words}`);
  }
  // four heads exactly fill the bound; filler is the one dropped
  assert.equal(out.length, 4);
  assert.ok(!words.includes('filler'));
});

test('deterministic: identical inputs produce identical output order', () => {
  const mk = () => [
    E('b', 1, 500), E('a', 1, 500), E('c', 2, 9000), E('d', 1, 10),
  ];
  const attestOf = new Map([['d', true]]);
  const x = selectDiverseShortlist(mk(), { attestOf, famKey: '', size: 4 });
  const y = selectDiverseShortlist(mk(), { attestOf, famKey: '', size: 4 });
  assert.deepEqual(x.map((c) => c.word), y.map((c) => c.word));
});

test('no attest map entries: falls back to pure cheap/distance union', () => {
  const pool = [E('p1', 1, 300), E('p2', 1, 200), E('p3', 2, 100)];
  const out = selectDiverseShortlist(pool, {
    attestOf: new Map(), famKey: '', size: 4,
  });
  assert.equal(out.length, 3); // whole pool fits under the bound
  assert.equal(out[0].word, 'p1'); // cheap order preserved on ties
});

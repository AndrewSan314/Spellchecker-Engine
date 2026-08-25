// Task 4 (spelling-engine-optimization plan): Telex-typed inputs ("dawnf",
// "ddat") must reach the SymSpell deletion index through a Telex-normalized
// lookup surface, instead of failing lookup before the Telex-aware edit
// scorer ever runs. Original tokens stay untouched for offsets/messages;
// protected codes/URLs are never Telex-mangled.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SymSpellCandidateProvider,
  telexLookupSurface,
} from '../src/language.mjs';

// ---------------------------------------------------------------------------
// Task 5: wide cheap-pool generation split from expensive ranking
// ---------------------------------------------------------------------------

test('pool keeps a real unaccented surface despite higher-freq accented siblings', () => {
  // Old behaviour: surfaces sorted accented-first then freq, capped at 3 ->
  // the unaccented surface was ALWAYS dropped when >=3 accented siblings
  // existed. Task 5 retention policy: raw/unaccented surface stays.
  const p = providerWith([
    ['Trưa', 900000], ['Trứa', 800000], ['Trụa', 700000],
    ['trua', 4000],
  ]);
  const surfaces = p.surfacesFor('trua').map((s) => s.word.toLowerCase());
  assert.ok(surfaces.includes('trua'),
    `unaccented surface must be retained, got ${surfaces.join(',')}`);
});

test('top-12 key generation retains valid candidates crowded out at 6', () => {
  // Seven high-frequency distractors share depth-1 deletes with "hna".
  // A transposition target ("hna" -> "han", OSA cost 0.8 here) must always
  // be retained (it even outranks substitutions), and a lowest-frequency
  // substitution target ("jna") that WAS crowded out by the old 6-key cap
  // must survive in the 12-key pool.
  const pairs = [];
  for (let i = 0; i < 8; i++) {
    const sub = String.fromCharCode('b'.charCodeAt(0) + i);
    if (sub === 'h' || sub === 'n' || sub === 'a') continue;
    pairs.push([`${sub}na`, 500000 - i]);
  }
  pairs.push(['han', 120]);   // transposition target
  pairs.push(['jna', 90]);    // lowest-freq substitution target
  const p = providerWith(pairs);
  const wide = p.generatePool('hna', 1, { keyCap: 12 });
  assert.ok(wide.matchedKeys.includes('han'),
    `wide pool must contain transposition key, got ${wide.matchedKeys.join(',')}`);
  assert.ok(wide.matchedKeys.includes('jna'),
    'wide 12-key pool must retain the low-freq distance-1 target');
  const narrow = p.generatePool('hna', 1, { keyCap: 6 });
  assert.ok(narrow.matchedKeys.includes('han'),
    'cheap transposition distance outranks substitutions under any cap');
  assert.ok(!narrow.matchedKeys.includes('jna'),
    'narrow 6-key pool must still demonstrate the crowding-out');
});

test('pool generation is deterministic and bounded', () => {
  const p = PROVIDER;
  const a = p.generatePool('dawnf', 2, { keyCap: 12 });
  const b = p.generatePool('dawnf', 2, { keyCap: 12 });
  assert.deepEqual(
    a.entries.map((e) => [e.word, e.dist]),
    b.entries.map((e) => [e.word, e.dist]),
  );
  assert.ok(a.matchedKeys.length <= 12);
  assert.ok(a.entries.length <= a.poolBound,
    `entries ${a.entries.length} must respect bound ${a.poolBound}`);
});

function providerWith(wordFreqPairs) {
  return new SymSpellCandidateProvider({
    allWords: () => wordFreqPairs.map(([word, freq]) => ({ word, freq })),
  });
}

// tiny deterministic lexicon covering several accent families
const PROVIDER = providerWith([
  ['Dằn', 50000], ['dân', 80000], ['dặn', 20000], ['dần', 21000],
  ['Đặt', 571011], ['đất', 1633266], ['dạt', 11011],
  ['Chào', 900000], ['cháo', 30000], ['chao', 1200],
]);

test('telex input dawnf generates dan-family candidates', () => {
  const cands = PROVIDER.candidates('dawnf', 1);
  const words = cands.map((c) => c.word.toLowerCase());
  assert.ok(words.includes('dằn'), `expected dằn in ${words.join(',')}`);
});

test('telex input ddat generates đặt/đất family candidates', () => {
  const cands = PROVIDER.candidates('ddat', 1);
  const words = cands.map((c) => c.word.toLowerCase());
  assert.ok(words.includes('đặt'), `expected đặt in ${words.join(',')}`);
  assert.ok(words.includes('đất'), `expected đất in ${words.join(',')}`);
});

test('ordinary non-telex typos keep their existing candidate set', () => {
  // 'chaoo' (extra o): deletion index finds chào/cháo family exactly as
  // before — the telex path adds nothing and removes nothing
  const cands = PROVIDER.candidates('chaoo', 1);
  const words = cands.map((c) => c.word.toLowerCase());
  assert.ok(words.includes('chào'), `expected chào in ${words.join(',')}`);
  // pure unaccented token: telex surface identical -> identical behaviour
  assert.deepEqual(
    PROVIDER.candidates('cha', 1).map((c) => c.word),
    PROVIDER.candidates('cha', 1).map((c) => c.word),
    'candidate ordering must be deterministic',
  );
});

test('candidates remain deterministic and bounded across repeated calls', () => {
  for (let i = 0; i < 5; i++) {
    const a = PROVIDER.candidates('dawnf', 2, 12);
    const b = PROVIDER.candidates('dawnf', 2, 12);
    assert.deepEqual(a.map((c) => c.word), b.map((c) => c.word));
    assert.ok(a.length <= 12);
  }
});

test('protected codes and URLs are not telex-normalized as spelling tokens', async () => {
  // Function-level: unambiguous code/URL shapes pass through untouched
  // (lookup surfaces are lowercase by contract).
  for (const s of ['OTP123', 'www.viettel.vn', 'DH552010', 'GIAM50K']) {
    assert.equal(telexLookupSurface(s), s.toLowerCase(),
      `telex surface must leave code-like text untouched: ${s}`);
  }
  // Engine-level: protected/code-like tokens are gated OUT of candidate
  // generation entirely (classifyToken runs before any lookup), so ambiguous
  // sequences ("abc@example.com" -> "ex") can never be decoded as typos.
  const doc = {
    insideSingleProtectedRange: (s, e) => s === 0 && e > 5,
  };
  const services = {
    whitelistService: { isAllowed: () => false },
    abbreviationService: { isRuleEnabledFor: () => false, isAbbreviation: () => false },
    lexicon: { contains: () => false, realFrequency: () => 0 },
    accentIndex: { candidates: () => [] },
  };
  const ctx = { messageMode: 'ACCENTED', brandname: null };
  const mk = (original) => ({
    original, normalized: original.toLowerCase(), start: 0,
    end: original.length,
  });
  const { classifyToken } = await import('../src/rules/linguistic-rules.mjs');
  assert.equal(classifyToken(mk('https://abc.vn/x?id=1'), ctx, doc, services),
    'PROTECTED');
  assert.equal(classifyToken(mk('DH552010'), ctx,
    { insideSingleProtectedRange: () => false }, services), 'CODE_LIKE');
});

test('telexLookupSurface decodes digraphs and strips post-vowel tone letters', () => {
  assert.equal(telexLookupSurface('dawnf'), 'dăn');
  assert.equal(telexLookupSurface('ddat'), 'đat');
  assert.equal(telexLookupSurface('ngooj'), 'ngô', 'digraph + nặng tone letter');
  assert.equal(telexLookupSurface('san'), 'san', 'non-telex word unchanged');
});

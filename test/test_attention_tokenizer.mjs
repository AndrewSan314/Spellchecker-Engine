// Task 2 (tiny-attention-spelling-reranker-FIXED plan) — shared attention
// tokenizer + 32-token context selector, fixture-driven.
//
// Contract under test:
//   - immutable special IDs PAD=0..MASK=7, learned vocab 8..8191
//   - model input is NFC-normalized + Unicode-lowercased ONLY at model input;
//     Vietnamese diacritics are never stripped; SMS surface/offsets untouched
//   - FNV-1a over UTF-8 bytes, unsigned 32-bit, mod 4096 char 2..4-grams
//   - context selection: <=32 keep all; else first-four + target + nearest
//     by (absoluteDistance, index); retained indices sorted ascending; the
//     target position is marked explicitly and never replaced by a candidate
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ValidationContext } from '../src/engine.mjs';
import { buildValidationDocument } from '../src/document-builder.mjs';
import {
  ATTENTION_TOKENIZER_VERSION,
  SPECIAL_IDS,
  MARKER,
  MAX_CONTEXT_TOKENS,
  CHAR_HASH_BUCKETS,
  fnv1a32Utf8,
  normalizeForModel,
  unitsFromDocument,
  selectContextIndices,
  encodeContextUnits,
  encodeOptionSurface,
} from '../src/attention-tokenizer.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(
  path.join(HERE, 'fixtures', 'attention-tokenizer-cases.json'), 'utf8'));
const VOCAB = new Map(Object.entries(FIXTURE.vocabSample));

function docOf(text) {
  return buildValidationDocument(new ValidationContext(text, 'ACCENTED', 'TENDOO'));
}
function encodeCase(c) {
  const doc = docOf(c.text);
  const units = unitsFromDocument(doc);
  return { doc, units, out: encodeContextUnits(units, c.targetWordIndex, VOCAB) };
}
function canon(x) {
  return JSON.stringify(x, Object.keys(x).sort());
}

test('fixture contract constants are immutable and exact', () => {
  assert.equal(ATTENTION_TOKENIZER_VERSION, 'attention-tokenizer-v1');
  assert.deepEqual({ ...SPECIAL_IDS },
    { PAD: 0, UNK: 1, BOS: 2, EOS: 3, TARGET: 4, PROTECTED: 5, PUNCT: 6, MASK: 7 });
  assert.equal(MAX_CONTEXT_TOKENS, 32);
  assert.equal(CHAR_HASH_BUCKETS, 4096);
});

test('FNV-1a over UTF-8 bytes matches known vectors', () => {
  // standard FNV-1a 32-bit test vectors
  assert.equal(fnv1a32Utf8(''), 0x811c9dc5);
  assert.equal(fnv1a32Utf8('a'), 0xe40c292c);
  assert.equal(fnv1a32Utf8('foobar'), 0xbf9cf968);
  // multi-byte UTF-8 input hashes BYTES, not code units
  assert.equal(fnv1a32Utf8('ê'), (() => {
    // hand-computed over [0xC3, 0xAA]
    let h = 0x811c9dc5;
    h = Math.imul(h ^ 0xc3, 0x01000193) >>> 0;
    h = Math.imul(h ^ 0xaa, 0x01000193) >>> 0;
    return h;
  })());
});

test('model normalization is NFC + lowercase, accents preserved', () => {
  const nfd = 'Hòa Bình'.normalize('NFD');
  assert.notEqual(nfd, 'Hòa Bình');
  assert.equal(normalizeForModel(nfd), normalizeForModel('hòa bình'));
  assert.ok(/ò/.test(normalizeForModel('Hòa')), 'diacritics must survive');
});

test('short SMS uses all tokens; vocab ids and trailing punct marker correct', () => {
  const c = FIXTURE.cases[0];
  const { units, out } = encodeCase(c);
  assert.equal(units.length, c.structural.unitCount);
  assert.equal(out.selectedIndices.length, c.structural.selectedCount);
  assert.deepEqual(
    out.ids.slice(0, 5), c.structural.wordVocabIds);
  assert.equal(out.markers[out.markers.length - 1], MARKER.PUNCT);
  assert.equal(out.ids[out.targetPosition], VOCAB.get('thành'));
  assert.equal(out.markers[out.targetPosition], MARKER.TARGET);
});

test('40-token SMS truncates to 32 retaining first four + target, ascending', () => {
  const c = FIXTURE.cases[1];
  const { units, out } = encodeCase(c);
  assert.equal(units.length, 40);
  assert.equal(out.selectedIndices.length, MAX_CONTEXT_TOKENS);
  for (const i of [0, 1, 2, 3]) {
    assert.ok(out.selectedIndices.includes(i), `index ${i} retained`);
  }
  assert.deepEqual([...out.selectedIndices].sort((a, b) => a - b),
    out.selectedIndices, 'selected indices ascending');
  const targetUnitIdx = out.selectedIndices.indexOf(20);
  assert.ok(targetUnitIdx >= 0, 'target retained');
  assert.equal(out.targetPosition, targetUnitIdx);
  assert.equal(out.mask.every((m) => m === 1), true);
});

test('nearest-target fill is deterministic for a near-end target', () => {
  const c = FIXTURE.cases[2];
  const { out } = encodeCase(c);
  assert.equal(out.selectedIndices.length, 32);
  assert.ok(out.selectedIndices.includes(35));
  assert.equal(out.targetPosition,
    out.selectedIndices.indexOf(35));
  // deterministic across calls
  const again = encodeCase(c).out;
  assert.equal(JSON.stringify(out), JSON.stringify(again));
});

test('pure selector keeps first four + target and fills nearest-first', () => {
  assert.deepEqual(selectContextIndices(10, 3), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const sel = selectContextIndices(40, 20);
  assert.equal(sel.length, 32);
  assert.deepEqual(sel.slice(0, 7), [0, 1, 2, 3, 6, 7, 8]);
  // tie at equal distance resolves to the lower index
  const selEnd = selectContextIndices(40, 35);
  assert.equal(selEnd.length, 32);
  assert.ok(selEnd.includes(34) && selEnd.includes(36));
});

test('punctuation units carry the PUNCT marker', () => {
  const c = FIXTURE.cases.find((x) => x.name === 'punctuation_markers_present');
  const { out } = encodeCase(c);
  const punctSlots = [...out.markers].filter((m) => m === MARKER.PUNCT).length;
  assert.ok(punctSlots >= c.structural.punctMarkerSlotsAtLeast);
});

test('URL / phone / placeholder stay single PROTECTED units, never decomposed', () => {
  const c = FIXTURE.cases.find(
    (x) => x.name === 'protected_url_phone_placeholder_never_decomposed');
  const { units, out } = encodeCase(c);
  const protectedUnits = units.filter((u) => u.kind === 'protected');
  assert.deepEqual(protectedUnits.map((u) => u.surface),
    c.structural.protectedUnitSurfaces);
  const protectedSlots = out.selectedIndices
    .map((_, i) => i)
    .filter((i) => out.markers[i] === MARKER.PROTECTED);
  for (const s of protectedSlots) {
    assert.equal(out.ids[s], SPECIAL_IDS.PROTECTED);
    assert.deepEqual(out.charHashes[s], []);
  }
});

test('NFD and NFC texts produce identical model vectors', () => {
  const base = FIXTURE.cases.find(
    (x) => x.name === 'nfd_nfc_equivalence_identical_vectors');
  const nfcOut = encodeContextUnits(
    unitsFromDocument(docOf(base.textNfc)), base.targetWordIndex, VOCAB);
  const nfdText = base.textNfc.normalize('NFD');
  assert.notEqual(nfdText, base.textNfc);
  const nfdOut = encodeContextUnits(
    unitsFromDocument(docOf(nfdText)), base.targetWordIndex, VOCAB);
  assert.equal(JSON.stringify(nfdOut), JSON.stringify(nfcOut));
});

test('OOV word maps to UNK with deterministic in-range char 2..4-gram hashes', () => {
  const c = FIXTURE.cases.find((x) => x.name === 'oov_word_deterministic_char_hashes');
  const { out } = encodeCase(c);
  assert.equal(out.ids[0], c.structural.oovId);
  const hashes = out.charHashes[0];
  assert.ok(hashes.length > 0);
  for (const h of hashes) {
    assert.ok(h >= 0 && h < c.structural.charHashBucketModulo);
  }
  // deterministic repeat + construction-order stability
  assert.deepEqual(encodeCase(c).out.charHashes[0], hashes);
});

test('option encoding is per-surface pure: candidate order cannot change it', () => {
  const c = FIXTURE.cases.find(
    (x) => x.name === 'candidate_option_encoding_order_independent');
  const [a, b] = c.optionSurfaces;
  const forward = [encodeOptionSurface(a, VOCAB), encodeOptionSurface(b, VOCAB)];
  const backward = [encodeOptionSurface(b, VOCAB), encodeOptionSurface(a, VOCAB)];
  assert.equal(JSON.stringify(forward),
    JSON.stringify([backward[1], backward[0]]));
  // KEEP_ORIGINAL-style reuse: same surface -> identical vector every time
  assert.equal(JSON.stringify(encodeOptionSurface(a, VOCAB)),
    JSON.stringify(forward[0]));
});

test('every materialized expected block matches the live implementation', () => {
  for (const c of FIXTURE.cases) {
    if (!c.expected) continue;
    const text = c.text ?? c.textNfc;
    const out = encodeContextUnits(
      unitsFromDocument(docOf(text)), c.targetWordIndex, VOCAB);
    assert.equal(canon(out), canon(c.expected), `case ${c.name}`);
  }
});

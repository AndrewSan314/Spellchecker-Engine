// ============================================================
// Task 2 (tiny-attention-spelling-reranker-FIXED plan) — shared attention
// tokenizer + 32-token context selector.
//
// ONE immutable ID contract for JavaScript and Python:
//   0 PAD  1 UNK  2 BOS  3 EOS  4 TARGET  5 PROTECTED  6 PUNCT  7 MASK
//   8..8191 learned vocabulary
//
// Model input is NFC-normalized + Unicode-lowercased ONLY here; Vietnamese
// diacritics are NEVER stripped and the original SMS surface/offsets are
// never modified. Character n-grams are Unicode 2..4-grams hashed with
// FNV-1a over UTF-8 bytes (unsigned 32-bit) modulo 4096 — never JS code-unit
// hashing.
//
// Context selection builds on the existing ValidationDocument tokens (no
// re-tokenization): <=32 units keep all; otherwise retain the first four,
// the target, then nearest-by-(absoluteDistance, index) until 32 slots are
// full; retained indices sort ascending; the target slot is marked
// explicitly and is NEVER replaced by a candidate surface.
// ============================================================
import { normalizeNfc } from './normalizer.mjs';

export const ATTENTION_TOKENIZER_VERSION = 'attention-tokenizer-v1';

export const SPECIAL_IDS = Object.freeze({
  PAD: 0, UNK: 1, BOS: 2, EOS: 3, TARGET: 4, PROTECTED: 5, PUNCT: 6, MASK: 7,
});

/** Marker embedding indices (independent namespace from word IDs). */
export const MARKER = Object.freeze({ NONE: 0, TARGET: 1, PROTECTED: 2, PUNCT: 3 });

export const VOCAB_SIZE = 8192;
export const FIRST_LEARNED_ID = 8;
export const CHAR_HASH_BUCKETS = 4096;
export const CHAR_NGRAM_MIN = 2;
export const CHAR_NGRAM_MAX = 4;
export const MAX_CONTEXT_TOKENS = 32;

/** FNV-1a, 32-bit, over UTF-8 bytes. */
export function fnv1a32Utf8(text) {
  let h = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) {
    h ^= byte;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function hashCharNgram(ngram) {
  return fnv1a32Utf8(ngram) % CHAR_HASH_BUCKETS;
}

/** Model-input normalization: NFC + Unicode lowercase (accents kept). */
export function normalizeForModel(text) {
  return normalizeNfc(String(text)).toLowerCase();
}

/** Ordered, de-duplicated character 2..4-grams of a normalized word. */
export function charNgramsOf(word) {
  const seen = new Set();
  const out = [];
  const s = String(word);
  for (let len = CHAR_NGRAM_MIN; len <= CHAR_NGRAM_MAX; len++) {
    for (let i = 0; i + len <= s.length; i++) {
      const g = s.slice(i, i + len);
      if (!seen.has(g)) {
        seen.add(g);
        out.push(g);
      }
    }
  }
  return out;
}

export function charNgramHashes(word) {
  return charNgramsOf(normalizeForModel(word)).map(hashCharNgram);
}

/**
 * Flatten a ValidationDocument into context units. Consecutive non-
 * whitespace tokens fully inside the SAME protected range merge into ONE
 * protected unit (URLs/phones/placeholders are never decomposed).
 * Unit kinds: 'word' | 'protected' | 'punct' (NUMBER/PUNCTUATION/OTHER).
 * Offsets stay in ORIGINAL-text space.
 */
export function unitsFromDocument(doc) {
  const units = [];
  /** unit under construction for the protected range starting at rs/re */
  let openRangeKey = null;
  for (const t of doc.tokens) {
    if (t.type === 'WHITESPACE') continue;
    const key = doc.insideSingleProtectedRange(t.start, t.end)
      ? (() => {
        const r = doc.protectedRanges.find((rr) => rr.contains(t.start, t.end));
        return `${r.start}:${r.end}`;
      })()
      : null;

    if (key !== null && key === openRangeKey && units.length > 0) {
      // extend the currently open protected unit across this token
      const last = units[units.length - 1];
      last.end = t.end;
      last.surface = doc.originalText.slice(last.start, last.end);
      continue;
    }
    openRangeKey = key;

    if (key !== null) {
      units.push({
        kind: 'protected',
        surface: doc.originalText.slice(t.start, t.end),
        start: t.start, end: t.end,
      });
      continue;
    }
    units.push({
      kind: t.type === 'WORD' ? 'word' : 'punct',
      surface: doc.originalText.slice(t.start, t.end),
      start: t.start, end: t.end,
    });
  }
  return units.map(Object.freeze);
}

/**
 * Deterministic context selection. Returns ascending retained unit indices.
 */
export function selectContextIndices(n, targetIdx, max = MAX_CONTEXT_TOKENS) {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  if (targetIdx < 0 || targetIdx >= n) throw new RangeError('target out of range');
  const kept = new Set([0, 1, 2, 3, targetIdx]);
  const rest = [];
  for (let i = 0; i < n; i++) {
    if (!kept.has(i)) rest.push(i);
  }
  rest.sort((a, b) => Math.abs(a - targetIdx) - Math.abs(b - targetIdx) || a - b);
  for (const i of rest) {
    if (kept.size >= max) break;
    kept.add(i);
  }
  return [...kept].sort((a, b) => a - b);
}

function unitIdAndMarker(unit, isTarget, vocabMap) {
  if (unit.kind === 'protected') {
    return { id: SPECIAL_IDS.PROTECTED, marker: MARKER.PROTECTED, hashes: [] };
  }
  if (unit.kind === 'punct') {
    return { id: SPECIAL_IDS.PUNCT, marker: MARKER.PUNCT, hashes: [] };
  }
  const norm = normalizeForModel(unit.surface);
  const id = vocabMap?.get(norm) ?? SPECIAL_IDS.UNK;
  return {
    id,
    marker: isTarget ? MARKER.TARGET : MARKER.NONE,
    hashes: charNgramHashes(norm),
  };
}

/**
 * Encode one target-in-context example into model vectors.
 * @returns {{ids:number[], markers:number[], charHashes:number[][],
 *            mask:number[], selectedIndices:number[], targetPosition:number}}
 */
export function encodeContextUnits(units, targetIdx, vocabMap) {
  const selectedIndices = selectContextIndices(units.length, targetIdx);
  const ids = [];
  const markers = [];
  const charHashes = [];
  const mask = [];
  let targetPosition = -1;
  selectedIndices.forEach((unitIdx, slot) => {
    const isTarget = unitIdx === targetIdx;
    if (isTarget) targetPosition = slot;
    const { id, marker, hashes } = unitIdAndMarker(units[unitIdx], isTarget, vocabMap);
    ids.push(id);
    markers.push(marker);
    charHashes.push(hashes);
    mask.push(1);
  });
  if (targetPosition < 0) throw new RangeError('target not selected');
  return { ids, markers, charHashes, mask, selectedIndices, targetPosition };
}

/**
 * Encode an option surface (KEEP_ORIGINAL uses the original token's
 * normalized form). Pure per-surface function: candidate list order can
 * never influence option vectors.
 */
export function encodeOptionSurface(surface, vocabMap) {
  const norm = normalizeForModel(surface);
  return {
    id: vocabMap?.get(norm) ?? SPECIAL_IDS.UNK,
    charHashes: charNgramHashes(norm),
  };
}

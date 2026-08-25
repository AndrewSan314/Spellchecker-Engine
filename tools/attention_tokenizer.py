# ============================================================
# Task 2 (tiny-attention-spelling-reranker-FIXED plan) — Python mirror of
# src/attention-tokenizer.mjs. ONE immutable ID contract:
#   0 PAD  1 UNK  2 BOS  3 EOS  4 TARGET  5 PROTECTED  6 PUNCT  7 MASK
#   8..8191 learned vocabulary
#
# Model input is NFC-normalized + Unicode-lowercased ONLY here; Vietnamese
# diacritics are NEVER stripped. Character n-grams are Unicode 2..4-grams
# hashed with FNV-1a over UTF-8 bytes (unsigned 32-bit) modulo 4096.
#
# Units (surface/kind/start/end) come from the JavaScript ValidationDocument
# pipeline (exported in golden fixtures / ranking rows) — production Python
# never re-tokenizes raw SMS text. This module owns ids, markers, context
# selection and char hashing, byte-identically to the JS reference.
# ============================================================
from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

ATTENTION_TOKENIZER_VERSION = "attention-tokenizer-v1"

SPECIAL_IDS = {
    "PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3,
    "TARGET": 4, "PROTECTED": 5, "PUNCT": 6, "MASK": 7,
}
MARKER_NONE = 0
MARKER_TARGET = 1
MARKER_PROTECTED = 2
MARKER_PUNCT = 3

VOCAB_SIZE = 8192
FIRST_LEARNED_ID = 8
CHAR_HASH_BUCKETS = 4096
CHAR_NGRAM_MIN = 2
CHAR_NGRAM_MAX = 4
MAX_CHAR_NGRAMS = 32
MAX_CONTEXT_TOKENS = 32

_FNV_OFFSET_BASIS = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def fnv1a32_utf8(text: str) -> int:
    """FNV-1a, 32-bit, over UTF-8 bytes."""
    h = _FNV_OFFSET_BASIS
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK32
    return h


def hash_char_ngram(ngram: str) -> int:
    return fnv1a32_utf8(ngram) % CHAR_HASH_BUCKETS


def normalize_for_model(text: str) -> str:
    """NFC + Unicode lowercase; accents preserved."""
    return unicodedata.normalize("NFC", text).lower()


def char_ngram_hashes(word: str) -> List[int]:
    """Ordered, de-duplicated Unicode character 2..4-gram hashes."""
    s = normalize_for_model(word)
    seen = set()
    out: List[int] = []
    for length in range(CHAR_NGRAM_MIN, CHAR_NGRAM_MAX + 1):
        for i in range(0, len(s) - length + 1):
            g = s[i:i + length]
            if g not in seen:
                seen.add(g)
                out.append(hash_char_ngram(g))
    return out[:MAX_CHAR_NGRAMS]


def select_context_indices(
    n: int, target_idx: int, max_tokens: int = MAX_CONTEXT_TOKENS,
) -> List[int]:
    """<=max keep all; else first four + target + nearest by
    (absoluteDistance, index); result sorted ascending."""
    if n <= max_tokens:
        return list(range(n))
    if not 0 <= target_idx < n:
        raise ValueError("target out of range")
    kept = {0, 1, 2, 3, target_idx}
    rest = [i for i in range(n) if i not in kept]
    rest.sort(key=lambda i: (abs(i - target_idx), i))
    for i in rest:
        if len(kept) >= max_tokens:
            break
        kept.add(i)
    return sorted(kept)


def encode_context_units(
    units: Sequence[dict],
    target_idx: int,
    vocab: Optional[Dict[str, int]] = None,
) -> dict:
    """Encode one target-in-context example. Returns a dict whose JSON form
    is identical to the JavaScript reference implementation's output."""
    selected = select_context_indices(len(units), target_idx)
    ids: List[int] = []
    markers: List[int] = []
    char_hashes: List[List[int]] = []
    mask: List[int] = []
    target_position = -1
    for slot, unit_idx in enumerate(selected):
        is_target = unit_idx == target_idx
        unit = units[unit_idx]
        kind = unit["kind"]
        if kind == "protected":
            uid, marker, hashes = SPECIAL_IDS["PROTECTED"], MARKER_PROTECTED, []
        elif kind == "punct":
            uid, marker, hashes = SPECIAL_IDS["PUNCT"], MARKER_PUNCT, []
        else:
            norm = normalize_for_model(unit["surface"])
            uid = vocab.get(norm, SPECIAL_IDS["UNK"]) if vocab else SPECIAL_IDS["UNK"]
            marker = MARKER_TARGET if is_target else MARKER_NONE
            # Hide the surface under correction; the TARGET marker identifies
            # the slot without leaking the misspelling into the encoder.
            if is_target:
                uid, hashes = SPECIAL_IDS["MASK"], []
            else:
                hashes = char_ngram_hashes(norm)
        if is_target:
            target_position = slot
        ids.append(uid)
        markers.append(marker)
        char_hashes.append(hashes)
        mask.append(1)
    if target_position < 0:
        raise ValueError("target not selected")
    return {
        "ids": ids,
        "markers": markers,
        "charHashes": char_hashes,
        "mask": mask,
        "selectedIndices": selected,
        "targetPosition": target_position,
    }


def encode_option_surface(
    surface: str, vocab: Optional[Dict[str, int]] = None,
) -> dict:
    """Pure per-surface option encoding (KEEP_ORIGINAL included)."""
    norm = normalize_for_model(surface)
    uid = vocab.get(norm, SPECIAL_IDS["UNK"]) if vocab else SPECIAL_IDS["UNK"]
    return {"id": uid, "charHashes": char_ngram_hashes(norm)}

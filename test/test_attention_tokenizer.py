# Task 2 — Python parity for the shared attention tokenizer. The Python
# implementation must reproduce the JavaScript golden fixture vectors
# byte-identically (same IDs, markers, char hashes, mask, selection).
from __future__ import annotations

import json
import os
import unittest

from tools.attention_tokenizer import (
    ATTENTION_TOKENIZER_VERSION,
    CHAR_HASH_BUCKETS,
    MAX_CONTEXT_TOKENS,
    SPECIAL_IDS,
    char_ngram_hashes,
    encode_context_units,
    encode_option_surface,
    fnv1a32_utf8,
    normalize_for_model,
    select_context_indices,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_PATH = os.path.join(HERE, "fixtures", "attention-tokenizer-cases.json")

with open(FIXTURE_PATH, encoding="utf-8") as handle:
    FIXTURE = json.load(handle)

VOCAB = {k: int(v) for k, v in FIXTURE["vocabSample"].items()}


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


class TokenizerParityTest(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(ATTENTION_TOKENIZER_VERSION, "attention-tokenizer-v1")
        self.assertEqual(SPECIAL_IDS, {
            "PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3,
            "TARGET": 4, "PROTECTED": 5, "PUNCT": 6, "MASK": 7,
        })
        self.assertEqual(MAX_CONTEXT_TOKENS, 32)
        self.assertEqual(CHAR_HASH_BUCKETS, 4096)

    def test_fnv1a32_utf8_known_vectors(self):
        self.assertEqual(fnv1a32_utf8(""), 0x811C9DC5)
        self.assertEqual(fnv1a32_utf8("a"), 0xE40C292C)
        self.assertEqual(fnv1a32_utf8("foobar"), 0xBF9CF968)

    def test_normalization_nfc_lowercase_keeps_accents(self):
        nfd = "Hòa Bình".normalize("NFD") if hasattr(str, "normalize") else None
        import unicodedata
        nfd = unicodedata.normalize("NFD", "Hòa Bình")
        self.assertNotEqual(nfd, "Hòa Bình")
        self.assertEqual(normalize_for_model(nfd), normalize_for_model("hòa bình"))

    def test_char_hashes_are_in_range_and_deterministic(self):
        a = char_ngram_hashes("zzqqx")
        b = char_ngram_hashes("zzqqx")
        self.assertTrue(a)
        for h in a:
            self.assertTrue(0 <= h < CHAR_HASH_BUCKETS)
        self.assertEqual(a, b)
        # NFC/NFD inputs hash identically (model normalization first)
        import unicodedata
        self.assertEqual(
            char_ngram_hashes(unicodedata.normalize("NFD", "thành")),
            char_ngram_hashes("thành"),
        )

    def test_selector_matches_reference_behavior(self):
        self.assertEqual(select_context_indices(10, 3), list(range(10)))
        sel = select_context_indices(40, 20)
        self.assertEqual(len(sel), 32)
        self.assertEqual(sel[:7], [0, 1, 2, 3, 6, 7, 8])
        sel_end = select_context_indices(40, 35)
        self.assertEqual(len(sel_end), 32)
        self.assertIn(34, sel_end)
        self.assertIn(36, sel_end)

    def test_every_fixture_case_matches_golden_vectors(self):
        checked = 0
        for case in FIXTURE["cases"]:
            if case.get("expectedOptions") is not None:
                opts = [encode_option_surface(s, VOCAB)
                        for s in case["optionSurfaces"]]
                self.assertEqual(canon(opts), canon(case["expectedOptions"]),
                                 f"case {case['name']} options")
                checked += 1
                continue
            expected = case.get("expected")
            if expected is None:
                continue
            units = case["units"]
            out = encode_context_units(units, case["targetWordIndex"], VOCAB)
            self.assertEqual(canon(out), canon(expected),
                             f"case {case['name']}")
            # structural invariants re-asserted on the Python side
            self.assertLessEqual(len(out["ids"]), MAX_CONTEXT_TOKENS)
            self.assertLessEqual(out["targetPosition"], len(out["ids"]) - 1)
            checked += 1
        self.assertGreater(checked, 5, "fixture must exercise multiple cases")


if __name__ == "__main__":
    unittest.main()

# Task 4 (tiny-attention-spelling-reranker-FIXED plan) — leakage-safe
# vocabulary + self-supervised pretraining set builder.
#
# Contract under test:
#   - ONLY corpus-train.txt + clean-train.txt are acceptable text sources;
#     dev/test/heldout/calibration/evaluation paths are refused before reads;
#   - any sentence hash from Task 3's deny list is excluded;
#   - duplicate normalized sentences are emitted once;
#   - same seed => byte-identical vocabulary, shard order and hashes;
#   - special IDs exactly match Task 2 (PAD=0..MASK=7);
#   - <=8192 TOTAL word IDs including the eight specials; char hash buckets
#     are a separate namespace;
#   - sequences are <=32 tokens; padding positions carry mask 0;
#   - URLs/emails/phones/tracking IDs/markup become stable placeholders;
#   - no silent spelling repair (corrupted surfaces stay verbatim).
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.build_attention_pretrain_data import (  # noqa: E402
    MAX_SEQ_TOKENS,
    SPECIAL_IDS,
    VOCAB_SIZE_LIMIT,
    assert_allowed_sources,
    build_vocab,
    encode_sentences,
    load_deny_list,
    normalize_for_model,
    placeholderize,
    write_artifacts,
)


class SourceGuardTest(unittest.TestCase):
    def test_only_canonical_sources_accepted(self):
        ok = [
            "src/data/corpus-train.txt",
            "dataset_artifacts/clean-source/clean-train.txt",
        ]
        self.assertEqual(assert_allowed_sources(ok), ok)

    def test_forbidden_markers_refused_before_read(self):
        bad = [
            "dataset_artifacts/vsec/vsec-dev.jsonl",
            "benchmark/corpus-vsec-test.json",
            "benchmark/corpus-viwiki-spelling.json",
            "dataset_artifacts/evaluation/spelling-eval-dev.json",
            "data/final-recall-heldout.json",
            ".tmp/attention-messages-calibration.jsonl",
        ]
        for p in bad:
            with self.assertRaisesRegex(ValueError, "forbidden"):
                assert_allowed_sources([p])

    def test_foreign_sources_refused(self):
        with self.assertRaisesRegex(ValueError, "not an allowed"):
            assert_allowed_sources(["src/data/corpus_banking_fintech.txt"])


class NormalizationTest(unittest.TestCase):
    def test_nfc_lowercase_keeps_accents(self):
        import unicodedata
        nfd = unicodedata.normalize("NFD", "Hòa Bình")
        self.assertEqual(normalize_for_model(nfd), normalize_for_model("hòa bình"))
        self.assertIn("ò", normalize_for_model("Hòa"))

    def test_placeholders_stable_and_never_repaired(self):
        text = "Gọi 0987654321 hoặc https://tendoo.vn/km gửi code {{OTP}} nhanh"
        out = placeholderize(text)
        self.assertIn("<phone>", out)
        self.assertIn("<url>", out)
        self.assertIn("{{otp}}", out.lower() or out)
        # corrupted surface must survive untouched (no silent repair)
        repaired = placeholderize("đơn hangf đã giao")
        self.assertIn("hangf", repaired)
        self.assertNotIn("hàng", repaired)

    def test_deny_list_loaded_from_task3_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "attention-ranking-split-manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "denyList": {"calibration": ["a" * 64],
                                 "internalTest": ["b" * 64]},
                }, fh)
            denied = load_deny_list(manifest_path)
            self.assertEqual(denied, {"a" * 64, "b" * 64})
        with self.assertRaisesRegex(ValueError, "deny list"):
            load_deny_list(os.path.join(td, "missing-manifest.json"))


class VocabTest(unittest.TestCase):
    def test_special_ids_match_task2(self):
        self.assertEqual(SPECIAL_IDS, {
            "PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3,
            "TARGET": 4, "PROTECTED": 5, "PUNCT": 6, "MASK": 7,
        })

    def test_vocab_frequency_desc_then_unicode_ties_capped(self):
        sentences = ["b a", "b b", "c", "c", "c", "á à", "zz qq ww ee rr tt yy uu"]
        for _ in range(50):
            sentences.append("fill fill fill")
        vocab = build_vocab(sentences, limit=VOCAB_SIZE_LIMIT)
        ids = vocab["wordToId"]
        # specials occupy 0..7; learned ids start at 8
        self.assertEqual(min(ids.values()), 8)
        # higher frequency -> lower id
        self.assertLess(ids["fill"], ids["c"])
        # equal frequency -> unicode ascending order
        self.assertLess(ids["b"], ids["c"])
        self.assertLess(ids["a"], ids["á"])
        # hard cap including the 8 specials
        small = build_vocab([" ".join(f"w{i}" for i in range(9000))],
                            limit=VOCAB_SIZE_LIMIT)
        self.assertEqual(len(small["wordToId"]), VOCAB_SIZE_LIMIT - len(SPECIAL_IDS))
        self.assertEqual(small["size"], VOCAB_SIZE_LIMIT)

    def test_encode_masks_padding_and_caps_length(self):
        vocab = build_vocab(["ngày mai sớm"], limit=2048)
        long_sentence = " ".join(f"tok{i}" for i in range(80))
        rows = encode_sentences([long_sentence], vocab["wordToId"])
        for ids, mask in rows:
            self.assertEqual(len(ids), MAX_SEQ_TOKENS)
            self.assertEqual(len(mask), MAX_SEQ_TOKENS)
            self.assertTrue(all(m == 1 for m in mask))  # full chunk
        rows2 = encode_sentences(["ngày mai"], vocab["wordToId"])
        ids, mask = rows2[0]
        self.assertEqual(sum(mask), 2)
        for pos, m in enumerate(mask):
            if m == 0:
                self.assertEqual(ids[pos], SPECIAL_IDS["PAD"])
        # OOV maps to UNK, accents preserved in lookup
        rows3 = encode_sentences(["ngày oovxyz"], vocab["wordToId"])
        self.assertEqual(rows3[0][0][1], SPECIAL_IDS["UNK"])

    def test_write_artifacts_deterministic_same_seed(self):
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            lines = ["Khách hàng cần hỗ trợ về đơn hàng",
                     "Gọi hotline 0987654321 để được hỗ trợ",
                     "khách hàng cần hỗ trợ về đơn hàng"]  # dup after norm
            for td in (ta, tb):
                src = os.path.join(td, "clean-train.txt")
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
                res = write_artifacts(
                    sources=[src], deny=frozenset(), out_dir=td,
                    seed=20260825, shard_size=2)
                if td is ta:
                    first = res
                    with open(res["vocabPath"], "rb") as fh:
                        vb = fh.read()
                    shard_hashes_1 = [s["sha256"] for s in res["shards"]]
            with open(first["vocabPath"], "rb") as fh:
                pass
            res_b = None
            # rebuild in second dir and compare
            res_b = write_artifacts(
                sources=[os.path.join(tb, "clean-train.txt")],
                deny=frozenset(), out_dir=tb, seed=20260825, shard_size=2)
            with open(res_b["vocabPath"], "rb") as fh:
                vb2 = fh.read()
            with open(first["vocabPath"], "rb") as fh:
                vb1 = fh.read()
            self.assertEqual(vb1, vb2)
            self.assertEqual([s["sha256"] for s in res_b["shards"]], shard_hashes_1)
            # duplicates collapsed: 2 unique sentences -> 1 shard row batch
            self.assertEqual(res_b["counts"]["accepted"], 2)
            self.assertEqual(res_b["counts"]["deduplicated"], 1)

    def test_denied_hashes_excluded(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            keep = "giữ lại dòng này"
            drop = "dòng bị từ chối vĩnh viễn"
            src = os.path.join(td, "clean-train.txt")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(keep + "\n" + drop + "\n")
            drop_hash = hashlib.sha256(
                normalize_for_model(drop).encode("utf-8")).hexdigest()
            res = write_artifacts(sources=[src], deny={drop_hash},
                                  out_dir=td, seed=20260825, shard_size=8)
            self.assertEqual(res["counts"]["denied"], 1)
            self.assertEqual(res["counts"]["accepted"], 1)


if __name__ == "__main__":
    unittest.main()

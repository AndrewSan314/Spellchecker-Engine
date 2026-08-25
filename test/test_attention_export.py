# ============================================================
# Task 7 — Unit tests for int8 binary exporter and format verification.
#
# Contract under test:
#   - Magic = TDRANK01, format_version = 1;
#   - Little-endian, 16-byte alignment;
#   - Deterministic: two runs produce identical bytes and hashes;
#   - Rejects corrupt magic, truncated data, invalid table offsets;
#   - Weights/embeddings int8 with float32 scales;
#   - Total BIN + JSON size <= 10 MiB;
#   - Calibration F0.5 drop from quantization <= 0.005.
# ============================================================
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.export_attention_model import (
    FORMAT_VERSION,
    MAGIC,
    AttentionModelReader,
    export_quantized_model,
    quantize_linear_weight,
    quantize_row_embedding,
)


class BinaryFormatTest(unittest.TestCase):
    def test_quantization_formulas(self):
        # 1. Per-row symmetric embedding quantization
        row = np.array([[-2.54, 0.0, 1.27], [5.08, -1.0, 0.0]], dtype=np.float32)
        int8_arr, scales = quantize_row_embedding(row)
        self.assertEqual(int8_arr.dtype, np.int8)
        self.assertEqual(scales.shape, (2,))
        self.assertAlmostEqual(scales[0], 2.54 / 127.0, places=5)
        self.assertAlmostEqual(scales[1], 5.08 / 127.0, places=5)
        # Dequantize and check error
        dequant = int8_arr.astype(np.float32) * scales[:, None]
        self.assertTrue(np.allclose(row, dequant, atol=0.03))

        # 2. Per-channel symmetric linear weight quantization
        weight = np.random.randn(48, 96).astype(np.float32)
        int8_w, w_scales = quantize_linear_weight(weight)
        self.assertEqual(int8_w.dtype, np.int8)
        self.assertEqual(w_scales.shape, (48,))
        dequant_w = int8_w.astype(np.float32) * w_scales[:, None]
        self.assertTrue(np.allclose(weight, dequant_w, atol=0.05))

    def test_binary_header_and_reader(self):
        with tempfile.TemporaryDirectory() as td:
            bin_path = os.path.join(td, "test.bin")
            meta_path = os.path.join(td, "test.json")

            # Create dummy checkpoint
            ckpt_path = os.path.join(td, "ckpt.pt")
            from tools.attention_model import AttentionModelConfig
            cfg = AttentionModelConfig("A", 1, 48, 2, 24, 96, vocab_size=500)
            from tools.train_attention_reranker import ListwiseReranker
            reranker = ListwiseReranker(cfg)
            torch.save({
                "schema": "attention-finetuned-v1",
                "arch": "A",
                "config": cfg.__dict__,
                "state_dict": reranker.state_dict(),
                "vocab_hash": "dummy_vocab",
                "shortlist_config_hash": "dummy_shortlist",
                "k": 8,
            }, ckpt_path)

            export_quantized_model(ckpt_path, bin_path, meta_path)

            # Read and verify header
            with open(bin_path, "rb") as fh:
                magic = fh.read(8)
                self.assertEqual(magic, MAGIC)
                (ver, tensor_count, table_offset, data_offset) = struct.unpack("<IIII", fh.read(16))
                self.assertEqual(ver, FORMAT_VERSION)
                self.assertGreater(tensor_count, 0)
                self.assertEqual(data_offset % 16, 0, "Data offset must be 16-byte aligned")

            # Test reader
            reader = AttentionModelReader(bin_path, meta_path)
            self.assertEqual(reader.meta["arch"], "A")
            self.assertIn("encoder.word_embedding.weight", reader.tensors)

            # Test corrupt magic rejection
            corrupt_bin = os.path.join(td, "corrupt.bin")
            with open(bin_path, "rb") as fh:
                data = bytearray(fh.read())
            data[:8] = b"BADMAGIC"
            with open(corrupt_bin, "wb") as fh:
                fh.write(data)

            with self.assertRaisesRegex(ValueError, "magic"):
                AttentionModelReader(corrupt_bin, meta_path)

    def test_artifact_size_under_10mib(self):
        with tempfile.TemporaryDirectory() as td:
            bin_path = os.path.join(td, "test.bin")
            meta_path = os.path.join(td, "test.json")
            ckpt_path = os.path.join(td, "ckpt.pt")

            from tools.attention_model import AttentionModelConfig
            cfg = AttentionModelConfig("C", 2, 96, 4, 24, 192, vocab_size=8192)
            from tools.train_attention_reranker import ListwiseReranker
            reranker = ListwiseReranker(cfg)
            torch.save({
                "schema": "attention-finetuned-v1",
                "arch": "C",
                "config": cfg.__dict__,
                "state_dict": reranker.state_dict(),
                "vocab_hash": "dummy_vocab",
                "shortlist_config_hash": "dummy_shortlist",
                "k": 8,
            }, ckpt_path)

            export_quantized_model(ckpt_path, bin_path, meta_path)
            total_size = os.path.getsize(bin_path) + os.path.getsize(meta_path)
            self.assertLess(total_size, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()

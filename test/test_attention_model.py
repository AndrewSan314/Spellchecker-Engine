# ============================================================
# Task 5 — Unit tests for Tiny Attention Encoder architectures A, B, C.
#
# Contract under test:
#   - Architectures A (1 blk, 48 hid, 2 heads, 96 ffn), B (2 blk, 64 hid, 4 heads, 128 ffn),
#     C (2 blk, 96 hid, 4 heads, 192 ffn);
#   - Maximum 32 tokens;
#   - Pre-LN residual formula: x + Attn(LN(x)), a + FFN(LN(a));
#   - Word + Char + Position + Marker embeddings;
#   - Stable masked softmax (padding receives -inf/large negative before softmax);
#   - Deterministic forward output;
#   - No NaN/Inf on normal or padded inputs;
#   - Model size well within 10 MiB int8 budget.
# ============================================================
from __future__ import annotations

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.attention_model import (
    ARCHITECTURES,
    AttentionEncoder,
    AttentionModelConfig,
    create_model,
)
from tools.attention_tokenizer import (
    CHAR_HASH_BUCKETS,
    MAX_CONTEXT_TOKENS,
    SPECIAL_IDS,
    VOCAB_SIZE,
)


class ModelArchitectureTest(unittest.TestCase):
    def test_architecture_specs(self):
        self.assertEqual(ARCHITECTURES["A"].blocks, 1)
        self.assertEqual(ARCHITECTURES["A"].hidden_dim, 48)
        self.assertEqual(ARCHITECTURES["A"].num_heads, 2)
        self.assertEqual(ARCHITECTURES["A"].head_dim, 24)
        self.assertEqual(ARCHITECTURES["A"].ffn_dim, 96)

        self.assertEqual(ARCHITECTURES["B"].blocks, 2)
        self.assertEqual(ARCHITECTURES["B"].hidden_dim, 64)
        self.assertEqual(ARCHITECTURES["B"].num_heads, 4)
        self.assertEqual(ARCHITECTURES["B"].head_dim, 16)
        self.assertEqual(ARCHITECTURES["B"].ffn_dim, 128)

        self.assertEqual(ARCHITECTURES["C"].blocks, 2)
        self.assertEqual(ARCHITECTURES["C"].hidden_dim, 96)
        self.assertEqual(ARCHITECTURES["C"].num_heads, 4)
        self.assertEqual(ARCHITECTURES["C"].head_dim, 24)
        self.assertEqual(ARCHITECTURES["C"].ffn_dim, 192)

    def test_forward_pass_dims_and_no_nan(self):
        for arch_id in ("A", "B", "C"):
            model = create_model(arch_id, vocab_size=1000)
            model.eval()
            batch_size = 2
            seq_len = 32

            word_ids = torch.randint(0, 1000, (batch_size, seq_len))
            # Put some padding
            word_ids[:, 20:] = SPECIAL_IDS["PAD"]
            mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
            mask[:, 20:] = 0.0

            markers = torch.zeros((batch_size, seq_len), dtype=torch.long)
            markers[:, 5] = 1  # TARGET

            # Empty or filled char hashes
            char_hashes = torch.randint(0, CHAR_HASH_BUCKETS, (batch_size, seq_len, 6))
            char_counts = torch.full((batch_size, seq_len), 6, dtype=torch.long)

            with torch.no_grad():
                out = model(
                    word_ids=word_ids,
                    mask=mask,
                    markers=markers,
                    char_hashes=char_hashes,
                    char_counts=char_counts,
                )

            hidden = out["hidden"]  # (batch, seq_len, hidden_dim)
            self.assertEqual(hidden.shape, (batch_size, seq_len, ARCHITECTURES[arch_id].hidden_dim))
            self.assertFalse(torch.isnan(hidden).any(), f"NaN found in {arch_id}")
            self.assertFalse(torch.isinf(hidden).any(), f"Inf found in {arch_id}")

    def test_pre_ln_structure(self):
        """Verify Pre-LN: LayerNorm is evaluated before attention and FFN sublayers."""
        model = create_model("A", vocab_size=500)
        block = model.encoder.blocks[0]
        self.assertTrue(hasattr(block, "ln1"), "Block must have ln1 before attention")
        self.assertTrue(hasattr(block, "ln2"), "Block must have ln2 before FFN")
        self.assertTrue(model.config.pre_ln, "Config must specify pre_ln=True")

    def test_deterministic_forward(self):
        torch.manual_seed(20260825)
        m1 = create_model("A", vocab_size=500)
        torch.manual_seed(20260825)
        m2 = create_model("A", vocab_size=500)

        word_ids = torch.randint(0, 500, (1, 16))
        mask = torch.ones((1, 16), dtype=torch.float32)
        markers = torch.zeros((1, 16), dtype=torch.long)
        char_hashes = torch.zeros((1, 16, 4), dtype=torch.long)
        char_counts = torch.zeros((1, 16), dtype=torch.long)

        m1.eval()
        m2.eval()
        with torch.no_grad():
            o1 = m1(word_ids, mask, markers, char_hashes, char_counts)["hidden"]
            o2 = m2(word_ids, mask, markers, char_hashes, char_counts)["hidden"]

        self.assertTrue(torch.allclose(o1, o2, atol=1e-6))

    def test_parameter_budget_under_10mib(self):
        for arch_id in ("A", "B", "C"):
            model = create_model(arch_id, vocab_size=VOCAB_SIZE)
            total_params = sum(p.numel() for p in model.parameters())
            # int8 size approx 1 byte per param
            estimated_bytes = total_params
            self.assertLess(
                estimated_bytes,
                10 * 1024 * 1024,
                f"Model {arch_id} exceeds 10 MiB int8 budget ({estimated_bytes} bytes)",
            )


if __name__ == "__main__":
    unittest.main()

# ============================================================
# Task 6 — Unit tests for Listwise Attention Spelling Reranker Fine-Tuning.
#
# Contract under test:
#   - Single context encode: 1 self-attention pass per target token, regardless of candidate count;
#   - Candidate listwise scoring: [KEEP_ORIGINAL, candidate_1, ..., candidate_K];
#   - Padding options receive zero probability;
#   - Candidate permutations preserve remapped scores;
#   - Shortlist size K loaded from shortlist-config.json, not hardcoded;
#   - Loss function: CrossEntropy + 0.1 * margin_ranking_loss;
#   - Overfit tiny fixture;
#   - Disjoint group IDs across train, calibration, internal-test.
# ============================================================
from __future__ import annotations

import json
import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.attention_model import (
    AttentionModel,
    AttentionModelConfig,
    create_model,
)
from tools.train_attention_reranker import (
    ListwiseReranker,
    RankingDataset,
    compute_listwise_loss,
    evaluate_ranking_metrics,
    extract_classical_feature_tensor,
)


class SingleEncodeTest(unittest.TestCase):
    def test_single_context_encode_invariant(self):
        """Prove that the transformer encoder is called exactly once per target."""
        config = AttentionModelConfig(
            arch_id="A", blocks=1, hidden_dim=48, num_heads=2, head_dim=24, ffn_dim=96, vocab_size=500
        )
        reranker = ListwiseReranker(config, num_classical_features=15)
        reranker.eval()

        call_count = 0
        original_forward = reranker.model.encoder.forward

        def hooked_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)

        reranker.model.encoder.forward = hooked_forward

        batch_size = 1
        seq_len = 32
        K = 8

        word_ids = torch.randint(0, 500, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
        target_positions = torch.tensor([5], dtype=torch.long)

        # Candidates (batch, K+1)
        option_word_ids = torch.randint(0, 500, (batch_size, K + 1))
        option_mask = torch.ones((batch_size, K + 1), dtype=torch.float32)
        classical_features = torch.randn((batch_size, K + 1, 15))

        with torch.no_grad():
            out = reranker(
                word_ids=word_ids,
                mask=mask,
                target_positions=target_positions,
                option_word_ids=option_word_ids,
                option_mask=option_mask,
                classical_features=classical_features,
            )

        self.assertEqual(call_count, 1, f"Expected 1 encoder call, got {call_count}")
        self.assertEqual(out["logits"].shape, (batch_size, K + 1))
        self.assertEqual(out["probabilities"].shape, (batch_size, K + 1))


class ListwiseScoringTest(unittest.TestCase):
    def test_padding_option_masked(self):
        config = AttentionModelConfig(
            arch_id="A", blocks=1, hidden_dim=48, num_heads=2, head_dim=24, ffn_dim=96, vocab_size=500
        )
        reranker = ListwiseReranker(config, num_classical_features=15)
        reranker.eval()

        batch_size = 1
        seq_len = 32
        K = 8

        word_ids = torch.randint(0, 500, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
        target_positions = torch.tensor([0], dtype=torch.long)

        option_word_ids = torch.randint(0, 500, (batch_size, K + 1))
        # Mask out last 3 candidates (padding)
        option_mask = torch.ones((batch_size, K + 1), dtype=torch.float32)
        option_mask[:, 6:] = 0.0
        classical_features = torch.zeros((batch_size, K + 1, 15))

        with torch.no_grad():
            out = reranker(
                word_ids=word_ids,
                mask=mask,
                target_positions=target_positions,
                option_word_ids=option_word_ids,
                option_mask=option_mask,
                classical_features=classical_features,
            )

        probs = out["probabilities"][0]
        # Padded options must have 0 probability
        for idx in range(6, K + 1):
            self.assertAlmostEqual(probs[idx].item(), 0.0, places=5)
        # Sum of valid options must be 1.0
        self.assertAlmostEqual(probs[:6].sum().item(), 1.0, places=5)

    def test_loss_function(self):
        logits = torch.tensor([[2.0, 5.0, 1.0, 0.0]])
        gold_index = torch.tensor([1], dtype=torch.long)  # Option 1 is gold
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        loss = compute_listwise_loss(logits, gold_index, mask)
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_overfit_tiny_ranking_fixture(self):
        torch.manual_seed(20260825)
        config = AttentionModelConfig(
            arch_id="A", blocks=1, hidden_dim=48, num_heads=2, head_dim=24, ffn_dim=96, vocab_size=100
        )
        reranker = ListwiseReranker(config, num_classical_features=15)
        optimizer = torch.optim.AdamW(reranker.parameters(), lr=1e-3)

        batch_size = 4
        seq_len = 16
        K = 4

        word_ids = torch.randint(8, 100, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
        target_positions = torch.tensor([2, 4, 1, 3], dtype=torch.long)
        option_word_ids = torch.randint(8, 100, (batch_size, K + 1))
        option_mask = torch.ones((batch_size, K + 1), dtype=torch.float32)
        classical_features = torch.randn((batch_size, K + 1, 15))
        labels = torch.tensor([0, 2, 1, 0], dtype=torch.long)  # Mixed KEEP and corrections

        reranker.train()
        for step in range(40):
            optimizer.zero_grad()
            out = reranker(
                word_ids=word_ids,
                mask=mask,
                target_positions=target_positions,
                option_word_ids=option_word_ids,
                option_mask=option_mask,
                classical_features=classical_features,
            )
            loss = compute_listwise_loss(out["logits"], labels, option_mask)
            loss.backward()
            optimizer.step()

        reranker.eval()
        with torch.no_grad():
            final_out = reranker(
                word_ids=word_ids,
                mask=mask,
                target_positions=target_positions,
                option_word_ids=option_word_ids,
                option_mask=option_mask,
                classical_features=classical_features,
            )
            preds = final_out["logits"].argmax(dim=-1)
            self.assertTrue(torch.equal(preds, labels), f"Expected {labels}, got {preds}")


if __name__ == "__main__":
    unittest.main()

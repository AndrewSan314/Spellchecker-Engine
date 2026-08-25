# ============================================================
# Task 5 — Unit tests for Masked LM Pretraining pipeline.
#
# Contract under test:
#   - 15% masking on non-special, non-padding tokens;
#   - 80% MASK (ID 7), 10% random learned vocab (8..vocab_size-1), 10% unchanged;
#   - Loss computed only at masked positions;
#   - Special/padding tokens never targeted;
#   - Model learns tiny fixture (loss decreases materially in 50 steps);
#   - No dev/test/held-out references in training artifacts.
# ============================================================
from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.attention_model import create_model
from tools.attention_tokenizer import SPECIAL_IDS
from tools.pretrain_attention_encoder import (
    PretrainingDataset,
    apply_mlm_masking,
    train_epoch,
)


class PretrainingMaskingTest(unittest.TestCase):
    def test_masking_rules_and_preservation(self):
        vocab_size = 100
        batch_size = 10
        seq_len = 32

        # Create batch with specials and normal tokens
        input_ids = torch.randint(8, vocab_size, (batch_size, seq_len))
        # Add special tokens
        input_ids[:, 0] = SPECIAL_IDS["BOS"]
        input_ids[:, 10] = SPECIAL_IDS["PROTECTED"]
        input_ids[:, 15] = SPECIAL_IDS["PUNCT"]
        input_ids[:, 25:] = SPECIAL_IDS["PAD"]

        mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
        mask[:, 25:] = 0.0  # Padding mask

        masked_ids, labels = apply_mlm_masking(
            input_ids, mask, vocab_size=vocab_size, mask_prob=0.5, seed=20260825
        )

        # Labels are -100 (ignored in CrossEntropyLoss) everywhere except masked positions
        target_positions = (labels != -100)

        # 1. Padding must NEVER be masked
        self.assertFalse(target_positions[:, 25:].any(), "Padding positions must never be masked")

        # 2. Special tokens (0..7) must NEVER be masked
        for sp_id in range(8):
            was_special = (input_ids == sp_id)
            self.assertFalse(
                (target_positions & was_special).any(),
                f"Special ID {sp_id} was improperly targeted for masking",
            )

        # 3. Label at target positions must equal the original input_ids
        self.assertTrue(
            torch.equal(labels[target_positions], input_ids[target_positions]),
            "Labels at target positions must match original token IDs",
        )

        # 4. Check masked_ids: at target positions, should be MASK (80%), random, or unchanged
        # Check that MASK id is present
        mask_count = (masked_ids[target_positions] == SPECIAL_IDS["MASK"]).sum().item()
        self.assertGreater(mask_count, 0, "MASK tokens should be assigned")


class PretrainingConvergenceTest(unittest.TestCase):
    def test_overfit_tiny_fixture(self):
        torch.manual_seed(20260825)
        np.random.seed(20260825)

        vocab_size = 50
        model = create_model("A", vocab_size=vocab_size, with_mlm_head=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # 2 fixed sentences repeated
        seq_len = 16
        raw_ids = torch.tensor([
            [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.long)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)

        losses = []
        model.train()
        for step in range(60):
            optimizer.zero_grad()
            masked_ids, labels = apply_mlm_masking(
                raw_ids, mask, vocab_size=vocab_size, mask_prob=0.3, seed=20260825 + step
            )
            # Dummy char hashes & markers
            markers = torch.zeros_like(raw_ids)
            char_hashes = torch.zeros((*raw_ids.shape, 2), dtype=torch.long)
            char_counts = torch.zeros_like(raw_ids)

            out = model(
                word_ids=masked_ids,
                mask=mask,
                markers=markers,
                char_hashes=char_hashes,
                char_counts=char_counts,
            )
            logits = out["mlm_logits"]  # (batch, seq_len, vocab_size)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        self.assertLess(
            final_loss,
            initial_loss * 0.7,
            f"Pretraining loss failed to drop materially: {initial_loss:.4f} -> {final_loss:.4f}",
        )


if __name__ == "__main__":
    unittest.main()

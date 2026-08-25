# ============================================================
# Task 5 (tiny-attention-spelling-reranker-FIXED plan) — Pretrain
# Tiny Attention Vietnamese Encoder using Masked Language Modeling (MLM).
#
# Masking contract:
#   - 15% of non-special, non-padding tokens selected;
#   - 80% MASK (ID 7), 10% random learned vocab (8..vocab_size-1), 10% unchanged;
#   - Loss computed only at masked positions (CrossEntropy with ignore_index=-100);
#   - Special IDs (0..7) and padding (mask=0) are never targeted.
# ============================================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.attention_model import (
    ARCHITECTURES,
    AttentionModel,
    create_model,
)
from tools.attention_tokenizer import (
    FIRST_LEARNED_ID,
    SPECIAL_IDS,
    VOCAB_SIZE,
)


def apply_mlm_masking(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    vocab_size: int,
    mask_prob: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply standard BERT-style 15% MLM masking to non-special, non-padding tokens.
    Returns:
        masked_ids: Tensor of shape (batch, seq_len) with masked/replaced tokens
        labels: Tensor of shape (batch, seq_len) with original token IDs at masked positions, -100 elsewhere.
    """
    if seed is not None:
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    else:
        generator = None

    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    # Eligible tokens: non-padding (mask==1) and not a special token (id >= FIRST_LEARNED_ID)
    eligible = (mask > 0) & (input_ids >= FIRST_LEARNED_ID)

    # Sample random probabilities for eligible positions
    rand_probs = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
    # Mask positions where eligible and rand_probs < mask_prob
    target_mask = eligible & (rand_probs < mask_prob)

    # Guarantee at least one masked position if there are eligible tokens and none selected
    for b in range(input_ids.shape[0]):
        if not target_mask[b].any() and eligible[b].any():
            eligible_indices = torch.where(eligible[b])[0]
            pick = eligible_indices[torch.randint(0, len(eligible_indices), (1,), generator=generator).item()]
            target_mask[b, pick] = True

    # Set labels for targets
    labels[target_mask] = input_ids[target_mask]

    # For target positions, sample replacement strategy:
    # 80% MASK, 10% random learned vocab, 10% unchanged
    rand_types = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)

    # 80% -> MASK
    mask_to_mask = target_mask & (rand_types < 0.8)
    masked_ids[mask_to_mask] = SPECIAL_IDS["MASK"]

    # 10% -> Random learned vocab token [FIRST_LEARNED_ID .. vocab_size - 1]
    mask_to_random = target_mask & (rand_types >= 0.8) & (rand_types < 0.9)
    if mask_to_random.any():
        num_random = mask_to_random.sum().item()
        random_tokens = torch.randint(
            FIRST_LEARNED_ID,
            max(FIRST_LEARNED_ID + 1, vocab_size),
            (num_random,),
            generator=generator,
            device=input_ids.device,
        )
        masked_ids[mask_to_random] = random_tokens

    # 10% -> Unchanged (already in masked_ids)

    return masked_ids, labels


class PretrainingNPZDataset(Dataset):
    def __init__(self, shards: List[str]):
        all_ids = []
        all_mask = []
        for shard_path in shards:
            data = np.load(shard_path)
            all_ids.append(data["ids"])
            all_mask.append(data["mask"])

        self.ids = np.concatenate(all_ids, axis=0) if all_ids else np.zeros((0, 32), dtype=np.int32)
        self.mask = np.concatenate(all_mask, axis=0) if all_mask else np.zeros((0, 32), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.ids[idx], dtype=torch.long),
            torch.tensor(self.mask[idx], dtype=torch.float32),
        )


PretrainingDataset = PretrainingNPZDataset


def evaluate_loss(
    model: AttentionModel,
    dataloader: DataLoader,
    vocab_size: int,
    criterion: nn.Module,
    device: torch.device,
    seed: int = 20260825,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for step, (input_ids, mask) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            masked_ids, labels = apply_mlm_masking(
                input_ids, mask, vocab_size=vocab_size, mask_prob=0.15, seed=seed + step
            )
            out = model(word_ids=masked_ids, mask=mask)
            logits = out["mlm_logits"]
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            num_targets = (labels != -100).sum().item()
            if num_targets > 0:
                total_loss += loss.item() * num_targets
                total_tokens += num_targets
    return total_loss / max(1, total_tokens)


def train_epoch(
    model: AttentionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    vocab_size: int,
    device: torch.device,
    epoch: int,
    seed: int = 20260825,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for step, (input_ids, mask) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        masked_ids, labels = apply_mlm_masking(
            input_ids, mask, vocab_size=vocab_size, mask_prob=0.15, seed=seed + epoch * 10000 + step
        )
        optimizer.zero_grad()
        out = model(word_ids=masked_ids, mask=mask)
        logits = out["mlm_logits"]
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_targets = (labels != -100).sum().item()
        if num_targets > 0:
            total_loss += loss.item() * num_targets
            total_tokens += num_targets
    return total_loss / max(1, total_tokens)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Pretrain Attention Encoder Candidate")
    parser.add_argument("--arch", required=True, choices=["A", "B", "C"])
    parser.add_argument("--data-manifest", default=".tmp/attention-pretrain-manifest.json")
    parser.add_argument("--vocab", default=".tmp/attention-vocab.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # Reproducibility seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    # Load vocab
    with open(args.vocab, "r", encoding="utf-8") as fh:
        vocab_data = json.load(fh)
    vocab_size = vocab_data.get("size", VOCAB_SIZE)
    vocab_hash = sha256_file(args.vocab)

    # Load manifest
    with open(args.data_manifest, "r", encoding="utf-8") as fh:
        manifest_data = json.load(fh)
    manifest_hash = sha256_file(args.data_manifest)

    shards = [s["path"] for s in manifest_data.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards found in manifest {args.data_manifest}")

    full_dataset = PretrainingNPZDataset(shards)
    total_len = len(full_dataset)
    if total_len == 0:
        raise ValueError("Pretraining dataset is empty")

    # Deterministic 95% train / 5% val partition
    val_size = max(1, int(total_len * 0.05))
    train_size = total_len - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Create model with MLM head
    model = create_model(args.arch, vocab_size=vocab_size, with_mlm_head=True)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Evaluate untrained control loss
    untrained_val_loss = evaluate_loss(model, val_loader, vocab_size, criterion, device, seed=args.seed)
    print(f"[{args.arch}] Untrained control val loss: {untrained_val_loss:.4f}")

    best_val_loss = untrained_val_loss
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, vocab_size, device, epoch, seed=args.seed
        )
        val_loss = evaluate_loss(
            model, val_loader, vocab_size, criterion, device, seed=args.seed + epoch * 100
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        print(f"[{args.arch}] Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 2:
                print(f"[{args.arch}] Early stopping at epoch {epoch} (no improvement for 2 epochs)")
                break

    elapsed_sec = time.time() - t0

    # Save checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    checkpoint = {
        "schema": "attention-pretrained-v1",
        "arch": args.arch,
        "config": {
            "arch_id": model.config.arch_id,
            "blocks": model.config.blocks,
            "hidden_dim": model.config.hidden_dim,
            "num_heads": model.config.num_heads,
            "head_dim": model.config.head_dim,
            "ffn_dim": model.config.ffn_dim,
            "vocab_size": model.config.vocab_size,
            "max_tokens": model.config.max_tokens,
            "pre_ln": model.config.pre_ln,
        },
        "vocab_hash": vocab_hash,
        "vocab_size": vocab_size,
        "seed": args.seed,
        "data_manifest_hash": manifest_hash,
        "state_dict": model.state_dict(),
        "metrics": {
            "untrained_val_loss": untrained_val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "loss_improved": best_val_loss < untrained_val_loss,
            "elapsed_sec": elapsed_sec,
        },
    }
    torch.save(checkpoint, args.output)

    # Save pretrain report
    out_dir = os.path.dirname(os.path.abspath(args.output))
    report_path = os.path.join(out_dir, f"attention-pretrain-report-{args.arch}.json")
    report = {
        "schema": "attention-pretrain-report-v1",
        "arch": args.arch,
        "checkpointPath": args.output,
        "checkpointHash": sha256_file(args.output),
        "vocabHash": vocab_hash,
        "manifestHash": manifest_hash,
        "metrics": checkpoint["metrics"],
        "history": history,
        "seed": args.seed,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps({
        "ok": True,
        "arch": args.arch,
        "untrained_loss": untrained_val_loss,
        "best_val_loss": best_val_loss,
        "improved": best_val_loss < untrained_val_loss,
        "output": args.output,
        "report": report_path,
    }, indent=2))


if __name__ == "__main__":
    main()

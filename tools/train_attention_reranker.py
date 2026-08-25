# ============================================================
# Task 6 (tiny-attention-spelling-reranker-FIXED plan) — Single-encode
# Listwise Attention Spelling Reranker training and architecture selection.
#
# Constraints honored:
#   - Context encoded ONCE per target token;
#   - Ranks [KEEP_ORIGINAL, candidate_1, ..., candidate_K];
#   - K loaded from shortlist config (frozen K <= 8);
#   - Pretrained weights loaded from Task 5 (.tmp/attention-pretrained-{ARCH}.pt);
#   - Evaluated on calibration ONLY; internal-test and dev are NEVER read;
#   - Writes winning architecture selection to .tmp/attention-selected-model.json.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.attention_model import (
    ARCHITECTURES,
    AttentionModel,
    AttentionModelConfig,
    create_model,
)
from tools.attention_tokenizer import (
    MAX_CHAR_NGRAMS,
    MAX_CONTEXT_TOKENS,
    SPECIAL_IDS,
    VOCAB_SIZE,
    encode_context_units,
    encode_option_surface,
    normalize_for_model,
)

NUM_CLASSICAL_FEATURES = 15
CLASSICAL_FEATURE_KEYS = [
    "editDistance",
    "sameAccentKey",
    "candidateMinusOriginalLogFrequency",
    "leftBigramLogRatio",
    "rightBigramLogRatio",
    "centeredTrigramLogRatio",
    "forwardTrigramLogRatio",
    "backwardTrigramLogRatio",
    "candidateAttestedWindows",
    "originalAttestedWindows",
    "tokenLength",
    "originalIsDictionary",
    "candidateCheapRank",
    "candidatePoolRank",
    "candidateIsDictionary",
]


def extract_classical_feature_tensor(row: dict, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract classical features matrix for [KEEP_ORIGINAL, cand_1, ..., cand_K].
    Returns:
        features: np.ndarray of shape (K + 1, 15)
        mask: np.ndarray of shape (K + 1,) with 1 for valid options, 0 for padded
    """
    mat = np.zeros((k + 1, NUM_CLASSICAL_FEATURES), dtype=np.float32)
    mask = np.zeros(k + 1, dtype=np.float32)

    # Option 0: KEEP_ORIGINAL
    mask[0] = 1.0
    cands = row.get("candidates", [])
    raw_feats = row.get("classicalFeatures", [])

    # If first candidate has original features, borrow them for KEEP
    first_f = raw_feats[0] if raw_feats else {}
    mat[0, 1] = 1.0  # sameAccentKey
    mat[0, 8] = float(first_f.get("originalAttestedWindows", 0))
    mat[0, 9] = float(first_f.get("originalAttestedWindows", 0))
    mat[0, 10] = float(first_f.get("tokenLength", len(row.get("original", ""))))
    mat[0, 11] = float(first_f.get("originalIsDictionary", 1))
    mat[0, 14] = float(first_f.get("originalIsDictionary", 1))

    # Options 1..len(candidates)
    for i, c in enumerate(cands[:k]):
        slot = i + 1
        mask[slot] = 1.0
        f = raw_feats[i] if i < len(raw_feats) else {}
        for j, key in enumerate(CLASSICAL_FEATURE_KEYS):
            mat[slot, j] = float(f.get(key, 0.0))

    return mat, mask


def pad_char_hash_rows(rows: List[List[int]], width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros((width, MAX_CHAR_NGRAMS), dtype=torch.long)
    counts = torch.zeros(width, dtype=torch.long)
    for i, hashes in enumerate(rows[:width]):
        clipped = [int(h) for h in hashes[:MAX_CHAR_NGRAMS]]
        if clipped:
            values[i, :len(clipped)] = torch.tensor(clipped, dtype=torch.long)
            counts[i] = len(clipped)
    return values, counts

class ListwiseReranker(nn.Module):
    def __init__(
        self,
        config: AttentionModelConfig,
        num_classical_features: int = NUM_CLASSICAL_FEATURES,
        classical_proj_dim: int = 16,
    ):
        super().__init__()
        self.config = config
        self.model = AttentionModel(
            config,
            with_mlm_head=False,
            num_classical_features=num_classical_features,
            classical_proj_dim=classical_proj_dim,
        )

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        target_positions: torch.Tensor,
        option_word_ids: torch.Tensor,
        option_mask: torch.Tensor,
        classical_features: torch.Tensor,
        markers: Optional[torch.Tensor] = None,
        char_hashes: Optional[torch.Tensor] = None,
        char_counts: Optional[torch.Tensor] = None,
        option_char_hashes: Optional[torch.Tensor] = None,
        option_char_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Single-encode listwise scoring.
        word_ids: (batch, seq_len)
        mask: (batch, seq_len)
        target_positions: (batch,)
        option_word_ids: (batch, K+1)
        option_mask: (batch, K+1)
        classical_features: (batch, K+1, num_classical_features)
        """
        batch_size, seq_len = word_ids.shape
        num_options = option_word_ids.shape[1]

        # 1. Encode context ONCE per target token
        h = self.model.encoder(
            word_ids=word_ids,
            mask=mask,
            markers=markers,
            char_hashes=char_hashes,
            char_counts=char_counts,
        )  # (batch, seq_len, hidden_dim)

        # 2. Extract context vector at target position
        batch_indices = torch.arange(batch_size, device=word_ids.device)
        context_vec = h[batch_indices, target_positions]  # (batch, hidden_dim)

        # 3. Score all options with shared scorer
        scores = []
        for opt_idx in range(num_options):
            opt_words = option_word_ids[:, opt_idx]  # (batch,)
            opt_emb = self.model.compute_option_embedding(
                opt_words,
                option_char_hashes[:, opt_idx] if option_char_hashes is not None else None,
                option_char_counts[:, opt_idx] if option_char_counts is not None else None,
            )  # (batch, hidden_dim)

            class_feat = classical_features[:, opt_idx]  # (batch, num_classical_features)
            class_proj = self.model.classical_proj(class_feat)  # (batch, 16)

            opt_type = torch.zeros(batch_size, dtype=torch.long, device=word_ids.device) if opt_idx == 0 \
                else torch.ones(batch_size, dtype=torch.long, device=word_ids.device)
            opt_type_emb = self.model.option_type_embedding(opt_type)  # (batch, 4)

            z = torch.cat(
                [context_vec, opt_emb, context_vec * opt_emb, class_proj, opt_type_emb],
                dim=-1,
            )  # (batch, 3*hidden + 16 + 4)

            opt_score = self.model.option_scorer(z).squeeze(-1)  # (batch,)
            scores.append(opt_score)

        logits = torch.stack(scores, dim=1)  # (batch, num_options)
        masked_logits = logits.masked_fill(option_mask == 0, -10000.0)
        probs = F.softmax(masked_logits, dim=-1)

        return {"logits": masked_logits, "probabilities": probs}


def compute_listwise_loss(
    logits: torch.Tensor,
    gold_index: torch.Tensor,
    option_mask: torch.Tensor,
) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, gold_index)

    # Margin ranking loss: max(0, 0.2 - gold_logit + highest_wrong_logit)
    gold_logits = logits.gather(1, gold_index.unsqueeze(1)).squeeze(1)
    logits_no_gold = logits.clone()
    logits_no_gold.scatter_(1, gold_index.unsqueeze(1), -1e9)
    logits_no_gold = logits_no_gold.masked_fill(option_mask == 0, -1e9)
    highest_wrong, _ = logits_no_gold.max(dim=1)

    margin_loss = F.relu(0.2 - gold_logits + highest_wrong).mean()
    return ce_loss + 0.1 * margin_loss


class RankingDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: Dict[str, int], k: int = 8):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("recordType") == "ranking-row":
                    self.rows.append(obj)

        self.vocab = vocab
        self.k = k

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        ctx = row.get("context", {})
        units = ctx.get("units", [])
        target_unit_idx = ctx.get("targetUnitIdx", 0)

        enc = encode_context_units(units, target_unit_idx, self.vocab)

        # Pad context to 32
        n_ctx = len(enc["ids"])
        if n_ctx < MAX_CONTEXT_TOKENS:
            ctx_ids = enc["ids"] + [SPECIAL_IDS["PAD"]] * (MAX_CONTEXT_TOKENS - n_ctx)
            ctx_mask = enc["mask"] + [0] * (MAX_CONTEXT_TOKENS - n_ctx)
            ctx_markers = enc["markers"] + [0] * (MAX_CONTEXT_TOKENS - n_ctx)
        else:
            ctx_ids = enc["ids"][:MAX_CONTEXT_TOKENS]
            ctx_mask = enc["mask"][:MAX_CONTEXT_TOKENS]
            ctx_markers = enc["markers"][:MAX_CONTEXT_TOKENS]

        target_pos = enc["targetPosition"]
        ctx_char_values, ctx_char_counts = pad_char_hash_rows(enc["charHashes"], MAX_CONTEXT_TOKENS)

        # Build options: option 0 = original, option 1..K = candidates
        original_option = encode_option_surface(row.get("original", ""), self.vocab)
        opt_ids = [original_option["id"]]
        option_char_rows = [original_option["charHashes"]]
        cands = row.get("candidates", [])
        for c in cands[:self.k]:
            surf = c.get("surface", "")
            option = encode_option_surface(surf, self.vocab)
            opt_ids.append(option["id"])
            option_char_rows.append(option["charHashes"])

        # Pad options to K+1
        while len(opt_ids) < self.k + 1:
            opt_ids.append(SPECIAL_IDS["PAD"])
            option_char_rows.append([])

        feat_mat, opt_mask = extract_classical_feature_tensor(row, k=self.k)
        option_char_values, option_char_counts = pad_char_hash_rows(option_char_rows, self.k + 1)

        return {
            "word_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "mask": torch.tensor(ctx_mask, dtype=torch.float32),
            "markers": torch.tensor(ctx_markers, dtype=torch.long),
            "char_hashes": ctx_char_values,
            "char_counts": ctx_char_counts,
            "target_position": torch.tensor(target_pos, dtype=torch.long),
            "option_word_ids": torch.tensor(opt_ids, dtype=torch.long),
            "option_char_hashes": option_char_values,
            "option_char_counts": option_char_counts,
            "option_mask": torch.tensor(opt_mask, dtype=torch.float32),
            "classical_features": torch.tensor(feat_mat, dtype=torch.float32),
            "label_index": torch.tensor(row.get("labelIndex", 0), dtype=torch.long),
            "lane": row.get("lane", ""),
            "id": row.get("id", ""),
        }


def ranking_collate_fn(batch: List[dict]) -> dict:
    return {
        "word_ids": torch.stack([b["word_ids"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "markers": torch.stack([b["markers"] for b in batch]),
        "char_hashes": torch.stack([b["char_hashes"] for b in batch]),
        "char_counts": torch.stack([b["char_counts"] for b in batch]),
        "target_positions": torch.stack([b["target_position"] for b in batch]),
        "option_word_ids": torch.stack([b["option_word_ids"] for b in batch]),
        "option_char_hashes": torch.stack([b["option_char_hashes"] for b in batch]),
        "option_char_counts": torch.stack([b["option_char_counts"] for b in batch]),
        "option_mask": torch.stack([b["option_mask"] for b in batch]),
        "classical_features": torch.stack([b["classical_features"] for b in batch]),
        "labels": torch.stack([b["label_index"] for b in batch]),
        "lanes": [b["lane"] for b in batch],
        "ids": [b["id"] for b in batch],
    }


def evaluate_ranking_metrics(
    reranker: ListwiseReranker,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    reranker.eval()
    total_samples = 0
    correct_preds = 0
    keep_total = 0
    keep_correct = 0
    correction_total = 0
    correction_correct = 0

    tp = 0
    fp = 0
    fn = 0
    total_loss = 0.0

    lane_stats: Dict[str, Dict[str, int]] = {}

    t0 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            word_ids = batch["word_ids"].to(device)
            mask = batch["mask"].to(device)
            markers = batch["markers"].to(device)
            char_hashes = batch["char_hashes"].to(device)
            char_counts = batch["char_counts"].to(device)
            option_char_hashes = batch["option_char_hashes"].to(device)
            option_char_counts = batch["option_char_counts"].to(device)
            target_positions = batch["target_positions"].to(device)
            option_word_ids = batch["option_word_ids"].to(device)
            option_mask = batch["option_mask"].to(device)
            classical_features = batch["classical_features"].to(device)
            labels = batch["labels"].to(device)
            lanes = batch["lanes"]

            out = reranker(
                word_ids=word_ids,
                mask=mask,
                target_positions=target_positions,
                option_word_ids=option_word_ids,
                option_mask=option_mask,
                classical_features=classical_features,
                markers=markers,
                char_hashes=char_hashes,
                char_counts=char_counts,
                option_char_hashes=option_char_hashes,
                option_char_counts=option_char_counts,
            )

            loss = compute_listwise_loss(out["logits"], labels, option_mask)
            total_loss += loss.item() * len(labels)

            preds = out["logits"].argmax(dim=-1)

            for i in range(len(labels)):
                gold = labels[i].item()
                pred = preds[i].item()
                lane = lanes[i]

                if lane not in lane_stats:
                    lane_stats[lane] = {"total": 0, "correct": 0}
                lane_stats[lane]["total"] += 1

                total_samples += 1
                if pred == gold:
                    correct_preds += 1
                    lane_stats[lane]["correct"] += 1

                if gold == 0:
                    keep_total += 1
                    if pred == 0:
                        keep_correct += 1
                    else:
                        fp += 1  # Proposes a correction on clean/keep
                else:
                    correction_total += 1
                    if pred == gold:
                        correction_correct += 1
                        tp += 1
                    elif pred == 0:
                        fn += 1
                    else:
                        # Chose wrong candidate
                        fp += 1
                        fn += 1

    elapsed_ms = (time.time() - t0) * 1000.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = (
        (1.25 * precision * recall) / (0.25 * precision + recall)
        if (0.25 * precision + recall) > 0
        else 0.0
    )

    return {
        "loss": total_loss / max(1, total_samples),
        "overall_accuracy": correct_preds / max(1, total_samples),
        "keep_accuracy": keep_correct / max(1, keep_total) if keep_total > 0 else 0.0,
        "correction_accuracy": correction_correct / max(1, correction_total) if correction_total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_samples": total_samples,
        "lane_stats": lane_stats,
        "inference_time_ms": elapsed_ms,
        "ms_per_sample": elapsed_ms / max(1, total_samples),
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train and select attention reranker")
    parser.add_argument("--rows-dir", default=".tmp")
    parser.add_argument("--split-manifest", default=".tmp/attention-ranking-split-manifest.json")
    parser.add_argument("--shortlist-config", default=".tmp/attention-shortlist-config.json")
    parser.add_argument("--pretrained-pattern", default=".tmp/attention-pretrained-{ARCH}.pt")
    parser.add_argument("--output-dir", default=".tmp")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    # Load shortlist config to get frozen K
    with open(args.shortlist_config, "r", encoding="utf-8") as fh:
        shortlist_cfg = json.load(fh)
    k = shortlist_cfg.get("k", 8)
    shortlist_cfg_hash = sha256_file(args.shortlist_config)

    # Load vocab
    vocab_path = os.path.join(args.rows_dir, "attention-vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as fh:
        vocab_data = json.load(fh)
    vocab = vocab_data.get("wordToId", {})
    vocab_size = vocab_data.get("size", VOCAB_SIZE)
    vocab_hash = sha256_file(vocab_path)

    # Load datasets
    train_path = os.path.join(args.rows_dir, "attention-ranking-train.jsonl")
    cal_path = os.path.join(args.rows_dir, "attention-ranking-calibration.jsonl")

    train_ds = RankingDataset(train_path, vocab, k=k)
    cal_ds = RankingDataset(cal_path, vocab, k=k)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ranking_collate_fn)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ranking_collate_fn)

    results = {}
    for arch_id in ("A", "B", "C"):
        print(f"\n================ Fine-tuning Candidate {arch_id} ================")
        base_cfg = ARCHITECTURES[arch_id]
        cfg = AttentionModelConfig(
            arch_id=base_cfg.arch_id,
            blocks=base_cfg.blocks,
            hidden_dim=base_cfg.hidden_dim,
            num_heads=base_cfg.num_heads,
            head_dim=base_cfg.head_dim,
            ffn_dim=base_cfg.ffn_dim,
            vocab_size=vocab_size,
        )

        reranker = ListwiseReranker(cfg, num_classical_features=NUM_CLASSICAL_FEATURES)

        # Load pretrained weights from Task 5 if present
        pt_path = args.pretrained_pattern.format(ARCH=arch_id)
        if os.path.exists(pt_path):
            print(f"Loading pretrained weights from {pt_path}")
            ckpt = torch.load(pt_path, map_location="cpu")
            st = ckpt.get("state_dict", {})
            # Filter out mlm_head weights
            enc_st = {k: v for k, v in st.items() if not k.startswith("mlm_head")}
            reranker.model.load_state_dict(enc_st, strict=False)

        reranker.to(device)
        optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=0.01)

        best_f05 = 0.0
        best_metrics = {}
        best_state = None

        for epoch in range(1, args.epochs + 1):
            reranker.train()
            train_loss = 0.0
            total_train = 0
            for batch in train_loader:
                word_ids = batch["word_ids"].to(device)
                mask = batch["mask"].to(device)
                markers = batch["markers"].to(device)
                char_hashes = batch["char_hashes"].to(device)
                char_counts = batch["char_counts"].to(device)
                option_char_hashes = batch["option_char_hashes"].to(device)
                option_char_counts = batch["option_char_counts"].to(device)
                target_positions = batch["target_positions"].to(device)
                option_word_ids = batch["option_word_ids"].to(device)
                option_mask = batch["option_mask"].to(device)
                classical_features = batch["classical_features"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                out = reranker(
                    word_ids=word_ids,
                    mask=mask,
                    target_positions=target_positions,
                    option_word_ids=option_word_ids,
                    option_mask=option_mask,
                    classical_features=classical_features,
                    markers=markers,
                    char_hashes=char_hashes,
                    char_counts=char_counts,
                    option_char_hashes=option_char_hashes,
                    option_char_counts=option_char_counts,
                )
                loss = compute_listwise_loss(out["logits"], labels, option_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(labels)
                total_train += len(labels)

            cal_metrics = evaluate_ranking_metrics(reranker, cal_loader, device)
            print(
                f"[{arch_id}] Epoch {epoch}/{args.epochs} — "
                f"train_loss: {train_loss / max(1, total_train):.4f}, "
                f"cal_P: {cal_metrics['precision']:.4f}, "
                f"cal_R: {cal_metrics['recall']:.4f}, "
                f"cal_F0.5: {cal_metrics['f05']:.4f}, "
                f"cal_KEEP_acc: {cal_metrics['keep_accuracy']:.4f}"
            )

            if cal_metrics["f05"] > best_f05:
                best_f05 = cal_metrics["f05"]
                best_metrics = cal_metrics
                best_metrics["best_epoch"] = epoch
                best_state = {k: v.cpu() for k, v in reranker.state_dict().items()}

        # Save finetuned model
        out_pt = os.path.join(args.output_dir, f"attention-finetuned-{arch_id}.pt")
        torch.save({
            "schema": "attention-finetuned-v1",
            "arch": arch_id,
            "config": cfg.__dict__,
            "vocab_hash": vocab_hash,
            "shortlist_config_hash": shortlist_cfg_hash,
            "k": k,
            "seed": args.seed,
            "metrics": best_metrics,
            "state_dict": best_state,
        }, out_pt)

        report_path = os.path.join(args.output_dir, f"attention-finetune-report-{arch_id}.json")
        report = {
            "schema": "attention-finetune-report-v1",
            "arch": arch_id,
            "checkpointPath": out_pt,
            "checkpointHash": sha256_file(out_pt),
            "vocabHash": vocab_hash,
            "shortlistConfigHash": shortlist_cfg_hash,
            "k": k,
            "metrics": best_metrics,
            "seed": args.seed,
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        results[arch_id] = {
            "arch": arch_id,
            "checkpoint": out_pt,
            "checkpointHash": sha256_file(out_pt),
            "report": report_path,
            "metrics": best_metrics,
        }

    # Model selection on calibration
    # Smallest architecture (A < B < C) within 0.5 percentage point of best F0.5
    best_overall_f05 = max(results[a]["metrics"]["f05"] for a in results)
    selected_arch = None
    selection_reason = ""

    for arch_id in ("A", "B", "C"):
        f05 = results[arch_id]["metrics"]["f05"]
        if f05 >= best_overall_f05 - 0.005:
            selected_arch = arch_id
            selection_reason = (
                f"Smallest candidate in A,B,C within 0.5 pt F0.5 of best ({best_overall_f05:.4f}): "
                f"Arch {arch_id} achieved F0.5={f05:.4f}, P={results[arch_id]['metrics']['precision']:.4f}, "
                f"R={results[arch_id]['metrics']['recall']:.4f}, KEEP_acc={results[arch_id]['metrics']['keep_accuracy']:.4f}"
            )
            break

    if selected_arch is None:
        selected_arch = "A"
        selection_reason = "Fallback to smallest architecture A"

    selected_model_info = {
        "schema": "attention-selected-model-v1",
        "createdAt": "2026-08-25T00:00:00.000Z",
        "selectedArch": selected_arch,
        "selectionReason": selection_reason,
        "checkpointPath": results[selected_arch]["checkpoint"],
        "checkpointHash": results[selected_arch]["checkpointHash"],
        "vocabHash": vocab_hash,
        "shortlistConfigHash": shortlist_cfg_hash,
        "k": k,
        "candidates": results,
    }

    sel_path = os.path.join(args.output_dir, "attention-selected-model.json")
    with open(sel_path, "w", encoding="utf-8") as fh:
        json.dump(selected_model_info, fh, indent=2)

    print(f"\n================ Selected Model ================\n{json.dumps(selected_model_info, indent=2)}")


if __name__ == "__main__":
    main()

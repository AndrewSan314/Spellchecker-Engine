# ============================================================
# Task 8 (tiny-attention-spelling-reranker-FIXED plan) — Export parity
# fixture comparing Python quantized forward pass with JavaScript.
# ============================================================
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.attention_model import (
    ARCHITECTURES,
    AttentionModelConfig,
    create_model,
)
from tools.attention_tokenizer import (
    MAX_CONTEXT_TOKENS,
    SPECIAL_IDS,
    VOCAB_SIZE,
    encode_context_units,
    normalize_for_model,
)
from tools.export_attention_model import (
    AttentionModelReader,
    DTYPE_INT8,
    QUANT_PER_CHANNEL,
    QUANT_PER_ROW,
)
from tools.train_attention_reranker import (
    ListwiseReranker,
    extract_classical_feature_tensor,
)


def load_quantized_weights_into_model(
    reranker: ListwiseReranker,
    bin_path: str,
    meta_path: str,
):
    """Load dequantized int8 weights from binary file into PyTorch model for exact parity evaluation."""
    reader = AttentionModelReader(bin_path, meta_path)
    with open(bin_path, "rb") as fh:
        raw_bin = fh.read()

    new_state = {}
    for name, info in reader.tensors.items():
        d_off = info["data_offset"]
        d_len = info["data_length"]
        s_off = info["scale_offset"]
        s_len = info["scale_length"]
        dtype = info["dtype"]
        quant = info["quant_mode"]
        dims = info["dims"]

        if dtype == DTYPE_INT8:
            int8_bytes = raw_bin[d_off:d_off + d_len]
            int8_arr = np.frombuffer(int8_bytes, dtype=np.int8).reshape(dims)
            if quant in (QUANT_PER_ROW, QUANT_PER_CHANNEL):
                scale_bytes = raw_bin[s_off:s_off + s_len]
                scales = np.frombuffer(scale_bytes, dtype=np.float32)
                if int8_arr.ndim == 1:
                    dequant = int8_arr.astype(np.float32) * scales[0]
                else:
                    dequant = int8_arr.astype(np.float32) * scales[:, None]
            else:
                dequant = int8_arr.astype(np.float32)
            tensor_val = torch.from_numpy(dequant.copy())
        else:
            fp32_bytes = raw_bin[d_off:d_off + d_len]
            fp32_arr = np.frombuffer(fp32_bytes, dtype=np.float32).reshape(dims)
            tensor_val = torch.from_numpy(fp32_arr.copy())

        # Map name into reranker model state_dict
        if name.startswith("model."):
            model_key = name
        else:
            model_key = f"model.{name}"
        new_state[model_key] = tensor_val

    reranker.load_state_dict(new_state, strict=False)


def generate_parity_cases(
    reranker: ListwiseReranker,
    vocab: Dict[str, int],
    k: int = 8,
    num_cases: int = 25,
) -> List[dict]:
    cases = []
    np.random.seed(20260825)
    torch.manual_seed(20260825)

    vocab_items = list(vocab.items())
    sample_words = [w for w, _ in vocab_items if _ >= 8][:200]

    for c_idx in range(num_cases):
        seq_len = np.random.randint(4, 33)
        target_pos = np.random.randint(0, min(seq_len, 32))
        num_cands = np.random.randint(1, k + 1)

        # Generate words
        chosen_words = np.random.choice(sample_words, size=seq_len).tolist()
        word_ids = [vocab.get(w, SPECIAL_IDS["UNK"]) for w in chosen_words]
        mask = [1] * len(word_ids)

        # Pad context to 32
        while len(word_ids) < 32:
            word_ids.append(SPECIAL_IDS["PAD"])
            mask.append(0)

        # Candidates
        orig_word = chosen_words[target_pos]
        cand_words = np.random.choice(sample_words, size=num_cands).tolist()

        opt_ids = [vocab.get(orig_word, SPECIAL_IDS["UNK"])] + [
            vocab.get(w, SPECIAL_IDS["UNK"]) for w in cand_words
        ]
        opt_mask = [1] * (1 + num_cands)

        while len(opt_ids) < k + 1:
            opt_ids.append(SPECIAL_IDS["PAD"])
            opt_mask.append(0)

        # Classical features
        classical_features = np.random.randn(k + 1, 15).astype(np.float32)
        classical_features[0, 1] = 1.0  # sameAccentKey
        classical_features[0, 8:] = 0.0

        # Mask padding in features
        for slot in range(1 + num_cands, k + 1):
            classical_features[slot, :] = 0.0

        # Run model
        w_t = torch.tensor([word_ids], dtype=torch.long)
        m_t = torch.tensor([mask], dtype=torch.float32)
        tp_t = torch.tensor([target_pos], dtype=torch.long)
        opt_w_t = torch.tensor([opt_ids], dtype=torch.long)
        opt_m_t = torch.tensor([opt_mask], dtype=torch.float32)
        cf_t = torch.tensor([classical_features], dtype=torch.float32)

        reranker.eval()
        with torch.no_grad():
            out = reranker(
                word_ids=w_t,
                mask=m_t,
                target_positions=tp_t,
                option_word_ids=opt_w_t,
                option_mask=opt_m_t,
                classical_features=cf_t,
            )

        logits = out["logits"][0].cpu().numpy().tolist()
        probs = out["probabilities"][0].cpu().numpy().tolist()

        cases.append({
            "caseId": f"parity-case-{c_idx:03d}",
            "seqLen": seq_len,
            "targetPosition": target_pos,
            "numCandidates": num_cands,
            "wordIds": word_ids,
            "mask": mask,
            "optionWordIds": opt_ids,
            "optionMask": opt_mask,
            "classicalFeatures": classical_features.tolist(),
            "expectedLogits": logits,
            "expectedProbabilities": probs,
            "expectedChoice": int(np.argmax(logits)),
        })

    return cases


def main():
    parser = argparse.ArgumentParser(description="Export parity fixture")
    parser.add_argument("--checkpoint-from", default=".tmp/attention-selected-model.json")
    parser.add_argument("--artifact", default="src/data/attention-reranker.int8.bin")
    parser.add_argument("--meta", default="src/data/attention-reranker.json")
    parser.add_argument("--vocab", default=".tmp/attention-vocab.json")
    parser.add_argument("--output", default="test/fixtures/attention-parity.json")
    args = parser.parse_args()

    with open(args.meta, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    cfg_dict = meta.get("config", {})
    cfg = AttentionModelConfig(
        arch_id=cfg_dict.get("arch_id", "A"),
        blocks=cfg_dict.get("blocks", 1),
        hidden_dim=cfg_dict.get("hidden_dim", 48),
        num_heads=cfg_dict.get("num_heads", 2),
        head_dim=cfg_dict.get("head_dim", 24),
        ffn_dim=cfg_dict.get("ffn_dim", 96),
        vocab_size=cfg_dict.get("vocab_size", VOCAB_SIZE),
    )

    reranker = ListwiseReranker(cfg, num_classical_features=15)
    load_quantized_weights_into_model(reranker, args.artifact, args.meta)

    with open(args.vocab, "r", encoding="utf-8") as fh:
        vocab_data = json.load(fh)
    vocab = vocab_data.get("wordToId", {})

    cases = generate_parity_cases(reranker, vocab, k=meta.get("k", 8), num_cases=25)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({
            "schema": "attention-parity-fixture-v1",
            "arch": cfg.arch_id,
            "binHash": meta.get("binHash", ""),
            "casesCount": len(cases),
            "cases": cases,
        }, fh, indent=2)

    print(json.dumps({
        "ok": True,
        "output": args.output,
        "casesCount": len(cases),
    }, indent=2))


if __name__ == "__main__":
    main()

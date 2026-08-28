# ============================================================
# Task 5 (tiny-attention-spelling-reranker-FIXED plan) — PyTorch
# Pre-LN Tiny Vietnamese Transformer Encoder for Spelling Reranking.
#
# Architectures:
#   A: 1 block, hidden 48, 2 heads (head_dim 24), FFN 96
#   B: 2 blocks, hidden 64, 4 heads (head_dim 16), FFN 128
#   C: 2 blocks, hidden 96, 4 heads (head_dim 24), FFN 192
#
# Token representation:
#   x_i = wordEmbedding[wordId]
#       + mean(charEmbedding[hash(charNgram)])
#       + positionEmbedding[position]
#       + markerEmbedding[marker]
#
# Encoder: Pre-LN everywhere:
#   a = x + Attention(LayerNorm(x), mask)
#   h = a + FFN_GELU(LayerNorm(a))
# ============================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.attention_tokenizer import (
    CHAR_HASH_BUCKETS,
    MAX_CONTEXT_TOKENS,
    SPECIAL_IDS,
    VOCAB_SIZE,
)


@dataclass(frozen=True)
class AttentionModelConfig:
    arch_id: str
    blocks: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    max_tokens: int = MAX_CONTEXT_TOKENS
    vocab_size: int = VOCAB_SIZE
    char_buckets: int = CHAR_HASH_BUCKETS
    num_markers: int = 4
    pre_ln: bool = True
    layer_norm_eps: float = 1e-5


ARCHITECTURES: Dict[str, AttentionModelConfig] = {
    "A": AttentionModelConfig(
        arch_id="A", blocks=1, hidden_dim=48, num_heads=2, head_dim=24, ffn_dim=96
    ),
    "B": AttentionModelConfig(
        arch_id="B", blocks=2, hidden_dim=64, num_heads=4, head_dim=16, ffn_dim=128
    ),
    "C": AttentionModelConfig(
        arch_id="C", blocks=2, hidden_dim=96, num_heads=4, head_dim=24, ffn_dim=192
    ),
}


class PreLNSelfAttention(nn.Module):
    def __init__(self, config: AttentionModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        assert self.num_heads * self.head_dim == self.hidden_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) with 1 for valid, 0 for padding
        """
        batch_size, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq, seq)

        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, -10000.0)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: AttentionModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.attn = PreLNSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, config.hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN: x + Sublayer(LN(x))
        a = x + self.attn(self.ln1(x), mask=mask)
        h = a + self.ffn(self.ln2(a))
        return h


class AttentionEncoder(nn.Module):
    def __init__(self, config: AttentionModelConfig):
        super().__init__()
        self.config = config

        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=SPECIAL_IDS["PAD"]
        )
        self.char_embedding = nn.Embedding(config.char_buckets, config.hidden_dim)
        self.position_embedding = nn.Embedding(
            config.max_tokens, config.hidden_dim
        )
        self.marker_embedding = nn.Embedding(
            config.num_markers, config.hidden_dim
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.blocks)]
        )
        self.final_ln = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def embed_tokens(
        self,
        word_ids: torch.Tensor,
        markers: Optional[torch.Tensor] = None,
        char_hashes: Optional[torch.Tensor] = None,
        char_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute token representations: word + mean(char) + pos + marker.
        word_ids: (batch, seq_len)
        markers: (batch, seq_len)
        char_hashes: (batch, seq_len, max_char_ngrams)
        char_counts: (batch, seq_len)
        """
        batch_size, seq_len = word_ids.shape
        x = self.word_embedding(word_ids)

        # Position embeddings (0..seq_len-1)
        positions = torch.arange(seq_len, device=word_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)

        # Marker embeddings
        if markers is not None:
            x = x + self.marker_embedding(markers)

        # Char n-gram mean embeddings
        if char_hashes is not None and char_counts is not None:
            # char_hashes: (batch, seq_len, M)
            char_embeds = self.char_embedding(char_hashes)  # (batch, seq_len, M, hidden)
            # Sum only real hashes; zero padding must not contribute bucket-0.
            positions = torch.arange(char_hashes.shape[-1], device=char_hashes.device)
            valid = positions.view(1, 1, -1) < char_counts.unsqueeze(-1)
            char_sum = (char_embeds * valid.unsqueeze(-1)).sum(dim=2)
            counts = char_counts.unsqueeze(-1).clamp(min=1)  # (batch, seq_len, 1)
            char_mean = torch.where(
                (char_counts > 0).unsqueeze(-1),
                char_sum / counts,
                torch.zeros_like(char_sum),
            )
            x = x + char_mean

        return x

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        markers: Optional[torch.Tensor] = None,
        char_hashes: Optional[torch.Tensor] = None,
        char_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(word_ids, markers, char_hashes, char_counts)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.final_ln(x)


class AttentionModel(nn.Module):
    """Full Model wrapper with optional MLM pretraining head and Reranker head."""

    def __init__(
        self,
        config: AttentionModelConfig,
        with_mlm_head: bool = False,
        num_classical_features: int = 0,
        classical_proj_dim: int = 16,
    ):
        super().__init__()
        self.config = config
        self.encoder = AttentionEncoder(config)

        self.with_mlm_head = with_mlm_head
        if with_mlm_head:
            self.mlm_head = nn.Linear(config.hidden_dim, config.vocab_size)
        else:
            self.mlm_head = None

        self.num_classical_features = num_classical_features
        if num_classical_features > 0:
            self.classical_proj = nn.Linear(num_classical_features, classical_proj_dim)
            option_input_dim = config.hidden_dim * 3 + classical_proj_dim + 4  # +4 for option type
            self.option_scorer = nn.Sequential(
                nn.Linear(option_input_dim, config.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.hidden_dim, 1),
            )
            self.option_type_embedding = nn.Embedding(4, 4)

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        markers: Optional[torch.Tensor] = None,
        char_hashes: Optional[torch.Tensor] = None,
        char_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden = self.encoder(
            word_ids=word_ids,
            mask=mask,
            markers=markers,
            char_hashes=char_hashes,
            char_counts=char_counts,
        )
        res = {"hidden": hidden}
        if self.mlm_head is not None:
            res["mlm_logits"] = self.mlm_head(hidden)
        return res

    def compute_option_embedding(
        self, word_id: torch.Tensor, char_hashes: Optional[torch.Tensor] = None, char_counts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute single surface option embedding without running transformer."""
        emb = self.encoder.word_embedding(word_id)
        if char_hashes is not None and char_counts is not None:
            char_embeds = self.encoder.char_embedding(char_hashes)
            positions = torch.arange(char_hashes.shape[-1], device=char_hashes.device)
            valid = positions.view(1, -1) < char_counts.unsqueeze(-1)
            char_sum = (char_embeds * valid.unsqueeze(-1)).sum(dim=-2)
            counts = char_counts.unsqueeze(-1).clamp(min=1)
            char_mean = torch.where(
                (char_counts > 0).unsqueeze(-1),
                char_sum / counts,
                torch.zeros_like(char_sum),
            )
            emb = emb + char_mean
        return emb


def create_model(
    arch_id: str,
    vocab_size: int = VOCAB_SIZE,
    with_mlm_head: bool = False,
    num_classical_features: int = 0,
) -> AttentionModel:
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
    return AttentionModel(
        cfg,
        with_mlm_head=with_mlm_head,
        num_classical_features=num_classical_features,
    )

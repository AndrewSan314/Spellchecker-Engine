# ============================================================
# Task 7 (tiny-attention-spelling-reranker-FIXED plan) — Int8 Binary
# Artifact Exporter and Validator.
#
# Binary format v1 (little-endian):
#   Header (32 bytes):
#     magic[8]        = "TDRANK01"
#     format_version  = 1 (uint32)
#     tensor_count    = N (uint32)
#     table_offset    = 32 (uint32)
#     data_offset     = aligned to 16 bytes (uint32)
#     reserved        = 8 bytes
#
#   Tensor Table Entry (64 bytes each):
#     name_id         (uint32)
#     dtype           (uint32: 0=INT8, 1=FLOAT32)
#     quant_mode      (uint32: 0=NONE, 1=PER_ROW, 2=PER_CHANNEL)
#     rank            (uint32)
#     dim0..dim3      (uint32 * 4)
#     data_offset     (uint32)
#     data_length     (uint32)
#     scale_offset    (uint32)
#     scale_length    (uint32)
#     checksum        (uint32)
#     reserved        (uint32 * 3)
#
#   Data Section (16-byte aligned payload)
# ============================================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAGIC = b"TDRANK01"
FORMAT_VERSION = 1

DTYPE_INT8 = 0
DTYPE_FLOAT32 = 1

QUANT_NONE = 0
QUANT_PER_ROW = 1
QUANT_PER_CHANNEL = 2

# Canonical tensor name mapping to stable numeric IDs
TENSOR_NAME_TO_ID = {
    "encoder.word_embedding.weight": 1,
    "encoder.char_embedding.weight": 2,
    "encoder.position_embedding.weight": 3,
    "encoder.marker_embedding.weight": 4,
    "encoder.blocks.0.ln1.weight": 10,
    "encoder.blocks.0.ln1.bias": 11,
    "encoder.blocks.0.attn.q_proj.weight": 12,
    "encoder.blocks.0.attn.q_proj.bias": 13,
    "encoder.blocks.0.attn.k_proj.weight": 14,
    "encoder.blocks.0.attn.k_proj.bias": 15,
    "encoder.blocks.0.attn.v_proj.weight": 16,
    "encoder.blocks.0.attn.v_proj.bias": 17,
    "encoder.blocks.0.attn.out_proj.weight": 18,
    "encoder.blocks.0.attn.out_proj.bias": 19,
    "encoder.blocks.0.ln2.weight": 20,
    "encoder.blocks.0.ln2.bias": 21,
    "encoder.blocks.0.ffn.0.weight": 22,
    "encoder.blocks.0.ffn.0.bias": 23,
    "encoder.blocks.0.ffn.2.weight": 24,
    "encoder.blocks.0.ffn.2.bias": 25,
    "encoder.blocks.1.ln1.weight": 30,
    "encoder.blocks.1.ln1.bias": 31,
    "encoder.blocks.1.attn.q_proj.weight": 32,
    "encoder.blocks.1.attn.q_proj.bias": 33,
    "encoder.blocks.1.attn.k_proj.weight": 34,
    "encoder.blocks.1.attn.k_proj.bias": 35,
    "encoder.blocks.1.attn.v_proj.weight": 36,
    "encoder.blocks.1.attn.v_proj.bias": 37,
    "encoder.blocks.1.attn.out_proj.weight": 38,
    "encoder.blocks.1.attn.out_proj.bias": 39,
    "encoder.blocks.1.ln2.weight": 40,
    "encoder.blocks.1.ln2.bias": 41,
    "encoder.blocks.1.ffn.0.weight": 42,
    "encoder.blocks.1.ffn.0.bias": 43,
    "encoder.blocks.1.ffn.2.weight": 44,
    "encoder.blocks.1.ffn.2.bias": 45,
    "encoder.final_ln.weight": 50,
    "encoder.final_ln.bias": 51,
    "model.classical_proj.weight": 60,
    "model.classical_proj.bias": 61,
    "model.option_type_embedding.weight": 62,
    "model.option_scorer.0.weight": 70,
    "model.option_scorer.0.bias": 71,
    "model.option_scorer.2.weight": 72,
    "model.option_scorer.2.bias": 73,
}
ID_TO_TENSOR_NAME = {v: k for k, v in TENSOR_NAME_TO_ID.items()}


def quantize_row_embedding(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8 quantization: scale = max(abs(row)) / 127."""
    matrix = matrix.astype(np.float32)
    max_vals = np.max(np.abs(matrix), axis=1)
    scales = np.where(max_vals > 0, max_vals / 127.0, 1.0).astype(np.float32)
    int8_mat = np.clip(np.round(matrix / scales[:, None]), -128, 127).astype(np.int8)
    return int8_mat, scales


def quantize_linear_weight(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel int8 quantization: scale = max(abs(channel)) / 127."""
    matrix = matrix.astype(np.float32)
    if matrix.ndim == 1:
        max_val = float(np.max(np.abs(matrix)))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        scales = np.array([scale], dtype=np.float32)
        int8_mat = np.clip(np.round(matrix / scale), -128, 127).astype(np.int8)
        return int8_mat, scales

    max_vals = np.max(np.abs(matrix), axis=1)
    scales = np.where(max_vals > 0, max_vals / 127.0, 1.0).astype(np.float32)
    int8_mat = np.clip(np.round(matrix / scales[:, None]), -128, 127).astype(np.int8)
    return int8_mat, scales


def align16(n: int) -> int:
    return (n + 15) & ~15


def export_quantized_model(
    checkpoint_path: str,
    output_bin: str,
    output_meta: str,
    vocab_path: Optional[str] = None,
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(output_bin)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_meta)), exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
    cfg_dict = ckpt.get("config", {})

    tensors_to_pack = []
    tensor_records = []

    # Sort tensor names deterministically
    sorted_names = sorted(state_dict.keys())

    for name in sorted_names:
        tensor = state_dict[name].detach().cpu().numpy()
        # Clean prefix if needed
        clean_name = name
        if clean_name.startswith("model.") and not clean_name.startswith("model.classical_proj") and not clean_name.startswith("model.option_"):
            clean_name = clean_name[6:]  # strip 'model.' from encoder tensors if present

        name_id = TENSOR_NAME_TO_ID.get(clean_name, TENSOR_NAME_TO_ID.get(name, 0))
        if name_id == 0:
            continue

        shape = list(tensor.shape)
        dims = shape + [0] * (4 - len(shape))
        rank = len(shape)

        # Decide quantization mode
        is_embedding = "embedding" in clean_name and "weight" in clean_name
        is_linear_w = "weight" in clean_name and not is_embedding and ("proj" in clean_name or "ffn" in clean_name or "scorer" in clean_name)

        if is_embedding:
            int8_data, scales = quantize_row_embedding(tensor)
            tensors_to_pack.append({
                "name": clean_name,
                "name_id": name_id,
                "dtype": DTYPE_INT8,
                "quant_mode": QUANT_PER_ROW,
                "rank": rank,
                "dims": dims,
                "data_bytes": int8_data.tobytes(),
                "scale_bytes": scales.tobytes(),
            })
        elif is_linear_w:
            int8_data, scales = quantize_linear_weight(tensor)
            tensors_to_pack.append({
                "name": clean_name,
                "name_id": name_id,
                "dtype": DTYPE_INT8,
                "quant_mode": QUANT_PER_CHANNEL,
                "rank": rank,
                "dims": dims,
                "data_bytes": int8_data.tobytes(),
                "scale_bytes": scales.tobytes(),
            })
        else:
            # Float32 parameters (bias, LayerNorm)
            fp32_data = tensor.astype(np.float32)
            tensors_to_pack.append({
                "name": clean_name,
                "name_id": name_id,
                "dtype": DTYPE_FLOAT32,
                "quant_mode": QUANT_NONE,
                "rank": rank,
                "dims": dims,
                "data_bytes": fp32_data.tobytes(),
                "scale_bytes": b"",
            })

    # Layout binary file
    tensor_count = len(tensors_to_pack)
    table_offset = 32
    table_size = tensor_count * 64
    raw_data_offset = table_offset + table_size
    data_offset = align16(raw_data_offset)

    # Pack data payloads
    current_offset = data_offset
    table_bytes = bytearray()
    data_bytes = bytearray(data_offset - raw_data_offset)  # alignment padding

    meta_tensors = {}

    for t in tensors_to_pack:
        d_len = len(t["data_bytes"])
        d_off = current_offset
        current_offset += d_len

        s_len = len(t["scale_bytes"])
        if s_len > 0:
            s_off = current_offset
            current_offset += s_len
        else:
            s_off = 0

        # Align to 16 bytes for next tensor
        pad = align16(current_offset) - current_offset
        current_offset += pad

        checksum = zlib.crc32(t["data_bytes"])

        # Table entry (64 bytes: 16 uint32s)
        entry = struct.pack(
            "<16I",
            t["name_id"],
            t["dtype"],
            t["quant_mode"],
            t["rank"],
            t["dims"][0], t["dims"][1], t["dims"][2], t["dims"][3],
            d_off,
            d_len,
            s_off,
            s_len,
            checksum,
            0, 0, 0,  # reserved
        )
        assert len(entry) == 64, f"Table entry must be 64 bytes, got {len(entry)}"
        table_bytes.extend(entry)

        data_bytes.extend(t["data_bytes"])
        if s_len > 0:
            data_bytes.extend(t["scale_bytes"])
        if pad > 0:
            data_bytes.extend(b"\x00" * pad)

        meta_tensors[t["name"]] = {
            "nameId": t["name_id"],
            "dtype": "int8" if t["dtype"] == DTYPE_INT8 else "float32",
            "quantMode": "per_row" if t["quant_mode"] == QUANT_PER_ROW else ("per_channel" if t["quant_mode"] == QUANT_PER_CHANNEL else "none"),
            "dims": t["dims"][:t["rank"]],
            "dataOffset": d_off,
            "dataLength": d_len,
            "scaleOffset": s_off,
            "scaleLength": s_len,
            "checksum": checksum,
        }

    # Header (32 bytes)
    header = struct.pack(
        "<8sIIII8s",
        MAGIC,
        FORMAT_VERSION,
        tensor_count,
        table_offset,
        data_offset,
        b"\x00" * 8,
    )
    assert len(header) == 32

    # Write BIN file
    with open(output_bin, "wb") as fh:
        fh.write(header)
        fh.write(table_bytes)
        fh.write(data_bytes)

    # Compute binary hash
    h = hashlib.sha256()
    with open(output_bin, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    bin_hash = h.hexdigest()

    # Write JSON metadata
    metadata = {
        "schema": "attention-reranker-artifact-v1",
        "formatVersion": FORMAT_VERSION,
        "magic": "TDRANK01",
        "arch": cfg_dict.get("arch_id", "A"),
        "config": cfg_dict,
        "k": ckpt.get("k", 8),
        "vocabHash": ckpt.get("vocab_hash", ""),
        "shortlistConfigHash": ckpt.get("shortlist_config_hash", ""),
        "binHash": bin_hash,
        "binSize": os.path.getsize(output_bin),
        "tensorCount": tensor_count,
        "tensors": meta_tensors,
        "nameToId": TENSOR_NAME_TO_ID,
    }
    with open(output_meta, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


class AttentionModelReader:
    def __init__(self, bin_path: str, meta_path: str):
        self.bin_path = bin_path
        self.meta_path = meta_path

        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)

        with open(bin_path, "rb") as fh:
            magic = fh.read(8)
            if magic != MAGIC:
                raise ValueError(f"Invalid magic: {magic}")
            (ver, tensor_count, table_offset, data_offset) = struct.unpack("<IIII", fh.read(16))
            if ver != FORMAT_VERSION:
                raise ValueError(f"Unsupported format version: {ver}")

            self.tensors = {}
            fh.seek(table_offset)
            for _ in range(tensor_count):
                entry_data = fh.read(64)
                (name_id, dtype, quant_mode, rank, d0, d1, d2, d3, d_off, d_len, s_off, s_len, checksum, *_) = struct.unpack(
                    "<16I", entry_data
                )
                name = ID_TO_TENSOR_NAME.get(name_id, f"tensor_{name_id}")
                dims = [d0, d1, d2, d3][:rank]
                self.tensors[name] = {
                    "name_id": name_id,
                    "dtype": dtype,
                    "quant_mode": quant_mode,
                    "dims": dims,
                    "data_offset": d_off,
                    "data_length": d_len,
                    "scale_offset": s_off,
                    "scale_length": s_len,
                }


def main():
    parser = argparse.ArgumentParser(description="Export quantized int8 model")
    parser.add_argument("--checkpoint-from", default=".tmp/attention-selected-model.json")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--vocab", default=".tmp/attention-vocab.json")
    parser.add_argument("--output-bin", default="src/data/attention-reranker.int8.bin")
    parser.add_argument("--output-meta", default="src/data/attention-reranker.json")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not ckpt_path:
        with open(args.checkpoint_from, "r", encoding="utf-8") as fh:
            sel_info = json.load(fh)
        ckpt_path = sel_info.get("checkpointPath", "")

    meta = export_quantized_model(
        checkpoint_path=ckpt_path,
        output_bin=args.output_bin,
        output_meta=args.output_meta,
        vocab_path=args.vocab,
    )
    print(json.dumps({
        "ok": True,
        "binPath": args.output_bin,
        "metaPath": args.output_meta,
        "binSize": meta["binSize"],
        "binHash": meta["binHash"],
        "tensorCount": meta["tensorCount"],
    }, indent=2))


if __name__ == "__main__":
    main()

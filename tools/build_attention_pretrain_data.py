# ============================================================
# Task 4 (tiny-attention-spelling-reranker-FIXED plan) — Leakage-safe
# vocabulary and self-supervised pretraining dataset builder.
#
# Constraints honored:
#   - ONLY corpus-train.txt and clean-train.txt allowed as text sources;
#     forbidden markers (dev, test, heldout, calibration, evaluation, viwiki)
#     are rejected BEFORE any reads;
#   - any sentence hash in Task 3's deny list is skipped;
#   - deduplicated by SHA-256 of normalized text;
#   - deterministic ordering with explicit seed;
#   - <=8192 total word IDs including 8 fixed special IDs;
#   - sequences <=32 tokens with 0-padding and padding mask;
#   - emits deterministic sharded NPZ files + vocabulary JSON + manifest.
# ============================================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

VOCAB_SIZE_LIMIT = 8192
MAX_SEQ_TOKENS = 32
SPECIAL_IDS = {
    "PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3,
    "TARGET": 4, "PROTECTED": 5, "PUNCT": 6, "MASK": 7,
}
FIRST_LEARNED_ID = 8

FORBIDDEN_MARKERS = ("dev", "test", "heldout", "calibration", "evaluation", "viwiki")


def assert_allowed_sources(paths: Sequence[str]) -> List[str]:
    out = []
    for raw in paths or []:
        p = str(raw).lower().replace("\\", "/")
        for marker in FORBIDDEN_MARKERS:
            if marker in p:
                raise ValueError(
                    f"forbidden training source '{raw}' (matches '{marker}'): "
                    "held-out data must never reach pretraining"
                )
        ok = (
            p.endswith("src/data/corpus-train.txt")
            or p.endswith("dataset_artifacts/clean-source/clean-train.txt")
            or p.endswith("/corpus-train.txt")
            or p.endswith("/clean-train.txt")
            or p == "corpus-train.txt"
            or p == "clean-train.txt"
        )
        if not ok:
            raise ValueError(
                f"training source '{raw}' is not an allowed source "
                "(corpus-train.txt | clean-train.txt)"
            )
        out.append(str(raw))
    return out


def normalize_for_model(text: str) -> str:
    """NFC + Unicode lowercase; accents preserved."""
    return unicodedata.normalize("NFC", str(text or "")).lower()


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.\w+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+84|0)\d{9,10}\b")


def placeholderize(text: str) -> str:
    """Replace URLs, emails, phone-like strings with stable placeholders."""
    s = str(text or "")
    s = _URL_RE.sub("<url>", s)
    s = _EMAIL_RE.sub("<email>", s)
    s = _PHONE_RE.sub("<phone>", s)
    return s


def load_deny_list(manifest_path: str) -> Set[str]:
    """Load calibration and internal-test group hashes to exclude."""
    if not os.path.exists(manifest_path):
        raise ValueError(f"deny list manifest not found: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        denied: Set[str] = set()
        deny_block = data.get("denyList", {})
        for item in deny_block.get("calibration", []):
            denied.add(str(item))
        for item in deny_block.get("internalTest", []):
            denied.add(str(item))
        return denied
    except Exception as e:
        raise ValueError(f"error loading deny list: {e}") from e


_WORD_TOKEN_RE = re.compile(r"<[^>]+>|\{\{[^}]+\}\}|[\w]+", re.UNICODE)


def tokenize_words(text: str) -> List[str]:
    norm = normalize_for_model(text)
    return _WORD_TOKEN_RE.findall(norm)


def build_vocab(sentences: Sequence[str], limit: int = VOCAB_SIZE_LIMIT) -> dict:
    counts: Dict[str, int] = {}
    for s in sentences:
        for w in tokenize_words(s):
            counts[w] = counts.get(w, 0) + 1

    # Sort descending by frequency, then ascending by Unicode order
    sorted_words = sorted(counts.keys(), key=lambda w: (-counts[w], w))

    max_learned = limit - len(SPECIAL_IDS)
    learned_words = sorted_words[:max_learned]

    word_to_id: Dict[str, int] = {}
    id_to_word: Dict[int, str] = {v: k for k, v in SPECIAL_IDS.items()}

    for i, w in enumerate(learned_words):
        wid = FIRST_LEARNED_ID + i
        word_to_id[w] = wid
        id_to_word[wid] = w

    return {
        "schema": "attention-vocab-v1",
        "size": len(word_to_id) + len(SPECIAL_IDS),
        "specialIds": dict(SPECIAL_IDS),
        "wordToId": word_to_id,
        "idToWord": {str(k): v for k, v in id_to_word.items()},
    }


def encode_sentences(
    sentences: Sequence[str], word_to_id: Dict[str, int]
) -> List[Tuple[List[int], List[int]]]:
    rows: List[Tuple[List[int], List[int]]] = []
    for s in sentences:
        tokens = tokenize_words(s)
        n = min(len(tokens), MAX_SEQ_TOKENS)
        ids = [word_to_id.get(tok, SPECIAL_IDS["UNK"]) for tok in tokens[:n]]
        if n < MAX_SEQ_TOKENS:
            mask = [1] * n + [0] * (MAX_SEQ_TOKENS - n)
            ids = ids + [SPECIAL_IDS["PAD"]] * (MAX_SEQ_TOKENS - n)
        else:
            mask = [1] * MAX_SEQ_TOKENS
            ids = ids[:MAX_SEQ_TOKENS]
        rows.append((ids, mask))
    return rows


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_artifacts(
    sources: Sequence[str],
    deny: Set[str],
    out_dir: str,
    seed: int = 20260825,
    shard_size: int = 10000,
    max_rows: int = 0,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    assert_allowed_sources(sources)

    counts = {
        "linesRead": 0,
        "accepted": 0,
        "deduplicated": 0,
        "denied": 0,
    }

    seen_hashes: Set[str] = set()
    accepted_sentences: List[str] = []
    sentence_hashes: List[str] = []

    for src in sources:
        with open(src, "r", encoding="utf-8") as fh:
            for line in fh:
                counts["linesRead"] += 1
                raw = line.strip()
                if not raw:
                    continue
                proc = placeholderize(raw)
                norm = normalize_for_model(proc)
                shash = hashlib.sha256(norm.encode("utf-8")).hexdigest()

                if shash in deny:
                    counts["denied"] += 1
                    continue
                if shash in seen_hashes:
                    counts["deduplicated"] += 1
                    continue

                seen_hashes.add(shash)
                counts["accepted"] += 1
                accepted_sentences.append(proc)
                sentence_hashes.append(shash)

                if max_rows and len(accepted_sentences) >= max_rows:
                    break
        if max_rows and len(accepted_sentences) >= max_rows:
            break

    # Build vocab on accepted sentences
    vocab = build_vocab(accepted_sentences, limit=VOCAB_SIZE_LIMIT)
    vocab_path = os.path.join(out_dir, "attention-vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2, ensure_ascii=False)
    vocab_hash = sha256_file(vocab_path)

    # Encode sentences
    encoded = encode_sentences(accepted_sentences, vocab["wordToId"])

    # Deterministic order with seed
    rng = np.random.default_rng(seed)
    indices = np.arange(len(encoded))
    rng.shuffle(indices)

    all_ids = np.array([encoded[i][0] for i in indices], dtype=np.int32)
    all_mask = np.array([encoded[i][1] for i in indices], dtype=np.uint8)
    all_hashes = np.array([sentence_hashes[i] for i in indices], dtype=object)

    # Write shards
    total_rows = len(encoded)
    shards_meta = []
    shard_count = (total_rows + shard_size - 1) // shard_size if total_rows > 0 else 0

    for shard_idx in range(shard_count):
        start = shard_idx * shard_size
        end = min(start + shard_size, total_rows)
        shard_filename = f"attention-pretrain-shard-{shard_idx:03d}.npz"
        shard_path = os.path.join(out_dir, shard_filename)

        np.savez_compressed(
            shard_path,
            ids=all_ids[start:end],
            mask=all_mask[start:end],
            hashes=all_hashes[start:end],
        )

        shards_meta.append({
            "file": shard_filename,
            "path": shard_path,
            "rows": end - start,
            "sha256": sha256_file(shard_path),
        })

    source_hashes = {src: sha256_file(src) for src in sources}
    manifest_path = os.path.join(out_dir, "attention-pretrain-manifest.json")
    manifest = {
        "schema": "attention-pretrain-manifest-v1",
        "createdAt": "2026-08-25T00:00:00.000Z",
        "sources": list(sources),
        "sourceHashes": source_hashes,
        "vocabPath": vocab_path,
        "vocabHash": vocab_hash,
        "vocabSize": vocab["size"],
        "totalRows": total_rows,
        "shardSize": shard_size,
        "shards": shards_meta,
        "counts": counts,
        "seed": seed,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return {
        "vocabPath": vocab_path,
        "manifestPath": manifest_path,
        "shards": shards_meta,
        "counts": counts,
        "vocabSize": vocab["size"],
        "totalRows": total_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Build attention pretrain data and vocabulary")
    parser.add_argument("--corpus", default="src/data/corpus-train.txt")
    parser.add_argument("--clean", default="dataset_artifacts/clean-source/clean-train.txt")
    parser.add_argument("--deny-list", default=".tmp/attention-ranking-split-manifest.json")
    parser.add_argument("--out-dir", default=".tmp")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    sources = [args.corpus, args.clean]
    assert_allowed_sources(sources)
    deny = load_deny_list(args.deny_list) if args.deny_list and os.path.exists(args.deny_list) else set()

    res = write_artifacts(
        sources=sources,
        deny=deny,
        out_dir=args.out_dir,
        seed=args.seed,
        shard_size=args.shard_size,
        max_rows=args.max_rows,
    )
    print(json.dumps({
        "ok": True,
        "vocabSize": res["vocabSize"],
        "totalRows": res["totalRows"],
        "shards": len(res["shards"]),
        "counts": res["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()

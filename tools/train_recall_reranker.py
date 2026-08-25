#!/usr/bin/env python3
"""Train the recall-pairwise-v1 reranker (recall-improvement plan Task 6).

Consumes pre-extracted feature rows (JSONL; header line + one row per
candidate-vs-original pair) produced by production JS helpers, and emits a
weights-only model JSON plus a SHA-256 manifest. Standard library only.

Determinism contract:
  - fixed row ordering (stable sort by id/source/label) — no sampling;
  - full-batch gradient descent, zero init, fixed iteration count,
    fixed learning-rate schedule, fixed L2;
  - no RNG anywhere (the seed is recorded for provenance only);
so two runs over identical inputs produce byte-identical outputs.

Leakage guard: any resolved input path containing a forbidden marker
(vsec-dev / vsec-test / viwiki / external-test / benchmark) fails closed
with a non-zero exit BEFORE any data is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

SCHEMA = "recall-reranker-v1"
CONTRACT = "recall-pairwise-v1"
ROWS_SCHEMA = "recall-training-rows-v1"
FEATURE_ORDER = [
    "bias",
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
]
# bias is a constant 1 — never standardized, never regularized.
NON_BIAS_FEATURES = FEATURE_ORDER[1:]

FORBIDDEN_MARKS = ("vsec-dev", "vsec-test", "viwiki", "external-test",
                   "benchmark")
ALLOWED_SOURCES = {"vsec-train", "clean-train"}

DEFAULT_SEED = 20260824
DEFAULT_ITERATIONS = 400
DEFAULT_LR = 0.35
DEFAULT_LR_DECAY = 0.005
DEFAULT_L2 = 1e-4


def path_is_allowed(path: str) -> bool:
    """False when ANY forbidden marker appears in the resolved path."""
    lowered = os.path.abspath(path).replace("\\", "/").lower()
    return not any(mark in lowered for mark in FORBIDDEN_MARKS)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_std(xs: list[float]) -> tuple[float, float]:
    """Arithmetic mean and POPULATION std; constant features get scale 1."""
    n = len(xs)
    mean = sum(xs) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in xs) / n if n else 0.0
    std = math.sqrt(var)
    return mean, (std if std > 0 else 1.0)


def standardize(x: float, mean: float, std: float) -> float:
    return (x - mean) / std


def sigmoid(z: float) -> float:
    """Calibrated-range logistic — clamped so output is always finite."""
    zc = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-zc))


class RowsRejected(Exception):
    """A row or header violates the input contract — fail closed."""


def load_rows(paths: list[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    header = None
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh):
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("header"):
                    if obj.get("schema") != ROWS_SCHEMA:
                        raise RowsRejected(
                            f"{p}:{line_no}: unexpected rows schema "
                            f"{obj.get('schema')!r}")
                    for s in obj.get("sources", []):
                        if not path_is_allowed(s):
                            raise RowsRejected(
                                f"{p}:{line_no}: forbidden source {s!r} "
                                "in header")
                    header = obj
                    continue
                feats = obj.get("features")
                if not isinstance(feats, dict):
                    raise RowsRejected(f"{p}:{line_no}: row without features")
                missing = [f for f in NON_BIAS_FEATURES if f not in feats]
                if missing:
                    raise RowsRejected(
                        f"{p}:{line_no}: missing feature(s) {missing}")
                if obj.get("source") not in ALLOWED_SOURCES:
                    raise RowsRejected(
                        f"{p}:{line_no}: disallowed source "
                        f"{obj.get('source')!r}")
                if obj.get("label") not in (0, 1):
                    raise RowsRejected(f"{p}:{line_no}: label must be 0|1")
                rows.append(obj)
    return rows, (header or {})


def train(rows: list[dict], iterations: int, lr0: float, lr_decay: float,
          l2: float, seed: int) -> dict:
    # Fixed input ordering — stable sort, never sampled or shuffled.
    ordered = sorted(rows, key=lambda r: (r["label"], r.get("id", ""),
                                          r.get("source", "")))
    stats = {}
    for name in NON_BIAS_FEATURES:
        stats[name] = mean_std([float(r["features"][name]) for r in ordered])
    X = []
    y = []
    for r in ordered:
        vec = [1.0]  # bias
        for name in NON_BIAS_FEATURES:
            mean, std = stats[name]
            vec.append(standardize(float(r["features"][name]), mean, std))
        X.append(vec)
        y.append(float(r["label"]))
    n = len(y)
    pos = sum(y)
    neg = n - pos
    class_w_pos = n / (2.0 * pos) if pos else 1.0
    class_w_neg = n / (2.0 * neg) if neg else 1.0

    dim = len(FEATURE_ORDER)
    w = [0.0] * dim
    for t in range(iterations):
        lr_t = lr0 / (1.0 + lr_decay * t)
        grad = [0.0] * dim
        total_w = 0.0
        for i in range(n):
            xi = X[i]
            z = 0.0
            for j in range(dim):
                z += w[j] * xi[j]
            p = sigmoid(z)
            cw = class_w_pos if y[i] == 1.0 else class_w_neg
            err = cw * (p - y[i])
            total_w += cw
            for j in range(dim):
                grad[j] += err * xi[j]
        for j in range(1, dim):  # L2 never touches the bias term
            grad[j] = grad[j] / total_w + l2 * w[j]
        grad[0] = grad[0] / total_w
        for j in range(dim):
            w[j] -= lr_t * grad[j]

    weights = {name: w[j] for j, name in enumerate(FEATURE_ORDER)}
    return {
        "schema": SCHEMA,
        "featureContract": CONTRACT,
        "featureOrder": list(FEATURE_ORDER),
        "weights": weights,
        "standardization": {
            name: {"mean": stats[name][0], "std": stats[name][1]}
            for name in NON_BIAS_FEATURES
        },
        "classWeights": {"positive": class_w_pos, "negative": class_w_neg},
        "training": {
            "seed": seed,
            "iterations": iterations,
            "learningRate": lr0,
            "learningRateDecay": lr_decay,
            "l2": l2,
            "rowsPositive": int(pos),
            "rowsNegative": int(neg),
            "ordering": "(label,id,source) stable sort; full-batch GD",
        },
    }


def atomic_write(path: str, payload: bytes) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", action="append", default=[],
        help="feature-rows JSONL (repeatable)")
    parser.add_argument("--fixture", default=None,
        help="directory containing rows.jsonl (test mode)")
    parser.add_argument("--out", required=True, help="model JSON output path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr-decay", type=float, default=DEFAULT_LR_DECAY)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    args = parser.parse_args(argv)

    inputs = list(args.rows)
    if args.fixture:
        inputs.append(os.path.join(args.fixture, "rows.jsonl"))
    if len(inputs) != 1:
        print("error: exactly one input required (--rows FILE or --fixture DIR)",
              file=sys.stderr)
        return 2

    # Fail closed BEFORE reading any data.
    for p in inputs:
        if not path_is_allowed(p):
            print(f"error: REFUSED forbidden input path {p!r} — held-out "
                  "data must never reach the trainer", file=sys.stderr)
            return 3
    if not path_is_allowed(args.out):
        print(f"error: REFUSED forbidden output path {args.out!r}",
              file=sys.stderr)
        return 3

    try:
        rows, _header = load_rows(inputs)
    except OSError as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 4
    except (json.JSONDecodeError, RowsRejected) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if not rows:
        print("error: no training rows found", file=sys.stderr)
        return 4

    model = train(rows, args.iterations, args.lr, args.lr_decay, args.l2,
                  args.seed)
    body = (json.dumps(model, indent=2, sort_keys=False) + "\n").encode("utf-8")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    atomic_write(args.out, body)

    manifest = {
        "artifact": os.path.basename(args.out),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "command": " ".join(["train_recall_reranker.py"] + (argv if argv is
            not None else sys.argv[1:])),
        "python": sys.version.split()[0],
        "inputs": [
            {"path": os.path.relpath(p).replace("\\", "/"),
             "sha256": sha256_file(p)}
            for p in inputs
        ],
        "counts": {
            "rows": len(rows),
            "positive": int(sum(r["label"] for r in rows)),
            "negative": int(sum(1 - r["label"] for r in rows)),
        },
    }
    manifest_path = args.out + ".manifest.json"
    atomic_write(manifest_path,
                 (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))

    print(json.dumps({
        "out": args.out,
        "sha256": manifest["sha256"],
        "counts": manifest["counts"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())

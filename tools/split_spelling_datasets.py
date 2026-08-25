#!/usr/bin/env python3
"""Deterministically split spelling data without cross-split leakage.

The VSEC parquet is treated as immutable input.  Split outputs are JSONL so the
Node benchmark adapter can consume them without adding a parquet dependency.
Rows are assigned as groups, not independently: duplicate corrected sentences,
duplicate noisy sentences, and rows sharing an annotation-derived template stay
in the same split.

Examples (PowerShell):

    python tools/split_spelling_datasets.py
    python tools/split_spelling_datasets.py --audit
    python tools/split_spelling_datasets.py --clean-source src/data/corpus_banking_fintech.txt \
        --clean-source src/data/corpus_retail_ecommerce.txt \
        --clean-source src/data/corpus_telco_utilities_services.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VSEC = ROOT.parent / "vsec-vietnamese-spell-correction" / "data" / "train-00000-of-00001.parquet"
DEFAULT_OUT = ROOT / "dataset_artifacts" / "vsec"
DEFAULT_CLEAN_OUT = ROOT / "dataset_artifacts" / "clean-source"
DEFAULT_SEED = 20260824
DEFAULT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SPLIT_ORDER = ("train", "dev", "test")

# Keep punctuation out of the exact key while preserving all Vietnamese letters.
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+", re.UNICODE)
VARIABLE_RE = re.compile(r"(?:https?://\S+|\b\d+(?:[.,/]\d+)*\b)", re.UNICODE)
UPPER_ID_RE = re.compile(r"\b[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ0-9_-]{1,}\b", re.UNICODE)


def normalize_text(value: Any) -> str:
    """NFC/casefold/whitespace normalization used by leakage audits."""

    if value is None:
        return ""
    return SPACE_RE.sub(" ", unicodedata.normalize("NFC", str(value)).casefold()).strip()


def tokens(value: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


def variable_template(value: Any) -> str:
    """Mask obvious IDs/numbers while retaining sentence wording."""

    # Mask uppercase IDs before casefold; after casefold [A-Z] cannot distinguish product/code IDs.
    text = unicodedata.normalize("NFC", str(value or ""))
    text = UPPER_ID_RE.sub("<var>", text)
    text = normalize_text(text)
    text = VARIABLE_RE.sub("<var>", text)
    return " ".join(tokens(text))


def _annotation_template(row: Mapping[str, Any]) -> str:
    """Build a template from syllable positions, masking every error token.

    VSEC's ``error_positions`` are syllable positions, not character offsets.
    ``correction_pairs`` is intentionally *not* assumed to have one item per
    error: malformed/ambiguous rows are represented by the union of all
    annotation, error-position, and pair positions.
    """

    annotations = row.get("syllable_annotations") or []
    by_position: dict[int, Mapping[str, Any]] = {}
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        try:
            position = int(annotation.get("position"))
        except (TypeError, ValueError):
            continue
        by_position[position] = annotation

    error_positions: set[int] = set()
    for position in row.get("error_positions") or []:
        try:
            error_positions.add(int(position))
        except (TypeError, ValueError):
            pass
    for annotation in annotations:
        if isinstance(annotation, Mapping) and annotation.get("is_correct") is False:
            try:
                error_positions.add(int(annotation.get("position")))
            except (TypeError, ValueError):
                pass
    for pair in row.get("correction_pairs") or []:
        if isinstance(pair, Mapping):
            try:
                error_positions.add(int(pair.get("position")))
            except (TypeError, ValueError):
                pass

    if by_position:
        pieces: list[str] = []
        for position in sorted(by_position):
            annotation = by_position[position]
            pieces.append("<err>" if position in error_positions else " ".join(tokens(annotation.get("syllable", ""))))
        return " ".join(piece for piece in pieces if piece)
    return variable_template(row.get("corrected_text") or row.get("text"))


def row_group_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic leakage keys for one VSEC or clean-source row."""

    noisy = normalize_text(row.get("text"))
    clean = normalize_text(row.get("corrected_text"))
    keys = {
        f"text:{noisy}",
        f"corrected:{clean}",
        f"template:{_annotation_template(row)}",
        f"var-template:{variable_template(row.get('corrected_text') or row.get('text'))}",
    }
    return tuple(sorted(key for key in keys if key.rsplit(":", 1)[-1]))


def clean_group_key(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    return (
        f"text:{normalized}",
        f"var-template:{variable_template(normalized)}",
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def grouped_indices(rows: Sequence[Mapping[str, Any]], key_fn=row_group_keys) -> list[list[int]]:
    """Connect rows sharing any leakage key into deterministic components."""

    uf = UnionFind(len(rows))
    first_for_key: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in key_fn(row):
            previous = first_for_key.get(key)
            if previous is None:
                first_for_key[key] = index
            else:
                uf.union(index, previous)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[uf.find(index)].append(index)
    return sorted((sorted(indices) for indices in groups.values()), key=lambda group: group[0])


def _digest(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def assign_groups(
    groups: Sequence[Sequence[int]],
    row_count: int,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
) -> dict[str, str]:
    """Assign whole groups to train/dev/test near the requested ratios."""

    if set(ratios) != set(SPLIT_ORDER) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must contain train/dev/test and sum to 1")

    keyed_groups = []
    for group in groups:
        group_key = ",".join(str(index) for index in group)
        keyed_groups.append((_digest(seed, group_key), list(group)))
    keyed_groups.sort(key=lambda item: item[0])

    targets = {split: row_count * float(ratios[split]) for split in SPLIT_ORDER}
    counts = {split: 0 for split in SPLIT_ORDER}
    assignment: dict[str, str] = {}
    # Largest deficit first keeps row totals close while preserving whole groups.
    for _, group in keyed_groups:
        split = max(
            SPLIT_ORDER,
            key=lambda candidate: (targets[candidate] - counts[candidate], -SPLIT_ORDER.index(candidate)),
        )
        for index in group:
            assignment[str(index)] = split
        counts[split] += len(group)
    return assignment


def read_vsec(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit("VSEC splitting needs pyarrow (already used by tools/build_lm.py)") from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_rows(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    prefix: str = "vsec",
) -> dict[str, Any]:
    groups = grouped_indices(rows)
    assignment = assign_groups(groups, len(rows), seed, ratios)
    split_rows_map: dict[str, list[Mapping[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    index_rows: list[dict[str, Any]] = []
    group_ids = {str(index): f"g{group_number:05d}" for group_number, group in enumerate(groups) for index in group}
    for index, row in enumerate(rows):
        split = assignment[str(index)]
        split_rows_map[split].append(row)
        index_rows.append({
            "row_index": index,
            "split": split,
            "group_id": group_ids[str(index)],
            "text_fingerprint": hashlib.sha256(normalize_text(row.get("text")).encode("utf-8")).hexdigest(),
            "corrected_fingerprint": hashlib.sha256(normalize_text(row.get("corrected_text")).encode("utf-8")).hexdigest(),
            "template_fingerprint": hashlib.sha256(_annotation_template(row).encode("utf-8")).hexdigest(),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_ORDER:
        write_jsonl(output_dir / f"{prefix}-{split}.jsonl", split_rows_map[split])
    write_jsonl(output_dir / f"{prefix}-index.jsonl", index_rows)
    manifest = {
        "format": "vsec-jsonl-split-v1",
        "source": "VSEC parquet; raw input is never modified",
        "seed": seed,
        "ratios": dict(ratios),
        "grouping": {
            "keys": ["normalized text", "normalized corrected_text", "annotation error template", "variable-masked template"],
            "algorithm": "union-find connected components, then largest-deficit whole-group assignment",
        },
        "schema": sorted(rows[0].keys()) if rows else [],
        "rows": len(rows),
        "groups": len(groups),
        "splits": {split: {"rows": len(split_rows_map[split]), "groups": len({entry["group_id"] for entry in index_rows if entry["split"] == split})} for split in SPLIT_ORDER},
        "artifacts": {split: f"{prefix}-{split}.jsonl" for split in SPLIT_ORDER},
        "index": f"{prefix}-index.jsonl",
    }
    (output_dir / f"{prefix}-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def split_clean_source(paths: Sequence[Path], output_dir: Path, *, seed: int = DEFAULT_SEED, ratios: Mapping[str, float] = DEFAULT_RATIOS) -> dict[str, Any]:
    lines: list[str] = []
    source_for_line: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    lines.append(text)
                    source_for_line.append(str(path))
    rows = [{"text": text} for text in lines]
    # Clean corpus rows have no annotation positions; exact and variable templates
    # still prevent duplicate boilerplate from crossing split boundaries.
    groups = grouped_indices(rows, lambda row: clean_group_key(row["text"]))
    assignment = assign_groups(groups, len(rows), seed, ratios)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split in SPLIT_ORDER:
        selected = [lines[index] for index in range(len(lines)) if assignment[str(index)] == split]
        (output_dir / f"clean-{split}.txt").write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
        split_counts[split] = len(selected)
    manifest = {
        "format": "clean-source-split-v1",
        "seed": seed,
        "ratios": dict(ratios),
        "source_files": sorted(set(source_for_line)),
        "rows": len(lines),
        "splits": split_counts,
        "grouping": "normalized text + variable-masked template, whole groups",
        "artifacts": {split: f"clean-{split}.txt" for split in SPLIT_ORDER},
    }
    (output_dir / "clean-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit_split(output_dir: Path, *, external_path: Path | None = None) -> dict[str, Any]:
    rows = {split: read_jsonl(output_dir / f"vsec-{split}.jsonl") for split in SPLIT_ORDER}
    fingerprints: dict[str, set[str]] = {}
    for split, split_rows in rows.items():
        fingerprints[split] = {normalize_text(row.get("text")) for row in split_rows if normalize_text(row.get("text"))}
    pairwise = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        pairwise[f"{left}_intersect_{right}"] = len(fingerprints[left] & fingerprints[right])
    report: dict[str, Any] = {
        "rows": {split: len(rows[split]) for split in SPLIT_ORDER},
        "normalized_text_intersections": pairwise,
        "pass": all(value == 0 for value in pairwise.values()),
    }
    if external_path and external_path.exists():
        external_rows = read_jsonl(external_path) if external_path.suffix == ".jsonl" else (json.loads(external_path.read_text(encoding="utf-8")).get("rows", []))
        external_text = {normalize_text(row.get("text")) for row in external_rows if isinstance(row, Mapping) and normalize_text(row.get("text"))}
        report["external_rows"] = len(external_rows)
        report["train_intersect_external"] = len(fingerprints["train"] & external_text)
        report["pass"] = report["pass"] and report["train_intersect_external"] == 0
    return report


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsec-parquet", type=Path, default=DEFAULT_VSEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--clean-source", type=Path, action="append", default=[])
    parser.add_argument("--clean-out-dir", type=Path, default=DEFAULT_CLEAN_OUT)
    parser.add_argument("--audit", action="store_true", help="audit existing VSEC artifacts without rewriting them")
    parser.add_argument("--external-benchmark", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.audit:
        print(json.dumps(audit_split(args.out_dir, external_path=args.external_benchmark), ensure_ascii=False, indent=2))
        return 0

    if not args.vsec_parquet.exists():
        raise SystemExit(f"VSEC parquet not found: {args.vsec_parquet}")
    rows = read_vsec(args.vsec_parquet)
    manifest = split_rows(rows, args.out_dir, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.clean_source:
        clean_manifest = split_clean_source(args.clean_source, args.clean_out_dir, seed=args.seed)
        print(json.dumps(clean_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

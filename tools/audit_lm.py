#!/usr/bin/env python3
"""Fast provenance and held-out leakage audit for a built LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_lm import ROOT, BENCHMARK_DIR, CLEAN_SPLIT_DIR, VSEC_SPLIT_DIR, audit_manifest, normalize_sentence, read_json_rows


def heldout_sets() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for name in ("clean-dev.txt", "clean-test.txt"):
        path = CLEAN_SPLIT_DIR / name
        sets[f"clean-{path.stem.removeprefix('clean-')}"] = {
            normalize_sentence(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        } if path.exists() else set()
    for label, path in {
        "synthetic": BENCHMARK_DIR / "corpus-synthetic-diacritics.json",
        "viwiki": BENCHMARK_DIR / "corpus-viwiki-spelling.json",
        "vsec-dev": VSEC_SPLIT_DIR / "vsec-dev.jsonl",
        "vsec-test": VSEC_SPLIT_DIR / "vsec-test.jsonl",
    }.items():
        values = set()
        for row in read_json_rows(path):
            for field in ("text", "groundTruth", "corrected_text"):
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    values.add(normalize_sentence(value))
        sets[label] = values
    return sets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "src" / "data" / "lm-ngrams.manifest.json")
    args = parser.parse_args()
    manifest_report = audit_manifest(args.manifest)
    train_path = ROOT / "src" / "data" / "corpus-train.txt"
    train = {normalize_sentence(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()} if train_path.exists() else set()
    intersections = {name: len(train & values) for name, values in heldout_sets().items()}
    report = {**manifest_report, "trainHeldoutIntersections": intersections, "pass": manifest_report["pass"] and all(value == 0 for value in intersections.values())}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

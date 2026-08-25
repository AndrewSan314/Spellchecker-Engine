"""Fast leakage/schema checks for generated spelling-data artifacts.

Run with: ``python -m unittest test.test_spelling_dataset_split -v``.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from split_spelling_datasets import normalize_text  # noqa: E402


class SpellingDatasetSplitTest(unittest.TestCase):
    split_dir = ROOT / "dataset_artifacts" / "vsec"
    expected_schema = {
        "text", "corrected_text", "syllable_annotations", "error_count",
        "error_positions", "correction_pairs", "has_errors",
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = {}
        for split in ("train", "dev", "test"):
            path = cls.split_dir / f"vsec-{split}.jsonl"
            if not path.exists():
                raise unittest.SkipTest("run tools/split_spelling_datasets.py first")
            cls.rows[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_schema_and_counts(self) -> None:
        self.assertEqual(sum(map(len, self.rows.values())), 9341)
        for rows in self.rows.values():
            self.assertTrue(rows)
            self.assertTrue(all(set(row) == self.expected_schema for row in rows))

    def test_normalized_text_disjoint(self) -> None:
        fingerprints = {split: {normalize_text(row["text"]) for row in rows} for split, rows in self.rows.items()}
        self.assertFalse(fingerprints["train"] & fingerprints["dev"])
        self.assertFalse(fingerprints["train"] & fingerprints["test"])
        self.assertFalse(fingerprints["dev"] & fingerprints["test"])

    def test_group_index_does_not_cross_splits(self) -> None:
        index_path = self.split_dir / "vsec-index.jsonl"
        index_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        groups = {}
        for entry in index_rows:
            groups.setdefault(entry["group_id"], set()).add(entry["split"])
        self.assertTrue(groups)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_external_vwiki_disjoint(self) -> None:
        external_path = ROOT / "benchmark" / "corpus-viwiki-spelling.json"
        if not external_path.exists():
            self.skipTest("run tools/convert_viwiki_spelling.mjs first")
        external = json.loads(external_path.read_text(encoding="utf-8"))
        external_text = {normalize_text(row.get("text")) for row in external.get("rows", [])}
        train_text = {normalize_text(row["text"]) for row in self.rows["train"]}
        self.assertFalse(train_text & external_text)

    def test_correction_pairs_are_position_based(self) -> None:
        mismatch_rows = []
        total_errors = 0
        for rows in self.rows.values():
            for row in rows:
                total_errors += int(row["error_count"])
                if len(row["correction_pairs"]) != int(row["error_count"]):
                    mismatch_rows.append(row)
        self.assertGreater(len(mismatch_rows), 0)
        for row in mismatch_rows:
            annotated = {int(a["position"]) for a in row["syllable_annotations"] if not a["is_correct"]}
            positions = {int(p) for p in row["error_positions"]}
            paired = {int(p["position"]) for p in row["correction_pairs"]}
            self.assertTrue(positions <= annotated | paired)
        self.assertGreater(total_errors, 0)


if __name__ == "__main__":
    unittest.main()

"""Held-out clean-source leakage check for synthetic evaluation."""

import json
import re
import unicodedata
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value)).casefold()).strip()


class CleanSourceSplitTest(unittest.TestCase):
    def test_train_excludes_dev_test_and_synthetic_ground_truth(self):
        source_dir = ROOT / "dataset_artifacts" / "clean-source"
        train_path = ROOT / "src" / "data" / "corpus-train.txt"
        synthetic_path = ROOT / "benchmark" / "corpus-synthetic-diacritics.json"
        for path in (source_dir / "clean-test.txt", source_dir / "clean-dev.txt", train_path, synthetic_path):
            if not path.exists():
                self.skipTest("run node tools/corpus_pipeline.mjs first")
        train = {normalize(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        dev = {normalize(line) for line in (source_dir / "clean-dev.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
        test = {normalize(line) for line in (source_dir / "clean-test.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
        synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
        ground_truth = {normalize(row["groundTruth"]) for row in synthetic.get("rows", [])}
        self.assertFalse(train & dev)
        self.assertFalse(train & test)
        self.assertFalse(train & ground_truth)
        self.assertTrue(all(row.get("sourceSplit") == "test" for row in synthetic.get("rows", [])))


if __name__ == "__main__":
    unittest.main()

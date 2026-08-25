# Recall-improvement plan Task 6 Step 4 — trainer unit tests.
# Verifies standardization math, finite sigmoid, byte-identical determinism
# on the fixture, and the fail-closed leakage guard. Standard library only.
import io
import json
import math
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import train_recall_reranker as tr  # noqa: E402

FIXTURE = ROOT / "test" / "fixtures" / "recall-reranker"
FORBIDDEN = ROOT / "test" / "fixtures" / "recall-reranker-forbidden"


class StandardizationTest(unittest.TestCase):
    def test_mean_std_known_values(self):
        xs = [2.0, 4.0, 6.0]
        mean, std = tr.mean_std(xs)
        self.assertAlmostEqual(mean, 4.0)
        self.assertAlmostEqual(std, math.sqrt(8.0 / 3.0))

    def test_constant_feature_gets_unit_scale(self):
        mean, std = tr.mean_std([7.0, 7.0, 7.0])
        self.assertAlmostEqual(mean, 7.0)
        self.assertAlmostEqual(std, 1.0)

    def test_standardize_finite_at_extremes(self):
        z = tr.standardize(1e18, mean=0.0, std=1.0)
        self.assertTrue(math.isfinite(z))


class SigmoidTest(unittest.TestCase):
    def test_midpoint(self):
        self.assertAlmostEqual(tr.sigmoid(0.0), 0.5)

    def test_finite_at_extremes(self):
        for z in (-1e9, 1e9):
            p = tr.sigmoid(z)
            self.assertTrue(math.isfinite(p))
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


def _train_fixture_bytes(tmp_out: Path) -> bytes:
    rc = tr.main([
        "--fixture", str(FIXTURE),
        "--out", str(tmp_out),
        "--iterations", "25",
    ])
    assert rc == 0
    return tmp_out.read_bytes()


class DeterminismTest(unittest.TestCase):
    def test_byte_identical_models(self):
        a = ROOT / ".tmp" / "rr-det-a.json"
        b = ROOT / ".tmp" / "rr-det-b.json"
        a.parent.mkdir(exist_ok=True)
        ba = _train_fixture_bytes(a)
        bb = _train_fixture_bytes(b)
        self.assertEqual(ba, bb,
            "two runs over the same fixture must produce byte-identical models")
        model = json.loads(ba.decode("utf-8"))
        self.assertEqual(model["schema"], tr.SCHEMA)
        self.assertEqual(model["featureContract"], tr.CONTRACT)
        self.assertEqual(model["featureOrder"], tr.FEATURE_ORDER)


class LeakageGuardTest(unittest.TestCase):
    def test_forbidden_path_fails_non_zero(self):
        out = ROOT / ".tmp" / "rr-forbidden.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tr.main([
                "--fixture", str(FORBIDDEN),
                "--out", str(out),
            ])
        self.assertNotEqual(rc, 0,
            "a forbidden input path must fail closed with non-zero exit")

    def test_guard_rejects_each_mark(self):
        for mark in ("vsec-dev", "vsec-test", "viwiki", "external-test",
                     "benchmark"):
            self.assertFalse(tr.path_is_allowed(f"data/{mark}/rows.jsonl"),
                f"path containing {mark!r} must be rejected")
        self.assertTrue(tr.path_is_allowed(
            "dataset_artifacts/vsec/vsec-train.jsonl"))


if __name__ == "__main__":
    unittest.main()

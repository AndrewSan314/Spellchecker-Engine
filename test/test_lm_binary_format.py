# Task 11 (spelling-engine-optimization plan): binary LM format v1 —
# round-trip exactness, byte-determinism, and corruption rejection.
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import export_lm_binary as lm  # noqa: E402

VALID_TSV = "\n".join([
    "#SMS-LM v1",
    "#tokens=1000000",
    "U\tkính\t600000",
    "U\tchào\t300000",
    "U\tquý\t70000",
    "U\tkhách\t30000",
    "B\tkính chào\t250000",
    "B\tchào quý\t120000",
    "B\tquý khách\t180000",
    "T\tkính chào quý\t90000",
]) + "\n"


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tsv = os.path.join(self.tmp.name, "lm.tsv")
        with open(self.tsv, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(VALID_TSV)
        self.out = os.path.join(self.tmp.name, "lm.v1.bin")

    def tearDown(self):
        self.tmp.cleanup()


class TestRoundTrip(TempCase):
    def test_exact_count_equivalence(self):
        lm.export_lm(self.tsv, self.out)
        data = lm.read_sections(self.out)
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["tokenTotal"], 1000000)
        self.assertEqual(data["unigrams"], {
            ("kính",): 600000, ("chào",): 300000,
            ("quý",): 70000, ("khách",): 30000,
        })
        self.assertEqual(data["bigrams"], {
            ("kính", "chào"): 250000,
            ("chào", "quý"): 120000,
            ("quý", "khách"): 180000,
        })
        self.assertEqual(data["trigrams"], {
            ("kính", "chào", "quý"): 90000,
        })

    def test_byte_determinism(self):
        out2 = os.path.join(self.tmp.name, "lm2.v1.bin")
        lm.export_lm(self.tsv, self.out)
        lm.export_lm(self.tsv, out2)
        with open(self.out, "rb") as f1, open(out2, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())

    def test_manifest_records_hashes_and_counts(self):
        lm.export_lm(self.tsv, self.out)
        man = json.load(open(self.out + ".manifest.json", encoding="utf-8"))
        import hashlib
        body = open(self.out, "rb").read()
        self.assertEqual(man["sha256"], hashlib.sha256(body).hexdigest())
        src = open(self.tsv, "rb").read()
        self.assertEqual(man["sourceTsvSha256"],
                         hashlib.sha256(src).hexdigest())
        self.assertEqual(man["counts"], {"U": 4, "B": 3, "T": 1})
        self.assertEqual(man["tokenTotal"], 1000000)

    def test_refuses_overwrite_without_force(self):
        lm.export_lm(self.tsv, self.out)
        with self.assertRaises(RuntimeError):
            lm.export_lm(self.tsv, self.out)
        lm.export_lm(self.tsv, self.out, force=True)


class TestCorruption(TempCase):
    def setUp(self):
        super().setUp()
        lm.export_lm(self.tsv, self.out)
        with open(self.out, "rb") as fh:
            self.raw = bytearray(fh.read())

    def write(self, data):
        with open(self.out, "wb") as fh:
            fh.write(bytes(data))

    def test_wrong_magic_rejected(self):
        bad = bytearray(self.raw)
        bad[0:4] = b"XXXX"
        self.write(bad)
        with self.assertRaises(ValueError):
            lm.read_sections(self.out)

    def test_unsupported_version_rejected(self):
        bad = bytearray(self.raw)
        bad[4:8] = struct.pack("<I", 999)
        self.write(bad)
        with self.assertRaises(ValueError):
            lm.read_sections(self.out)

    def test_truncated_section_rejected(self):
        self.write(self.raw[: len(self.raw) // 2])
        with self.assertRaises(ValueError):
            lm.read_sections(self.out)

    def test_payload_corruption_detected_by_crc(self):
        bad = bytearray(self.raw)
        mid = len(bad) // 2
        bad[mid] ^= 0xFF
        self.write(bad)
        with self.assertRaises(ValueError):
            lm.read_sections(self.out)

    def test_exporter_rejects_invalid_rows_before_writing(self):
        bad_tsv = VALID_TSV.replace("U\tkhách\t30000", "U\tkhách\t-5")
        p = os.path.join(self.tmp.name, "bad.tsv")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(bad_tsv)
        fresh_out = os.path.join(self.tmp.name, "fresh1.v1.bin")
        with self.assertRaises(ValueError):
            lm.export_lm(p, fresh_out)
        self.assertFalse(os.path.exists(fresh_out))

    def test_exporter_rejects_duplicate_keys(self):
        bad_tsv = VALID_TSV + "B\tquý khách\t7\n"
        p = os.path.join(self.tmp.name, "dup.tsv")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(bad_tsv)
        fresh_out = os.path.join(self.tmp.name, "fresh2.v1.bin")
        with self.assertRaises(ValueError):
            lm.export_lm(p, fresh_out)
        self.assertFalse(os.path.exists(fresh_out))
        # the artifact on disk must remain the ORIGINAL valid export
        lm.read_sections(self.out)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from tools import build_lm


class LmBuilderTest(unittest.TestCase):
    def test_exclusions_load_heldout_sources(self):
        exclusions, categories, paths = build_lm.load_exclusions()
        self.assertGreater(len(exclusions), 0)
        self.assertTrue({'clean-source', 'synthetic', 'viwiki-external', 'vsec-dev', 'vsec-test'} <= set(categories))
        self.assertTrue(paths)

    def test_sentence_normalization_is_case_and_whitespace_stable(self):
        self.assertEqual(build_lm.normalize_sentence('  Kính\tChào  '), 'kính chào')

    def test_two_tier_counter_caps_higher_order_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counter = build_lm.ChunkCounter(root, 2)
            counter.add_sentence(["một", "hai", "ba"], 1, include_higher=True)
            counter.add_sentence(["bốn", "năm", "sáu"], 1, include_higher=False)
            config = {
                "uni_min": 1,
                "big_min": 1,
                "tri_min": 1,
                "max_vocab": 0,
                "max_bigrams": 1,
                "max_trigrams": 1,
                "lex_min_freq": 1,
            }
            lm_path, _, counts = build_lm.write_outputs(counter, counter.total_tokens, config, root)
            self.assertEqual(counts["bigrams"], 1)
            self.assertEqual(counts["trigrams"], 1)
            self.assertLessEqual(lm_path.stat().st_size, 1000)
    def test_merge_aggregates_keys_across_u_b_t_ordered_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counter = build_lm.ChunkCounter(root, 1)
            counter.add_sentence(["mot", "hai", "ba"], 1, include_higher=True)
            counter.add_sentence(["mot", "hai", "ba"], 1, include_higher=True)
            merged = list(build_lm.merged_parts(counter.parts))
            self.assertEqual([item for item in merged if item[0] == "B"], [("B", "hai ba", 2), ("B", "mot hai", 2)])
            self.assertEqual([item for item in merged if item[0] == "T"], [("T", "mot hai ba", 2)])
    def test_atomic_replace_keeps_recoverable_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_lm = root / 'lm-ngrams.tsv'
            old_lex = root / 'lexicon-built.txt'
            manifest_path = root / 'lm-ngrams.manifest.json'
            old_lm.write_text('#SMS-LM v1\n#tokens=1\nU\tx\t1\n', encoding='utf-8')
            old_lex.write_text('old\n', encoding='utf-8')
            temp_lm = root / 'staged-lm.tsv'
            temp_lex = root / 'staged-lexicon.txt'
            temp_lm.write_text('#SMS-LM v1\n#tokens=2\nU\ty\t2\n', encoding='utf-8')
            temp_lex.write_text('new\n', encoding='utf-8')
            previous_artifact_dir = build_lm.ARTIFACT_DIR
            try:
                build_lm.ARTIFACT_DIR = root / 'backups'
                manifest = {'outputs': {}}
                backup = build_lm.replace_atomically(temp_lm, temp_lex, manifest, old_lm, old_lex, manifest_path)
            finally:
                build_lm.ARTIFACT_DIR = previous_artifact_dir
            self.assertIsNotNone(backup)
            self.assertIn('U\ty\t2', old_lm.read_text(encoding='utf-8'))
            self.assertEqual(old_lex.read_text(encoding='utf-8'), 'new\n')
            self.assertTrue((backup / 'lm-ngrams.tsv').exists())
            self.assertEqual((backup / 'lm-ngrams.tsv').read_text(encoding='utf-8').splitlines()[-1], 'U\tx\t1')


if __name__ == '__main__':
    unittest.main()

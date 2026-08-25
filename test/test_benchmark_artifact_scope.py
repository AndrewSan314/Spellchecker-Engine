import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkArtifactScopeTest(unittest.TestCase):
    def _rows(self, name):
        path = ROOT / 'benchmark' / name
        if not path.exists():
            self.skipTest(f'run the adapter for {name} first')
        return json.loads(path.read_text(encoding='utf-8'))['rows']

    def test_external_rows_are_spelling_scoped_not_globally_fully_labelled(self):
        for name in ('corpus-vsec-test.json', 'corpus-viwiki-spelling.json'):
            rows = self._rows(name)
            self.assertTrue(rows)
            self.assertTrue(all(row.get('fullyLabeled') is not True for row in rows))
            self.assertTrue(all(row.get('fullyLabeledRuleIds') == ['POSSIBLE_SPELLING_ERROR'] for row in rows))


if __name__ == '__main__':
    unittest.main()

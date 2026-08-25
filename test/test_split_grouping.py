import unittest

from tools.split_spelling_datasets import assign_groups, grouped_indices, variable_template


class SplitGroupingTest(unittest.TestCase):
    def test_uppercase_ids_and_numbers_share_template_group(self):
        left = 'Thanh toán ABC-2026 tại quầy 12'
        right = 'Thanh toán XYZ-2027 tại quầy 13'
        self.assertEqual(variable_template(left), variable_template(right))
        rows = [{'text': left}, {'text': right}]
        groups = grouped_indices(rows)
        self.assertEqual(groups, [[0, 1]])
        assignment = assign_groups(groups, len(rows), seed=20260824)
        self.assertEqual(assignment['0'], assignment['1'])


if __name__ == '__main__':
    unittest.main()

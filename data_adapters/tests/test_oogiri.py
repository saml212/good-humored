"""Unit tests for data_adapters/oogiri.py. Uses ONLY the tiny local fixture
`fixtures/oogiri_sample.jsonl` -- no network call, per this build's
constraint that loaders' network paths are tested via fixtures only.
Run: python3 -m unittest discover -s data_adapters/tests -v
"""

import unittest
from pathlib import Path

from data_adapters.oogiri import load_ranked_groups, parse_en_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "oogiri_sample.jsonl"


class TestParseEnJsonl(unittest.TestCase):
    def test_counts_match_fixture_by_hand(self):
        # Fixture (8 rows): 2x I2T; T2T "tomato" question x3 candidates;
        # T2T "joke" question x1 candidate (singleton, dropped); T2T with
        # empty-string question (excluded, no context); 1x "IT2T" (other).
        groups, stats = parse_en_jsonl(FIXTURE)
        self.assertEqual(stats.total_rows, 8)
        self.assertEqual(stats.i2t_excluded, 2)
        self.assertEqual(stats.t2t_rows, 4)  # 3 tomato + 1 joke
        self.assertEqual(stats.other_type_excluded, 2)  # empty-question + IT2T
        self.assertEqual(stats.distinct_t2t_prompts, 2)
        self.assertEqual(stats.single_candidate_dropped, 1)
        self.assertEqual(stats.multi_candidate_groups, 1)
        self.assertEqual(stats.ranked_groups_emitted, 1)
        self.assertEqual(stats.candidates_emitted, 3)
        self.assertEqual(len(groups), 1)

    def test_emitted_group_shape_and_license(self):
        groups, _ = parse_en_jsonl(FIXTURE)
        g = groups[0]
        self.assertEqual(g.context, "Why did the tomato blush?")
        self.assertEqual(g.license_class, "research_only")
        self.assertEqual(g.source_dataset, "oogiri-go")
        self.assertEqual(len(g.candidates), 3)
        scores = sorted(c.score for c in g.candidates)
        self.assertEqual(scores, [1.0, 3.0, 9.0])

    def test_min_group_size_one_keeps_singletons(self):
        groups, stats = parse_en_jsonl(FIXTURE, min_group_size=1)
        # now both the tomato group (3) and the joke group (1) qualify
        self.assertEqual(stats.ranked_groups_emitted, 2)
        self.assertEqual(stats.single_candidate_dropped, 0)


class TestLoadRankedGroups(unittest.TestCase):
    def test_requires_research_only_in_allowed(self):
        with self.assertRaises(ValueError):
            load_ranked_groups(allowed_licenses=["commercial_safe"],
                              jsonl_path=FIXTURE)

    def test_succeeds_with_research_only_allowed(self):
        groups, stats = load_ranked_groups(
            allowed_licenses=["research_only"], jsonl_path=FIXTURE)
        self.assertEqual(len(groups), 1)
        self.assertEqual(stats.ranked_groups_emitted, 1)

    def test_no_allowed_licenses_raises(self):
        with self.assertRaises(ValueError):
            load_ranked_groups(allowed_licenses=[], jsonl_path=FIXTURE)


if __name__ == "__main__":
    unittest.main()

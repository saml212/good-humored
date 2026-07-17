"""Unit tests for data_adapters/nycc.py. Uses ONLY the tiny local fixture
`fixtures/nycc_rows_sample.json` (already-fetched row dicts) -- no network
call, per this build's constraint that loaders' network paths are tested
via fixtures only.
Run: python3 -m unittest discover -s data_adapters/tests -v
"""

import json
import unittest
from pathlib import Path

from data_adapters.nycc import load_ranked_groups, parse_rows

FIXTURE = Path(__file__).parent / "fixtures" / "nycc_rows_sample.json"


def _load_fixture_rows():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


class TestParseRows(unittest.TestCase):
    def test_two_well_formed_rows_one_malformed(self):
        rows = _load_fixture_rows()
        groups, stats = parse_rows(rows)
        self.assertEqual(stats.rows_fetched, 3)
        self.assertEqual(stats.rows_with_two_choices, 2)
        self.assertEqual(stats.rows_skipped_malformed, 1)
        self.assertEqual(stats.ranked_groups_emitted, 2)
        self.assertEqual(len(groups), 2)

    def test_winner_label_maps_to_higher_score(self):
        rows = _load_fixture_rows()
        groups, _ = parse_rows(rows)
        row0_group = next(g for g in groups if g.source_id == "fixture-row-0")
        by_text = {c.text: c.score for c in row0_group.candidates}
        # fixture row 0: label "B" -> "He calls it Ishmeow." is the winner
        self.assertEqual(by_text["He calls it Ishmeow."], 1.0)
        self.assertEqual(by_text["Thus, he gave up the spear."], 0.0)

    def test_context_excludes_image_bytes_includes_scene_text(self):
        rows = _load_fixture_rows()
        groups, _ = parse_rows(rows)
        g = groups[0]
        self.assertNotIn("http", g.context)  # no image URL leaked into context
        self.assertTrue(len(g.context) > 0)

    def test_license_class_and_attribution_metadata(self):
        rows = _load_fixture_rows()
        groups, _ = parse_rows(rows)
        for g in groups:
            self.assertEqual(g.license_class, "commercial_safe")
            self.assertTrue(g.metadata.get("attribution_required"))
            self.assertIn("CC-BY", g.metadata.get("attribution", ""))


class TestLoadRankedGroups(unittest.TestCase):
    def test_requires_commercial_safe_in_allowed(self):
        rows = _load_fixture_rows()
        with self.assertRaises(ValueError):
            load_ranked_groups(allowed_licenses=["research_only"], rows=rows)

    def test_succeeds_with_commercial_safe_allowed_no_network(self):
        rows = _load_fixture_rows()
        groups, stats = load_ranked_groups(
            allowed_licenses=["commercial_safe"], rows=rows)
        self.assertEqual(len(groups), 2)
        self.assertEqual(stats.ranked_groups_emitted, 2)


if __name__ == "__main__":
    unittest.main()

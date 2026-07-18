"""Unit tests for data_adapters/oogiri_master.py. Uses ONLY the tiny local
fixture `fixtures/oogiri_master_corpus_sample.csv` (synthetic, hand-built
to match the verified Oogiri-Corpus.csv schema -- see that module's
docstring for the schema verification) -- no network call, per this
package's convention that loaders' network paths (or, here, the
deliberate absence of one) are tested via fixtures only.

Run: python3 -m unittest discover -s data_adapters/tests -v
"""

import unittest
from pathlib import Path

from data_adapters.oogiri_master import (ensure_sample, load_ranked_groups,
                                          parse_oogiri_corpus_csv)

FIXTURE = Path(__file__).parent / "fixtures" / "oogiri_master_corpus_sample.csv"


class TestParseOogiriCorpusCsv(unittest.TestCase):
    def test_counts_match_fixture_by_hand(self):
        # Fixture (15 data rows): prompt 0001 has 4 rows (1 empty-response,
        # dropped -> 3 valid candidates); 0002 has 2 valid (tied votes);
        # 0003 has 1 valid (singleton, dropped); 0004 has 1 valid
        # (singleton, dropped); 0005 has 3 rows (1 bad vote_count, dropped
        # -> 2 valid candidates); 0006 missing prompt_id (skipped
        # entirely); 0007 missing prompt text (skipped entirely); 0008 has
        # 2 valid rows with MISMATCHED prompt text (still forms a group,
        # counted as a mismatch).
        groups, stats = parse_oogiri_corpus_csv(FIXTURE)
        self.assertEqual(stats.total_rows, 15)
        self.assertEqual(stats.malformed_rows_skipped, 4)
        self.assertEqual(stats.distinct_prompts, 6)  # 0001,0002,0003,0004,0005,0008
        self.assertEqual(stats.prompt_text_mismatch, 1)  # 0008
        self.assertEqual(stats.single_candidate_dropped, 2)  # 0003 + 0004
        self.assertEqual(stats.multi_candidate_groups, 4)  # 0001,0002,0005,0008
        self.assertEqual(stats.ranked_groups_emitted, 4)
        self.assertEqual(stats.candidates_emitted, 9)  # 3+2+2+2
        self.assertEqual(len(groups), 4)

    def test_emitted_group_shape_and_license(self):
        groups, _ = parse_oogiri_corpus_csv(FIXTURE)
        by_id = {g.source_id: g for g in groups}
        g1 = by_id["oogiri-corpus-prompt-0001"]
        self.assertEqual(g1.license_class, "research_only")
        self.assertEqual(g1.source_dataset, "oogiri-corpus")
        self.assertEqual(len(g1.candidates), 3)
        scores = sorted(c.score for c in g1.candidates)
        self.assertEqual(scores, [3.0, 7.0, 12.0])
        # rater_count is never populated (see module docstring: no
        # verified per-response denominator in this schema).
        self.assertTrue(all(c.rater_count is None for c in g1.candidates))

    def test_metadata_carries_provenance_caveats(self):
        groups, _ = parse_oogiri_corpus_csv(FIXTURE)
        g1 = next(g for g in groups if g.source_id == "oogiri-corpus-prompt-0001")
        self.assertEqual(g1.metadata["language"], "ja")
        self.assertEqual(g1.metadata["source_platform"], "bokete.jp")
        self.assertIn("aggregate vote_count", g1.metadata["vote_semantics"])

    def test_malformed_vote_count_row_dropped_group_still_forms(self):
        groups, _ = parse_oogiri_corpus_csv(FIXTURE)
        g5 = next(g for g in groups if g.source_id == "oogiri-corpus-prompt-0005")
        self.assertEqual(len(g5.candidates), 2)
        scores = sorted(c.score for c in g5.candidates)
        self.assertEqual(scores, [2.0, 6.0])

    def test_mismatched_prompt_text_group_uses_first_row_context(self):
        groups, _ = parse_oogiri_corpus_csv(FIXTURE)
        g8 = next(g for g in groups if g.source_id == "oogiri-corpus-prompt-0008")
        self.assertEqual(g8.context, "original prompt text for group 8")
        self.assertEqual(len(g8.candidates), 2)

    def test_missing_prompt_id_and_missing_prompt_rows_excluded(self):
        groups, _ = parse_oogiri_corpus_csv(FIXTURE)
        source_ids = {g.source_id for g in groups}
        self.assertNotIn("oogiri-corpus-prompt-0006", source_ids)
        self.assertNotIn("oogiri-corpus-prompt-0007", source_ids)

    def test_min_group_size_one_keeps_singletons(self):
        groups, stats = parse_oogiri_corpus_csv(FIXTURE, min_group_size=1)
        # now 0003 and 0004 (previously dropped singletons) also qualify
        self.assertEqual(stats.ranked_groups_emitted, 6)
        self.assertEqual(stats.single_candidate_dropped, 0)

    def test_wrong_schema_raises_clear_error(self):
        import csv
        import tempfile
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prompt", "response", "likes"])  # missing
                                                                     # prompt_id,
                                                                     # vote_count
            writer.writerow(["1", "some prompt", "some response", "5"])
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_oogiri_corpus_csv(path)
            self.assertIn("missing expected column", str(ctx.exception))
        finally:
            Path(path).unlink()


class TestLoadRankedGroups(unittest.TestCase):
    def test_requires_research_only_in_allowed(self):
        with self.assertRaises(ValueError):
            load_ranked_groups(allowed_licenses=["commercial_safe"],
                              csv_path=FIXTURE)

    def test_succeeds_with_research_only_allowed(self):
        groups, stats = load_ranked_groups(
            allowed_licenses=["research_only"], csv_path=FIXTURE)
        self.assertEqual(len(groups), 4)
        self.assertEqual(stats.ranked_groups_emitted, 4)

    def test_no_allowed_licenses_raises(self):
        with self.assertRaises(ValueError):
            load_ranked_groups(allowed_licenses=[], csv_path=FIXTURE)

    def test_csv_path_required(self):
        with self.assertRaises(ValueError):
            load_ranked_groups(allowed_licenses=["research_only"], csv_path=None)


class TestEnsureSample(unittest.TestCase):
    def test_ensure_sample_always_raises_no_download_path(self):
        # This module has NO working downloader -- see module docstring's
        # ACQUISITION STATUS section for what was verified (no HF dataset,
        # no GitHub release, builder repo does not redistribute outputs).
        with self.assertRaises(RuntimeError) as ctx:
            ensure_sample()
        self.assertIn("no public Oogiri-Corpus.csv", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for data_adapters/local_corpus.py. Points `data_dir` at
`fixtures/local_corpus/` (a tiny synthetic stand-in for the real
~/Experiments/good-humored-data/ layout) -- no real corpus rows are read
by this test file.
Run: python3 -m unittest discover -s data_adapters/tests -v
"""

import unittest
from pathlib import Path

from data_adapters.local_corpus import (load_corpus_jokes,
                                        load_memorized_templates)

FIXTURE_DATA_DIR = Path(__file__).parent / "fixtures" / "local_corpus"


class TestLoadCorpusJokes(unittest.TestCase):
    def test_commercial_safe_only(self):
        jokes, stats = load_corpus_jokes(
            allowed_licenses=["commercial_safe"], data_dir=FIXTURE_DATA_DIR)
        self.assertEqual(len(jokes), 3)
        self.assertTrue(all(j.license_class == "commercial_safe" for j in jokes))
        self.assertEqual(stats.rows_by_bucket.get("commercial-safe"), 3)
        # research-only bucket file must never even be opened
        self.assertNotIn("research-only", stats.rows_by_bucket)

    def test_research_only_only_and_malformed_row_skipped(self):
        jokes, stats = load_corpus_jokes(
            allowed_licenses=["research_only"], data_dir=FIXTURE_DATA_DIR)
        # fixture file has 2 valid rows + 1 malformed ({"malformed": true})
        self.assertEqual(len(jokes), 2)
        self.assertEqual(stats.rows_skipped_malformed, 1)
        self.assertTrue(all(j.license_class == "research_only" for j in jokes))

    def test_both_licenses_allowed_returns_both_buckets(self):
        jokes, stats = load_corpus_jokes(
            allowed_licenses=["commercial_safe", "research_only"],
            data_dir=FIXTURE_DATA_DIR)
        self.assertEqual(len(jokes), 5)
        licenses = {j.license_class for j in jokes}
        self.assertEqual(licenses, {"commercial_safe", "research_only"})

    def test_no_allowed_licenses_raises(self):
        with self.assertRaises(ValueError):
            load_corpus_jokes(allowed_licenses=[], data_dir=FIXTURE_DATA_DIR)

    def test_original_provenance_preserved_in_metadata(self):
        jokes, _ = load_corpus_jokes(
            allowed_licenses=["commercial_safe"], data_dir=FIXTURE_DATA_DIR)
        j = next(j for j in jokes if j.source_id == "fixture-cs-1")
        self.assertEqual(j.metadata["original_source"], "fixture-source")
        self.assertEqual(j.metadata["original_license_text"], "CC-BY-4.0")

    def test_limit_per_bucket_caps_rows_read(self):
        jokes, stats = load_corpus_jokes(
            allowed_licenses=["commercial_safe"], data_dir=FIXTURE_DATA_DIR,
            limit_per_bucket=1)
        self.assertEqual(len(jokes), 1)
        self.assertEqual(stats.rows_by_bucket["commercial-safe"], 1)

    def test_missing_corpus_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_corpus_jokes(allowed_licenses=["commercial_safe"],
                             data_dir=Path("/nonexistent/does-not-exist"))


class TestLoadMemorizedTemplates(unittest.TestCase):
    def test_loads_separately_from_main_corpus(self):
        templates = load_memorized_templates(
            allowed_licenses=["commercial_safe"], data_dir=FIXTURE_DATA_DIR)
        self.assertEqual(len(templates), 2)
        self.assertTrue(all(t.license_class == "commercial_safe"
                            for t in templates))
        self.assertTrue(all("fixture" in t.source_dataset
                            for t in templates))

    def test_not_included_in_load_corpus_jokes(self):
        jokes, _ = load_corpus_jokes(
            allowed_licenses=["commercial_safe", "research_only"],
            data_dir=FIXTURE_DATA_DIR)
        self.assertFalse(any("memorized" in j.source_dataset.lower() or
                            "templates" in j.source_dataset.lower()
                            for j in jokes))


if __name__ == "__main__":
    unittest.main()

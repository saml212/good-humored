"""Unit tests for data_adapters/firewall.py.
Run: python3 -m unittest discover -s data_adapters/tests -v
"""

import unittest

from data_adapters.firewall import assert_license_class, split_by_license
from data_adapters.schema import CorpusJoke


def _joke(license_class, source_id="j1", source_dataset="fixture"):
    return CorpusJoke(text="a joke", source_dataset=source_dataset,
                      license_class=license_class, source_id=source_id)


class TestAssertLicenseClass(unittest.TestCase):
    def test_no_allowed_arg_raises(self):
        with self.assertRaises(ValueError):
            assert_license_class([_joke("commercial_safe")], allowed=[])

    def test_unknown_license_class_in_allowed_raises(self):
        with self.assertRaises(ValueError):
            assert_license_class([_joke("commercial_safe")],
                                allowed=["commercial_safe", "totally_open"])

    def test_all_commercial_safe_passes(self):
        records = [_joke("commercial_safe"), _joke("commercial_safe")]
        out = assert_license_class(records, allowed=["commercial_safe"])
        self.assertEqual(len(out), 2)

    def test_research_only_record_blocked_by_commercial_only_allow(self):
        records = [_joke("commercial_safe"), _joke("research_only")]
        with self.assertRaises(ValueError) as ctx:
            assert_license_class(records, allowed=["commercial_safe"])
        # error names the offending record's provenance
        self.assertIn("j1", str(ctx.exception))

    def test_research_only_passes_when_explicitly_allowed(self):
        records = [_joke("research_only")]
        out = assert_license_class(records, allowed=["research_only"])
        self.assertEqual(len(out), 1)

    def test_consumes_generator_into_list(self):
        gen = (_joke("commercial_safe") for _ in range(3))
        out = assert_license_class(gen, allowed=["commercial_safe"])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)


class TestSplitByLicense(unittest.TestCase):
    def test_splits_into_both_buckets(self):
        records = [_joke("commercial_safe", "a"), _joke("research_only", "b"),
                  _joke("commercial_safe", "c")]
        buckets = split_by_license(records)
        self.assertEqual(len(buckets["commercial_safe"]), 2)
        self.assertEqual(len(buckets["research_only"]), 1)

    def test_never_raises_on_empty_input(self):
        self.assertEqual(split_by_license([]), {})


if __name__ == "__main__":
    unittest.main()

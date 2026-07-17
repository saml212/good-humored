"""Unit tests for the field-coverage checker (benchmark/field_coverage.py).

Updated for the adversarial audit's two-tier redesign: labelers now
return (label, tier) pairs, and the report has no baked-in pass/fail --
it reports escape_rate (free-tier fraction), canon_rate, and
unparseable_rate, each with a per-tier label histogram.

No network, ever: every `complete` here is a scripted fake, same
discipline as test_relabel.py / test_rejector_v4.py. All directory
fixtures are written to a tempfile.TemporaryDirectory, never under
experiment-runs/.

Run: python3 -m pytest benchmark/tests/test_field_coverage.py -q
"""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark import field_coverage
from benchmark.relabel import LabelCache
from benchmark.rejector import TIER_CANON, TIER_FREE, TIER_UNPARSEABLE, UNPARSEABLE


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _raw_rec(run_id, turn, joke, topic="placeholder"):
    return {"run_id": run_id, "turn": turn, "joke": joke, "topic": topic,
            "refusal": False, "ts": float(turn)}


def _relabel_rec(run_id, turn, joke, labeler, label, topic_v2="orig"):
    return {"run_id": run_id, "turn": turn, "joke": joke,
            "topic_v2": topic_v2, ("topic_%s" % labeler): label,
            "refusal": False, "ts": float(turn), "relabeled_with": labeler}


class FakeLabelerFn:
    """label_topic_v4-shaped fake: looks the joke up in a dict of
    (label, tier) pairs, falls back to a free-tier "unmatched" label for
    anything unscripted so a typo fails loudly rather than silently
    matching nothing."""

    def __init__(self, joke_to_result):
        self.joke_to_result = joke_to_result
        self.calls = []

    def __call__(self, joke, complete):
        self.calls.append(joke)
        return self.joke_to_result.get(joke, ("unmatched", TIER_FREE))


class FieldCoverageTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------- infer_tier


class TestInferTier(unittest.TestCase):
    def test_v4_canon_and_free(self):
        self.assertEqual(field_coverage.infer_tier("v4", "egg"), TIER_CANON)
        self.assertEqual(field_coverage.infer_tier("v4", "ghost"), TIER_FREE)

    def test_v3_other_is_canon_not_free(self):
        # v3 has no free tier at all -- `other` is still a v3 vocabulary
        # entry, so it must report as canon (this is what makes the old
        # catch-all failure visible in histogram_canon rather than
        # disappearing).
        self.assertEqual(field_coverage.infer_tier("v3", "other"), TIER_CANON)

    def test_v2_always_free_unless_unparseable(self):
        self.assertEqual(field_coverage.infer_tier("v2", "anything"), TIER_FREE)
        self.assertEqual(field_coverage.infer_tier("v2", UNPARSEABLE),
                         TIER_UNPARSEABLE)

    def test_unparseable_always_its_own_tier(self):
        for labeler in ("v2", "v3", "v4"):
            self.assertEqual(field_coverage.infer_tier(labeler, UNPARSEABLE),
                             TIER_UNPARSEABLE)

    def test_none_label_is_none_tier(self):
        self.assertIsNone(field_coverage.infer_tier("v4", None))


# ---------------------------------------------------------- file discovery


class TestFindTurnFiles(FieldCoverageTestCase):
    def test_find_raw_turn_files_excludes_relabel_files(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-x-r00.jsonl", [_raw_rec("x-r00", 0, "j1")])
        _write_jsonl(lane / "turns-x-r00.relabel-v3.jsonl",
                    [_relabel_rec("x-r00", 0, "j1", "v3", "cat")])
        raw = field_coverage.find_raw_turn_files(self.root)
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0].name, "turns-x-r00.jsonl")

    def test_find_relabel_files_narrows_to_exact_labeler(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-x-r00.relabel-v3.jsonl",
                    [_relabel_rec("x-r00", 0, "j1", "v3", "cat")])
        _write_jsonl(lane / "turns-x-r00.relabel-v4.jsonl",
                    [_relabel_rec("x-r00", 0, "j1", "v4", "cat")])
        v3_only = field_coverage.find_relabel_files(self.root, "v3")
        self.assertEqual([p.name for p in v3_only],
                         ["turns-x-r00.relabel-v3.jsonl"])

    def test_find_relabel_files_any_labeler(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-x-r00.relabel-v3.jsonl",
                    [_relabel_rec("x-r00", 0, "j1", "v3", "cat")])
        anyl = field_coverage.find_relabel_files(self.root)
        self.assertEqual(len(anyl), 1)


# ---------------------------------------------------------- collect_records


class TestCollectRecords(FieldCoverageTestCase):
    def test_exact_relabel_files_used_when_present_tier_derived(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-x-r00.relabel-v4.jsonl", [
            _relabel_rec("x-r00", 0, "egg joke", "v4", "egg"),
            _relabel_rec("x-r00", 1, "mystery joke", "v4", "ghost"),
        ])
        records = field_coverage.collect_records(self.root, "v4")
        self.assertEqual(len(records), 2)
        by_joke = {r["joke"]: r for r in records}
        self.assertEqual(by_joke["egg joke"]["label"], "egg")
        self.assertEqual(by_joke["egg joke"]["tier"], TIER_CANON)
        self.assertEqual(by_joke["mystery joke"]["label"], "ghost")
        self.assertEqual(by_joke["mystery joke"]["tier"], TIER_FREE)
        self.assertEqual(records[0]["model"], "x")

    def test_exact_relabel_file_v3_other_derives_canon_tier(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-x-r00.relabel-v3.jsonl",
                    [_relabel_rec("x-r00", 0, "j1", "v3", "other")])
        records = field_coverage.collect_records(self.root, "v3")
        self.assertEqual(records[0]["label"], "other")
        self.assertEqual(records[0]["tier"], TIER_CANON)

    def test_falls_back_to_raw_pilot_files_when_no_matching_relabel(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-x-r00.jsonl",
                    [_raw_rec("x-r00", 0, "egg joke")])
        records = field_coverage.collect_records(self.root, "v4")
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["label"])
        self.assertIsNone(records[0]["tier"])
        self.assertEqual(records[0]["joke"], "egg joke")

    def test_falls_back_to_other_labelers_relabel_files_as_joke_source(self):
        # No v4 relabel files and no raw pilot files -- only a v3
        # relabel dir. The joke text must still be recoverable; the
        # label must NOT be (that would silently use the wrong labeler's
        # answer for a field-coverage report about a different labeler).
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-x-r00.relabel-v3.jsonl",
                    [_relabel_rec("x-r00", 0, "egg joke", "v3", "other")])
        records = field_coverage.collect_records(self.root, "v4")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["joke"], "egg joke")
        self.assertIsNone(records[0]["label"],
                          "must not reuse v3's label as if it were v4's")
        self.assertIsNone(records[0]["tier"])

    def test_raises_when_nothing_found(self):
        (self.root / "lane-a").mkdir(parents=True)
        with self.assertRaises(SystemExit):
            field_coverage.collect_records(self.root, "v4")

    def test_malformed_run_id_does_not_crash_model_attribution(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-weird.jsonl",
                    [_raw_rec("no-run-index-suffix", 0, "j1")])
        records = field_coverage.collect_records(self.root, "v4")
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["model"])


# ---------------------------------------------------------- resolve_labels


class TestResolveLabels(FieldCoverageTestCase):
    def _records(self, jokes):
        return [{"run_id": "m-r00", "model": "m", "turn": i, "joke": j,
                "label": None, "tier": None, "source_file": "x"}
               for i, j in enumerate(jokes)]

    def test_cache_hit_resolves_and_derives_tier_without_calling_labeler(self):
        cache = LabelCache(self.root / "cache.jsonl")
        cache.put("v4", "egg joke", "egg")
        records = self._records(["egg joke"])
        fake_fn = FakeLabelerFn({})
        stats = {}
        field_coverage.resolve_labels(records, "v4", fake_fn, None, cache,
                                      dry_run=True, stats=stats)
        self.assertEqual(records[0]["label"], "egg")
        self.assertEqual(records[0]["tier"], TIER_CANON)
        self.assertEqual(fake_fn.calls, [])
        self.assertEqual(stats["cache_hits"], 1)
        cache.close()

    def test_cache_hit_free_tier_label_derives_free(self):
        cache = LabelCache(self.root / "cache.jsonl")
        cache.put("v4", "ghost joke", "ghost")
        records = self._records(["ghost joke"])
        fake_fn = FakeLabelerFn({})
        stats = {}
        field_coverage.resolve_labels(records, "v4", fake_fn, None, cache,
                                      dry_run=True, stats=stats)
        self.assertEqual(records[0]["label"], "ghost")
        self.assertEqual(records[0]["tier"], TIER_FREE)
        cache.close()

    def test_dry_run_cache_miss_is_missing_not_a_call(self):
        cache = LabelCache(self.root / "cache.jsonl")
        records = self._records(["unknown joke"])
        fake_fn = FakeLabelerFn({})
        stats = {}
        field_coverage.resolve_labels(records, "v4", fake_fn, None, cache,
                                      dry_run=True, stats=stats)
        self.assertIsNone(records[0]["label"])
        self.assertIsNone(records[0]["tier"])
        self.assertEqual(fake_fn.calls, [])
        self.assertEqual(stats["missing"], 1)
        cache.close()

    def test_non_dry_run_cache_miss_calls_labeler_and_caches(self):
        cache = LabelCache(self.root / "cache.jsonl")
        records = self._records(["egg joke"])
        fake_fn = FakeLabelerFn({"egg joke": ("egg", TIER_CANON)})
        stats = {}
        field_coverage.resolve_labels(records, "v4", fake_fn,
                                      complete=lambda p: "unused", cache=cache,
                                      dry_run=False, stats=stats)
        self.assertEqual(records[0]["label"], "egg")
        self.assertEqual(records[0]["tier"], TIER_CANON)
        self.assertEqual(fake_fn.calls, ["egg joke"])
        self.assertEqual(stats["calls"], 1)
        cache.close()

        # Second resolve_labels call against a fresh LabelCache instance
        # reading the same file must hit the cache, not call again --
        # only the label is persisted; tier is re-derived.
        cache2 = LabelCache(self.root / "cache.jsonl")
        records2 = self._records(["egg joke"])
        fake_fn2 = FakeLabelerFn({"egg joke": ("egg", TIER_CANON)})
        stats2 = {}
        field_coverage.resolve_labels(records2, "v4", fake_fn2, None, cache2,
                                      dry_run=True, stats=stats2)
        self.assertEqual(records2[0]["label"], "egg")
        self.assertEqual(records2[0]["tier"], TIER_CANON)
        self.assertEqual(fake_fn2.calls, [])
        self.assertEqual(stats2["cache_hits"], 1)
        cache2.close()

    def test_already_known_label_is_never_touched(self):
        cache = LabelCache(self.root / "cache.jsonl")
        records = self._records(["egg joke"])
        records[0]["label"] = "egg"  # as if read from an exact relabel file
        records[0]["tier"] = TIER_CANON
        fake_fn = FakeLabelerFn({})
        stats = {}
        field_coverage.resolve_labels(records, "v4", fake_fn, None, cache,
                                      dry_run=True, stats=stats)
        self.assertEqual(records[0]["label"], "egg")
        self.assertEqual(records[0]["tier"], TIER_CANON)
        self.assertEqual(fake_fn.calls, [])
        self.assertEqual(stats, {})
        cache.close()


# ---------------------------------------------------------- build_report


class TestBuildReport(unittest.TestCase):
    def _rec(self, label, tier, model="m"):
        return {"run_id": "%s-r00" % model, "model": model, "turn": 0,
               "joke": "j", "label": label, "tier": tier, "source_file": "x"}

    def test_histogram_and_rates(self):
        records = ([self._rec("other", TIER_CANON)] * 4
                  + [self._rec("cat", TIER_CANON)] * 4
                  + [self._rec("ghost", TIER_FREE)] * 2
                  + [self._rec(UNPARSEABLE, TIER_UNPARSEABLE)] * 2)
        report = field_coverage.build_report(records)
        self.assertEqual(report["total_turns"], 12)
        self.assertEqual(report["labeled_turns"], 12)
        self.assertEqual(report["missing_turns"], 0)
        self.assertEqual(report["canon_count"], 8)
        self.assertAlmostEqual(report["canon_rate"], 8 / 12)
        self.assertEqual(report["escape_count"], 2)
        self.assertAlmostEqual(report["escape_rate"], 2 / 12)
        self.assertEqual(report["unparseable_count"], 2)
        self.assertAlmostEqual(report["unparseable_rate"], 2 / 12)
        self.assertEqual(report["histogram_canon"]["other"], 4)
        self.assertEqual(report["histogram_canon"]["cat"], 4)
        self.assertEqual(report["histogram_free"]["ghost"], 2)
        self.assertNotIn("other", report["histogram_free"])

    def test_no_pass_fail_fields_present(self):
        # Audit: "no pass/fail baked in -- EXP-010's bars live in the
        # experiment, not the tool."
        report = field_coverage.build_report([self._rec("cat", TIER_CANON)])
        for stale_key in ("field_bar_pass", "field_bar_metric",
                         "field_bar_threshold", "field_bar_note",
                         "catch_all_count", "catch_all_rate"):
            self.assertNotIn(stale_key, report)

    def test_empty_records_all_none_not_crash(self):
        report = field_coverage.build_report([])
        self.assertEqual(report["total_turns"], 0)
        self.assertIsNone(report["canon_rate"])
        self.assertIsNone(report["escape_rate"])
        self.assertIsNone(report["unparseable_rate"])

    def test_missing_turns_excluded_from_rates(self):
        records = ([self._rec("cat", TIER_CANON)] * 3
                  + [self._rec(None, None)] * 2)
        report = field_coverage.build_report(records)
        self.assertEqual(report["total_turns"], 5)
        self.assertEqual(report["labeled_turns"], 3)
        self.assertEqual(report["missing_turns"], 2)
        self.assertAlmostEqual(report["canon_rate"], 1.0)

    def test_per_model_breakdown(self):
        records = ([self._rec("ghost", TIER_FREE, model="a")] * 3
                  + [self._rec("cat", TIER_CANON, model="a")]
                  + [self._rec("cat", TIER_CANON, model="b")] * 4)
        report = field_coverage.build_report(records)
        self.assertEqual(report["per_model"]["a"]["n"], 4)
        self.assertAlmostEqual(report["per_model"]["a"]["escape_rate"], 0.75)
        self.assertAlmostEqual(report["per_model"]["a"]["canon_rate"], 0.25)
        self.assertEqual(report["per_model"]["b"]["n"], 4)
        self.assertAlmostEqual(report["per_model"]["b"]["escape_rate"], 0.0)
        self.assertAlmostEqual(report["per_model"]["b"]["canon_rate"], 1.0)


# ---------------------------------------------------- end-to-end (fakes only)


class TestEndToEndWithFakes(FieldCoverageTestCase):
    def test_full_pipeline_pilot_dir_non_dry_run(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-m-r00.jsonl", [
            _raw_rec("m-r00", 0, "egg joke"),
            _raw_rec("m-r00", 1, "mystery joke"),
        ])
        records = field_coverage.collect_records(self.root, "v4")
        cache = LabelCache(self.root / "cache.jsonl")
        fake_fn = FakeLabelerFn({"egg joke": ("egg", TIER_CANON),
                                 "mystery joke": ("ghost", TIER_FREE)})
        stats = {}
        field_coverage.resolve_labels(records, "v4", fake_fn,
                                      complete=lambda p: "unused", cache=cache,
                                      dry_run=False, stats=stats)
        cache.close()
        report = field_coverage.build_report(records)
        self.assertEqual(report["labeled_turns"], 2)
        self.assertEqual(report["escape_count"], 1)
        self.assertEqual(report["canon_count"], 1)
        self.assertEqual(stats["calls"], 2)

    def test_full_pipeline_relabel_dir_dry_run_zero_calls(self):
        lane = self.root / "lane-a"
        _write_jsonl(lane / "turns-m-r00.relabel-v3.jsonl", [
            _relabel_rec("m-r00", 0, "egg joke", "v3", "other"),
            _relabel_rec("m-r00", 1, "cat joke", "v3", "cat"),
        ])
        records = field_coverage.collect_records(self.root, "v3")
        cache = LabelCache(self.root / "cache.jsonl")
        fake_fn = FakeLabelerFn({})  # must never be called
        stats = {}
        field_coverage.resolve_labels(records, "v3", fake_fn, None, cache,
                                      dry_run=True, stats=stats)
        cache.close()
        report = field_coverage.build_report(records)
        self.assertEqual(report["missing_turns"], 0)
        self.assertEqual(fake_fn.calls, [])
        # v3 has no free tier -- `other` reports as canon, escape is 0.
        self.assertEqual(report["escape_rate"], 0.0)
        self.assertAlmostEqual(report["canon_rate"], 1.0)
        self.assertEqual(report["histogram_canon"]["other"], 1)


if __name__ == "__main__":
    unittest.main()

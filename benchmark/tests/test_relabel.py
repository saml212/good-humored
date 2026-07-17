"""Unit tests for the instrument-robustness relabeler (benchmark/relabel.py).

No network, ever: every `complete` callable here is a scripted fake (same
discipline as test_rejector.py's FakeComplete / test_run_pilot.py's
make_fake_model) -- label_topic_v3 itself is real (it is pure Python, no
network, and already validated live by EXP-008), only the provider
callable underneath it is faked. Covers, per the build spec:
  - cache-hit path (identical joke text -> one call, not two)
  - resume-from-cache path (interrupted run: zero re-spend)
  - agreement-report math (rate, confusion pairs, per-model breakdown)
  - metric-structure parity with run_pilot.py's own summary.json

Run: python3 -m pytest benchmark/tests/test_relabel.py -q
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark import relabel, run_pilot
from benchmark.rejector import label_topic_v3

JOKE_TAG_RE = re.compile(r"<joke>(.*?)</joke>", re.DOTALL)


def make_fake_complete(joke_to_label, call_log=None):
    """A fake `complete` for label_topic_v3: looks the joke text (pulled
    out of the real LABEL_PROMPT_V3-rendered prompt) up in a dict the
    test controls, rather than depending on any wording convention.
    Returns "other" (the vocabulary's mandatory escape hatch) for an
    unrecognized joke so a test typo fails loudly via a wrong label
    rather than an opaque retry loop.

    LABEL_PROMPT_V3 has its own three `<joke>...</joke>` few-shot
    examples baked in ahead of the real one -- the joke under test is
    always the LAST `<joke>` tag in the rendered prompt, not the first.
    """
    def complete(prompt):
        if call_log is not None:
            call_log.append(prompt)
        matches = JOKE_TAG_RE.findall(prompt)
        joke = matches[-1] if matches else ""
        return joke_to_label.get(joke, "other")
    return complete


def _turn_rec(run_id, turn, joke, topic, refusal=False, ts=0.0):
    return {"run_id": run_id, "turn": turn, "joke": joke, "topic": topic,
            "refusal": refusal, "ts": ts}


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- model_of


class TestModelOf(unittest.TestCase):
    def test_bare_alias(self):
        self.assertEqual(relabel.model_of("haiku-r00"), "haiku")

    def test_colon_prefixed(self):
        self.assertEqual(relabel.model_of("api:deepseek-r03"), "api:deepseek")

    def test_dotted_alias(self):
        self.assertEqual(relabel.model_of("codex:5.4-r02"), "codex:5.4")

    def test_double_digit_run_index(self):
        self.assertEqual(relabel.model_of("api:kimi-r12"), "api:kimi")

    def test_malformed_raises(self):
        with self.assertRaises(ValueError):
            relabel.model_of("no-run-suffix-here")


class TestJokeHash(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(relabel.joke_hash("abc"), relabel.joke_hash("abc"))

    def test_distinct_for_distinct_text(self):
        self.assertNotEqual(relabel.joke_hash("abc"), relabel.joke_hash("abd"))

    def test_sensitive_to_exact_text_not_normalized(self):
        # a rewording (even trivial casing) must NOT hash the same --
        # only byte-identical repeats are the degradation signal this
        # cache exists to dedupe; a near-miss must still get its own call.
        self.assertNotEqual(relabel.joke_hash("Cats."),
                            relabel.joke_hash("cats."))


# --------------------------------------------------------------- LabelCache


class TestLabelCache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cache-test-"))
        self.cache_path = self.tmp / "cache.jsonl"

    def test_miss_then_hit(self):
        cache = relabel.LabelCache(self.cache_path)
        self.assertIsNone(cache.get("v3", "joke A"))
        cache.put("v3", "joke A", "cat")
        self.assertEqual(cache.get("v3", "joke A"), "cat")
        cache.close()

    def test_persists_across_instances(self):
        cache = relabel.LabelCache(self.cache_path)
        cache.put("v3", "joke A", "cat")
        cache.close()
        cache2 = relabel.LabelCache(self.cache_path)
        self.assertEqual(cache2.get("v3", "joke A"), "cat")
        self.assertEqual(len(cache2), 1)

    def test_keyed_by_labeler_no_collision(self):
        cache = relabel.LabelCache(self.cache_path)
        cache.put("v2", "joke A", "kitty")
        cache.put("v3", "joke A", "cat")
        self.assertEqual(cache.get("v2", "joke A"), "kitty")
        self.assertEqual(cache.get("v3", "joke A"), "cat")
        self.assertEqual(len(cache), 2)
        cache.close()

    def test_flushes_immediately(self):
        # A reader opening the file WHILE the writer instance is still
        # open must see the entry -- proves put() flushes rather than
        # buffering until close() (the crash-safety this class exists
        # for).
        cache = relabel.LabelCache(self.cache_path)
        cache.put("v3", "joke A", "cat")
        with open(self.cache_path) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        cache.close()


# ------------------------------------------------------------- label_cached


class TestLabelCachedCacheHit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="labelcached-test-"))
        self.cache = relabel.LabelCache(self.tmp / "cache.jsonl")
        self.stats = {}

    def tearDown(self):
        self.cache.close()

    def test_cache_miss_calls_labeler(self):
        calls = []
        complete = make_fake_complete({"joke text": "cat"}, calls)
        label = relabel.label_cached("joke text", "v3", label_topic_v3,
                                     complete, self.cache, self.stats)
        self.assertEqual(label, "cat")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.stats["calls"], 1)
        self.assertEqual(self.stats.get("cache_hits", 0), 0)

    def test_cache_hit_does_not_call_labeler_again(self):
        calls = []
        complete = make_fake_complete({"joke text": "cat"}, calls)
        relabel.label_cached("joke text", "v3", label_topic_v3, complete,
                             self.cache, self.stats)
        label2 = relabel.label_cached("joke text", "v3", label_topic_v3,
                                      complete, self.cache, self.stats)
        self.assertEqual(label2, "cat")
        self.assertEqual(len(calls), 1, "second call for the identical "
                         "joke must be served from cache, not re-call "
                         "the labeler")
        self.assertEqual(self.stats["cache_hits"], 1)

    def test_repeats_label_identically(self):
        # The exact scenario the build spec calls out: identical jokes
        # (the degradation signal -- repeats ARE the point) must label
        # identically across every occurrence.
        complete = make_fake_complete({"same joke": "travel"})
        labels = [relabel.label_cached("same joke", "v3", label_topic_v3,
                                       complete, self.cache, self.stats)
                  for _ in range(5)]
        self.assertEqual(set(labels), {"travel"})
        self.assertEqual(self.stats["calls"], 1)
        self.assertEqual(self.stats["cache_hits"], 4)


# ------------------------------------------------------------ build_output_record


class TestBuildOutputRecord(unittest.TestCase):
    def test_v3_default_naming(self):
        rec = {"topic": "kitty", "run_id": "x-r00", "turn": 0}
        out = relabel.build_output_record(rec, "v3", "cat")
        self.assertEqual(out["topic_v2"], "kitty")
        self.assertEqual(out["topic_v3"], "cat")
        self.assertNotIn("topic", out)

    def test_v2_relabel_does_not_collide_with_original(self):
        # hypothetical future v2-vs-v2 A/B: the new label's field name
        # would otherwise collide with ORIGINAL_LABEL_FIELD.
        rec = {"topic": "kitty", "run_id": "x-r00", "turn": 0}
        out = relabel.build_output_record(rec, "v2", "cat")
        self.assertEqual(out["topic_v2_original"], "kitty")
        self.assertEqual(out["topic_v2"], "cat")

    def test_does_not_mutate_input(self):
        rec = {"topic": "kitty", "run_id": "x-r00", "turn": 0}
        relabel.build_output_record(rec, "v3", "cat")
        self.assertIn("topic", rec)  # original dict untouched


# ------------------------------------------------------------------ relabel_run


class TestRelabelRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="relabelrun-test-"))
        self.cache = relabel.LabelCache(self.tmp / "cache.jsonl")
        self.stats = {}

    def tearDown(self):
        self.cache.close()

    def _write(self, name, records):
        p = self.tmp / name
        _write_jsonl(p, records)
        return p

    def test_writes_topic_v2_and_topic_v3(self):
        turns = self._write("turns-modelA-r00.jsonl", [
            _turn_rec("modelA-r00", 0, "j0", "kitty")])
        out_path = self.tmp / "out.jsonl"
        complete = make_fake_complete({"j0": "cat"})
        path = relabel.relabel_run(turns, out_path, "v3", label_topic_v3,
                                   complete, self.cache, self.stats)
        self.assertEqual(path, ["cat"])
        with open(out_path) as f:
            rec = json.loads(f.readline())
        self.assertEqual(rec["topic_v2"], "kitty")
        self.assertEqual(rec["topic_v3"], "cat")
        self.assertNotIn("topic", rec)
        self.assertEqual(rec["run_id"], "modelA-r00")  # original fields kept

    def test_sorts_by_turn_even_if_file_out_of_order(self):
        turns = self._write("turns-modelA-r00.jsonl", [
            _turn_rec("modelA-r00", 1, "j1", "x"),
            _turn_rec("modelA-r00", 0, "j0", "y"),
        ])
        out_path = self.tmp / "out.jsonl"
        complete = make_fake_complete({"j0": "cat", "j1": "dog"})
        path = relabel.relabel_run(turns, out_path, "v3", label_topic_v3,
                                   complete, self.cache, self.stats)
        self.assertEqual(path, ["cat", "dog"])

    def test_skips_records_with_no_joke(self):
        turns = self._write("turns-modelA-r00.jsonl", [
            _turn_rec("modelA-r00", 0, "j0", "y"),
            {"run_id": "modelA-r00", "turn": 1, "topic": "y",
             "refusal": False, "ts": 1.0},  # malformed: no joke text
        ])
        out_path = self.tmp / "out.jsonl"
        complete = make_fake_complete({"j0": "cat"})
        path = relabel.relabel_run(turns, out_path, "v3", label_topic_v3,
                                   complete, self.cache, self.stats)
        self.assertEqual(path, ["cat"])

    def test_dedupes_repeated_joke_within_one_run(self):
        turns = self._write("turns-modelA-r00.jsonl", [
            _turn_rec("modelA-r00", 0, "same", "a"),
            _turn_rec("modelA-r00", 1, "same", "b"),
        ])
        out_path = self.tmp / "out.jsonl"
        calls = []
        complete = make_fake_complete({"same": "travel"}, calls)
        path = relabel.relabel_run(turns, out_path, "v3", label_topic_v3,
                                   complete, self.cache, self.stats)
        self.assertEqual(path, ["travel", "travel"])
        self.assertEqual(len(calls), 1, "identical joke text repeated "
                         "within a run must be a single labeler call")


class TestAlreadyDone(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="alreadydone-test-"))

    def _write(self, name, n_lines):
        p = self.tmp / name
        with open(p, "w") as f:
            for i in range(n_lines):
                f.write(json.dumps({"i": i}) + "\n")
        return p

    def test_false_when_out_missing(self):
        turns = self._write("in.jsonl", 3)
        self.assertFalse(
            relabel.already_done(turns, self.tmp / "missing.jsonl"))

    def test_false_when_partial(self):
        turns = self._write("in.jsonl", 3)
        out = self._write("out.jsonl", 2)
        self.assertFalse(relabel.already_done(turns, out))

    def test_true_when_complete(self):
        turns = self._write("in.jsonl", 3)
        out = self._write("out.jsonl", 3)
        self.assertTrue(relabel.already_done(turns, out))

    def test_false_when_input_empty(self):
        turns = self._write("in.jsonl", 0)
        out = self._write("out.jsonl", 0)
        self.assertFalse(relabel.already_done(turns, out))


# -------------------------------------------------------------- process_lane


class TestProcessLaneFailureFencing(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lane-test-"))
        self.lane = self.tmp / "lane-x"
        self.lane.mkdir()
        self.out_dir = self.tmp / "out"
        self.cache = relabel.LabelCache(self.out_dir / "cache.jsonl")
        self.stats = {}

    def tearDown(self):
        self.cache.close()

    def _write(self, name, records):
        _write_jsonl(self.lane / name, records)

    def test_one_bad_run_does_not_lose_others(self):
        self._write("turns-good-r00.jsonl",
                    [_turn_rec("good-r00", 0, "j-good", "x")])
        self._write("turns-good-r01.jsonl",
                    [_turn_rec("good-r01", 0, "j-good-2", "x")])
        self._write("turns-bad-r00.jsonl",
                    [_turn_rec("bad-r00", 0, "j-bad", "x")])

        def raising_labeler(joke, complete):
            if joke == "j-bad":
                raise RuntimeError("boom")
            return "cat"

        failures = []
        paths = relabel.process_lane(
            self.lane, self.out_dir, "v3", raising_labeler, lambda p: "cat",
            self.cache, self.stats, failures)
        self.assertEqual(len(failures), 1)
        self.assertIn("bad", failures[0]["run_id"])
        self.assertIn("good", paths)
        self.assertEqual(len(paths["good"]), 2)


# ---------------------------------------------------- build_summary parity


class TestBuildSummaryParityWithRunPilot(unittest.TestCase):
    def test_per_model_and_cross_model_match_run_pilot_exactly(self):
        """Same underlying path data through BOTH run_pilot's own summary
        construction (a real run_pilot.main() sweep against fake
        providers) and relabel.build_summary. If this tool truly reuses
        path_divergence / depth_to_degradation / cross_model_overlap /
        LabelSpace rather than reimplementing them, the two summaries'
        per_model and cross_model sections must come out identical over
        identical input paths."""
        labels_a = ["cat", "dog", "coffee"]
        labels_b = ["dog", "cat", "coffee"]

        def make_model(labels):
            state = {"i": 0}

            def complete(prompt):
                i = state["i"]
                state["i"] += 1
                return "A joke about %s." % labels[i % len(labels)]
            return complete

        def fake_rejector(prompt):
            m = re.search(r"about ([a-z]+)", prompt)
            return m.group(1) if m else "unknown"

        rp_out = Path(tempfile.mkdtemp(prefix="rp-parity-test-"))
        argv = ["run_pilot", "--models", "a,b", "--runs", "2", "--depth", "3",
                "--rejector", "fakerej", "--out", str(rp_out)]
        providers = {"a": make_model(labels_a), "b": make_model(labels_b),
                    "fakerej": fake_rejector}

        def fake_get_provider(s, temperature=None):
            return providers[s]

        with mock.patch.object(run_pilot, "get_provider",
                               side_effect=fake_get_provider), \
             mock.patch.object(sys, "argv", argv):
            run_pilot.main()
        with open(rp_out / "summary.json") as f:
            rp_summary = json.load(f)

        paths = {m: pm["paths"] for m, pm in rp_summary["per_model"].items()}
        summary = relabel.build_summary(paths, [], "v3", "v3-test",
                                        "fake:provider", "lane-test")

        self.assertEqual(summary["per_model"], rp_summary["per_model"])
        self.assertIn("cross_model", summary)
        self.assertEqual(summary["cross_model"], rp_summary["cross_model"])
        self.assertEqual(summary["cross_model_semantic"],
                         rp_summary["cross_model_semantic"])


# -------------------------------------------------------- instrument agreement


class TestBuildAgreementReport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agree-test-"))
        self.lane = self.tmp / "lane-x"
        self.lane.mkdir()

    def _write_relabeled(self, name, records):
        _write_jsonl(self.lane / name, records)

    def test_overall_rate_and_counts(self):
        self._write_relabeled("turns-modelA-r00.relabel-v3.jsonl", [
            {"run_id": "modelA-r00", "turn": 0, "topic_v2": "cat",
             "topic_v3": "cat"},
            {"run_id": "modelA-r00", "turn": 1, "topic_v2": "Coffee!",
             "topic_v3": "coffee"},  # agree after normalize_label
            {"run_id": "modelA-r00", "turn": 2, "topic_v2": "puppy",
             "topic_v3": "dog"},  # disagree
        ])
        report = relabel.build_agreement_report(self.tmp, [self.lane], "v3")
        self.assertEqual(report["overall"]["n_turns"], 3)
        self.assertEqual(report["overall"]["n_agree"], 2)
        self.assertAlmostEqual(report["overall"]["agreement_rate"], 2 / 3)

    def test_confusion_pairs_sorted_by_frequency(self):
        recs = ([{"run_id": "modelA-r00", "turn": i, "topic_v2": "puppy",
                  "topic_v3": "dog"} for i in range(3)]
               + [{"run_id": "modelA-r00", "turn": 3, "topic_v2": "kitty",
                   "topic_v3": "cat"}])
        self._write_relabeled("turns-modelA-r00.relabel-v3.jsonl", recs)
        report = relabel.build_agreement_report(self.tmp, [self.lane], "v3")
        pairs = report["confusion_pairs"]
        self.assertEqual(pairs[0]["topic_v2"], "puppy")
        self.assertEqual(pairs[0]["topic_v3"], "dog")
        self.assertEqual(pairs[0]["count"], 3)
        self.assertEqual(pairs[1]["count"], 1)

    def test_per_model_breakdown(self):
        self._write_relabeled("turns-modelA-r00.relabel-v3.jsonl", [
            {"run_id": "modelA-r00", "turn": 0, "topic_v2": "cat",
             "topic_v3": "cat"}])
        self._write_relabeled("turns-modelB-r00.relabel-v3.jsonl", [
            {"run_id": "modelB-r00", "turn": 0, "topic_v2": "puppy",
             "topic_v3": "dog"}])
        report = relabel.build_agreement_report(self.tmp, [self.lane], "v3")
        self.assertEqual(report["per_model"]["modelA"]["agreement_rate"], 1.0)
        self.assertEqual(report["per_model"]["modelB"]["agreement_rate"], 0.0)

    def test_empty_report_has_none_rate_not_zero_division(self):
        report = relabel.build_agreement_report(self.tmp, [self.lane], "v3")
        self.assertEqual(report["overall"]["n_turns"], 0)
        self.assertIsNone(report["overall"]["agreement_rate"])


# ------------------------------------------------------------- main() e2e


class TestMainEndToEnd(unittest.TestCase):
    def setUp(self):
        self.pilot = Path(tempfile.mkdtemp(prefix="pilot-fixture-"))
        self.out = Path(tempfile.mkdtemp(prefix="relabel-out-"))
        lane = self.pilot / "lane-x"
        lane.mkdir()
        # modelA: 2 runs sharing one joke verbatim (the dedup/cache case)
        # plus one distinct joke each.
        _write_jsonl(lane / "turns-modelA-r00.jsonl", [
            _turn_rec("modelA-r00", 0, "shared joke text", "kitty"),
            _turn_rec("modelA-r00", 1, "A only joke", "job"),
        ])
        _write_jsonl(lane / "turns-modelA-r01.jsonl", [
            _turn_rec("modelA-r01", 0, "shared joke text", "kitty"),
            _turn_rec("modelA-r01", 1, "another A joke", "trip"),
        ])
        # modelB: 2 runs, all-distinct jokes, for cross-model coverage.
        _write_jsonl(lane / "turns-modelB-r00.jsonl", [
            _turn_rec("modelB-r00", 0, "B joke one", "x"),
            _turn_rec("modelB-r00", 1, "B joke two", "y"),
        ])
        _write_jsonl(lane / "turns-modelB-r01.jsonl", [
            _turn_rec("modelB-r01", 0, "B joke three", "z"),
            _turn_rec("modelB-r01", 1, "B joke four", "w"),
        ])
        self.calls = []

    def _fake_complete_factory(self):
        # Deterministic v3 label per distinct joke, drawn from the REAL
        # vocabulary so label_topic_v3's membership check passes on the
        # first try (no retries -> exactly one call per unique joke).
        mapping = {
            "shared joke text": "cat",
            "A only joke": "work",
            "another A joke": "travel",
            "B joke one": "coffee",
            "B joke two": "dog",
            "B joke three": "bird",
            "B joke four": "fish",
        }
        return make_fake_complete(mapping, self.calls)

    def _run_main(self):
        argv = ["relabel", "--pilot", str(self.pilot), "--labeler", "v3",
                "--out", str(self.out)]
        fake_complete = self._fake_complete_factory()
        with mock.patch.object(relabel, "get_provider",
                               return_value=fake_complete), \
             mock.patch.object(sys, "argv", argv):
            relabel.main()

    def test_output_structure(self):
        self._run_main()
        lane_out = self.out / "lane-x"
        self.assertTrue((lane_out / "summary.json").exists())
        self.assertTrue(
            (lane_out / "turns-modelA-r00.relabel-v3.jsonl").exists())
        self.assertTrue((self.out / "label_cache.jsonl").exists())
        self.assertTrue((self.out / "instrument_agreement.json").exists())

        with open(lane_out / "turns-modelA-r00.relabel-v3.jsonl") as f:
            recs = [json.loads(l) for l in f]
        self.assertEqual(recs[0]["topic_v2"], "kitty")
        self.assertEqual(recs[0]["topic_v3"], "cat")
        self.assertEqual(recs[0]["run_id"], "modelA-r00")

    def test_cache_dedupes_repeated_joke_across_runs(self):
        self._run_main()
        # "shared joke text" appears in BOTH modelA-r00 and modelA-r01 --
        # must be exactly one call for it, not two.
        joke_calls = [c for c in self.calls if "shared joke text" in c]
        self.assertEqual(len(joke_calls), 1)
        # 7 distinct jokes total across both models -> 7 calls, no more.
        self.assertEqual(len(self.calls), 7)

    def test_resume_does_not_recall(self):
        self._run_main()
        self.assertEqual(len(self.calls), 7)
        self.calls.clear()
        self._run_main()  # same --out: everything already fully relabeled
        self.assertEqual(len(self.calls), 0,
                         "a fully-relabeled resume must make zero calls")

    def test_resume_after_partial_loss_only_recomputes_missing_files(self):
        # Simulate a crash that lost modelB's output files (but not the
        # cache, which is fsynced per-entry) and confirm resuming neither
        # re-calls the labeler (cache still has every label) nor leaves
        # the deleted files missing.
        self._run_main()
        (self.out / "lane-x" / "turns-modelB-r00.relabel-v3.jsonl").unlink()
        (self.out / "lane-x" / "turns-modelB-r01.relabel-v3.jsonl").unlink()
        self.calls.clear()
        self._run_main()
        self.assertEqual(len(self.calls), 0,
                         "every joke was already cached from the first "
                         "run -- rebuilding the missing files must not "
                         "re-spend any calls")
        self.assertTrue((self.out / "lane-x" /
                        "turns-modelB-r00.relabel-v3.jsonl").exists())

    def test_summary_has_cross_model_and_per_model(self):
        self._run_main()
        with open(self.out / "lane-x" / "summary.json") as f:
            summary = json.load(f)
        self.assertIn("modelA", summary["per_model"])
        self.assertIn("modelB", summary["per_model"])
        self.assertEqual(summary["per_model"]["modelA"]["paths"],
                         [["cat", "work"], ["cat", "travel"]])
        self.assertIn("cross_model", summary)

    def test_instrument_agreement_written(self):
        self._run_main()
        with open(self.out / "instrument_agreement.json") as f:
            agreement = json.load(f)
        self.assertEqual(agreement["overall"]["n_turns"], 8)
        # every v2 label in this fixture was deliberately chosen distinct
        # from its v3 counterpart -- zero agreement, by construction.
        self.assertEqual(agreement["overall"]["agreement_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

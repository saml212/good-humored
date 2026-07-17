"""Unit tests for benchmark/analyze_pilot.py's cross-lane pooling.

load_lanes() used to OVERWRITE paths[model] when the same model key
appeared in more than one lane's summary.json -- a real shape once
"-fill-" lanes exist (a lane that reruns just the runs that failed in the
main lane for a provider family, keyed by the same model string, e.g.
"api:glm" in both lane-api and lane-api-fill-glm). An overwrite silently
drops whichever lane's summary.json sorts first, losing completed runs
from the final analysis without any error. These tests lock in the fix:
same-model-across-lanes MERGES (concatenates) run lists instead.

Run: python3 -m pytest benchmark/tests/test_analyze_pilot.py -q
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark import analyze_pilot
from benchmark.metrics import path_divergence


def _write_summary(path: Path, per_model=None, failures=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"per_model": per_model or {}, "failures": failures or []}
    with open(path, "w") as f:
        json.dump(body, f)


class TestLoadLanesMerge(unittest.TestCase):
    def setUp(self):
        self.pilot = Path(tempfile.mkdtemp(prefix="analyze-pilot-test-"))

    def test_same_model_two_lanes_merges_runs(self):
        # lane-api: 2 surviving api:glm runs; lane-api-fill-glm: 2 more,
        # topping up the ones that failed in the main lane. Same model
        # key, two lanes -- must concatenate to 4, not overwrite to 2.
        run1, run2 = ["cat", "dog"], ["cat", "cat"]
        run3, run4 = ["dog", "dog"], ["cat", "dog"]
        _write_summary(
            self.pilot / "lane-api" / "summary.json",
            per_model={"api:glm": {"paths": [run1, run2]}},
            failures=[{"run_id": "api:glm-r00", "error": "boom"}])
        _write_summary(
            self.pilot / "lane-api-fill-glm" / "summary.json",
            per_model={"api:glm": {"paths": [run3, run4]}})

        paths, failures = analyze_pilot.load_lanes(self.pilot)

        self.assertEqual(paths["api:glm"], [run1, run2, run3, run4])
        self.assertEqual(len(paths["api:glm"]), 4)
        self.assertEqual(failures, [{"run_id": "api:glm-r00", "error": "boom"}])

    def test_divergence_hand_computed_over_union_of_merged_runs(self):
        """Hand-computed (not just re-derived) expected divergence over
        the 4-run union, so this test would fail if the merge dropped
        either lane's runs or double-counted one of them.

        Paths: r1=[cat,dog] r2=[cat,cat] r3=[dog,dog] r4=[cat,dog]
        (r1,r2 from lane-api; r3,r4 from the fill lane.)

        set_jaccard (6 pairs): {cat,dog}vs{cat}=.5, vs{dog}=.5,
          vs{cat,dog}=1.0, {cat}vs{dog}=0.0, {cat}vs{cat,dog}=.5,
          {dog}vs{cat,dog}=.5 -> mean = 3.0/6 = 0.5

        prefix_depth (6 pairs, order-sensitive leading match): (r1,r2)=1
          (cat matches, dog!=cat), (r1,r3)=0 (cat!=dog), (r1,r4)=2 (full
          match), (r2,r3)=0, (r2,r4)=1, (r3,r4)=0 -> mean = 4/6 = 0.6667

        norm_edit_distance (len 2 throughout, denom=2 every pair):
          (r1,r2)=1/2, (r1,r3)=1/2, (r1,r4)=0/2, (r2,r3)=2/2, (r2,r4)=1/2,
          (r3,r4)=1/2 -> mean = 3.0/6 = 0.5
        """
        run1, run2 = ["cat", "dog"], ["cat", "cat"]
        run3, run4 = ["dog", "dog"], ["cat", "dog"]
        _write_summary(self.pilot / "lane-api" / "summary.json",
                       per_model={"api:glm": {"paths": [run1, run2]}})
        _write_summary(self.pilot / "lane-api-fill-glm" / "summary.json",
                       per_model={"api:glm": {"paths": [run3, run4]}})

        paths, _ = analyze_pilot.load_lanes(self.pilot)
        div = path_divergence(paths["api:glm"])

        self.assertAlmostEqual(div["set_jaccard"], 0.5)
        self.assertAlmostEqual(div["prefix_depth"], 0.6666666666666666)
        self.assertAlmostEqual(div["norm_edit_distance"], 0.5)
        self.assertEqual(div["n_pairs"], 6.0)

    def test_zero_surviving_runs_in_one_lane_four_in_other_merges_to_four(self):
        # api:kimi: every run failed in the main lane (0 surviving --
        # doesn't even appear in lane-api's per_model, only in its
        # failures); the fill lane supplies all 4 successful runs. Merge
        # must land on exactly 4, not fewer (nothing to overwrite away)
        # and not more (no phantom runs invented).
        kimi_runs = [["a"], ["a"], ["b"], ["b"]]
        _write_summary(
            self.pilot / "lane-api" / "summary.json",
            per_model={},  # api:kimi has no surviving runs here
            failures=[{"run_id": "api:kimi-r%02d" % i, "error": "boom"}
                     for i in range(4)])
        _write_summary(
            self.pilot / "lane-api-fill-kimi" / "summary.json",
            per_model={"api:kimi": {"paths": kimi_runs}})

        paths, failures = analyze_pilot.load_lanes(self.pilot)

        self.assertEqual(len(paths["api:kimi"]), 4)
        self.assertEqual(paths["api:kimi"], kimi_runs)
        self.assertEqual(len(failures), 4)
        div = path_divergence(paths["api:kimi"])
        self.assertEqual(div["n_pairs"], 6.0)  # C(4,2) over ALL merged runs

    def test_no_collision_regression(self):
        # Disjoint model keys across lanes (the common case, no "-fill-"
        # lane involved) must still pool exactly as before -- each
        # model's run list is untouched by the merge machinery.
        _write_summary(
            self.pilot / "lane-claude" / "summary.json",
            per_model={"haiku": {"paths": [["work"], ["work"]]}})
        _write_summary(
            self.pilot / "lane-codex" / "summary.json",
            per_model={"codex:mini": {"paths": [["travel"], ["coffee"]]}})

        paths, failures = analyze_pilot.load_lanes(self.pilot)

        self.assertEqual(paths, {
            "haiku": [["work"], ["work"]],
            "codex:mini": [["travel"], ["coffee"]],
        })
        self.assertEqual(failures, [])

    def test_failures_pooled_across_lanes(self):
        _write_summary(self.pilot / "lane-a" / "summary.json",
                       per_model={"m": {"paths": [["x"], ["x"]]}},
                       failures=[{"run_id": "m-r00", "error": "e1"}])
        _write_summary(self.pilot / "lane-b" / "summary.json",
                       per_model={"m": {"paths": [["y"], ["y"]]}},
                       failures=[{"run_id": "m-r05", "error": "e2"}])

        paths, failures = analyze_pilot.load_lanes(self.pilot)

        self.assertEqual(len(paths["m"]), 4)
        self.assertEqual(len(failures), 2)
        run_ids = {f["run_id"] for f in failures}
        self.assertEqual(run_ids, {"m-r00", "m-r05"})


class TestAnalyzeMainConsumesMergedView(unittest.TestCase):
    """End-to-end: main() must feed the MERGED paths dict into
    cross_model_overlap / per_model / the census machinery, not
    recompute anything off a single lane's summary.json."""

    def setUp(self):
        self.pilot = Path(tempfile.mkdtemp(prefix="analyze-main-test-"))
        # api:glm: 2 runs in the main lane + 2 in the fill lane (topics
        # cat/dog); api:kimi: 0 in the main lane (all failed) + 4 in its
        # fill lane (topics x/y, vocabulary-disjoint from glm's so
        # cross-model jaccard/prefix-depth hand-computes to exactly 0).
        _write_summary(
            self.pilot / "lane-api" / "summary.json",
            per_model={"api:glm": {"paths": [["cat", "dog"], ["cat", "cat"]]}},
            failures=[{"run_id": "api:kimi-r%02d" % i, "error": "boom"}
                     for i in range(4)])
        _write_summary(
            self.pilot / "lane-api-fill-glm" / "summary.json",
            per_model={"api:glm": {"paths": [["dog", "dog"], ["cat", "dog"]]}})
        _write_summary(
            self.pilot / "lane-api-fill-kimi" / "summary.json",
            per_model={"api:kimi": {"paths": [["x"], ["x"], ["y"], ["y"]]}})
        self.json_out = self.pilot / "analysis.json"

    def _run_main(self):
        argv = ["analyze_pilot", "--pilot", str(self.pilot),
                "--json", str(self.json_out)]
        with mock.patch.object(sys, "argv", argv):
            analyze_pilot.main()
        with open(self.json_out) as f:
            return json.load(f)

    def test_per_model_n_runs_reflects_merge(self):
        analysis = self._run_main()
        self.assertEqual(analysis["per_model"]["api:glm"]["n_runs"], 4)
        self.assertEqual(analysis["per_model"]["api:kimi"]["n_runs"], 4)

    def test_per_model_divergence_matches_hand_computed_merge(self):
        analysis = self._run_main()
        div = analysis["per_model"]["api:glm"]["divergence"]
        self.assertAlmostEqual(div["set_jaccard"], 0.5)
        self.assertAlmostEqual(div["prefix_depth"], 0.6666666666666666)
        self.assertAlmostEqual(div["norm_edit_distance"], 0.5)
        self.assertEqual(div["n_pairs"], 6.0)

    def test_cross_model_uses_all_merged_runs(self):
        # glm's vocabulary (cat/dog) and kimi's (x/y) share nothing, so
        # every one of the 4x4=16 cross-model pairs has jaccard 0 and
        # prefix depth 0 -- a clean, hand-verifiable value that could
        # only be exactly 0.0 if ALL merged runs on both sides were
        # actually compared (a partial/overwritten pool could still
        # accidentally land on 0.0 by the same disjoint-vocabulary logic,
        # but n_runs above already rules that out; this checks the
        # overlap machinery consumed the same merged dict).
        analysis = self._run_main()
        cm = analysis["cross_model"]
        self.assertEqual(cm["mean_cross_jaccard"], 0.0)
        self.assertEqual(cm["mean_cross_prefix_depth"], 0.0)

    def test_opening_topic_census_counts_every_merged_run(self):
        # 4 (glm) + 4 (kimi) = 8 opening-topic observations. The old
        # overwrite bug would have landed on 2 (glm, fill lane only,
        # sorts last and clobbers lane-api's 2) + 4 (kimi, no collision)
        # = 6, so this distinguishes fixed from broken.
        analysis = self._run_main()
        total = sum(e["n_runs"] for e in analysis["opening_topic_census_raw"])
        self.assertEqual(total, 8)

    def test_failures_include_the_main_lane_kimi_failures(self):
        analysis = self._run_main()
        self.assertEqual(len(analysis["failures"]), 4)


if __name__ == "__main__":
    unittest.main()

"""Integration test for run_pilot orchestration — fake providers, no
network. Locks in the audit's findings: per-run failure fencing (W5),
and RAW labels as the primary metric surface (BLOCKER-1: canon() only
ever merges, so semantic scoring can only inflate overlap).
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark import run_pilot
from benchmark.metrics import path_divergence


def make_fake_model(labels, fail_at=None):
    """Emits 'A joke about <label>.' cycling through labels; optionally
    raises on the Nth call to exercise the failure fence."""
    state = {"i": 0}

    def complete(prompt):
        i = state["i"]
        state["i"] += 1
        if fail_at is not None and i == fail_at:
            raise RuntimeError("boom")
        return "A joke about %s." % labels[i % len(labels)]
    return complete


def fake_rejector(prompt):
    m = re.search(r"about ([a-z]+)", prompt)
    return m.group(1) if m else "unknown"


class TestRunPilot(unittest.TestCase):
    def _run(self, providers, models, runs=2, depth=3):
        out = tempfile.mkdtemp(prefix="pilot-test-")
        argv = ["run_pilot", "--models", ",".join(models),
                "--runs", str(runs), "--depth", str(depth),
                "--rejector", "fakerej", "--out", out]
        providers = dict(providers, fakerej=fake_rejector)
        with mock.patch.object(run_pilot, "get_provider",
                               side_effect=lambda s: providers[s]), \
             mock.patch.object(sys, "argv", argv):
            run_pilot.main()
        with open(Path(out) / "summary.json") as f:
            return json.load(f)

    def test_failure_fenced_sweep_continues(self):
        s = self._run(
            {"good": make_fake_model(["cat", "dog", "work"]),
             "flaky": make_fake_model(["coffee", "gym", "math"],
                                      fail_at=4)},
            ["good", "flaky"])
        self.assertEqual(len(s["failures"]), 1)
        self.assertIn("flaky", s["failures"][0]["run_id"])
        # good model unaffected; flaky keeps its surviving run's data
        self.assertIn("good", s["per_model"])
        self.assertEqual(len(s["per_model"]["good"]["paths"]), 2)

    def test_primary_metrics_are_raw(self):
        # fitness/gym merge semantically (calibrated must-merge pair) —
        # raw metrics must NOT reflect that merge.
        s = self._run(
            {"a": make_fake_model(["fitness", "cat", "work"]),
             "b": make_fake_model(["gym", "coffee", "math"])},
            ["a", "b"])
        pm = s["per_model"]["a"]
        # primary divergence == recompute over the RAW paths, exactly
        self.assertEqual(pm["divergence"], path_divergence(pm["paths"]))
        # raw cross-model overlap must be 0 (no shared raw strings)...
        self.assertEqual(s["cross_model"]["mean_cross_jaccard"], 0.0)
        # ...while the semantic view may only move UP, never down
        self.assertGreaterEqual(
            s["cross_model_semantic"]["mean_cross_jaccard"],
            s["cross_model"]["mean_cross_jaccard"])


if __name__ == "__main__":
    unittest.main()

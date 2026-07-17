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
    def setUp(self):
        # (spec, temperature) for every get_provider() call the run made —
        # lets tests assert WHICH specs got a temperature override.
        self.get_provider_calls = []

    def _run(self, providers, models, runs=2, depth=3, temperature=None):
        out = tempfile.mkdtemp(prefix="pilot-test-")
        self.last_out = Path(out)
        argv = ["run_pilot", "--models", ",".join(models),
                "--runs", str(runs), "--depth", str(depth),
                "--rejector", "fakerej", "--out", out]
        if temperature is not None:
            argv += ["--temperature", str(temperature)]
        providers = dict(providers, fakerej=fake_rejector)

        def fake_get_provider(s, temperature=None):
            self.get_provider_calls.append((s, temperature))
            return providers[s]

        with mock.patch.object(run_pilot, "get_provider",
                               side_effect=fake_get_provider), \
             mock.patch.object(sys, "argv", argv):
            run_pilot.main()
        with open(self.last_out / "summary.json") as f:
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

    def test_temperature_plumbed_to_model_not_rejector(self):
        s = self._run(
            {"a": make_fake_model(["cat", "dog", "work"])},
            ["a"], runs=2, depth=2, temperature=0.2)
        self.assertEqual(s["temperature"], 0.2)
        calls = dict(self.get_provider_calls)
        # model-under-test got the override...
        self.assertEqual(calls["a"], 0.2)
        # ...the rejector never did — the instrument must not vary with
        # the manipulation.
        self.assertIsNone(calls["fakerej"])
        # every turn record on disk carries the same number (recoverable
        # from logs alone, per the "no protocol violation" requirement).
        turns_files = list(self.last_out.glob("turns-a-r*.jsonl"))
        self.assertEqual(len(turns_files), 2)
        for tf in turns_files:
            with open(tf) as f:
                for line in f:
                    self.assertEqual(json.loads(line)["temperature"], 0.2)

    def test_no_temperature_flag_is_none_everywhere(self):
        s = self._run(
            {"a": make_fake_model(["cat", "dog", "work"])},
            ["a"], runs=1, depth=2)
        self.assertIsNone(s["temperature"])
        self.assertIsNone(dict(self.get_provider_calls)["a"])


if __name__ == "__main__":
    unittest.main()

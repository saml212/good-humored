"""Unit tests for env/validate_incongruity_gate.py (EXP-014) -- fake
predictor/embed_fn callables and small synthetic fixtures only, NO
network/CLI calls and NO real sentence_transformers/torch dependency.
Mirrors env/tests/test_validate_bvt_gate.py's shape (fake scripted/lookup
calls, cache resume/miss-then-hit, budget-guard raises-before-exceeding)
and env/tests/test_incongruity_gate.py's `_v`/fake-embed_fn convention
(unit vectors chosen so cosine similarity to a fixed reference is exactly
a caller-picked number, avoiding hand trig).
Run: python3 -m unittest discover -s env/tests -v
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest import mock

from benchmark.relabel import LabelCache
from env.validate_incongruity_gate import (BudgetExceeded, PredictorEmpty,
                                           REPEAT_CONSISTENCY_CLASS,
                                           build_bars, class_gate_rates,
                                           load_fixture,
                                           make_budgeted_complete,
                                           make_hybrid_predictor,
                                           pooled_gate1_rate,
                                           repeat_consistency, run_validation,
                                           score_repeat)


def _item(id_, gold_class, setup, punchline):
    return {"id": id_, "gold_class": gold_class, "setup": setup,
           "punchline": punchline, "text": setup + " " + punchline}


def _v(cos_to_ref):
    """A 2D unit vector whose cosine similarity to (1.0, 0.0) is exactly
    `cos_to_ref` -- see env/tests/test_incongruity_gate.py's identical
    helper."""
    return (cos_to_ref, math.sqrt(max(0.0, 1.0 - cos_to_ref ** 2)))


def make_vector_embed_fn(vector_map: Dict[str, tuple]):
    def embed(texts):
        return [vector_map[t] for t in texts]
    return embed


def make_scripted_complete(responses: List[str]):
    it = iter(responses)

    def complete(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(
                "make_scripted_complete: called more times than scripted "
                "responses were provided") from None
    return complete


# ------------------------------------------------------------- budget guard


class TestBudgetGuard(unittest.TestCase):
    def test_calls_under_budget_pass_through(self):
        log = []
        complete = lambda p: (log.append(p), "cold answer")[1]  # noqa: E731
        stats = {"calls": 0}
        budgeted = make_budgeted_complete(complete, stats, max_calls=3)
        for _ in range(3):
            budgeted("p")
        self.assertEqual(stats["calls"], 3)

    def test_raises_before_exceeding_and_never_overcalls(self):
        log = []
        complete = lambda p: (log.append(p), "x")[1]  # noqa: E731
        stats = {"calls": 0}
        budgeted = make_budgeted_complete(complete, stats, max_calls=1)
        budgeted("p1")
        with self.assertRaises(BudgetExceeded):
            budgeted("p2")
        self.assertEqual(len(log), 1)


# --------------------------------------------------------- hybrid predictor


class TestMakeHybridPredictor(unittest.TestCase):
    def test_split_prompt_is_replayed_at_zero_cost(self):
        item = _item("i1", "real_joke", "the setup", "the punchline")

        def _must_not_be_called(prompt):
            raise AssertionError("split must never reach the real provider")

        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            predictor = make_hybrid_predictor(item, 0, _must_not_be_called,
                                              cache, {}, retries=1)
            from env.incongruity_gate import SPLIT_PROMPT
            reply = predictor(SPLIT_PROMPT.format(completion=item["text"]))
            self.assertEqual(reply, "SETUP: the setup\nPUNCHLINE: the punchline")
            cache.close()

    def test_cold_and_primed_are_real_cached_calls(self):
        from env.incongruity_gate import (PREDICT_COLD_PROMPT,
                                          PREDICT_PRIMED_PROMPT)
        item = _item("i1", "real_joke", "setup text", "punch text")
        complete = make_scripted_complete(["cold reply", "primed reply"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            stats = {}
            predictor = make_hybrid_predictor(item, 0, complete, cache,
                                              stats, retries=1)
            cold = predictor(PREDICT_COLD_PROMPT.format(setup=item["setup"]))
            primed = predictor(PREDICT_PRIMED_PROMPT.format(setup=item["setup"]))
            self.assertEqual(cold, "cold reply")
            self.assertEqual(primed, "primed reply")
            # second predictor with same item/repeat hits cache -- no more
            # scripted responses left, so a real call would raise
            predictor2 = make_hybrid_predictor(item, 0, complete, cache,
                                               stats, retries=1)
            cold2 = predictor2(PREDICT_COLD_PROMPT.format(setup=item["setup"]))
            self.assertEqual(cold2, "cold reply")
            self.assertEqual(stats["cache_hits"], 1)
            cache.close()

    def test_different_repeat_index_is_independent(self):
        from env.incongruity_gate import PREDICT_COLD_PROMPT
        item = _item("i1", "real_joke", "setup text", "punch text")
        complete = make_scripted_complete(["reply r0", "reply r1"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            p0 = make_hybrid_predictor(item, 0, complete, cache, {}, retries=1)
            p1 = make_hybrid_predictor(item, 1, complete, cache, {}, retries=1)
            r0 = p0(PREDICT_COLD_PROMPT.format(setup=item["setup"]))
            r1 = p1(PREDICT_COLD_PROMPT.format(setup=item["setup"]))
            self.assertEqual((r0, r1), ("reply r0", "reply r1"))
            cache.close()

    def test_empty_response_after_retries_raises_predictor_empty(self):
        from env.incongruity_gate import PREDICT_COLD_PROMPT
        item = _item("i1", "real_joke", "setup text", "punch text")
        complete = make_scripted_complete(["", "   "])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            stats = {}
            predictor = make_hybrid_predictor(item, 0, complete, cache,
                                              stats, retries=1)
            with self.assertRaises(PredictorEmpty):
                predictor(PREDICT_COLD_PROMPT.format(setup=item["setup"]))
            self.assertEqual(stats["unparseable"], 1)
            cache.close()

    def test_retry_recovers_from_one_empty_response(self):
        from env.incongruity_gate import PREDICT_COLD_PROMPT
        item = _item("i1", "real_joke", "setup text", "punch text")
        complete = make_scripted_complete(["", "a real reply"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            predictor = make_hybrid_predictor(item, 0, complete, cache, {},
                                              retries=1)
            reply = predictor(PREDICT_COLD_PROMPT.format(setup=item["setup"]))
            self.assertEqual(reply, "a real reply")
            cache.close()


# --------------------------------------------------------------- score_repeat


class TestScoreRepeat(unittest.TestCase):
    def test_hand_computed_gates_both_pass(self):
        item = _item("i1", "real_joke", "setup", "punchline")
        # cold far from punchline (dist 1.0 >= 0.5 surprise threshold);
        # primed close (dist 0.1, drop of 0.9 >= 0.15 drop threshold)
        vecs = {
            "cold reply": _v(0.0), "primed reply": _v(0.9),
            "punchline": (1.0, 0.0),
        }
        embed_fn = make_vector_embed_fn(vecs)
        complete = make_scripted_complete(["cold reply", "primed reply"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, complete, cache, {}, retries=1,
                                  embed_fn=embed_fn, surprise_threshold=0.5,
                                  drop_threshold=0.15)
            cache.close()
        self.assertAlmostEqual(result["d_cold"], 1.0)
        self.assertAlmostEqual(result["d_primed"], 0.1, places=6)
        self.assertTrue(result["gate_1"])
        self.assertTrue(result["gate_2"])
        self.assertTrue(result["passes"])
        self.assertFalse(result["unparseable"])

    def test_boring_expected_fails_gate1(self):
        item = _item("i1", "boring_expected", "setup", "punchline")
        # cold prediction lands almost exactly on the real punchline --
        # tiny distance, well under the 0.5 surprise threshold.
        vecs = {"cold reply": _v(0.98), "primed reply": _v(0.5),
               "punchline": (1.0, 0.0)}
        embed_fn = make_vector_embed_fn(vecs)
        complete = make_scripted_complete(["cold reply", "primed reply"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, complete, cache, {}, retries=1,
                                  embed_fn=embed_fn, surprise_threshold=0.5,
                                  drop_threshold=0.15)
            cache.close()
        self.assertFalse(result["gate_1"])
        self.assertFalse(result["passes"])

    def test_surprising_but_primed_does_not_get_closer_fails_gate2(self):
        item = _item("i1", "setup_nonsequitur", "setup", "punchline")
        # both cold and primed are far from the real (unrelated) punchline
        # -- surprising, but nothing resolves.
        vecs = {"cold reply": _v(0.0), "primed reply": _v(0.05),
               "punchline": (1.0, 0.0)}
        embed_fn = make_vector_embed_fn(vecs)
        complete = make_scripted_complete(["cold reply", "primed reply"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, complete, cache, {}, retries=1,
                                  embed_fn=embed_fn, surprise_threshold=0.5,
                                  drop_threshold=0.15)
            cache.close()
        self.assertTrue(result["gate_1"])
        self.assertFalse(result["gate_2"])  # drop of 0.05 < 0.15 threshold
        self.assertFalse(result["passes"])

    def test_empty_predictor_response_marks_unparseable(self):
        item = _item("i1", "real_joke", "setup", "punchline")
        complete = make_scripted_complete(["", ""])  # cold empty even after retry
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, complete, cache, {}, retries=1,
                                  embed_fn=lambda ts: [(1.0, 0.0)] * len(ts),
                                  surprise_threshold=0.5, drop_threshold=0.15)
            cache.close()
        self.assertTrue(result["unparseable"])
        self.assertIsNone(result["gate_1"])
        self.assertIsNone(result["passes"])

    def test_empty_setup_or_punchline_short_circuits_before_any_call(self):
        # bypasses load_fixture's own validation (which forbids this) to
        # exercise score_repeat's defensive "setup is None" branch
        # directly -- TwoStageIncongruityGate._split's real code returns
        # (None, None) when either side of a parsed "SETUP:.../PUNCHLINE:
        # ..." reply is empty.
        item = {"id": "broken", "gold_class": "real_joke", "setup": "",
               "punchline": "", "text": " "}

        def _must_not_be_called(prompt):
            raise AssertionError("cold/primed must never be reached")

        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, _must_not_be_called, cache, {},
                                  retries=1, embed_fn=lambda ts: [(1.0, 0.0)] * len(ts),
                                  surprise_threshold=0.5, drop_threshold=0.15)
            cache.close()
        self.assertTrue(result["unparseable"])


# ---------------------------------------------------------- run_validation


class TestRunValidation(unittest.TestCase):
    def _items(self):
        return [_item("a", "real_joke", "setup a", "punch a"),
               _item("b", "boring_expected", "setup b", "punch b")]

    def _embed(self, ts):
        return [(1.0, 0.0)] * len(ts)

    def test_completes_under_generous_budget(self):
        complete = make_scripted_complete(
            ["c0a", "p0a", "c1a", "p1a", "c0b", "p0b", "c1b", "p1b"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = run_validation(self._items(), complete, cache,
                                    max_calls=100, repeats=2, retries=1,
                                    embed_fn=self._embed,
                                    surprise_threshold=0.5, drop_threshold=0.15)
            cache.close()
        self.assertFalse(result["budget_exhausted"])
        self.assertEqual(len(result["per_item"]["a"]), 2)
        self.assertEqual(len(result["per_item"]["b"]), 2)
        self.assertEqual(result["stats"]["calls"], 8)

    def test_budget_stop_drops_incomplete_repeat(self):
        # item a repeat 0 costs 2 calls (cold+primed); repeat 1's cold call
        # is the 3rd call, budget=3 refuses the 4th (primed).
        complete = make_scripted_complete(["c0a", "p0a", "c1a"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = run_validation(self._items(), complete, cache,
                                    max_calls=3, repeats=2, retries=1,
                                    embed_fn=self._embed,
                                    surprise_threshold=0.5, drop_threshold=0.15)
            cache.close()
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(len(result["per_item"]["a"]), 1)
        self.assertNotIn("b", result["per_item"])

    def test_resumed_run_serves_everything_from_cache(self):
        complete = make_scripted_complete(
            ["c0a", "p0a", "c1a", "p1a", "c0b", "p0b", "c1b", "p1b"])
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "c.jsonl"
            cache1 = LabelCache(cache_path)
            run_validation(self._items(), complete, cache1, max_calls=100,
                          repeats=2, retries=1, embed_fn=self._embed,
                          surprise_threshold=0.5, drop_threshold=0.15)
            cache1.close()

            cache2 = LabelCache(cache_path)

            def _must_not_be_called(prompt):
                raise AssertionError("resumed run re-spent a cached call")

            result = run_validation(self._items(), _must_not_be_called,
                                    cache2, max_calls=100, repeats=2,
                                    retries=1, embed_fn=self._embed,
                                    surprise_threshold=0.5, drop_threshold=0.15)
            cache2.close()
        self.assertEqual(result["stats"]["calls"], 0)
        self.assertEqual(result["stats"]["cache_hits"], 8)


# ------------------------------------------------------------ class_gate_rates


class TestClassGateRates(unittest.TestCase):
    def test_pooled_fraction_excludes_unparseable(self):
        items = [_item("a", "real_joke", "s", "p"),
                _item("b", "real_joke", "s2", "p2")]
        per_item = {
            "a": [{"gate_1": True, "gate_2": True, "passes": True,
                  "d_cold": 0.9, "d_primed": 0.1, "unparseable": False},
                 {"gate_1": True, "gate_2": False, "passes": False,
                  "d_cold": 0.7, "d_primed": 0.6, "unparseable": False}],
            "b": [{"gate_1": None, "gate_2": None, "passes": None,
                  "d_cold": None, "d_primed": None, "unparseable": True}],
        }
        rates = class_gate_rates(items, per_item, "real_joke")
        self.assertEqual(rates["n_items"], 2)
        self.assertEqual(rates["n_observations"], 2)  # b's repeat excluded
        self.assertAlmostEqual(rates["gate_1_pass_rate"], 1.0)
        self.assertAlmostEqual(rates["gate_2_pass_rate"], 0.5)
        self.assertAlmostEqual(rates["both_gates_pass_rate"], 0.5)

    def test_no_observations_is_none(self):
        rates = class_gate_rates([], {}, "real_joke")
        self.assertIsNone(rates["gate_1_pass_rate"])


class TestPooledGate1Rate(unittest.TestCase):
    def test_pools_across_two_classes(self):
        items = [_item("a", "real_joke", "s", "p"),
                _item("b", "setup_nonsequitur", "s2", "p2")]
        per_item = {
            "a": [{"gate_1": True, "unparseable": False}],
            "b": [{"gate_1": False, "unparseable": False}],
        }
        rate = pooled_gate1_rate(items, per_item, ("real_joke", "setup_nonsequitur"))
        self.assertAlmostEqual(rate, 0.5)


# ----------------------------------------------------------- repeat_consistency


class TestRepeatConsistency(unittest.TestCase):
    def test_only_scores_the_configured_class(self):
        items = [_item("rj1", "real_joke", "s", "p"),
                _item("be1", "boring_expected", "s2", "p2")]
        per_item = {
            "rj1": [{"passes": True, "unparseable": False},
                   {"passes": True, "unparseable": False}],
            # boring_expected disagreement would tank the score if it
            # leaked into the real_joke-only computation
            "be1": [{"passes": True, "unparseable": False},
                   {"passes": False, "unparseable": False}],
        }
        self.assertAlmostEqual(
            repeat_consistency(items, per_item, cls="real_joke"), 1.0)

    def test_disagreement_within_class_lowers_score(self):
        items = [_item("rj1", "real_joke", "s", "p")]
        per_item = {"rj1": [{"passes": True, "unparseable": False},
                           {"passes": False, "unparseable": False}]}
        self.assertAlmostEqual(
            repeat_consistency(items, per_item, cls="real_joke"), 0.0)

    def test_default_class_is_real_joke(self):
        self.assertEqual(REPEAT_CONSISTENCY_CLASS, "real_joke")

    def test_no_valid_pairs_defaults_to_one(self):
        items = [_item("rj1", "real_joke", "s", "p")]
        per_item = {"rj1": [{"passes": True, "unparseable": False}]}
        self.assertAlmostEqual(
            repeat_consistency(items, per_item, cls="real_joke"), 1.0)


# --------------------------------------------------------------- build_bars


class TestBuildBars(unittest.TestCase):
    def _rates(self, **overrides):
        base = {
            "real_joke": {"gate_1_pass_rate": 0.9, "gate_2_pass_rate": 0.8},
            "setup_nonsequitur": {"gate_1_pass_rate": 0.9, "gate_2_pass_rate": 0.1},
            "boring_expected": {"gate_1_pass_rate": 0.1, "gate_2_pass_rate": 0.0},
            "vague_abstract_gaming_probe": {"gate_1_pass_rate": 0.5, "gate_2_pass_rate": 0.1},
        }
        base.update(overrides)
        return base

    def test_all_pass_when_metrics_clear_every_bar(self):
        rates = self._rates()
        bars = build_bars(rates, sep_gate1=0.85, sep_gate2=0.7, consistency=0.9)
        self.assertTrue(all(b["passed"] for b in bars))

    def test_none_separation_fails_its_bar(self):
        rates = self._rates()
        bars = build_bars(rates, sep_gate1=None, sep_gate2=0.7, consistency=0.9)
        b = next(x for x in bars if x["name"] == "separation_surprising_minus_boring_gate1")
        self.assertFalse(b["passed"])

    def test_vague_probe_bar_direction(self):
        rates_pass = self._rates(
            vague_abstract_gaming_probe={"gate_1_pass_rate": 0.5, "gate_2_pass_rate": 0.25})
        rates_fail = self._rates(
            vague_abstract_gaming_probe={"gate_1_pass_rate": 0.5, "gate_2_pass_rate": 0.26})
        bars_pass = build_bars(rates_pass, 0.85, 0.7, 0.9)
        bars_fail = build_bars(rates_fail, 0.85, 0.7, 0.9)
        bp = next(x for x in bars_pass
                 if x["name"] == "vague_abstract_gaming_probe_gate2_pass_rate")
        bf = next(x for x in bars_fail
                 if x["name"] == "vague_abstract_gaming_probe_gate2_pass_rate")
        self.assertTrue(bp["passed"])
        self.assertFalse(bf["passed"])


# ------------------------------------------------------------- load_fixture


class TestLoadFixture(unittest.TestCase):
    def test_real_fixture_loads_forty_items(self):
        items = load_fixture()
        self.assertEqual(len(items), 40)

    def test_duplicate_ids_raise(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.jsonl"
            with open(p, "w") as f:
                rec = {"id": "x", "gold_class": "real_joke", "setup": "s",
                      "punchline": "p", "text": "s p"}
                f.write(json.dumps(rec) + "\n")
                f.write(json.dumps(rec) + "\n")
            with mock.patch("env.validate_incongruity_gate.FIXTURE", p):
                with self.assertRaises(ValueError):
                    load_fixture()

    def test_unknown_gold_class_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"id": "x", "gold_class": "nonsense",
                                   "setup": "s", "punchline": "p",
                                   "text": "s p"}) + "\n")
            with mock.patch("env.validate_incongruity_gate.FIXTURE", p):
                with self.assertRaises(ValueError):
                    load_fixture()

    def test_missing_setup_or_punchline_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"id": "x", "gold_class": "real_joke",
                                   "setup": "", "punchline": "p",
                                   "text": "s p"}) + "\n")
            with mock.patch("env.validate_incongruity_gate.FIXTURE", p):
                with self.assertRaises(ValueError):
                    load_fixture()


if __name__ == "__main__":
    unittest.main()

"""Unit tests for env/validate_bvt_gate.py (EXP-013) -- fake judge
callables and small synthetic fixtures only, NO network or CLI calls.
Mirrors env/tests/test_certify_judge_oogiri.py's shape (fake scripted/
lookup judges, cache resume/miss-then-hit, budget-guard raises-before-
exceeding) plus benchmark/tests coverage of per-class/separation/echo
metric hand-computation.
Run: python3 -m unittest discover -s env/tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

from benchmark.relabel import LabelCache
from env.bvt_gate import BENIGN_PROMPT, VIOLATION_PROMPT
from env.validate_bvt_gate import (BudgetExceeded, REPEAT_PASS_THRESHOLD,
                                   _parse_score, build_bars,
                                   disclaimer_marker_count,
                                   echo_hackability_check, item_means,
                                   judge_raw_score, load_fixture,
                                   make_budgeted_complete,
                                   make_disproof_judge, per_class_stats,
                                   repeat_consistency, run_disproof_check,
                                   run_validation, score_repeat,
                                   shock_word_count)


# ------------------------------------------------------------ fake judges


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


def make_directional_complete(violation_responses, benign_responses):
    """Routes to violation_responses/benign_responses by which of
    VIOLATION_PROMPT/BENIGN_PROMPT's own distinguishing text appears in
    the rendered prompt -- mirrors env/tests/test_incongruity_gate.py's
    regex-routed fake predictor convention, applied to the two BVT
    prompts."""
    v_it = iter(violation_responses)
    b_it = iter(benign_responses)

    def complete(prompt: str) -> str:
        if "VIOLATES a" in prompt:
            return next(v_it)
        if "is BENIGN" in prompt:
            return next(b_it)
        raise AssertionError("unrecognized prompt: %r" % prompt)
    return complete


def _item(id_, gold_class, text):
    return {"id": id_, "gold_class": gold_class, "text": text}


# ------------------------------------------------------------ _parse_score


class TestParseScore(unittest.TestCase):
    def test_zero_is_valid(self):
        self.assertEqual(_parse_score("0"), 0)

    def test_ten_is_valid(self):
        self.assertEqual(_parse_score("10"), 10)

    def test_plain_integer_first_line(self):
        self.assertEqual(_parse_score("7\nsome trailing text"), 7)

    def test_two_digit_not_confused(self):
        self.assertIsNone(_parse_score("23"))

    def test_empty_or_none_is_none(self):
        self.assertIsNone(_parse_score(""))
        self.assertIsNone(_parse_score("   "))
        self.assertIsNone(_parse_score(None))


# ------------------------------------------------------------- budget guard


class TestBudgetGuard(unittest.TestCase):
    def test_calls_under_budget_pass_through(self):
        log = []
        complete = lambda p: (log.append(p), "5")[1]  # noqa: E731
        stats = {"calls": 0}
        budgeted = make_budgeted_complete(complete, stats, max_calls=3)
        for _ in range(3):
            budgeted("p")
        self.assertEqual(stats["calls"], 3)
        self.assertEqual(len(log), 3)

    def test_raises_before_exceeding_and_never_overcalls(self):
        log = []
        complete = lambda p: (log.append(p), "5")[1]  # noqa: E731
        stats = {"calls": 0}
        budgeted = make_budgeted_complete(complete, stats, max_calls=2)
        budgeted("p1")
        budgeted("p2")
        with self.assertRaises(BudgetExceeded):
            budgeted("p3")
        self.assertEqual(len(log), 2)


# --------------------------------------------------------- judge_raw_score


class TestJudgeRawScore(unittest.TestCase):
    def test_miss_then_hit_no_second_call(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            complete = make_scripted_complete(["6"])
            stats = {}
            first = judge_raw_score("violation", "some text", 0,
                                    VIOLATION_PROMPT, complete, cache, stats,
                                    retries=1)
            self.assertEqual(first, 6)
            second = judge_raw_score("violation", "some text", 0,
                                     VIOLATION_PROMPT, complete, cache,
                                     stats, retries=1)
            self.assertEqual(second, 6)
            self.assertEqual(stats["cache_hits"], 1)
            cache.close()

    def test_different_repeat_index_is_independent_cache_entry(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            complete = make_scripted_complete(["3", "8"])
            stats = {}
            r0 = judge_raw_score("violation", "txt", 0, VIOLATION_PROMPT,
                                 complete, cache, stats, retries=1)
            r1 = judge_raw_score("violation", "txt", 1, VIOLATION_PROMPT,
                                 complete, cache, stats, retries=1)
            self.assertEqual((r0, r1), (3, 8))
            self.assertEqual(stats.get("cache_hits", 0), 0)  # both were
                                                              # real calls
            cache.close()

    def test_violation_and_benign_directions_do_not_collide(self):
        # same text, same repeat, different direction -- must be two
        # independent cache entries (different labelers).
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            complete = make_scripted_complete(["9", "1"])
            stats = {}
            v = judge_raw_score("violation", "txt", 0, VIOLATION_PROMPT,
                                complete, cache, stats, retries=1)
            b = judge_raw_score("benign", "txt", 0, BENIGN_PROMPT, complete,
                                cache, stats, retries=1)
            self.assertEqual((v, b), (9, 1))
            cache.close()

    def test_retries_once_then_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            complete = make_scripted_complete(["not a number", "4"])
            stats = {}
            result = judge_raw_score("violation", "txt", 0, VIOLATION_PROMPT,
                                     complete, cache, stats, retries=1)
            self.assertEqual(result, 4)
            cache.close()

    def test_exhausts_retries_returns_none_never_cached(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            complete = make_scripted_complete(["nope", "still nope"])
            stats = {}
            result = judge_raw_score("violation", "txt", 0, VIOLATION_PROMPT,
                                     complete, cache, stats, retries=1)
            self.assertIsNone(result)
            self.assertEqual(stats["unparseable"], 1)
            self.assertIsNone(cache.get("bvt-violation-exp013-v1", "0\x1ftxt"))
            cache.close()


# ------------------------------------------------------------- score_repeat


class TestScoreRepeat(unittest.TestCase):
    def test_product_matches_hand_computed_v_times_b(self):
        item = _item("i1", "both", "some completion text")
        complete = make_directional_complete(["8"], ["9"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, complete, cache, {}, retries=1)
            cache.close()
        self.assertEqual(result["violation_raw"], 8)
        self.assertEqual(result["benign_raw"], 9)
        self.assertAlmostEqual(result["product"], 0.8 * 0.9)

    def test_unparseable_violation_makes_product_none(self):
        item = _item("i1", "violation_only", "text")
        complete = make_directional_complete(["garbage", "garbage"], ["9"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, complete, cache, {}, retries=1)
            cache.close()
        self.assertIsNone(result["violation_raw"])
        self.assertEqual(result["benign_raw"], 9)
        self.assertIsNone(result["product"])

    def test_zero_score_on_either_axis_zeroes_product(self):
        item = _item("i1", "neither", "text")
        complete = make_directional_complete(["0"], ["10"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = score_repeat(item, 0, complete, cache, {}, retries=1)
            cache.close()
        self.assertAlmostEqual(result["product"], 0.0)


# ----------------------------------------------------------- run_validation


class TestRunValidation(unittest.TestCase):
    def _items(self):
        return [_item("a", "both", "text a"), _item("b", "neither", "text b")]

    def test_completes_under_generous_budget(self):
        # each item x 2 repeats x 2 calls (v, b) = 8 calls total
        complete = make_scripted_complete(["8", "9", "7", "6", "1", "2", "0", "3"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = run_validation(self._items(), complete, cache,
                                    max_calls=100, repeats=2, retries=1)
            cache.close()
        self.assertFalse(result["budget_exhausted"])
        self.assertEqual(len(result["per_item"]["a"]), 2)
        self.assertEqual(len(result["per_item"]["b"]), 2)
        self.assertEqual(result["stats"]["calls"], 8)

    def test_budget_stop_drops_incomplete_repeat_but_keeps_prior_ones(self):
        # item a repeat 0: v=8,b=9 (2 calls) ; repeat 1: v-call succeeds (3rd
        # call) then budget (max_calls=3) refuses the benign call.
        complete = make_scripted_complete(["8", "9", "7"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            result = run_validation(self._items(), complete, cache,
                                    max_calls=3, repeats=2, retries=1)
            cache.close()
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(result["stats"]["calls"], 3)
        self.assertEqual(len(result["per_item"]["a"]), 1)  # only repeat 0
        self.assertNotIn("b", result["per_item"])  # item b never started

    def test_resumed_run_serves_everything_from_cache(self):
        complete = make_scripted_complete(["8", "9", "7", "6", "1", "2", "0", "3"])
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "c.jsonl"
            cache1 = LabelCache(cache_path)
            run_validation(self._items(), complete, cache1, max_calls=100,
                          repeats=2, retries=1)
            cache1.close()

            cache2 = LabelCache(cache_path)

            def _must_not_be_called(prompt):
                raise AssertionError("resumed run re-spent a cached call")

            result = run_validation(self._items(), _must_not_be_called,
                                    cache2, max_calls=100, repeats=2,
                                    retries=1)
            cache2.close()
        self.assertEqual(result["stats"]["calls"], 0)
        self.assertEqual(result["stats"]["cache_hits"], 8)


# --------------------------------------------------- item_means/per_class


class TestItemMeansAndPerClass(unittest.TestCase):
    def test_item_mean_ignores_unparseable_repeats(self):
        per_item = {
            "x": [{"violation_raw": 8, "benign_raw": 2, "product": 0.16},
                 {"violation_raw": None, "benign_raw": None, "product": None},
                 {"violation_raw": 10, "benign_raw": 4, "product": 0.4}],
        }
        means = item_means(per_item)
        self.assertAlmostEqual(means["x"]["violation_raw_mean"], 9.0)
        self.assertAlmostEqual(means["x"]["benign_raw_mean"], 3.0)
        self.assertAlmostEqual(means["x"]["product_mean"], 0.28)
        self.assertEqual(means["x"]["n_scored_product"], 2)

    def test_item_all_unparseable_is_none(self):
        per_item = {"x": [{"violation_raw": None, "benign_raw": None,
                          "product": None}]}
        means = item_means(per_item)
        self.assertIsNone(means["x"]["violation_raw_mean"])
        self.assertIsNone(means["x"]["product_mean"])

    def test_per_class_is_item_then_class_average_not_pooled(self):
        # class 'both' has 2 items: item1 has ONE repeat (product=1.0),
        # item2 has THREE repeats (product=0.0 each). Item-then-class
        # averaging must give (1.0 + 0.0) / 2 = 0.5, NOT the pooled
        # (1.0 + 0*3) / 4 = 0.25 -- this is the exact distinction
        # validate_banter_judge.py's item_mean_deltas makes.
        items = [_item("i1", "both", "t1"), _item("i2", "both", "t2")]
        per_item = {
            "i1": [{"violation_raw": 9, "benign_raw": 9, "product": 1.0}],
            "i2": [{"violation_raw": 0, "benign_raw": 0, "product": 0.0}] * 3,
        }
        means = item_means(per_item)
        pc = per_class_stats(items, means)
        self.assertAlmostEqual(pc["both"]["product_mean"], 0.5)


# --------------------------------------------------------- keyword counts


class TestKeywordCounts(unittest.TestCase):
    def test_shock_word_count_basic(self):
        self.assertEqual(shock_word_count("You are worthless and I hate you"), 2)
        self.assertEqual(shock_word_count("a warm, pleasant afternoon"), 0)

    def test_disclaimer_marker_count_basic(self):
        self.assertEqual(disclaimer_marker_count("jk!! just kidding lol"), 3)
        self.assertEqual(disclaimer_marker_count("a normal sentence"), 0)


# --------------------------------------------------- echo_hackability_check


class TestEchoHackabilityCheck(unittest.TestCase):
    def test_perfect_correlation_flags_risk(self):
        items = [
            _item("v1", "violation_only", "hate"),         # 1 shock word
            _item("v2", "violation_only", "hate hate"),     # 2
            _item("v3", "violation_only", "hate hate hate"),  # 3
            _item("v4", "violation_only", "no charged words here at all"),  # 0
        ]
        # violation_raw_mean scales exactly with shock word count -> r=1.0
        means = {
            "v1": {"violation_raw_mean": 2.0, "benign_raw_mean": None},
            "v2": {"violation_raw_mean": 4.0, "benign_raw_mean": None},
            "v3": {"violation_raw_mean": 6.0, "benign_raw_mean": None},
            "v4": {"violation_raw_mean": 0.0, "benign_raw_mean": None},
        }
        result = echo_hackability_check(items, means)
        self.assertAlmostEqual(result["violation_vs_shock_word_count"]["pearson_r"], 1.0)
        self.assertTrue(result["violation_vs_shock_word_count"]["risk_detected"])

    def test_no_correlation_does_not_flag(self):
        items = [
            _item("v1", "violation_only", "hate"),
            _item("v2", "violation_only", "hate hate"),
            _item("v3", "violation_only", "no charged words"),
            _item("v4", "violation_only", "worthless betray sabotage"),
        ]
        # constant violation score regardless of shock word count -> zero
        # variance on one axis -> _pearson returns None (undefined, not 0)
        means = {i: {"violation_raw_mean": 5.0, "benign_raw_mean": None}
                for i in ("v1", "v2", "v3", "v4")}
        result = echo_hackability_check(items, means)
        self.assertIsNone(result["violation_vs_shock_word_count"]["pearson_r"])
        self.assertFalse(result["violation_vs_shock_word_count"]["risk_detected"])

    def test_disclaimer_check_pools_washed_and_both_classes(self):
        items = [
            _item("w1", "disclaimer_washed_violation", "jk"),
            _item("b1", "both", "no markers here"),
        ]
        means = {
            "w1": {"violation_raw_mean": None, "benign_raw_mean": 3.0},
            "b1": {"violation_raw_mean": None, "benign_raw_mean": 7.0},
        }
        result = echo_hackability_check(items, means)
        self.assertEqual(result["benign_vs_disclaimer_marker_count"]["n"], 2)


# ----------------------------------------------------------- repeat_consistency


class TestRepeatConsistency(unittest.TestCase):
    def test_all_agree_is_1(self):
        per_item = {
            "a": [{"product": 0.5}, {"product": 0.6}, {"product": 0.9}],  # all PASS (>0.2)
            "b": [{"product": 0.0}, {"product": 0.1}, {"product": 0.05}],  # all FAIL
        }
        self.assertAlmostEqual(repeat_consistency(per_item), 1.0)

    def test_split_labels_lower_consistency(self):
        per_item = {
            "a": [{"product": 0.9}, {"product": 0.0}],  # disagree: 1 pair, 0 agree
        }
        self.assertAlmostEqual(repeat_consistency(per_item), 0.0)

    def test_item_with_lone_valid_repeat_contributes_zero_pairs(self):
        per_item = {
            "a": [{"product": 0.9}, {"product": None}, {"product": None}],
            "b": [{"product": 0.9}, {"product": 0.8}],  # 1 pair, agree
        }
        # item a contributes 0 pairs (only 1 valid repeat); item b
        # contributes 1 agreeing pair -> overall 1.0
        self.assertAlmostEqual(repeat_consistency(per_item), 1.0)

    def test_no_valid_pairs_anywhere_defaults_to_one(self):
        per_item = {"a": [{"product": None}]}
        self.assertAlmostEqual(repeat_consistency(per_item), 1.0)

    def test_threshold_is_strict_greater_than_0_2(self):
        # product exactly at REPEAT_PASS_THRESHOLD is FAIL, not PASS
        self.assertEqual(REPEAT_PASS_THRESHOLD, 0.2)
        per_item = {"a": [{"product": 0.2}, {"product": 0.21}]}
        # 0.2 -> FAIL, 0.21 -> PASS -> disagree -> 0.0
        self.assertAlmostEqual(repeat_consistency(per_item), 0.0)


# --------------------------------------------------------------- build_bars


class TestBuildBars(unittest.TestCase):
    def _per_class(self, **overrides):
        base = {
            "both": {"violation_raw_mean": 8.0, "benign_raw_mean": 8.0, "product_mean": 0.6},
            "violation_only": {"violation_raw_mean": 8.0, "benign_raw_mean": 1.0, "product_mean": 0.08},
            "benign_only": {"violation_raw_mean": 0.5, "benign_raw_mean": 9.0, "product_mean": 0.04},
            "neither": {"violation_raw_mean": 0.5, "benign_raw_mean": 5.0, "product_mean": 0.02},
        }
        base.update(overrides)
        return base

    def test_all_pass_when_metrics_clear_every_bar(self):
        pc = self._per_class()
        disclaimer_stats = {"benign_raw_mean": 2.0}
        echo = {"violation_vs_shock_word_count": {"risk_detected": False},
               "benign_vs_disclaimer_marker_count": {"risk_detected": False}}
        bars = build_bars(pc, disclaimer_stats, echo, consistency=0.9)
        self.assertTrue(all(b["passed"] for b in bars))

    def test_none_value_fails_its_bar(self):
        pc = self._per_class(both={"violation_raw_mean": None,
                                  "benign_raw_mean": None,
                                  "product_mean": None})
        disclaimer_stats = {"benign_raw_mean": 2.0}
        echo = {"violation_vs_shock_word_count": {"risk_detected": False},
               "benign_vs_disclaimer_marker_count": {"risk_detected": False}}
        bars = build_bars(pc, disclaimer_stats, echo, consistency=0.9)
        both_product_bar = next(b for b in bars if b["name"] == "both_product_mean")
        self.assertFalse(both_product_bar["passed"])

    def test_echo_risk_detected_fails_its_bar(self):
        pc = self._per_class()
        disclaimer_stats = {"benign_raw_mean": 2.0}
        echo = {"violation_vs_shock_word_count": {"risk_detected": True},
               "benign_vs_disclaimer_marker_count": {"risk_detected": False}}
        bars = build_bars(pc, disclaimer_stats, echo, consistency=0.9)
        echo_bar = next(b for b in bars
                       if b["name"] == "echo_violation_vs_shock_word_r_not_risky")
        self.assertFalse(echo_bar["passed"])

    def test_disclaimer_washing_bar_direction(self):
        pc = self._per_class()
        echo = {"violation_vs_shock_word_count": {"risk_detected": False},
               "benign_vs_disclaimer_marker_count": {"risk_detected": False}}
        bars_pass = build_bars(pc, {"benign_raw_mean": 3.0}, echo, 0.9)
        bars_fail = build_bars(pc, {"benign_raw_mean": 3.1}, echo, 0.9)
        washing_pass = next(b for b in bars_pass
                           if b["name"] == "disclaimer_washed_benign_raw_mean")
        washing_fail = next(b for b in bars_fail
                           if b["name"] == "disclaimer_washed_benign_raw_mean")
        self.assertTrue(washing_pass["passed"])  # 3.0 <= 3.0
        self.assertFalse(washing_fail["passed"])  # 3.1 > 3.0


# --------------------------------------------------------------- disproof


class TestMakeDisproofJudge(unittest.TestCase):
    def test_normalizes_and_caches(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            complete = make_scripted_complete(["7"])
            stats = {}
            judge = make_disproof_judge(complete, cache, stats, retries=1)
            score = judge(None, "a joke")
            self.assertAlmostEqual(score, 0.7)
            # second call is a cache hit -- complete() would raise if
            # invoked again (scripted list is exhausted)
            score2 = judge(None, "a joke")
            self.assertAlmostEqual(score2, 0.7)
            cache.close()

    def test_unparseable_falls_back_to_neutral_and_is_logged(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            complete = make_scripted_complete(["garbage", "still garbage"])
            stats = {}
            judge = make_disproof_judge(complete, cache, stats, retries=1)
            score = judge(None, "a joke")
            self.assertAlmostEqual(score, 0.5)
            self.assertEqual(stats["disproof_unparseable_fallback"], 1)
            cache.close()


class TestRunDisproofCheck(unittest.TestCase):
    def test_additive_stack_matches_judge_plus_comprehensibility(self):
        items = [
            _item("v1", "violation_only",
                 "This is a well formed cruel sentence with enough tokens."),
            _item("b1", "benign_only",
                 "This is a well formed pleasant sentence with enough tokens."),
            _item("both1", "both",
                 "This is a well formed dual appraisal sentence with tokens."),
        ]
        # every item gets judge score 8/10 -> normalized 0.8 -> * weight
        # 1.0 = 0.8. additive_stack_mean must equal judge_preference_mean +
        # comprehensibility_mean exactly -- the property this test checks
        # is that run_disproof_check WIRES the two real reward-class
        # instances together correctly (doesn't drop/double either term),
        # not a hand-reimplementation of ComprehensibilityReward's own
        # length/punctuation/unique-ratio band scoring (that class's own
        # unit tests in env/tests/test_rewards.py own that).
        from env.rewards import ComprehensibilityReward
        complete = make_scripted_complete(["8", "8", "8"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            stats = {"calls": 0}
            result = run_disproof_check(items, complete, cache, stats,
                                        max_calls=100, retries=1)
            cache.close()
        comp = ComprehensibilityReward(weight=0.3)
        for cls, it in (("violation_only", items[0]), ("benign_only", items[1]),
                       ("both", items[2])):
            self.assertAlmostEqual(result[cls]["judge_preference_mean"], 0.8)
            expected_comp = comp(prompts=[None], completions=[it["text"]])[0]
            self.assertAlmostEqual(result[cls]["comprehensibility_mean"],
                                   expected_comp)
            self.assertAlmostEqual(
                result[cls]["additive_stack_mean"],
                result[cls]["judge_preference_mean"] + expected_comp,
                places=6)

    def test_budget_exhaustion_is_recorded_not_raised(self):
        items = [_item("v1", "violation_only", "text one two three four"),
                _item("b1", "benign_only", "text five six seven eight")]
        complete = make_scripted_complete(["8"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            stats = {"calls": 0}
            result = run_disproof_check(items, complete, cache, stats,
                                        max_calls=1, retries=0)
            cache.close()
        self.assertTrue(result.get("budget_exhausted"))

    def test_max_calls_is_absolute_not_a_remaining_delta(self):
        # regression test: main() passes the SAME stats dict run_validation
        # already incremented, plus args.max_calls (the run-wide absolute
        # cap) -- NOT (args.max_calls - stats["calls"]). A stats dict that
        # arrives with calls already at 5 and an absolute max_calls=7 must
        # allow exactly 2 more real calls, not 7 more.
        items = [_item("v1", "violation_only", "one two three four five"),
                _item("b1", "benign_only", "six seven eight nine ten")]
        complete = make_scripted_complete(["8", "8"])
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "c.jsonl")
            stats = {"calls": 5}  # simulates 5 calls already spent upstream
            result = run_disproof_check(items, complete, cache, stats,
                                        max_calls=7, retries=0)
            cache.close()
        # both items scored (5 + 2 = 7, exactly at the absolute cap) --
        # neither the delta bug (would allow 7 MORE, i.e. up to 12) nor an
        # under-strict interpretation blocked this.
        self.assertNotIn("budget_exhausted", result)
        self.assertEqual(stats["calls"], 7)
        self.assertAlmostEqual(result["violation_only"]["judge_preference_mean"], 0.8)


# ------------------------------------------------------------- load_fixture


class TestLoadFixture(unittest.TestCase):
    def test_real_fixture_loads_forty_items(self):
        items = load_fixture()
        self.assertEqual(len(items), 40)

    def test_duplicate_ids_raise(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"id": "x", "gold_class": "both", "text": "a"}) + "\n")
                f.write(json.dumps({"id": "x", "gold_class": "both", "text": "b"}) + "\n")
            with mock.patch("env.validate_bvt_gate.FIXTURE", p):
                with self.assertRaises(ValueError):
                    load_fixture()

    def test_unknown_gold_class_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"id": "x", "gold_class": "not_a_real_class",
                                   "text": "a"}) + "\n")
            with mock.patch("env.validate_bvt_gate.FIXTURE", p):
                with self.assertRaises(ValueError):
                    load_fixture()


if __name__ == "__main__":
    unittest.main()

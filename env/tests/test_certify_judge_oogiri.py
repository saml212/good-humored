"""Unit tests for env/certify_judge_oogiri.py (EXP-012) -- fake judge
callables and the tiny local fixture only, NO network or CLI calls.
Run: python3 -m unittest discover -s env/tests -v
"""

import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional

from benchmark.relabel import LabelCache
from data_adapters.schema import Candidate, RankedGroup
from env.certify_judge_oogiri import (BudgetExceeded, _parse_score, _ranks,
                                      cache_key, certify_group,
                                      judge_once, load_and_sample,
                                      make_budgeted_complete, run_certification,
                                      sample_groups, score_candidate,
                                      spearman_rho)

FIXTURE = Path(__file__).parent / "fixtures" / "oogiri_certify_sample.jsonl"


# ------------------------------------------------------------ fake judges


def make_scripted_complete(responses: List[str]):
    """Returns each string in `responses` in order, one per call. Raises
    if called more times than responses were provided -- makes an
    over-budget or over-retried call an immediate, loud test failure
    rather than an IndexError several frames away."""
    it = iter(responses)

    def complete(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(
                "make_scripted_complete: called more times than scripted "
                "responses were provided") from None
    return complete


def make_lookup_judge(scores_by_candidate: Dict[str, int], call_log: Optional[List[str]] = None):
    """Fake judge that returns the score scripted for whichever candidate
    text appears (verbatim) in the formatted prompt -- lets a test build a
    RankedGroup with known candidate texts and assert exactly the rho
    those known judge scores must produce, without depending on
    certify_judge_oogiri's own prompt-formatting internals beyond
    'the candidate text appears in the prompt somewhere'. Longest text
    first, so one candidate's text being a substring of another's can't
    cause a false match."""
    texts_by_len = sorted(scores_by_candidate, key=len, reverse=True)

    def complete(prompt: str) -> str:
        if call_log is not None:
            call_log.append(prompt)
        for t in texts_by_len:
            if t in prompt:
                return str(scores_by_candidate[t])
        raise AssertionError(
            "make_lookup_judge: no known candidate text found in prompt: %r"
            % prompt)
    return complete


def _cand(text: str, score: float) -> Candidate:
    return Candidate(text=text, score=score)


def _group(context: str, candidates, source_id: str = "g1") -> RankedGroup:
    return RankedGroup(context=context, candidates=tuple(candidates),
                       source_dataset="fixture", license_class="research_only",
                       source_id=source_id)


# ------------------------------------------------------------- _ranks


class TestRanks(unittest.TestCase):
    def test_no_ties(self):
        self.assertEqual(_ranks([30, 10, 20]), [3.0, 1.0, 2.0])

    def test_one_tie_pair(self):
        # two 20s jointly occupy sorted positions 2 and 3 -> both rank 2.5
        self.assertEqual(_ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_all_tied(self):
        self.assertEqual(_ranks([7, 7, 7]), [2.0, 2.0, 2.0])


# --------------------------------------------------------- spearman_rho


class TestSpearmanRho(unittest.TestCase):
    """Hand-computed cases, including ties -- the tie-handling
    (average-rank Pearson-on-ranks) is exactly what the module docstring
    claims, worked by hand below rather than re-derived from the same
    code under test."""

    def test_perfect_positive_no_ties(self):
        self.assertAlmostEqual(
            spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0)

    def test_perfect_negative_no_ties(self):
        self.assertAlmostEqual(
            spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0)

    def test_hand_computed_with_one_tie_in_y(self):
        # x = [1,2,3,4,5] -> ranks [1,2,3,4,5] (no ties)
        # y = [5,6,7,8,7] -> sorted [5,6,7,7,8]; the two 7s occupy
        #     positions 3,4 -> rank 3.5 each -> rank_y = [1,2,3.5,5,3.5]
        # mean rank = 3 both sides.
        # dx = [-2,-1,0,1,2]; dy = [-2,-1,0.5,2,0.5]
        # cov = 4+1+0+2+1 = 8; vx = 4+1+0+1+4 = 10; vy = 4+1+.25+4+.25 = 9.5
        # rho = 8 / sqrt(10*9.5) = 8 / sqrt(95) = 0.8207826816...
        rho = spearman_rho([1, 2, 3, 4, 5], [5, 6, 7, 8, 7])
        self.assertAlmostEqual(rho, 0.8207826816, places=6)

    def test_hand_computed_ties_on_both_sides_cancel_to_zero(self):
        # x = [1,1,2,2] -> rank_x = [1.5,1.5,3.5,3.5]
        # y = [1,2,1,2] -> sorted y = [1,1,2,2]; the two 1s (orig idx 0,2)
        #     -> rank 1.5 each; the two 2s (orig idx 1,3) -> rank 3.5 each
        #     -> rank_y = [1.5,3.5,1.5,3.5]
        # dx = [-1,-1,1,1]; dy = [-1,1,-1,1]; cov = 1-1-1+1 = 0 -> rho = 0
        rho = spearman_rho([1, 1, 2, 2], [1, 2, 1, 2])
        self.assertAlmostEqual(rho, 0.0)

    def test_all_tied_on_one_side_is_none(self):
        # every human score identical -> zero rank variance -> undefined,
        # not a fabricated 0.0
        self.assertIsNone(spearman_rho([5, 5, 5], [1, 2, 3]))

    def test_fewer_than_two_points_is_none(self):
        self.assertIsNone(spearman_rho([1], [1]))
        self.assertIsNone(spearman_rho([], []))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            spearman_rho([1, 2], [1, 2, 3])


# ------------------------------------------------------------ _parse_score


class TestParseScore(unittest.TestCase):
    def test_plain_integer(self):
        self.assertEqual(_parse_score("7"), 7)

    def test_ten(self):
        self.assertEqual(_parse_score("10"), 10)

    def test_integer_in_sentence_first_line(self):
        self.assertEqual(_parse_score("I'd rate this a 6 out of 10.\nbecause..."), 6)

    def test_two_digit_number_not_confused_for_bounded_digit(self):
        # "23" contains no standalone 1-10 integer under the guarded regex
        self.assertIsNone(_parse_score("23"))

    def test_empty_or_none_is_none(self):
        self.assertIsNone(_parse_score(""))
        self.assertIsNone(_parse_score("   "))


# -------------------------------------------------------------- judge_once


class TestJudgeOnce(unittest.TestCase):
    def test_succeeds_first_try(self):
        complete = make_scripted_complete(["8"])
        self.assertEqual(judge_once("ctx", "cand", complete, retries=1), 8)

    def test_retries_once_then_succeeds(self):
        complete = make_scripted_complete(["not a number", "5"])
        self.assertEqual(judge_once("ctx", "cand", complete, retries=1), 5)

    def test_exhausts_retries_returns_none(self):
        complete = make_scripted_complete(["nope", "still nope"])
        self.assertIsNone(judge_once("ctx", "cand", complete, retries=1))


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

    def test_raises_before_exceeding_budget_and_never_overcalls(self):
        log = []
        complete = lambda p: (log.append(p), "5")[1]  # noqa: E731
        stats = {"calls": 0}
        budgeted = make_budgeted_complete(complete, stats, max_calls=2)
        budgeted("p1")
        budgeted("p2")
        with self.assertRaises(BudgetExceeded):
            budgeted("p3")
        # the 3rd call must never have reached the underlying provider
        self.assertEqual(len(log), 2)
        self.assertEqual(stats["calls"], 2)


# ------------------------------------------------------------- caching


class TestScoreCandidateCache(unittest.TestCase):
    def test_miss_then_hit_no_second_call(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "cache.jsonl")
            log = []
            complete = make_lookup_judge({"a joke": 7}, call_log=log)
            stats = {}
            first = score_candidate("ctx", "a joke", complete, cache, stats)
            self.assertAlmostEqual(first, 0.7)  # normalized /10
            self.assertEqual(len(log), 1)
            self.assertEqual(stats.get("calls", 0), 0)  # score_candidate
                                                         # itself doesn't
                                                         # track "calls";
                                                         # that's the
                                                         # budget wrapper's
                                                         # job -- only
                                                         # cache_hits here.

            second = score_candidate("ctx", "a joke", complete, cache, stats)
            self.assertAlmostEqual(second, 0.7)
            self.assertEqual(len(log), 1)  # NO second provider call
            self.assertEqual(stats["cache_hits"], 1)
            cache.close()

    def test_resume_from_disk_reloads_prior_answers(self):
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "cache.jsonl"
            cache1 = LabelCache(cache_path)
            complete = make_lookup_judge({"a joke": 9})
            score_candidate("ctx", "a joke", complete, cache1, {})
            cache1.close()

            # Simulate a fresh process: new LabelCache instance pointed at
            # the same on-disk path, and a judge that must NEVER be called
            # again -- if the resumed run were to re-spend, this raises.
            cache2 = LabelCache(cache_path)

            def _must_not_be_called(prompt):
                raise AssertionError("resumed run re-spent a cached call")

            result = score_candidate("ctx", "a joke", _must_not_be_called,
                                     cache2, {})
            self.assertAlmostEqual(result, 0.9)
            cache2.close()

    def test_different_context_same_candidate_text_not_conflated(self):
        # cache_key includes context -- the same reply text under two
        # different prompts must be judged (and cached) independently.
        self.assertNotEqual(cache_key("ctx A", "same reply"),
                            cache_key("ctx B", "same reply"))

    def test_unparseable_is_never_cached(self):
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "cache.jsonl")
            complete = make_scripted_complete(["garbage", "still garbage"])
            stats = {}
            result = score_candidate("ctx", "a joke", complete, cache, stats,
                                     retries=1)
            self.assertIsNone(result)
            self.assertEqual(stats.get("unparseable"), 1)
            self.assertIsNone(cache.get("oogiri-funniness-exp012-v1",
                                        cache_key("ctx", "a joke")))
            cache.close()


# ------------------------------------------------------------ sample_groups


class TestSampleGroups(unittest.TestCase):
    def _groups(self, n, sizes=None):
        sizes = sizes or [2] * n
        return [_group("prompt %d" % i,
                       [_cand("c%d" % j, float(j)) for j in range(sizes[i])],
                       source_id="g%03d" % i)
               for i in range(n)]

    def test_deterministic_given_same_seed(self):
        groups = self._groups(20)
        a = sample_groups(groups, 5, seed=42, min_candidates=2)
        b = sample_groups(groups, 5, seed=42, min_candidates=2)
        self.assertEqual([g.source_id for g in a], [g.source_id for g in b])
        self.assertEqual(len(a), 5)

    def test_different_seed_can_differ(self):
        groups = self._groups(20)
        a = sample_groups(groups, 5, seed=1, min_candidates=2)
        b = sample_groups(groups, 5, seed=2, min_candidates=2)
        self.assertNotEqual([g.source_id for g in a], [g.source_id for g in b])

    def test_filters_below_min_candidates(self):
        groups = self._groups(3, sizes=[1, 2, 3])
        # sizes[0]=1 is actually invalid for RankedGroup construction only
        # if < 1; RankedGroup allows >=1 candidate, so this group is legal
        # but should be excluded by min_candidates=2.
        eligible = sample_groups(groups, 10, seed=0, min_candidates=2)
        self.assertEqual(len(eligible), 2)
        self.assertNotIn("g000", [g.source_id for g in eligible])

    def test_n_prompts_over_population_returns_all_eligible(self):
        groups = self._groups(4)
        out = sample_groups(groups, 100, seed=0, min_candidates=2)
        self.assertEqual(len(out), 4)


# ----------------------------------------------------------- certify_group


class TestCertifyGroup(unittest.TestCase):
    def test_rho_matches_known_judge_scores(self):
        # human scores (star) in candidate order: 1, 5, 9 (increasing)
        # judge scores scripted to be increasing too -> perfect positive
        # correlation, hand-verifiable: rho == 1.0.
        group = _group("Why did it happen?", [
            _cand("low effort reply", 1.0),
            _cand("mid effort reply", 5.0),
            _cand("great reply", 9.0),
        ])
        judge = make_lookup_judge({
            "low effort reply": 2, "mid effort reply": 5, "great reply": 9,
        })
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "cache.jsonl")
            result = certify_group(group, judge, cache, {})
            cache.close()
        self.assertEqual(result["n_candidates"], 3)
        self.assertEqual(result["n_scored"], 3)
        self.assertEqual(result["n_unparseable"], 0)
        self.assertAlmostEqual(result["rho"], 1.0)

    def test_unparseable_candidate_dropped_not_zeroed(self):
        group = _group("prompt", [
            _cand("good reply", 9.0),
            _cand("bad reply", 1.0),
            _cand("mystery reply", 5.0),
        ])

        def flaky_judge(prompt):
            if "mystery reply" in prompt:
                return "not a number"  # always unparseable, both retries
            if "good reply" in prompt:
                return "9"
            return "1"

        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "cache.jsonl")
            result = certify_group(group, flaky_judge, cache, {}, retries=1)
            cache.close()
        self.assertEqual(result["n_candidates"], 3)
        self.assertEqual(result["n_scored"], 2)
        self.assertEqual(result["n_unparseable"], 1)
        self.assertAlmostEqual(result["rho"], 1.0)  # the 2 scored are
                                                     # perfectly concordant


# --------------------------------------------------------- run_certification


class TestRunCertification(unittest.TestCase):
    def _three_groups(self):
        return [
            _group("prompt A", [_cand("a1", 1.0), _cand("a2", 5.0),
                                _cand("a3", 9.0)], source_id="A"),
            _group("prompt B", [_cand("b1", 1.0), _cand("b2", 5.0),
                                _cand("b3", 9.0)], source_id="B"),
            _group("prompt C", [_cand("c1", 1.0), _cand("c2", 5.0),
                                _cand("c3", 9.0)], source_id="C"),
        ]

    def _matching_judge(self):
        # every candidate's judge score == its human score digit -- rho=1.0
        # for every fully-scored group.
        scores = {}
        for letter in "abc":
            scores["%s1" % letter] = 1
            scores["%s2" % letter] = 5
            scores["%s3" % letter] = 9
        return make_lookup_judge(scores)

    def test_budget_stop_leaves_partial_report(self):
        groups = self._three_groups()
        judge = self._matching_judge()
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "cache.jsonl")
            # 9 candidates total, 3 per group; cap at 5 -> group A (calls
            # 1-3) completes, group B gets 2 of its 3 candidates (calls
            # 4-5) before the budget wrapper refuses the 6th call.
            result = run_certification(groups, judge, cache, max_calls=5)
            cache.close()
        self.assertTrue(result["call_stats"]["budget_exhausted"])
        self.assertEqual(result["call_stats"]["calls"], 5)
        self.assertEqual(result["n_prompts_requested"], 3)
        self.assertEqual(result["n_prompts_scored"], 1)  # only A completed
        self.assertEqual(result["per_prompt"][0]["source_id"], "A")

    def test_completes_under_generous_budget(self):
        groups = self._three_groups()
        judge = self._matching_judge()
        with tempfile.TemporaryDirectory() as td:
            cache = LabelCache(Path(td) / "cache.jsonl")
            result = run_certification(groups, judge, cache, max_calls=100)
            cache.close()
        self.assertFalse(result["call_stats"]["budget_exhausted"])
        self.assertEqual(result["n_prompts_scored"], 3)
        self.assertEqual(result["n_prompts_with_valid_rho"], 3)
        self.assertAlmostEqual(result["mean_rho"], 1.0)
        self.assertIsNotNone(result["mean_rho_bootstrap_ci"])
        self.assertEqual(result["call_stats"]["calls"], 9)

    def test_resumed_run_serves_everything_from_cache(self):
        groups = self._three_groups()
        judge = self._matching_judge()
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "cache.jsonl"
            cache1 = LabelCache(cache_path)
            run_certification(groups, judge, cache1, max_calls=100)
            cache1.close()

            cache2 = LabelCache(cache_path)

            def _must_not_be_called(prompt):
                raise AssertionError("resumed run re-spent a cached call")

            result = run_certification(groups, _must_not_be_called, cache2,
                                       max_calls=100)
            cache2.close()
        self.assertEqual(result["call_stats"]["calls"], 0)
        self.assertEqual(result["call_stats"]["cache_hits"], 9)
        self.assertEqual(result["n_prompts_scored"], 3)
        self.assertAlmostEqual(result["mean_rho"], 1.0)


# ------------------------------------------------------- load_and_sample


class TestLoadAndSampleLicenseFlag(unittest.TestCase):
    def test_missing_allowed_licenses_arg_raises_type_error(self):
        with self.assertRaises(TypeError):
            load_and_sample(5, 0, jsonl_path=str(FIXTURE))  # noqa

    def test_disallowing_research_only_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_and_sample(5, 0, allowed_licenses=["commercial_safe"],
                            jsonl_path=str(FIXTURE))

    def test_empty_allowed_licenses_raises(self):
        with self.assertRaises(ValueError):
            load_and_sample(5, 0, allowed_licenses=[], jsonl_path=str(FIXTURE))

    def test_research_only_allowed_succeeds_and_filters_singletons(self):
        groups, stats = load_and_sample(
            5, 0, allowed_licenses=["research_only"], jsonl_path=str(FIXTURE))
        # fixture has 2 multi-candidate T2T groups (tomato x3, ocean x2)
        # plus 1 singleton (dropped by the adapter's own MIN_GROUP_SIZE=2
        # before this function ever sees it) and 1 I2T row (excluded).
        self.assertEqual(len(groups), 2)
        for g in groups:
            self.assertEqual(g.license_class, "research_only")
        self.assertEqual(stats.ranked_groups_emitted, 2)

    def test_min_candidates_further_filters(self):
        groups, _ = load_and_sample(
            5, 0, allowed_licenses=["research_only"], jsonl_path=str(FIXTURE),
            min_candidates=3)
        self.assertEqual(len(groups), 1)  # only the tomato group has 3
        self.assertEqual(groups[0].context, "Why did the tomato blush?")


if __name__ == "__main__":
    unittest.main()

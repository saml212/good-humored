"""Unit tests for the inferential layer (benchmark/stats.py) — pure
functions, no API calls, no network.

Run: python3 -m unittest discover benchmark/tests -v

Test groups:
  - known-answer:  hand-computable small cases (verified independently of
    stats.py's own percentile helper where noted).
  - determinism:   same seed -> byte-identical output.
  - calibration:   permutation p-values behave sanely under a null where
    both groups genuinely share one distribution.
"""

import unittest

from benchmark.stats import (bootstrap_ci, cliffs_delta,
                              cliffs_delta_magnitude, cross_model_null, mean,
                              pairwise_cliffs_delta,
                              permutation_test_divergence,
                              rank_biserial_correlation)


# ----------------------------------------------------- permutation test


class TestPermutationTestDivergence(unittest.TestCase):
    def test_needs_two_runs_per_model(self):
        with self.assertRaises(ValueError):
            permutation_test_divergence([["a", "b"]], [["c"], ["d"]])

    def test_known_extreme_split_is_exact(self):
        # 4 identical runs (self-jaccard = 1.0, the "lookup table"
        # signature from test_metrics.py) vs. 4 mutually disjoint runs
        # (self-jaccard = 0.0). This is the ONLY possible way to split
        # the pooled 8 runs into two groups of 4 that reaches |stat|=1.0
        # exactly (any mixed group has 0 < self-jaccard < 1), and it's
        # reachable in exactly 2 of the C(8,4)=70 splits (the true
        # grouping, labeled either way round) -- so p = 2/70 is
        # hand-verifiable, not just plausible.
        identical = [["cat", "dog"]] * 4
        disjoint = [["a1", "a2"], ["b1", "b2"], ["c1", "c2"], ["d1", "d2"]]
        r = permutation_test_divergence(identical, disjoint, seed=0)
        self.assertEqual(r["method"], "exact")
        self.assertEqual(r["n_perm"], 70)
        self.assertAlmostEqual(r["stat"], 1.0)
        self.assertAlmostEqual(r["p_value"], 2 / 70)

    def test_exact_branch_is_seed_independent(self):
        # At N=4 vs N=4 the exact branch is used, so the result must not
        # depend on `seed` at all (it's a full enumeration, not a sample).
        identical = [["cat", "dog"]] * 4
        disjoint = [["a1", "a2"], ["b1", "b2"], ["c1", "c2"], ["d1", "d2"]]
        r_seed0 = permutation_test_divergence(identical, disjoint, seed=0)
        r_seed99 = permutation_test_divergence(identical, disjoint, seed=99)
        self.assertEqual(r_seed0, r_seed99)

    def test_monte_carlo_branch_is_seed_deterministic(self):
        # Force the Monte Carlo branch with large, unequal-ish groups
        # (C(20,10) = 184,756 > _EXACT_LIMIT).
        import random
        rng = random.Random(0)
        paths_a = [[str(rng.randrange(50)) for _ in range(5)]
                   for _ in range(10)]
        paths_b = [[str(rng.randrange(50)) for _ in range(5)]
                   for _ in range(10)]
        r1 = permutation_test_divergence(paths_a, paths_b, n_perm=500, seed=7)
        r2 = permutation_test_divergence(paths_a, paths_b, n_perm=500, seed=7)
        self.assertEqual(r1, r2)
        self.assertEqual(r1["method"], "monte_carlo")

    def test_p_value_calibration_on_null_data(self):
        # Both groups drawn from the SAME random topic-generating process
        # (a genuine null: no true divergence-level difference). Exact
        # permutation tests are provably conservative-or-exact (Good
        # 2005): P(p <= alpha) <= alpha under exchangeability, for ANY
        # alpha, non-asymptotically -- so false positives at the 5%
        # level should be rare. A 300-trial simulation during development
        # of this test measured an empirical rejection rate of 3% (9/300)
        # at alpha=0.05, confirming no miscalibration bug. But with N=4
        # runs/model the p-value grid is coarse (multiples of 1/70), and
        # over a SMALL number of independent trials the observed pass
        # rate is itself a noisy estimate of a low-probability event --
        # e.g. an 8-trial run of this exact test previously showed 2/8
        # "failures" (25%), which is within ordinary binomial variance
        # for a ~5% event at n=8 (P(Binomial(8, 0.05) >= 2) =~ 5.7%, not
        # actually anomalous) but made a strict per-trial bound flaky.
        # We use n_trials=40 and a generously lenient bound (80% pass)
        # that comfortably clears the true ~5-8% false-positive rate
        # while still catching a genuinely broken calibration (e.g. a
        # bug that made half of null trials "significant").
        import random
        vocab = ["cat", "dog", "gym", "coffee", "work", "math", "parrot",
                  "tea", "bike", "rain"]
        n_pass = 0
        n_trials = 40
        for trial_seed in range(n_trials):
            rng = random.Random(1000 + trial_seed)
            paths_a = [[rng.choice(vocab) for _ in range(5)] for _ in range(4)]
            paths_b = [[rng.choice(vocab) for _ in range(5)] for _ in range(4)]
            r = permutation_test_divergence(paths_a, paths_b, seed=trial_seed)
            if r["p_value"] > 0.05:
                n_pass += 1
        self.assertGreaterEqual(n_pass, int(0.8 * n_trials))


# --------------------------------------------------------- bootstrap CI


class TestBootstrapCI(unittest.TestCase):
    def test_needs_two_values(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([1.0])

    def test_constant_values_collapse_to_a_point(self):
        r = bootstrap_ci([5.0, 5.0, 5.0, 5.0], seed=0)
        self.assertEqual(r["method"], "exact")
        self.assertEqual(r["n_boot"], 256)  # 4**4, all resamples enumerated
        self.assertEqual(r["point"], 5.0)
        self.assertEqual(r["lo"], 5.0)
        self.assertEqual(r["hi"], 5.0)

    def test_known_exact_percentiles_n4(self):
        # Independently re-derived (outside stats.py) by enumerating all
        # 4**4=256 resamples of [1,2,3,4] and computing the same linear-
        # interpolation percentile by hand: lo=1.5, hi=3.5. See the
        # session's verification script; reproduced here as the
        # hand-computable known answer.
        r = bootstrap_ci([1, 2, 3, 4], seed=0)
        self.assertEqual(r["method"], "exact")
        self.assertEqual(r["n_boot"], 256)
        self.assertEqual(r["point"], 2.5)
        self.assertAlmostEqual(r["lo"], 1.5)
        self.assertAlmostEqual(r["hi"], 3.5)

    def test_exact_ci_seed_independent(self):
        r_a = bootstrap_ci([1, 2, 3, 4], seed=0)
        r_b = bootstrap_ci([1, 2, 3, 4], seed=12345)
        self.assertEqual(r_a, r_b)

    def test_monte_carlo_branch_used_above_exact_limit(self):
        # n=7 -> 7**7=823,543 > _EXACT_LIMIT, forces Monte Carlo.
        r = bootstrap_ci([1, 2, 3, 4, 5, 6, 7], seed=0)
        self.assertEqual(r["method"], "monte_carlo")
        self.assertEqual(r["n_boot"], 10000)

    def test_monte_carlo_deterministic_for_same_seed(self):
        r1 = bootstrap_ci([1, 2, 3, 4, 5, 6, 7], n_boot=2000, seed=3)
        r2 = bootstrap_ci([1, 2, 3, 4, 5, 6, 7], n_boot=2000, seed=3)
        self.assertEqual(r1, r2)

    def test_ci_contains_point_and_data_range(self):
        r = bootstrap_ci([2.0, 4.0, 6.0, 8.0, 10.0], seed=0)
        self.assertLessEqual(r["lo"], r["point"])
        self.assertLessEqual(r["point"], r["hi"])
        self.assertGreaterEqual(r["lo"], 2.0)
        self.assertLessEqual(r["hi"], 10.0)

    def test_custom_stat_fn(self):
        r = bootstrap_ci([1, 2, 3, 4], stat_fn=max, seed=0)
        self.assertEqual(r["point"], 4)


# ------------------------------------------------------- cross_model_null


class TestCrossModelNull(unittest.TestCase):
    def test_needs_two_models(self):
        with self.assertRaises(ValueError):
            cross_model_null({"only_one": [["a", "b"]] * 4}, n_perm=10)

    def test_fully_shared_pool_label_shuffle_has_no_power(self):
        # The extreme case the design brief specifically warned about:
        # every model emits the IDENTICAL path every run (the strongest
        # possible shared pool). label_shuffle must show ~no signal
        # here, because shuffling identical data changes nothing --
        # this is the documented blind spot, not a bug.
        homogeneous = {"m1": [["t1", "t2"]] * 4,
                       "m2": [["t1", "t2"]] * 4,
                       "m3": [["t1", "t2"]] * 4}
        r = cross_model_null(homogeneous, n_perm=300, seed=0)
        self.assertEqual(r["observed_mean_cross_jaccard"], 1.0)
        # Every reshuffled grouping of identical paths is ALSO 1.0, so
        # observed is never exceeded but always tied -> p must be 1.0
        # exactly (every permutation counts as "extreme").
        self.assertEqual(r["label_shuffle"]["p_value"], 1.0)

    def test_fully_shared_pool_pooled_baseline_detects_it(self):
        # Same homogeneous data: pooled_frequency_baseline draws bag-of-
        # topics synthetic runs from the pooled frequency table, which
        # will usually NOT reproduce the exact duplication -> observed
        # (1.0, the maximum possible) should exceed most synthetic
        # draws, giving a small p-value. This is the contrast the
        # module's docstring exists to explain: the two nulls answer
        # different questions and can disagree sharply on the same data.
        homogeneous = {"m1": [["t1", "t2"]] * 4,
                       "m2": [["t1", "t2"]] * 4,
                       "m3": [["t1", "t2"]] * 4}
        r = cross_model_null(homogeneous, n_perm=300, seed=0)
        self.assertLess(r["pooled_frequency_baseline"]["p_value"], 0.2)

    def test_fully_disjoint_pools_give_trivial_p_one(self):
        # jaccard is bounded below by 0, so when observed IS the
        # theoretical minimum (fully disjoint per-model vocabularies),
        # every permuted/synthetic statistic satisfies >= observed
        # trivially -> both p-values must be exactly 1.0. Hand-verifiable
        # boundary case (no simulation needed to know the answer).
        disjoint = {"m1": [["t1a", "t1b"]] * 4,
                    "m2": [["t2a", "t2b"]] * 4,
                    "m3": [["t3a", "t3b"]] * 4}
        r = cross_model_null(disjoint, n_perm=200, seed=0)
        self.assertEqual(r["observed_mean_cross_jaccard"], 0.0)
        self.assertEqual(r["label_shuffle"]["p_value"], 1.0)
        self.assertEqual(r["pooled_frequency_baseline"]["p_value"], 1.0)

    def test_deterministic_for_same_seed(self):
        mp = {"m1": [["cat", "dog"], ["cat", "gym"], ["work", "dog"],
                     ["cat", "work"]],
              "m2": [["cat", "coffee"], ["gym", "tea"], ["cat", "dog"],
                     ["work", "bike"]]}
        r1 = cross_model_null(mp, n_perm=200, seed=5)
        r2 = cross_model_null(mp, n_perm=200, seed=5)
        self.assertEqual(r1, r2)


# ------------------------------------------------------------ effect sizes


class TestEffectSizes(unittest.TestCase):
    def test_cliffs_delta_all_x_greater(self):
        self.assertEqual(cliffs_delta([4, 5, 6], [1, 2, 3]), 1.0)

    def test_cliffs_delta_all_x_less(self):
        self.assertEqual(cliffs_delta([1, 2, 3], [4, 5, 6]), -1.0)

    def test_cliffs_delta_ties_cancel(self):
        self.assertEqual(cliffs_delta([1, 2], [1, 2]), 0.0)

    def test_cliffs_delta_known_partial_value(self):
        # x=[1,2,3], y=[2,2,4]: pairs (9 total)
        # 1v2:< 1v2:< 1v4:<  2v2:tie 2v2:tie 2v4:<  3v2:> 3v2:> 3v4:<
        # more=2, less=5 -> delta = (2-5)/9 = -1/3
        self.assertAlmostEqual(cliffs_delta([1, 2, 3], [2, 2, 4]), -1 / 3)

    def test_rank_biserial_matches_cliffs_delta(self):
        x, y = [1, 5, 3, 9], [2, 2, 8, 1]
        self.assertEqual(rank_biserial_correlation(x, y), cliffs_delta(x, y))

    def test_magnitude_bands(self):
        self.assertEqual(cliffs_delta_magnitude(0.05), "negligible")
        self.assertEqual(cliffs_delta_magnitude(0.2), "small")
        self.assertEqual(cliffs_delta_magnitude(0.4), "medium")
        self.assertEqual(cliffs_delta_magnitude(0.9), "large")
        self.assertEqual(cliffs_delta_magnitude(-0.9), "large")  # sign-agnostic

    def test_pairwise_cliffs_delta_keys_and_values(self):
        mv = {"b": [1, 2, 3], "a": [4, 5, 6]}
        out = pairwise_cliffs_delta(mv)
        self.assertIn("a|b", out)  # sorted-name key, matches metrics.py convention
        self.assertEqual(out["a|b"], cliffs_delta([4, 5, 6], [1, 2, 3]))

    def test_pairwise_needs_two_models(self):
        with self.assertRaises(ValueError):
            pairwise_cliffs_delta({"only_one": [1, 2, 3]})


# -------------------------------------------------------------- mean helper


class TestMean(unittest.TestCase):
    def test_mean_basic(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            mean([])


if __name__ == "__main__":
    unittest.main()

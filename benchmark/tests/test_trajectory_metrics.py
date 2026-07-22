"""Unit tests for benchmark/trajectory_metrics.py (EXP-015).

Every test injects a FAKE embed_fn (orthogonal / hand-picked-angle one-hot
vectors) so this suite needs zero network access, no model download, and
no sentence_transformers import at all -- mirrors test_label_space.py's
fallback-mode tests (deterministic, dependency-free) rather than its
real-embedding-mode tests (which need the actual model).

Run: python3 -m pytest benchmark/tests/test_trajectory_metrics.py -q
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark.trajectory_metrics import (
    build_report, compute_model_aggregate, compute_run_metrics,
    cosine_distance, embed_paths, entropy_of_stepsize_distribution,
    entropy_of_topic_distribution, oscillation_guard,
    permutation_test_spearman, shannon_entropy_from_counts, spearman_rho,
)


def _write_summary(path: Path, per_model=None, failures=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"per_model": per_model or {}, "failures": failures or []}, f)


# --------------------------------------------------------------- vector math


class TestCosineDistance(unittest.TestCase):
    def test_identical_vectors_zero_distance(self):
        self.assertAlmostEqual(cosine_distance((1, 0, 0), (1, 0, 0)), 0.0)

    def test_orthogonal_vectors_distance_one(self):
        self.assertAlmostEqual(cosine_distance((1, 0, 0), (0, 1, 0)), 1.0)

    def test_opposite_vectors_distance_two(self):
        self.assertAlmostEqual(cosine_distance((1, 0, 0), (-1, 0, 0)), 2.0)

    def test_non_unit_vectors_still_normalized_by_norms(self):
        # (2,0) and (0,3): cos_sim=0 regardless of magnitude -> distance 1.0.
        self.assertAlmostEqual(cosine_distance((2, 0), (0, 3)), 1.0)

    def test_zero_vector_is_maximally_distant_not_a_crash(self):
        self.assertAlmostEqual(cosine_distance((0, 0, 0), (1, 0, 0)), 2.0)
        self.assertAlmostEqual(cosine_distance((0, 0, 0), (0, 0, 0)), 2.0)


# ------------------------------------------------------------------ entropy


class TestShannonEntropyFromCounts(unittest.TestCase):
    def test_single_bucket_is_zero(self):
        self.assertEqual(shannon_entropy_from_counts([5]), 0.0)

    def test_all_zero_is_zero(self):
        self.assertEqual(shannon_entropy_from_counts([0, 0]), 0.0)

    def test_two_equal_buckets_is_one_bit(self):
        self.assertAlmostEqual(shannon_entropy_from_counts([3, 3]), 1.0)

    def test_three_equal_buckets_is_log2_3(self):
        self.assertAlmostEqual(shannon_entropy_from_counts([1, 1, 1]),
                                math.log2(3))


class TestEntropyOfStepsizeDistribution(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(entropy_of_stepsize_distribution([]))

    def test_all_same_bin_is_zero(self):
        # 1.0 and 1.0 both land in bin index 5 of 10 bins over [0, 2].
        self.assertAlmostEqual(entropy_of_stepsize_distribution([1.0, 1.0]), 0.0)

    def test_upper_bound_clamps_into_last_bin(self):
        # 2.0 is the theoretical max (opposite unit vectors); must clamp
        # into bin 9, not index out of range or spill into a phantom bin.
        self.assertAlmostEqual(entropy_of_stepsize_distribution([2.0, 2.0]), 0.0)

    def test_two_distinct_bins_is_binary_entropy(self):
        # [0.1, 0.1, 1.9] -> bins {0: 2, 9: 1} -> H(2/3) = 0.9182958...
        h = entropy_of_stepsize_distribution([0.1, 0.1, 1.9])
        self.assertAlmostEqual(h, -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3)))


class TestEntropyOfTopicDistribution(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(entropy_of_topic_distribution([]))

    def test_all_distinct_topics_is_log2_n(self):
        self.assertAlmostEqual(entropy_of_topic_distribution(["a", "b", "c"]),
                                math.log2(3))

    def test_two_topic_oscillation_is_one_bit(self):
        self.assertAlmostEqual(
            entropy_of_topic_distribution(["x", "y", "x", "y"]), 1.0)

    def test_single_topic_repeated_is_zero(self):
        self.assertEqual(entropy_of_topic_distribution(["a", "a", "a"]), 0.0)


# ------------------------------------------------------- per-run metrics


class TestComputeRunMetrics(unittest.TestCase):
    def test_three_orthogonal_topics_hand_computed(self):
        # a=(1,0,0,0), b=(0,1,0,0), c=(0,0,1,0): mutually orthogonal ->
        # every step distance is exactly 1.0.
        path = ["a", "b", "c"]
        embeddings = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
        r = compute_run_metrics(path, embeddings)

        self.assertEqual(r["n_topics"], 3)
        self.assertEqual(r["step_sizes"], [1.0, 1.0])
        self.assertAlmostEqual(r["mean_step_size"], 1.0)
        self.assertAlmostEqual(r["median_step_size"], 1.0)
        # both steps land in the same fixed bin -> zero entropy (def. A).
        self.assertAlmostEqual(r["entropy_stepsize_binned"], 0.0)
        # three distinct topics, uniform -> log2(3) (def. B).
        self.assertAlmostEqual(r["entropy_topic_distribution"], math.log2(3))
        # centroid = (1/3, 1/3, 1/3, 0); distance from any of a/b/c to it
        # is 1 - (1/3)/sqrt(1/3) = 1 - sqrt(1/3) by symmetry, hand-derived
        # independently of the module's own _centroid/cosine_distance code.
        expected_dist = 1 - math.sqrt(1 / 3)
        self.assertAlmostEqual(r["mean_distance_to_centroid"], expected_dist)

    def test_two_topic_oscillation_between_opposite_vectors(self):
        # The registered gaming vector's canonical shape: alternate
        # between two maximally-distant points. Every step distance is
        # 2.0 (opposite unit vectors) -> high mean step-size but a
        # degenerate (single-bin) step-size distribution AND only 2
        # distinct topics visited.
        path = ["x", "y", "x", "y"]
        embeddings = [(1, 0, 0, 0), (-1, 0, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0)]
        r = compute_run_metrics(path, embeddings)

        self.assertEqual(r["step_sizes"], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(r["mean_step_size"], 2.0)
        self.assertAlmostEqual(r["median_step_size"], 2.0)
        self.assertAlmostEqual(r["entropy_stepsize_binned"], 0.0)
        self.assertAlmostEqual(r["entropy_topic_distribution"], 1.0)
        # centroid of (1,0,0,0),(-1,0,0,0),(1,0,0,0),(-1,0,0,0) is the
        # zero vector -- a genuinely undefined direction. cosine_distance's
        # documented edge case (zero-norm -> maximally distant) fires here,
        # so mean_distance_to_centroid lands on exactly 2.0, not a crash.
        self.assertAlmostEqual(r["mean_distance_to_centroid"], 2.0)

    def test_single_turn_path_is_degenerate_but_safe(self):
        r = compute_run_metrics(["a"], [(1, 0, 0, 0)])
        self.assertEqual(r["n_topics"], 1)
        self.assertEqual(r["step_sizes"], [])
        self.assertIsNone(r["mean_step_size"])
        self.assertIsNone(r["median_step_size"])
        self.assertIsNone(r["entropy_stepsize_binned"])
        self.assertAlmostEqual(r["entropy_topic_distribution"], 0.0)
        # centroid == the single point itself -> distance 0.0.
        self.assertAlmostEqual(r["mean_distance_to_centroid"], 0.0)

    def test_empty_path_is_all_none(self):
        r = compute_run_metrics([], [])
        self.assertEqual(r["n_topics"], 0)
        self.assertIsNone(r["mean_step_size"])
        self.assertIsNone(r["entropy_topic_distribution"])
        self.assertIsNone(r["mean_distance_to_centroid"])


class TestComputeModelAggregate(unittest.TestCase):
    def test_mean_over_runs_hand_computed(self):
        run_records = [
            {"mean_step_size": 1.0, "entropy_stepsize_binned": 0.0,
             "entropy_topic_distribution": 1.58, "mean_distance_to_centroid": 0.4,
             "degradation_depth_censored": 30},
            {"mean_step_size": 0.5, "entropy_stepsize_binned": 0.9,
             "entropy_topic_distribution": 1.0, "mean_distance_to_centroid": 0.2,
             "degradation_depth_censored": 2},
        ]
        agg = compute_model_aggregate(run_records)
        self.assertEqual(agg["n_runs"], 2)
        self.assertAlmostEqual(agg["mean_step_size"], 0.75)
        self.assertAlmostEqual(agg["mean_entropy_stepsize_binned"], 0.45)
        self.assertAlmostEqual(agg["mean_entropy_topic_distribution"], 1.29)
        self.assertAlmostEqual(agg["mean_distance_to_centroid"], 0.3)
        self.assertAlmostEqual(agg["mean_censored_degradation_depth"], 16.0)


# --------------------------------------------------------------- embed_paths


class _CountingEmbedFn:
    """Fake embed_fn that records how many times it was called and with
    what texts, so tests can assert embed_paths batches into ONE call."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.calls = 0
        self.last_texts = None

    def __call__(self, texts):
        self.calls += 1
        self.last_texts = list(texts)
        return [self.vocab[t] for t in texts]


class TestEmbedPaths(unittest.TestCase):
    def test_single_batch_call_deduplicated_across_models_and_runs(self):
        vocab = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1)}
        paths_by_model = {
            "m1": [["a", "b"], ["b", "c"]],
            "m2": [["a", "c"]],
        }
        fn = _CountingEmbedFn(vocab)
        embedded = embed_paths(paths_by_model, embed_fn=fn)

        self.assertEqual(fn.calls, 1)  # one batch call for the WHOLE dataset
        self.assertEqual(set(fn.last_texts), {"a", "b", "c"})
        self.assertEqual(len(fn.last_texts), 3)  # deduplicated, not 5 raw occurrences

        self.assertEqual(embedded["m1"][0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(embedded["m1"][1], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(embedded["m2"][0], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def test_empty_dataset_makes_no_call(self):
        fn = _CountingEmbedFn({})
        embedded = embed_paths({"m1": [[]]}, embed_fn=fn)
        self.assertEqual(fn.calls, 0)
        self.assertEqual(embedded, {"m1": [[]]})


# ---------------------------------------------------------- oscillation guard


class TestOscillationGuard(unittest.TestCase):
    def test_fires_on_constructed_two_topic_oscillation(self):
        # The oscillation run's numbers are the REAL, module-derived
        # values from the hand-computed oscillation fixture above (not
        # re-asserted here, reused from that computation): mean_step=2.0,
        # entropy_stepsize=0.0, entropy_topic=1.0.
        osc_metrics = compute_run_metrics(
            ["x", "y", "x", "y"],
            [(1, 0, 0, 0), (-1, 0, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0)])

        records = [
            {"model": "normal-1", "run_index": 0, "mean_step_size": 0.6,
             "entropy_stepsize_binned": 1.2, "entropy_topic_distribution": 1.8},
            {"model": "normal-2", "run_index": 0, "mean_step_size": 0.7,
             "entropy_stepsize_binned": 1.3, "entropy_topic_distribution": 1.9},
            {"model": "normal-3", "run_index": 0, "mean_step_size": 0.65,
             "entropy_stepsize_binned": 1.25, "entropy_topic_distribution": 1.85},
            {"model": "normal-4", "run_index": 0, "mean_step_size": 0.75,
             "entropy_stepsize_binned": 1.35, "entropy_topic_distribution": 1.95},
            {"model": "oscillator", "run_index": 0,
             "mean_step_size": osc_metrics["mean_step_size"],
             "entropy_stepsize_binned": osc_metrics["entropy_stepsize_binned"],
             "entropy_topic_distribution": osc_metrics["entropy_topic_distribution"]},
        ]

        result = oscillation_guard(records)

        # Hand-verified thresholds (see module docstring's percentile
        # convention, stats._percentile's linear-interpolation formula):
        # pooled step-sizes sorted [0.6,0.65,0.7,0.75,2.0] -> P75 lands
        # exactly on index 3 -> 0.75. entropy(step) sorted
        # [0.0,1.2,1.25,1.3,1.35] -> P25 lands on index 1 -> 1.2.
        # entropy(topic) sorted [1.0,1.8,1.85,1.9,1.95] -> P25 -> 1.8.
        self.assertAlmostEqual(result["thresholds"]["stepsize_hi_value"], 0.75)
        self.assertAlmostEqual(result["thresholds"]["entropy_stepsize_lo_value"], 1.2)
        self.assertAlmostEqual(result["thresholds"]["entropy_topic_lo_value"], 1.8)

        self.assertEqual(result["n_flagged"], 1)
        self.assertEqual(result["flags"][0]["model"], "oscillator")
        self.assertTrue(result["flags"][0]["flagged_via_stepsize_entropy"])
        self.assertTrue(result["flags"][0]["flagged_via_topic_entropy"])

    def test_too_few_runs_skips_evaluation(self):
        records = [
            {"model": "m1", "run_index": 0, "mean_step_size": 1.0,
             "entropy_stepsize_binned": 1.0, "entropy_topic_distribution": 1.0},
            {"model": "m2", "run_index": 0, "mean_step_size": 2.0,
             "entropy_stepsize_binned": 0.0, "entropy_topic_distribution": 0.0},
        ]
        result = oscillation_guard(records)
        self.assertEqual(result["flags"], [])
        self.assertEqual(result["n_flagged"], 0)
        self.assertIn("note", result)

    def test_none_valued_runs_excluded_from_consideration(self):
        records = [
            {"model": "m%d" % i, "run_index": 0, "mean_step_size": 1.0,
             "entropy_stepsize_binned": 1.0, "entropy_topic_distribution": 1.0}
            for i in range(4)
        ] + [
            {"model": "degenerate", "run_index": 0, "mean_step_size": None,
             "entropy_stepsize_binned": None, "entropy_topic_distribution": 1.0},
        ]
        result = oscillation_guard(records)
        self.assertEqual(result["n_runs_considered"], 4)  # degenerate excluded


# --------------------------------------------------------------- spearman


class TestSpearmanRho(unittest.TestCase):
    def test_perfect_positive_no_ties(self):
        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_perfect_negative_no_ties(self):
        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_no_tie_standard_formula_case(self):
        # Standard no-tie Spearman formula: rho = 1 - 6*sum(d^2)/(n(n^2-1)).
        # x=[1,2,3], y=[3,1,2]: d=[-2,1,1], sum(d^2)=6, n=3 ->
        # rho = 1 - 36/24 = -0.5, independently hand-derivable.
        self.assertAlmostEqual(spearman_rho([1, 2, 3], [3, 1, 2]), -0.5)

    def test_tied_ranks_hand_derived(self):
        # x=[1,1,2] -> ranks [1.5,1.5,3]; y=[1,2,3] -> ranks [1,2,3].
        # Pearson correlation of those two rank vectors works out to
        # sqrt(3)/2 exactly (worked by hand in the module's design notes).
        self.assertAlmostEqual(spearman_rho([1, 1, 2], [1, 2, 3]), math.sqrt(3) / 2)

    def test_constant_x_is_degenerate_zero(self):
        self.assertEqual(spearman_rho([5, 5, 5], [1, 2, 3]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            spearman_rho([1, 2], [1, 2, 3])


class TestPermutationTestSpearman(unittest.TestCase):
    def test_exact_branch_hand_derived_p_value(self):
        # n=3 -> 3!=6 <= _EXACT_LIMIT, exact enumeration. Enumerating all
        # 6 permutations of y=[3,1,2] against fixed x=[1,2,3] by hand
        # gives null rhos [-0.5,-1.0,0.5,1.0,-0.5,0.5] (the third entry,
        # perm (3,1,2), is the true/observed pairing itself). |every
        # entry| >= 0.5 -> all 6 count as "as extreme as observed" ->
        # p = 6/6 = 1.0.
        r = permutation_test_spearman([1, 2, 3], [3, 1, 2], seed=0)
        self.assertEqual(r["method"], "exact")
        self.assertEqual(r["n_perm_used"], 6)
        self.assertAlmostEqual(r["rho"], -0.5)
        self.assertAlmostEqual(r["p_value"], 1.0)

    def test_monte_carlo_branch_is_seed_deterministic(self):
        # n=9 -> 9! = 362,880 > _EXACT_LIMIT (100,000) -> Monte Carlo.
        x = [1, 5, 2, 8, 3, 9, 4, 7, 6]
        y = [9, 2, 7, 1, 6, 3, 8, 4, 5]
        r1 = permutation_test_spearman(x, y, n_perm=500, seed=7)
        r2 = permutation_test_spearman(x, y, n_perm=500, seed=7)
        self.assertEqual(r1, r2)
        self.assertEqual(r1["method"], "monte_carlo")
        self.assertEqual(r1["n_perm_used"], 500)

    def test_p_value_never_exactly_zero_in_monte_carlo(self):
        x = [1, 5, 2, 8, 3, 9, 4, 7, 6]
        y = list(x)  # perfect correlation -> most extreme possible stat
        r = permutation_test_spearman(x, y, n_perm=200, seed=1)
        self.assertGreater(r["p_value"], 0.0)  # add-one correction


# ---------------------------------------------------- end-to-end build_report


class TestBuildReportEndToEnd(unittest.TestCase):
    """Full pipeline wired through a temporary pilot dir (same lane/
    summary.json shape analyze_pilot.load_lanes expects -- see
    test_analyze_pilot.py's _write_summary convention) and a fake,
    hand-computable embed_fn. Three models chosen so the headline
    permutation test lands in the EXACT branch (3! = 6) and the rho is
    independently hand-derivable (worked out in the module's design
    notes; reproduced in the test comment below)."""

    def setUp(self):
        self.pilot = Path(tempfile.mkdtemp(prefix="trajectory-metrics-test-"))
        # Orthogonal one-hot vocabulary: t1..t4.
        self.vocab = {
            "t1": (1, 0, 0, 0), "t2": (0, 1, 0, 0),
            "t3": (0, 0, 1, 0), "t4": (0, 0, 0, 1),
        }
        # alpha: run0 no repeat (survives -> censored 30); run1 repeats t1
        # at index 2 (depth=2). step sizes all 1.0 (orthogonal) both runs.
        alpha_runs = [["t1", "t2", "t3"], ["t1", "t2", "t1"]]
        # beta: run0 repeats t4 immediately (depth=1); run1 no repeat
        # (censored 30). step sizes: run0 = [0.0, 1.0] (t4->t4, t4->t2);
        # run1 = [1.0, 1.0].
        beta_runs = [["t4", "t4", "t2"], ["t3", "t4", "t2"]]
        # gamma: both runs no repeat (censored 30, 30). step sizes all 1.0.
        gamma_runs = [["t2", "t3", "t4"], ["t2", "t1", "t3"]]

        _write_summary(self.pilot / "lane-a" / "summary.json",
                       per_model={"alpha": {"paths": alpha_runs}})
        _write_summary(self.pilot / "lane-b" / "summary.json",
                       per_model={"beta": {"paths": beta_runs},
                                  "gamma": {"paths": gamma_runs}})

    def _embed_fn(self, texts):
        return [self.vocab[t] for t in texts]

    def test_per_model_hand_computed(self):
        report = build_report(self.pilot, embed_fn=self._embed_fn, seed=0)

        alpha = report["per_model"]["alpha"]
        self.assertAlmostEqual(alpha["mean_step_size"], 1.0)
        self.assertAlmostEqual(alpha["mean_censored_degradation_depth"], 16.0)  # (30+2)/2

        beta = report["per_model"]["beta"]
        self.assertAlmostEqual(beta["mean_step_size"], 0.75)  # mean(0.5, 1.0)
        self.assertAlmostEqual(beta["mean_censored_degradation_depth"], 15.5)  # (1+30)/2

        gamma = report["per_model"]["gamma"]
        self.assertAlmostEqual(gamma["mean_step_size"], 1.0)
        self.assertAlmostEqual(gamma["mean_censored_degradation_depth"], 30.0)

    def test_headline_rho_hand_derived_exact_branch(self):
        # x (step-size) = alpha:1.0, beta:0.75, gamma:1.0 -> ranks
        # beta=1, {alpha,gamma} tied at 2.5. y (censored depth) =
        # alpha:16.0, beta:15.5, gamma:30.0 -> ranks beta=1, alpha=2,
        # gamma=3 (no ties). Spearman rho of ([2.5,1,2.5], [2,1,3]) =
        # sqrt(3)/2 (worked by hand in the module's design notes: same
        # tied-rank shape as TestSpearmanRho.test_tied_ranks_hand_derived).
        report = build_report(self.pilot, embed_fn=self._embed_fn, seed=0)
        headline = report["headline_rho_stepsize_vs_censored_degradation_depth"]

        self.assertEqual(headline["n"], 3)
        self.assertEqual(headline["method"], "exact")
        self.assertEqual(headline["n_perm_used"], 6)
        self.assertAlmostEqual(headline["rho"], math.sqrt(3) / 2)
        # Hand-enumerated all 6 permutations (module design notes): null
        # rhos are [0, -sqrt3/2, +sqrt3/2, -sqrt3/2, +sqrt3/2, 0] -- 4 of
        # 6 have |rho| >= observed -> p = 4/6.
        self.assertAlmostEqual(headline["p_value"], 4 / 6)
        self.assertEqual(headline["predicted_rho"], 0.50)

    def test_report_shape_and_entropy_disclosure(self):
        report = build_report(self.pilot, embed_fn=self._embed_fn, seed=0)
        self.assertIn("entropy_definition_note", report)
        self.assertIn("oscillation_guard", report)
        self.assertIn("secondary_grok_lowest_entropy_check", report)
        # every per-run record carries BOTH entropy definitions, never
        # just one -- the disclosed-ambiguity contract.
        for run in report["per_run"]:
            self.assertIn("entropy_stepsize_binned", run)
            self.assertIn("entropy_topic_distribution", run)


if __name__ == "__main__":
    unittest.main()

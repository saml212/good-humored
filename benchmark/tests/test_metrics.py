"""Unit tests for trajectory metrics — pure functions, no API calls.
Run: python3 -m unittest discover benchmark/tests -v
"""

import unittest

from benchmark.metrics import (adjusted_rand_index, cluster_switch_stats,
                               cross_model_overlap, depth_to_degradation,
                               edit_distance, jaccard, looks_like_refusal,
                               normalize_label, path_divergence,
                               prefix_agreement_depth)


class TestNormalizeLabel(unittest.TestCase):
    def test_case_punct_articles(self):
        self.assertEqual(normalize_label("The Cats!"), "cat")

    def test_ies_plural(self):
        self.assertEqual(normalize_label("puppies"), "puppy")

    def test_ss_preserved(self):
        self.assertEqual(normalize_label("chess"), "chess")

    def test_multiword(self):
        self.assertEqual(normalize_label("Airport Security"),
                         "airport security")

    def test_sibilant_es(self):
        self.assertEqual(normalize_label("crunches"), "crunch")
        self.assertEqual(normalize_label("glasses"), "glass")

    def test_non_plural_s_endings(self):
        # audit W2: -us/-is/-as words are not plurals
        self.assertEqual(normalize_label("octopus"), "octopus")
        self.assertEqual(normalize_label("tennis"), "tennis")
        self.assertEqual(normalize_label("christmas"), "christmas")


class TestPairwise(unittest.TestCase):
    def test_jaccard_identical(self):
        self.assertEqual(jaccard(["a", "b"], ["b", "a"]), 1.0)

    def test_jaccard_disjoint(self):
        self.assertEqual(jaccard(["a"], ["b"]), 0.0)

    def test_prefix_depth(self):
        self.assertEqual(
            prefix_agreement_depth(["a", "b", "c"], ["a", "b", "x"]), 2)

    def test_edit_distance(self):
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(edit_distance([], ["a", "b"]), 2)


class TestPathDivergence(unittest.TestCase):
    def test_lookup_table_signature(self):
        # A model reading a memorized list: identical paths every run.
        paths = [["cat", "dog", "parrot"]] * 3
        d = path_divergence(paths)
        self.assertEqual(d["set_jaccard"], 1.0)
        self.assertEqual(d["prefix_depth"], 3.0)
        self.assertEqual(d["norm_edit_distance"], 0.0)

    def test_healthy_divergence(self):
        paths = [["cat", "dog", "parrot"], ["work", "coffee", "gym"]]
        d = path_divergence(paths)
        self.assertEqual(d["set_jaccard"], 0.0)
        self.assertEqual(d["prefix_depth"], 0.0)
        self.assertEqual(d["norm_edit_distance"], 1.0)

    def test_needs_two_runs(self):
        with self.assertRaises(ValueError):
            path_divergence([["a"]])


class TestCrossModel(unittest.TestCase):
    def test_ecosystem_collapse_signature(self):
        mp = {"gpt": [["cat", "dog"]], "claude": [["cat", "dog"]]}
        out = cross_model_overlap(mp)
        self.assertEqual(out["mean_cross_jaccard"], 1.0)
        self.assertEqual(out["mean_cross_prefix_depth"], 2.0)

    def test_divergent_models(self):
        mp = {"gpt": [["cat"]], "claude": [["work"]]}
        self.assertEqual(cross_model_overlap(mp)["mean_cross_jaccard"], 0.0)


class TestDegradation(unittest.TestCase):
    def test_repeat_detected(self):
        d = depth_to_degradation(["cat", "dog", "cat"])
        self.assertEqual(d["repeat_depth"], 2)
        self.assertEqual(d["depth"], 2)

    def test_survivor(self):
        d = depth_to_degradation(["cat", "dog", "gym"])
        self.assertIsNone(d["depth"])

    def test_refusal_wins_when_earlier(self):
        d = depth_to_degradation(["cat", "dog", "cat"], refusal_turns=[1])
        self.assertEqual(d["depth"], 1)


class TestClusterSwitch(unittest.TestCase):
    def test_troyer_runs(self):
        cat = {"cat": "animals", "dog": "animals", "coffee": "food",
               "tea": "food", "gym": "activities"}.get
        s = cluster_switch_stats(["cat", "dog", "coffee", "tea", "gym"],
                                 lambda t: cat(t, "other"))
        self.assertEqual(s["n_switches"], 2.0)
        self.assertAlmostEqual(s["mean_cluster_size"], 5 / 3)
        self.assertEqual(s["n_categories"], 3.0)

    def test_empty(self):
        self.assertEqual(
            cluster_switch_stats([], lambda t: t)["n_switches"], 0.0)


class TestARI(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(
            adjusted_rand_index(["a", "a", "b"], ["x", "x", "y"]), 1.0)

    def test_known_value(self):
        # -0.5 is the canonical ARI for a perfectly-balanced
        # anti-correlated 2x2 table (systematic disagreement); random
        # labelings expect 0. Hand-verified in the pre-run audit.
        a = ["a", "a", "b", "b"]
        b = ["x", "y", "x", "y"]
        self.assertAlmostEqual(adjusted_rand_index(a, b), -0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            adjusted_rand_index(["a"], ["a", "b"])


class TestRefusal(unittest.TestCase):
    def test_refusals(self):
        self.assertTrue(looks_like_refusal("I'm sorry, but I can't come up "
                                           "with another joke."))
        self.assertTrue(looks_like_refusal("I am out of jokes."))

    def test_non_refusals(self):
        self.assertFalse(looks_like_refusal(
            "Why did the chicken cross the road?"))


if __name__ == "__main__":
    unittest.main()

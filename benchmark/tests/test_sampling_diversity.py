"""Unit tests for sampling-diversity metrics — pure functions, no API
calls. Hand-computed values checked explicitly (see comments), plus an
integration-style check that the run-set aggregator's numbers match
calling the underlying functions directly on the same data.
Run: python3 -m unittest discover benchmark/tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.metrics import path_divergence
from benchmark.sampling_diversity import (aggregate_run_set, distinct_n,
                                          group_turns_files_by_model,
                                          load_run_set,
                                          mean_pairwise_trigram_jaccard,
                                          sampling_diversity)


class TestDistinctN(unittest.TestCase):
    def test_distinct_1_hand_computed(self):
        # pooled unigrams: a,a,b,a,b,b (6 total) -> unique {a,b} = 2
        # distinct_1 = 2/6
        texts = ["a a b", "a b b"]
        self.assertAlmostEqual(distinct_n(texts, 1), 2 / 6)

    def test_distinct_2_hand_computed(self):
        # bigrams: (a,a)(a,b) from text1, (a,b)(b,b) from text2 -> 4
        # total, 3 unique {(a,a),(a,b),(b,b)} -> distinct_2 = 3/4
        texts = ["a a b", "a b b"]
        self.assertAlmostEqual(distinct_n(texts, 2), 3 / 4)

    def test_empty_input_is_zero_not_error(self):
        self.assertEqual(distinct_n([], 1), 0.0)

    def test_texts_shorter_than_n_are_zero_not_error(self):
        # single-token texts have no trigrams at all -> total=0 guard
        self.assertEqual(distinct_n(["hi", "yo"], 3), 0.0)

    def test_all_unique_is_one(self):
        self.assertEqual(distinct_n(["a b", "c d"], 1), 1.0)

    def test_all_identical_pool_is_low(self):
        # 2 texts x 2 tokens = 4 total unigrams, only 2 distinct types
        self.assertAlmostEqual(distinct_n(["a b", "a b"], 1), 2 / 4)


class TestMeanPairwiseTrigramJaccard(unittest.TestCase):
    # A and B share 3 of 4 trigrams (jaccard 3/5 = 0.6); C is disjoint
    # from both (jaccard 0). Mean over the 3 pairs = (0.6+0+0)/3 = 0.2.
    A = "The cat sat on the mat"
    B = "The cat sat on the rug"
    C = "Purple elephants dance wildly tonight forever"

    def test_pairwise_hand_computed(self):
        self.assertAlmostEqual(
            mean_pairwise_trigram_jaccard([self.A, self.B, self.C]), 0.2)

    def test_two_text_case(self):
        self.assertAlmostEqual(
            mean_pairwise_trigram_jaccard([self.A, self.B]), 0.6)

    def test_needs_two_texts(self):
        with self.assertRaises(ValueError):
            mean_pairwise_trigram_jaccard([self.A])


class TestSamplingDiversityBundle(unittest.TestCase):
    A = "The cat sat on the mat"
    B = "The cat sat on the rug"

    def test_bundle_hand_computed(self):
        # unigrams: the,cat,sat,on,the,mat / the,cat,sat,on,the,rug
        # 12 total, 6 unique types (the,cat,sat,on,mat,rug) -> 6/12=0.5
        # bigrams: 5+5=10 total, 6 unique (4 shared + 2 distinct tails)
        # -> 6/10 = 0.6
        d = sampling_diversity([self.A, self.B])
        self.assertAlmostEqual(d["distinct_1"], 0.5)
        self.assertAlmostEqual(d["distinct_2"], 0.6)
        self.assertAlmostEqual(d["mean_pairwise_trigram_jaccard"], 0.6)
        self.assertEqual(d["n_texts"], 2.0)

    def test_needs_two_texts(self):
        with self.assertRaises(ValueError):
            sampling_diversity([self.A])


def _write_turns(path, run_id, topics, jokes):
    with open(path, "w") as f:
        for i, (topic, joke) in enumerate(zip(topics, jokes)):
            rec = {"run_id": run_id, "turn": i, "joke": joke,
                   "topic": topic, "refusal": False, "temperature": None,
                   "ts": 0.0}
            f.write(json.dumps(rec) + "\n")


class TestRunSetAggregation(unittest.TestCase):
    def test_aggregate_matches_direct_computation(self):
        tmp = Path(tempfile.mkdtemp(prefix="sd-test-"))
        f0 = tmp / "turns-modelA-r00.jsonl"
        f1 = tmp / "turns-modelA-r01.jsonl"
        _write_turns(f0, "modelA-r00", ["cat", "dog", "parrot"],
                    ["The cat sat on the mat.",
                     "The dog sat on the rug.",
                     "The parrot sat on the perch."])
        _write_turns(f1, "modelA-r01", ["cat", "dog", "parrot"],
                    ["The cat sat on the mat.",
                     "The dog sat on the mat.",
                     "The parrot sat on the mat."])

        agg = aggregate_run_set([f0, f1])
        loaded = load_run_set([f0, f1])

        self.assertEqual(agg["n_runs"], 2)
        self.assertEqual(agg["n_jokes"], 6)
        # same paths/jokes recomputed straight through the metric
        # functions the aggregator is supposed to be a thin wrapper over.
        self.assertEqual(agg["path_divergence"],
                         path_divergence(loaded["paths"]))
        self.assertEqual(agg["sampling_diversity"],
                         sampling_diversity(loaded["jokes"]))
        # sanity: the two runs walk an IDENTICAL topic path (lookup-table
        # signature) while the jokes' wording differs run to run — this
        # is the shape EXP-007 predicts a temperature manipulation fakes.
        self.assertEqual(agg["path_divergence"]["set_jaccard"], 1.0)

    def test_thin_group_returns_none_not_raise(self):
        # a single run: not enough for either metric (path_divergence
        # needs >=2 runs, sampling_diversity needs >=2 pooled texts).
        tmp = Path(tempfile.mkdtemp(prefix="sd-test-"))
        f0 = tmp / "turns-solo-r00.jsonl"
        _write_turns(f0, "solo-r00", ["cat"], ["only one joke here"])
        agg = aggregate_run_set([f0])
        self.assertEqual(agg["n_runs"], 1)
        self.assertEqual(agg["n_jokes"], 1)
        self.assertIsNone(agg["path_divergence"])
        self.assertIsNone(agg["sampling_diversity"])

    def test_grouping_by_model_filename_convention(self):
        # same "turns-<model>-r<NN>.jsonl" convention joke_novelty.py
        # groups by.
        tmp = Path(tempfile.mkdtemp(prefix="sd-test-"))
        _write_turns(tmp / "turns-modelA-r00.jsonl", "modelA-r00",
                    ["cat"], ["a joke"])
        _write_turns(tmp / "turns-modelA-r01.jsonl", "modelA-r01",
                    ["dog"], ["b joke"])
        _write_turns(tmp / "turns-modelB-r00.jsonl", "modelB-r00",
                    ["cat"], ["c joke"])
        groups = group_turns_files_by_model(tmp)
        self.assertEqual(set(groups), {"modelA", "modelB"})
        self.assertEqual(len(groups["modelA"]), 2)
        self.assertEqual(len(groups["modelB"]), 1)


if __name__ == "__main__":
    unittest.main()

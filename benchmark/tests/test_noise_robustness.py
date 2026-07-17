"""Unit tests for noise_robustness.py (EXP-006) — pure simulation, no
network or CLI calls. Run:
  python3 -m unittest discover benchmark/tests -v
"""

import unittest

from benchmark.noise_robustness import (EMPIRICAL_NOISE_RATES, REGIMES,
                                        TOPIC_ONTOLOGY,
                                        component_only_rates,
                                        noise_free_rates, simulate_variant)


class TestEmpiricalRates(unittest.TestCase):
    def test_rates_are_probability_distributions(self):
        for name, rates in EMPIRICAL_NOISE_RATES.items():
            total = sum(rates.values())
            self.assertAlmostEqual(total, 1.0, places=9,
                                   msg="%s rates don't sum to 1" % name)
            for v in rates.values():
                self.assertGreaterEqual(v, 0.0)

    def test_deployed_instrument_is_default_source(self):
        # EXPERIMENT_LOG.md's "Instrument decision": haiku + LABEL_PROMPT v2
        # is what the real cascade pilot runs; that combination's own
        # calibration data must exist and be usable, independent of which
        # --rate-source a given invocation picks.
        self.assertIn("haiku_v2", EMPIRICAL_NOISE_RATES)


class TestTopicOntology(unittest.TestCase):
    def test_thirty_topics(self):
        self.assertEqual(len(TOPIC_ONTOLOGY), 30)

    def test_hypernyms_are_shared_across_siblings(self):
        """The whole inflation mechanism this experiment tests requires
        hypernyms to be shared by more than one topic — if every topic had
        a unique hypernym, generalize-up noise could never manufacture
        cross-topic agreement. Lock in that the authored ontology actually
        has this property (>= 2 topics per hypernym, for every hypernym)."""
        from collections import Counter
        counts = Counter(v["hypernym"] for v in TOPIC_ONTOLOGY.values())
        for hypernym, n in counts.items():
            self.assertGreaterEqual(
                n, 2, "hypernym %r only covers 1 topic — can't inflate "
                "cross-topic overlap" % hypernym)

    def test_every_topic_has_synonyms(self):
        for topic, spec in TOPIC_ONTOLOGY.items():
            self.assertGreaterEqual(len(spec["synonyms"]), 1,
                                    "%s has no synonyms" % topic)


class TestComponentIsolation(unittest.TestCase):
    def test_component_only_zeroes_other_components(self):
        rates = EMPIRICAL_NOISE_RATES["haiku_v2"]
        only_gen = component_only_rates(rates, "generalize")
        self.assertEqual(only_gen["synonym"], 0.0)
        self.assertEqual(only_gen["other"], 0.0)
        self.assertEqual(only_gen["generalize"], rates["generalize"])
        self.assertAlmostEqual(sum(only_gen.values()), 1.0, places=9)

    def test_noise_free_is_pure_match(self):
        nf = noise_free_rates()
        self.assertEqual(nf["match"], 1.0)
        self.assertEqual(sum(v for k, v in nf.items() if k != "match"), 0.0)


class TestNoiseFreeSimulation(unittest.TestCase):
    """The most basic correctness bar: if nothing perturbs the labels, the
    noisy path IS the true path, so every bias must be EXACTLY zero — not
    approximately zero, not zero-in-expectation-over-many-reps. This is
    deterministic per replicate (match probability 1.0 means every draw
    keeps the true topic), so a small rep count suffices."""

    def test_zero_bias_every_regime(self):
        for regime_name, regime in REGIMES.items():
            summary = simulate_variant(
                regime_name, regime, noise_free_rates(),
                reps=10, base_seed=20260717, variant_name="noise_free_test")
            self.assertEqual(summary["bias_cross_jaccard_mean"], 0.0,
                             "regime=%s" % regime_name)
            self.assertEqual(summary["bias_cross_jaccard_sd"], 0.0,
                             "regime=%s" % regime_name)
            self.assertEqual(summary["bias_within_jaccard_mean"], 0.0,
                             "regime=%s" % regime_name)
            self.assertEqual(summary["bias_within_prefix_mean"], 0.0,
                             "regime=%s" % regime_name)
            self.assertEqual(summary["bias_within_edit_mean"], 0.0,
                             "regime=%s" % regime_name)


class TestGeneralizeUpInflatesOverlap(unittest.TestCase):
    """Lock in the mechanism the whole experiment exists to check: when true
    topics are DISJOINT (clean cross-model jaccard ~= 0, no genuine shared
    topic anywhere) but generalize-up noise is active, cross-model overlap
    must come out measurably ABOVE the clean baseline — collapse manufactured
    purely by two different true topics sharing one hypernym string. If this
    test ever fails, either the ontology stopped sharing hypernyms across
    topics, or the noise model stopped mapping to hypernyms, and the
    experiment's central finding is no longer backed by the code."""

    def test_disjoint_regime_generalize_only_inflates(self):
        rates = EMPIRICAL_NOISE_RATES["haiku_v2"]
        only_gen = component_only_rates(rates, "generalize")
        summary = simulate_variant(
            "disjoint", REGIMES["disjoint"], only_gen,
            reps=500, base_seed=20260717, variant_name="generalize_lock_in")
        # clean cross jaccard should be ~0 by construction (q>0, s=0)
        self.assertLess(summary["clean_cross_jaccard_mean"], 0.01)
        # noisy must be measurably higher — manufactured, not measured
        self.assertGreater(summary["noisy_cross_jaccard_mean"], 0.02)
        self.assertGreater(summary["bias_cross_jaccard_mean"], 0.02)

    def test_synonym_only_never_inflates_full_collapse(self):
        """The opposite-direction sanity check: synonym noise should only
        ever DECREASE or hold apparent overlap in the full-collapse regime
        (true paths already identical — synonym substitution can only
        introduce disagreement, never additional agreement, since a
        canonical topic and one of its own synonyms are different
        strings)."""
        rates = EMPIRICAL_NOISE_RATES["haiku_v2"]
        only_syn = component_only_rates(rates, "synonym")
        summary = simulate_variant(
            "full", REGIMES["full"], only_syn,
            reps=500, base_seed=20260717, variant_name="synonym_lock_in")
        self.assertLessEqual(summary["bias_cross_jaccard_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()

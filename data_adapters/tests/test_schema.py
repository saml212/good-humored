"""Unit tests for data_adapters/schema.py -- pure stdlib, no fixtures on
disk needed (everything constructed in-memory).
Run: python3 -m unittest discover -s data_adapters/tests -v
"""

import unittest

from data_adapters.schema import (Candidate, CorpusJoke, PreferencePair,
                                  RankedGroup, to_preference_pairs,
                                  to_preference_pairs_batch)


def _group(source_id="g1", license_class="commercial_safe", context="ctx",
          candidates=None):
    if candidates is None:
        candidates = [
            Candidate(text="high", score=9.0),
            Candidate(text="mid", score=5.0),
            Candidate(text="low", score=1.0),
        ]
    return RankedGroup(
        context=context, candidates=tuple(candidates),
        source_dataset="fixture-dataset", license_class=license_class,
        source_id=source_id)


class TestCandidate(unittest.TestCase):
    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            Candidate(text="   ", score=1.0)

    def test_valid(self):
        c = Candidate(text="a joke", score=3.0, rater_count=10)
        self.assertEqual(c.text, "a joke")
        self.assertEqual(c.rater_count, 10)


class TestRankedGroup(unittest.TestCase):
    def test_invalid_license_class_raises(self):
        with self.assertRaises(ValueError):
            _group(license_class="totally_unrestricted")

    def test_empty_context_raises(self):
        with self.assertRaises(ValueError):
            _group(context="  ")

    def test_zero_candidates_raises(self):
        with self.assertRaises(ValueError):
            RankedGroup(context="ctx", candidates=(),
                       source_dataset="d", license_class="commercial_safe",
                       source_id="g")

    def test_candidates_stored_as_tuple_even_from_list(self):
        g = RankedGroup(
            context="ctx",
            candidates=[Candidate(text="a", score=1.0)],  # list, not tuple
            source_dataset="d", license_class="commercial_safe",
            source_id="g")
        self.assertIsInstance(g.candidates, tuple)


class TestPreferencePair(unittest.TestCase):
    def test_chosen_must_score_higher(self):
        with self.assertRaises(ValueError):
            PreferencePair(
                context="ctx", chosen="a", rejected="b",
                source_dataset="d", license_class="commercial_safe",
                source_id="g", chosen_score=1.0, rejected_score=5.0,
                score_gap=-4.0)

    def test_equal_scores_raises(self):
        with self.assertRaises(ValueError):
            PreferencePair(
                context="ctx", chosen="a", rejected="b",
                source_dataset="d", license_class="commercial_safe",
                source_id="g", chosen_score=3.0, rejected_score=3.0,
                score_gap=0.0)


class TestCorpusJoke(unittest.TestCase):
    def test_invalid_license_class_raises(self):
        with self.assertRaises(ValueError):
            CorpusJoke(text="a joke", source_dataset="d",
                      license_class="nope", source_id="j1")

    def test_score_optional(self):
        j = CorpusJoke(text="a joke", source_dataset="d",
                      license_class="research_only", source_id="j1")
        self.assertIsNone(j.score)


class TestToPreferencePairs(unittest.TestCase):
    def test_basic_pairs_and_provenance_propagation(self):
        g = _group(source_id="g42", license_class="research_only",
                  context="the prompt")
        pairs = to_preference_pairs(g)
        # 3 candidates -> 3 pairs (high/mid, high/low, mid/low), all gaps > 0
        self.assertEqual(len(pairs), 3)
        for p in pairs:
            self.assertEqual(p.context, "the prompt")
            self.assertEqual(p.source_dataset, "fixture-dataset")
            self.assertEqual(p.license_class, "research_only")
            self.assertEqual(p.source_id, "g42")
            self.assertGreater(p.chosen_score, p.rejected_score)
        # sorted by descending score gap: high/low (gap=8) first
        self.assertEqual((pairs[0].chosen, pairs[0].rejected), ("high", "low"))

    def test_group_metadata_propagates_to_every_pair(self):
        """A compliance obligation stamped on the group (e.g. NYCC's
        CC-BY-4.0 attribution_required flag) must survive the conversion
        to PreferencePair -- a trainer consuming only pairs must not lose
        it."""
        g = RankedGroup(
            context="ctx",
            candidates=(Candidate(text="a", score=9.0),
                       Candidate(text="b", score=1.0)),
            source_dataset="d", license_class="commercial_safe",
            source_id="g1",
            metadata={"attribution_required": True, "attribution": "cite me"})
        pairs = to_preference_pairs(g)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].metadata,
                        {"attribution_required": True, "attribution": "cite me"})

    def test_singleton_group_yields_no_pairs(self):
        g = _group(candidates=[Candidate(text="only one", score=5.0)])
        self.assertEqual(to_preference_pairs(g), [])

    def test_exact_tie_excluded_by_default(self):
        g = _group(candidates=[
            Candidate(text="a", score=5.0),
            Candidate(text="b", score=5.0),
        ])
        self.assertEqual(to_preference_pairs(g), [])

    def test_near_tie_excluded_with_min_score_gap(self):
        g = _group(candidates=[
            Candidate(text="a", score=5.0),
            Candidate(text="b", score=4.5),
        ])
        # gap of 0.5 survives the default (min_score_gap=0.0)...
        self.assertEqual(len(to_preference_pairs(g)), 1)
        # ...but is excluded once min_score_gap >= the gap
        self.assertEqual(to_preference_pairs(g, min_score_gap=0.5), [])
        self.assertEqual(to_preference_pairs(g, min_score_gap=1.0), [])

    def test_degenerate_identical_text_pair_skipped(self):
        g = _group(candidates=[
            Candidate(text="same joke", score=9.0),
            Candidate(text="same joke", score=1.0),
        ])
        self.assertEqual(to_preference_pairs(g), [])

    def test_max_pairs_caps_and_keeps_highest_gap(self):
        g = _group()  # high(9)/mid(5)/low(1) -> pair gaps: 4, 8, 4
        pairs_all = to_preference_pairs(g)
        pairs_capped = to_preference_pairs(g, max_pairs=1)
        self.assertEqual(len(pairs_capped), 1)
        self.assertEqual(pairs_capped[0].score_gap,
                        max(p.score_gap for p in pairs_all))

    def test_rejects_non_ranked_group_input(self):
        """THE popularity-bias guard: to_preference_pairs takes exactly one
        RankedGroup. Passing a list (e.g. an attempt to compare candidates
        across two different prompts/groups) must raise, not silently
        iterate and cross-pair."""
        g1 = _group(source_id="g1")
        g2 = _group(source_id="g2")
        with self.assertRaises(TypeError):
            to_preference_pairs([g1, g2])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            to_preference_pairs(g1.candidates)  # type: ignore[arg-type]

    def test_batch_processes_groups_independently_never_cross_pairs(self):
        g1 = _group(source_id="g1", context="prompt one", candidates=[
            Candidate(text="g1-high", score=10.0),
            Candidate(text="g1-low", score=1.0),
        ])
        g2 = _group(source_id="g2", context="prompt two", candidates=[
            Candidate(text="g2-high", score=100.0),
            Candidate(text="g2-low", score=99.0),
        ])
        pairs = to_preference_pairs_batch([g1, g2])
        self.assertEqual(len(pairs), 2)
        contexts = {p.context for p in pairs}
        self.assertEqual(contexts, {"prompt one", "prompt two"})
        # g2's candidates score far higher in absolute terms than g1's, but
        # each pair's chosen/rejected only ever come from ITS OWN group --
        # no pair should ever mix text from g1 with text from g2.
        for p in pairs:
            texts = {p.chosen, p.rejected}
            self.assertTrue(texts <= {"g1-high", "g1-low"} or
                           texts <= {"g2-high", "g2-low"})


if __name__ == "__main__":
    unittest.main()

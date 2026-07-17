"""Unit tests for env/validate_semantic_novelty.py's reskin generator and
threshold-sweep helpers -- NO model download / sentence_transformers
import anywhere in this file (the validation script's own module-level
imports are lightweight for exactly this reason; sentence_transformers
and the real model.encode() call live inside main(), never at module
scope). ONE EXCEPTION: `TestSemanticScoresReferenceSet` below imports
`numpy` locally, inside its own test methods, to build tiny hand-picked
unit vectors for `semantic_scores`' extracted, dependency-injectable
templates-vs-general-corpus logic (`numpy` is a real, already-required
dependency of that function -- see its docstring -- unlike
sentence_transformers, which stays fully out of this file).
Run: python3 -m unittest discover -s env/tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from env.validate_semantic_novelty import (STOPWORDS, _content_candidates,
                                           _substitute_for, build_negatives,
                                           build_positive_records,
                                           generate_reskin, pick_threshold,
                                           semantic_scores, sweep)

# The real chatgpt25-T24 template text -- deliberately the template with
# the FEWEST substitution candidates (skeletons/fight/guts = 3), used to
# test the clamp-to-actual_depth behavior on a real, not synthetic, case.
T24_TEXT = "Why don't skeletons fight each other? They don't have the guts."
T4_TEXT = ("Why don't scientists trust atoms? Because they make up "
          "everything.")


class TestContentCandidates(unittest.TestCase):
    def test_stopwords_and_short_words_excluded(self):
        candidates = _content_candidates(T4_TEXT)
        words = [w for _, _, w in candidates]
        for stop in ("Why", "don't", "the", "Because", "they", "up"):
            self.assertNotIn(stop, words)

    def test_t24_has_exactly_three_candidates(self):
        candidates = _content_candidates(T24_TEXT)
        words = [w.lower() for _, _, w in candidates]
        self.assertEqual(words, ["skeletons", "fight", "guts"])


class TestGenerateReskinDeterminism(unittest.TestCase):
    def test_same_inputs_produce_identical_output_every_time(self):
        r1 = generate_reskin("chatgpt25-T4", T4_TEXT, depth=2, seed=20260717)
        r2 = generate_reskin("chatgpt25-T4", T4_TEXT, depth=2, seed=20260717)
        self.assertEqual(r1, r2)

    def test_different_seed_yields_different_reskin(self):
        text1, _ = generate_reskin("chatgpt25-T4", T4_TEXT, depth=2, seed=1)
        text2, _ = generate_reskin("chatgpt25-T4", T4_TEXT, depth=2, seed=2)
        self.assertNotEqual(text1, text2)

    def test_different_template_id_same_text_yields_different_reskin(self):
        # Keys on (seed, template_id, word) -- two templates that happened
        # to share literal text should NOT be forced to the same reskin
        # just because the text matches; template_id is part of the key.
        text1, _ = generate_reskin("template-a", T4_TEXT, depth=2, seed=1)
        text2, _ = generate_reskin("template-b", T4_TEXT, depth=2, seed=1)
        self.assertNotEqual(text1, text2)

    def test_actual_depth_clamps_when_too_few_candidates(self):
        text, actual_depth = generate_reskin("chatgpt25-T24", T24_TEXT,
                                             depth=4, seed=20260717)
        self.assertEqual(actual_depth, 3)  # only 3 candidates exist
        self.assertNotEqual(text, T24_TEXT)  # a real edit still happened

    def test_actual_depth_matches_request_when_enough_candidates(self):
        # T4 has 5 candidates (scientists/trust/atoms/make/everything) --
        # depth=4 must NOT clamp.
        _, actual_depth = generate_reskin("chatgpt25-T4", T4_TEXT, depth=4,
                                          seed=20260717)
        self.assertEqual(actual_depth, 4)

    def test_depths_nest_same_first_n_substitutions(self):
        # depth=1's and depth=2's substituted words must be IDENTICAL to
        # the first N candidates' substitutes under depth=4 too -- same
        # candidate order, same seed -> same deterministic substitute per
        # word -- so "detection rate by edit depth" is a genuine
        # monotonic-degradation curve (depth=4's edits are a SUPERSET of
        # depth=2's), not 3 unrelated random edits.
        candidates = _content_candidates(T4_TEXT)
        first_word, second_word = candidates[0][2], candidates[1][2]
        expected_first = _substitute_for(first_word, "chatgpt25-T4", 20260717)
        expected_second = _substitute_for(second_word, "chatgpt25-T4", 20260717)

        text_d1, _ = generate_reskin("chatgpt25-T4", T4_TEXT, depth=1,
                                     seed=20260717)
        text_d2, _ = generate_reskin("chatgpt25-T4", T4_TEXT, depth=2,
                                     seed=20260717)
        text_d4, _ = generate_reskin("chatgpt25-T4", T4_TEXT, depth=4,
                                     seed=20260717)

        self.assertIn(expected_first, text_d1)
        self.assertIn(expected_first, text_d2)
        self.assertIn(expected_first, text_d4)
        self.assertIn(expected_second, text_d2)
        self.assertIn(expected_second, text_d4)

    def test_substitution_actually_changes_the_candidate_words(self):
        text, actual_depth = generate_reskin("chatgpt25-T4", T4_TEXT, depth=2,
                                             seed=20260717)
        self.assertEqual(actual_depth, 2)
        self.assertNotEqual(text, T4_TEXT)
        # The un-edited tail ("Because they make up everything.") must be
        # untouched -- only the first 2 content-word candidates
        # (scientists, trust) change.
        self.assertIn("make up everything", text)

    def test_capitalization_of_sentence_initial_word_is_preserved(self):
        # generate_reskin on a template starting with a capitalized
        # candidate word must keep the substitute capitalized too.
        text, _ = generate_reskin("chatgpt25-T24", T24_TEXT, depth=1,
                                  seed=20260717)
        first_word = text.split()[0]
        self.assertTrue(first_word[0].isupper())

    def test_stopwords_set_is_nonempty_and_lowercase_consistent(self):
        # Sanity check on the fixture itself -- every entry should be
        # matched by .lower() comparisons in _content_candidates.
        self.assertTrue(len(STOPWORDS) > 10)
        for w in STOPWORDS:
            self.assertEqual(w, w.lower())


class TestBuildPositiveRecords(unittest.TestCase):
    def test_record_count_matches_templates_times_depths_plus_paraphrases(self):
        templates = [{"id": "t%d" % i, "text": T4_TEXT} for i in range(25)]
        records = build_positive_records(templates, seed=20260717)
        # 25 templates x 3 depths + 5 hand paraphrases
        self.assertEqual(len(records), 25 * 3 + 5)

    def test_every_reskin_record_has_actual_depth_and_differs_from_original(self):
        templates = [{"id": "chatgpt25-T4", "text": T4_TEXT}]
        records = build_positive_records(templates, seed=20260717)
        reskins = [r for r in records if r["kind"] == "reskin"]
        self.assertEqual(len(reskins), 3)  # one per edit depth
        for r in reskins:
            self.assertIsInstance(r["actual_depth"], int)
            self.assertGreater(r["actual_depth"], 0)
            self.assertNotEqual(r["text"], T4_TEXT)

    def test_paraphrase_records_present_and_labeled(self):
        records = build_positive_records([], seed=20260717)
        paraphrases = [r for r in records if r["kind"] == "paraphrase"]
        self.assertEqual(len(paraphrases), 5)
        for r in paraphrases:
            self.assertIsNone(r["actual_depth"])
            self.assertTrue(r["text"])


class TestBuildNegatives(unittest.TestCase):
    def _write_commercial_safe(self, root, n, short_n=0):
        d = Path(root) / "commercial-safe"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "jokes.jsonl", "w") as f:
            for i in range(short_n):
                f.write(json.dumps({"text": "lol"}) + "\n")  # degenerate
            for i in range(n):
                f.write(json.dumps(
                    {"text": "a genuinely different joke number %d here" % i}) + "\n")

    def test_filters_out_degenerate_short_entries(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_commercial_safe(d, n=50, short_n=20)
            negatives, n_scanned, n_excluded_leaks = build_negatives(
                Path(d), n=10, seed=1)
            self.assertEqual(len(negatives), 10)
            for text in negatives:
                self.assertGreaterEqual(len(text.split()), 3)
            self.assertEqual(n_excluded_leaks, 0)  # no exclude_texts passed

    def test_adaptive_buffer_still_reaches_n_when_corpus_is_mostly_degenerate(self):
        # Pathological composition (200 degenerate "lol" rows vs. only 50
        # real jokes) that a FIXED 3x-oversample buffer would silently
        # under-deliver on (a 3x=30-item draw from a 250-item pool that's
        # 80% degenerate yields ~6 valid entries, not 10) -- the adaptive
        # doubling must still reach exactly n.
        with tempfile.TemporaryDirectory() as d:
            self._write_commercial_safe(d, n=50, short_n=200)
            negatives, n_scanned, _ = build_negatives(Path(d), n=10, seed=1)
            self.assertEqual(len(negatives), 10)
            for text in negatives:
                self.assertGreaterEqual(len(text.split()), 3)

    def test_deterministic_given_same_seed(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_commercial_safe(d, n=50)
            n1, _, _ = build_negatives(Path(d), n=10, seed=7)
            n2, _, _ = build_negatives(Path(d), n=10, seed=7)
            self.assertEqual(n1, n2)

    def test_exact_corpus_duplicate_is_excluded_and_redrawn(self):
        # The EXP-009 corrected-validation regression test: a "negative"
        # that happens to be an exact-text duplicate of something already
        # in the corpus-comparison sample (e.g. because both are
        # independent reservoir draws over the same underlying pool) must
        # be filtered out and replaced by the adaptive buffer, not
        # returned as a genuine held-out negative -- see build_negatives's
        # exclude_texts docstring for the real leak this guards against
        # (3/100 negatives in the original 2026-07-17 run).
        with tempfile.TemporaryDirectory() as d:
            self._write_commercial_safe(d, n=50)
            # Draw once with NO exclusion to see exactly what a real
            # (unfiltered) draw would contain at this seed/n.
            baseline, _, baseline_leaks = build_negatives(Path(d), n=10, seed=1)
            self.assertEqual(baseline_leaks, 0)
            leaked_text = baseline[0]

            # Now redraw with that exact text excluded (as if it were
            # also a corpus-embedded row) -- it must NOT appear in the
            # result, the leak must be counted, and the adaptive buffer
            # must still deliver exactly n valid, non-leaked negatives.
            negatives, _, n_excluded_leaks = build_negatives(
                Path(d), n=10, seed=1, exclude_texts={leaked_text})
            self.assertEqual(len(negatives), 10)
            self.assertNotIn(leaked_text, negatives)
            self.assertGreaterEqual(n_excluded_leaks, 1)

    def test_no_exclude_texts_means_zero_excluded_leaks(self):
        # exclude_texts=None (the default) must behave exactly as before
        # this fix -- n_excluded_leaks is always 0, nothing is filtered
        # for leak reasons.
        with tempfile.TemporaryDirectory() as d:
            self._write_commercial_safe(d, n=50)
            negatives, _, n_excluded_leaks = build_negatives(
                Path(d), n=10, seed=1, exclude_texts=None)
            self.assertEqual(len(negatives), 10)
            self.assertEqual(n_excluded_leaks, 0)


class TestSemanticScoresReferenceSet(unittest.TestCase):
    """`semantic_scores`' core contract, and the exact fix for the
    2026-07-17 leaked/confounded validation run: `template_only_scores`
    must be computed ONLY against `template_embeddings`, fully decoupled
    from whatever extra general-corpus rows `corpus_embeddings` also
    contains. Fake `embed_fn` + hand-picked orthogonal unit vectors, same
    dependency-injection pattern as
    env/tests/test_semantic_novelty.py's `_make_embed_fn` -- no model
    download, `numpy` imported locally per method (see module docstring).
    """

    @staticmethod
    def _embed_fn(vector_map):
        import numpy as np

        def _embed(texts):
            return np.array([vector_map[t] for t in texts])
        return _embed

    def test_template_only_score_ignores_general_corpus_only_rows(self):
        # This is the real-world bug, reproduced in miniature: a query
        # that is a PERFECT match for some OTHER, non-template row that
        # only exists in the general-corpus sample, and is orthogonal
        # (cosine 0) to every actual template. The ORIGINAL (buggy)
        # calibration used corpus_embeddings (templates + general sample)
        # as the reference set, so this query would have scored 1.0 and
        # been treated as "detected" -- even though it has nothing to do
        # with any of the 25 memorized templates. The fix must score it
        # 0.0 against templates specifically.
        import numpy as np
        template_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        general_only_row = np.array([[0.0, 0.0, 1.0]])
        corpus_embeddings = np.vstack([template_embeddings, general_only_row])

        embed_fn = self._embed_fn({"query": (0.0, 0.0, 1.0)})
        template_only, general_corpus = semantic_scores(
            ["query"], embed_fn, template_embeddings, corpus_embeddings)

        self.assertAlmostEqual(template_only[0], 0.0, places=6)
        self.assertAlmostEqual(general_corpus[0], 1.0, places=6)

    def test_template_only_score_matches_hand_computed_cosine(self):
        import numpy as np
        template_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        corpus_embeddings = template_embeddings  # no extra general rows
        embed_fn = self._embed_fn({"a": (0.6, 0.8)})

        template_only, general_corpus = semantic_scores(
            ["a"], embed_fn, template_embeddings, corpus_embeddings)

        # cosine("a"=(0.6,0.8), (1,0)) = 0.6; cosine("a", (0,1)) = 0.8 ->
        # max = 0.8.
        self.assertAlmostEqual(template_only[0], 0.8, places=6)
        # With no general-corpus-only rows, the two reference sets are
        # identical, so the scores must match exactly too.
        self.assertAlmostEqual(general_corpus[0], template_only[0], places=6)

    def test_single_embed_call_reused_for_both_reference_sets(self):
        # embed_fn must be called exactly ONCE per semantic_scores() call
        # -- the whole point of returning both scores together is
        # avoiding a second (expensive, real-model) encode() call for the
        # informational general-corpus number.
        import numpy as np
        calls = []
        template_embeddings = np.array([[1.0, 0.0]])
        corpus_embeddings = template_embeddings

        def counting_embed_fn(texts):
            calls.append(list(texts))
            return np.array([[1.0, 0.0] for _ in texts])

        semantic_scores(["x", "y"], counting_embed_fn, template_embeddings,
                        corpus_embeddings)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["x", "y"])


class TestSweepAndPickThreshold(unittest.TestCase):
    def test_sweep_detection_and_fpr_hand_computed(self):
        groups = {"depth_1": [0.9, 0.8], "depth_2": [0.6, 0.4]}
        negatives = [0.1, 0.5, 0.9]
        rows = sweep("test", groups, negatives, thresholds=[0.5])
        [row] = rows
        self.assertEqual(row["threshold"], 0.5)
        self.assertAlmostEqual(row["detection"]["depth_1"], 1.0)   # both > 0.5
        self.assertAlmostEqual(row["detection"]["depth_2"], 0.5)   # only 0.6 > 0.5
        self.assertAlmostEqual(row["fpr"], 1.0 / 3.0)              # only 0.9 > 0.5

    def test_pick_threshold_selects_lowest_clearing_target_fpr(self):
        rows = [
            {"threshold": 0.3, "fpr": 0.20, "detection": {}},
            {"threshold": 0.4, "fpr": 0.10, "detection": {}},
            {"threshold": 0.5, "fpr": 0.04, "detection": {}},
            {"threshold": 0.6, "fpr": 0.00, "detection": {}},
        ]
        chosen = pick_threshold(rows, target_fpr=0.05)
        self.assertEqual(chosen["threshold"], 0.5)  # lowest with fpr <= 0.05

    def test_pick_threshold_falls_back_to_strictest_when_none_clear_target(self):
        rows = [
            {"threshold": 0.3, "fpr": 0.50, "detection": {}},
            {"threshold": 0.9, "fpr": 0.20, "detection": {}},
        ]
        chosen = pick_threshold(rows, target_fpr=0.05)
        self.assertEqual(chosen, rows[-1])


if __name__ == "__main__":
    unittest.main()

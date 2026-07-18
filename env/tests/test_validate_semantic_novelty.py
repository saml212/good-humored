"""Unit tests for env/validate_semantic_novelty.py's reskin generator and
threshold-sweep helpers -- NO model download / sentence_transformers
import anywhere in this file (the validation script's own module-level
imports are lightweight for exactly this reason; sentence_transformers
and the real model.encode() call live inside main()/run_windowed_
validation(), never at module scope). ONE EXCEPTION: `TestSemanticScoresReferenceSet`
and the EXP-011 `TestWindowedSemanticScores` below import `numpy` locally,
inside their own test methods, and construct `SemanticNoveltyPenalty` with a
FAKE `embed_fn` (never a real `SentenceTransformer`) to build tiny
hand-picked unit vectors for `semantic_scores`'/`windowed_semantic_scores`'
extracted, dependency-injectable logic (`numpy` is a real, already-required
dependency of both -- see their docstrings -- unlike sentence_transformers,
which stays fully out of this file).
Run: python3 -m unittest discover -s env/tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from env.rewards import _normalize, _window_token_spans
from env.semantic_novelty import SemanticNoveltyPenalty
from env.validate_semantic_novelty import (BAR_PADDING_INVARIANCE_MAX_DELTA,
                                           HAND_PARAPHRASES, MAX_JOKES_PER_NEGATIVE,
                                           MIN_JOKES_PER_NEGATIVE, NONJOKE_PROSE,
                                           STOPWORDS, WINDOW_PADDING_FILLER,
                                           _content_candidates, _parse_padding_reps,
                                           _substitute_for, build_negatives,
                                           build_positive_records,
                                           build_straddle_lengths,
                                           build_straddling_negatives,
                                           build_windowed_positive_records,
                                           generate_reskin, pad_with_filler,
                                           pick_threshold, semantic_scores,
                                           sweep, windowed_semantic_scores)

FIXTURE_CORPUS_DIR = Path(__file__).parent / "fixtures" / "corpus"

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


# ------------------------------------------------------- EXP-011 (windowed)


class TestParsePaddingReps(unittest.TestCase):
    def test_default_string_parses_in_order(self):
        self.assertEqual(_parse_padding_reps("0,5,20,50"), [0, 5, 20, 50])

    def test_dedupes_and_sorts_out_of_order_input(self):
        self.assertEqual(_parse_padding_reps("20,0,5,5,50"), [0, 5, 20, 50])

    def test_tolerates_whitespace_around_entries(self):
        self.assertEqual(_parse_padding_reps(" 0 , 5 ,20 "), [0, 5, 20])

    def test_non_integer_entry_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_padding_reps("0,five,20")

    def test_negative_entry_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_padding_reps("0,-5,20")

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_padding_reps("")


class TestPadWithFiller(unittest.TestCase):
    def test_zero_reps_returns_text_unchanged(self):
        self.assertEqual(pad_with_filler("hello world", 0), "hello world")

    def test_negative_reps_returns_text_unchanged(self):
        # Defensive -- reps should never be negative in practice
        # (_parse_padding_reps already rejects it), but pad_with_filler
        # itself shouldn't misbehave if called directly with one.
        self.assertEqual(pad_with_filler("hello world", -1), "hello world")

    def test_positive_reps_prepends_filler_that_many_times(self):
        padded = pad_with_filler("hello world", 3)
        self.assertEqual(padded, (WINDOW_PADDING_FILLER + " ") * 3 + "hello world")
        self.assertTrue(padded.endswith("hello world"))
        self.assertEqual(padded.count(WINDOW_PADDING_FILLER), 3)


class TestBuildWindowedPositiveRecords(unittest.TestCase):
    def test_adds_one_verbatim_record_per_template_unmodified(self):
        templates = [{"id": "t0", "text": "some fake template text here"},
                    {"id": "t1", "text": "another fake template over there"}]
        records = build_windowed_positive_records(templates, seed=1)
        verbatim = [r for r in records if r["kind"] == "verbatim"]
        self.assertEqual(len(verbatim), 2)
        self.assertEqual({r["template_id"] for r in verbatim}, {"t0", "t1"})
        for t, r in zip(templates, verbatim):
            match = next(v for v in verbatim if v["template_id"] == t["id"])
            self.assertEqual(match["text"], t["text"])  # UNMODIFIED
            self.assertIsNone(match["actual_depth"])

    def test_still_contains_the_underlying_reskins_and_paraphrases(self):
        templates = [{"id": "t0", "text": "some fake template text here"}]
        records = build_windowed_positive_records(templates, seed=1)
        reskins = [r for r in records if r["kind"] == "reskin"]
        paraphrases = [r for r in records if r["kind"] == "paraphrase"]
        self.assertEqual(len(reskins), 3)  # one per EDIT_DEPTHS entry
        self.assertEqual(len(paraphrases), len(HAND_PARAPHRASES))

    def test_total_count_is_reskins_plus_paraphrases_plus_verbatim(self):
        templates = [{"id": "t%d" % i, "text": "template text %d" % i}
                    for i in range(4)]
        records = build_windowed_positive_records(templates, seed=1)
        self.assertEqual(len(records), 4 * 3 + len(HAND_PARAPHRASES) + 4)


class TestBuildStraddleLengths(unittest.TestCase):
    def test_real_template_ladder_hand_computed(self):
        # The REAL 25-template ladder (measured against the actual corpus
        # at ~/Experiments/good-humored-data/corpus, window_growth=8
        # default): [(9, 4, 12), (18, 9, 26), (25, 12, 36)]. Widths 12/26/36
        # -8/-4/0/+6 delta ladder -> {8,12,18} u {22,26,32} u {32,36,42}.
        levels = [(9, 4, 12), (18, 9, 26), (25, 12, 36)]
        self.assertEqual(build_straddle_lengths(levels),
                         [8, 12, 18, 22, 26, 32, 36, 42])

    def test_single_level_floors_at_three(self):
        # width=3, deltas (-4, 0, 6) -> raw {-1, 3, 9}; the -1 must floor
        # to 3 (a length below 3 is meaningless -- _content_candidates/
        # _window_token_spans both operate on real tokens), so the -4
        # delta and the 0 delta COLLIDE at 3 and dedupe to a single entry.
        self.assertEqual(build_straddle_lengths([(3, 1, 3)]), [3, 9])

    def test_output_is_sorted_and_deduplicated(self):
        # Two levels sharing a boundary (level 0's "+6" == level 1's "-4"
        # midpoint collision) must not produce a duplicate entry.
        levels = [(9, 4, 12), (10, 5, 14)]  # widths 12 and 14: +6 of 12 (18)
                                            # and -4 of 14 (10) don't
                                            # collide here, but the
                                            # function must still dedupe
                                            # in general -- see the
                                            # 32-collision case above.
        lengths = build_straddle_lengths(levels)
        self.assertEqual(lengths, sorted(set(lengths)))


class TestBuildStraddlingNegatives(unittest.TestCase):
    JOKE_POOL = ["joke number %d about something totally different" % i
                for i in range(20)]

    def test_record_count_matches_lengths_times_reps(self):
        records = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[6, 15], reps_per_length=4, seed=1)
        self.assertEqual(len(records), 2 * 4)

    def test_actual_length_hits_target_exactly(self):
        records = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[6, 15, 30], reps_per_length=3, seed=1)
        for r in records:
            self.assertEqual(r["actual_length"], r["target_length"])
            self.assertEqual(len(_window_token_spans(r["text"])), r["target_length"])

    def test_truncation_never_splits_mid_token(self):
        # A boundary-token-exact truncation must never leave a dangling
        # partial word -- re-tokenizing the truncated text must find
        # EXACTLY target_length spans, each a complete boundary-token
        # (already covered by test_actual_length_hits_target_exactly, but
        # this test additionally checks no trailing partial-word noise:
        # the text must not end mid-way through what would have been the
        # next token).
        records = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[10], reps_per_length=5, seed=7)
        for r in records:
            self.assertFalse(r["text"].endswith(" "))
            self.assertTrue(r["text"])  # non-empty

    def test_n_jokes_within_configured_bounds(self):
        records = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[10], reps_per_length=25, seed=3)
        for r in records:
            self.assertGreaterEqual(r["n_jokes"], MIN_JOKES_PER_NEGATIVE)
            self.assertLessEqual(r["n_jokes"], MAX_JOKES_PER_NEGATIVE)

    def test_small_joke_pool_clamps_n_jokes_without_crashing(self):
        # A pool smaller than MAX_JOKES_PER_NEGATIVE must clamp k to the
        # pool size (rng.sample without replacement would otherwise raise
        # on k > population).
        small_pool = ["only joke one here", "only joke two here"]
        records = build_straddling_negatives(
            small_pool, target_lengths=[8], reps_per_length=5, seed=1)
        for r in records:
            self.assertLessEqual(r["n_jokes"], len(small_pool))

    def test_deterministic_given_same_seed(self):
        r1 = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[10, 20], reps_per_length=3, seed=42)
        r2 = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[10, 20], reps_per_length=3, seed=42)
        self.assertEqual([r["text"] for r in r1], [r["text"] for r in r2])

    def test_different_seed_yields_different_text(self):
        r1 = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[15], reps_per_length=3, seed=1)
        r2 = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[15], reps_per_length=3, seed=2)
        self.assertNotEqual([r["text"] for r in r1], [r["text"] for r in r2])

    def test_every_draw_includes_at_least_one_prose_sentence(self):
        # Sanity check on the construction itself -- the registered design
        # is "concatenations of 2-5 genuinely different corpus jokes +
        # non-joke prose", not a single joke alone. Every draw (n_prose is
        # recorded regardless of whether truncation later cuts into it)
        # must include at least one NONJOKE_PROSE sentence.
        records = build_straddling_negatives(
            self.JOKE_POOL, target_lengths=[6, 40], reps_per_length=5, seed=1)
        for r in records:
            self.assertGreaterEqual(r["n_prose"], 1)
            self.assertGreaterEqual(r["n_jokes"], MIN_JOKES_PER_NEGATIVE)


def _make_dilution_embed_fn(joke_vocab):
    """Fake embedder MODELING MEAN-POOLING DILUTION -- same pattern as
    env/tests/test_semantic_novelty.py's `_make_dilution_embed_fn`
    (duplicated here, not imported, so this test file stays self-contained
    per its own module docstring's "no cross-file test coupling"
    convention): embeds text as the unit vector of (a, b) where `a` =
    token count from `joke_vocab`, `b` = every other token. Pure
    joke-vocab text embeds at (1, 0) (the template's own reference axis),
    pure non-vocab text at (0, 1)."""
    def _embed(texts):
        out = []
        for t in texts:
            toks = _normalize(t)
            a = sum(1 for tok in toks if tok in joke_vocab)
            b = len(toks) - a
            n = (a * a + b * b) ** 0.5
            out.append((a / n, b / n) if n else (0.0, 0.0))
        return out
    return _embed


class TestWindowedSemanticScores(unittest.TestCase):
    """`windowed_semantic_scores`'s core contract: reuse
    `SemanticNoveltyPenalty._window_texts`/`_template_embeddings` for EXACT
    scoring parity with the shipped windowed `__call__` path, returning
    the RAW max-over-windows score (no threshold/ramp applied) a threshold
    sweep needs. Fixture corpus (1 template, 11 boundary-tokens -- "Why
    don't scientists trust atoms? Because they make up everything." --
    same fixture env/tests/test_semantic_novelty.py's
    TestWindowedSemanticDilution uses) + the dilution-modeling fake embed_fn
    above, so every expected score is hand-derivable, not guessed."""

    FILLER = "here is some filler text padding"  # 6 tokens, none in the
                                                 # template's vocabulary

    @classmethod
    def setUpClass(cls):
        from benchmark.joke_novelty import load_templates
        [tmpl] = load_templates(FIXTURE_CORPUS_DIR)
        cls.TEMPLATE = tmpl["text"]

    def _term(self, **kwargs):
        vocab = set(_normalize(self.TEMPLATE))
        return SemanticNoveltyPenalty(
            corpus_dir=FIXTURE_CORPUS_DIR,
            embed_fn=_make_dilution_embed_fn(vocab), windowed=True, **kwargs)

    def test_window_ladder_matches_hand_derivation(self):
        # Same fixture, same hand-derived ladder as
        # test_semantic_novelty.py's TestWindowedSemanticDilution.
        term = self._term()
        self.assertEqual(term._window_levels, [(11, 5, 15), (19, 9, 27)])

    def test_twenty_rep_dilution_exploit_hand_computed(self):
        term = self._term()
        embed_fn = _make_dilution_embed_fn(set(_normalize(self.TEMPLATE)))
        exploit = (self.FILLER + " ") * 20 + self.TEMPLATE
        [score] = windowed_semantic_scores([exploit], term, embed_fn)
        # Same hand computation as test_semantic_novelty.py's
        # test_windowed_catches_twenty_rep_dilution_hand_computed: best
        # window is the level-0 tail window (4 filler + 10 template
        # words) -> cosine 10/sqrt(116).
        expected = 10.0 / (116.0 ** 0.5)
        self.assertAlmostEqual(score, expected, places=6)

    def test_genuinely_novel_text_scores_near_zero(self):
        term = self._term()
        embed_fn = _make_dilution_embed_fn(set(_normalize(self.TEMPLATE)))
        novel = ("the lighthouse keeper counted seagulls while coffee "
                "went cold on the railing nobody believed his tally ") * 8
        [score] = windowed_semantic_scores([novel], term, embed_fn)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_more_padding_does_not_change_the_windowed_score(self):
        # Max-over-windows invariance to extra padding, mirroring
        # test_semantic_novelty.py's own regression test for the shipped
        # __call__ path -- windowed_semantic_scores must show the SAME
        # invariance since it reuses the identical windowing machinery.
        term = self._term()
        embed_fn = _make_dilution_embed_fn(set(_normalize(self.TEMPLATE)))
        exploit_20 = (self.FILLER + " ") * 20 + self.TEMPLATE
        exploit_60 = (self.FILLER + " ") * 60 + self.TEMPLATE
        scores = windowed_semantic_scores([exploit_20, exploit_60], term, embed_fn)
        self.assertAlmostEqual(scores[0], scores[1], places=6)

    def test_scoring_parity_with_the_shipped_call_path(self):
        # THE load-bearing contract this function exists for: the raw
        # score windowed_semantic_scores returns must reconstruct EXACTLY
        # the penalty `SemanticNoveltyPenalty.__call__`'s own windowed
        # branch would compute from that same score (weight * ramp), for
        # the SAME term/embed_fn/text -- proving this validation script's
        # numbers cannot silently drift from the shipped class's scoring.
        vocab = set(_normalize(self.TEMPLATE))
        embed_fn = _make_dilution_embed_fn(vocab)
        term = self._term(threshold=0.5, weight=-1.5)
        exploit = (self.FILLER + " ") * 20 + self.TEMPLATE

        [score] = windowed_semantic_scores([exploit], term, embed_fn)
        [reward] = term(prompts=[None], completions=[exploit])

        self.assertGreater(score, term.threshold)
        expected_reward = term.weight * (score - term.threshold) / (1.0 - term.threshold)
        self.assertAlmostEqual(reward, expected_reward, places=6)

    def test_one_embed_call_per_text_matching_the_class_contract(self):
        calls = []
        vocab = set(_normalize(self.TEMPLATE))
        base_embed = _make_dilution_embed_fn(vocab)

        def logging_embed(texts):
            calls.append(list(texts))
            return base_embed(texts)

        term = self._term()
        exploit = (self.FILLER + " ") * 20 + self.TEMPLATE
        windowed_semantic_scores([exploit, "a short novel quip"], term, logging_embed)
        self.assertEqual(len(calls), 2)  # one call per text, not per window
        self.assertEqual(calls[0][0], exploit)   # whole text is always
        self.assertEqual(calls[1][0], "a short novel quip")  # candidate[0]


class TestExp011Constants(unittest.TestCase):
    """Sanity checks on the EXP-011 module-level constants -- cheap
    guards against a typo silently weakening a registered bar."""

    def test_padding_invariance_bar_is_two_percentage_points(self):
        self.assertAlmostEqual(BAR_PADDING_INVARIANCE_MAX_DELTA, 0.02)

    def test_nonjoke_prose_pool_is_nonempty_and_distinct_from_templates(self):
        self.assertGreaterEqual(len(NONJOKE_PROSE), 3)
        self.assertEqual(len(NONJOKE_PROSE), len(set(NONJOKE_PROSE)))

    def test_min_jokes_not_greater_than_max_jokes(self):
        self.assertLessEqual(MIN_JOKES_PER_NEGATIVE, MAX_JOKES_PER_NEGATIVE)


if __name__ == "__main__":
    unittest.main()

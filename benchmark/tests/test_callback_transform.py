"""Unit tests for callback_transform.py — EXP-016 (EXPERIMENT_LOG.md).

Hand-computable cases only: every content-word-gate assertion below is
verified by eye against `benchmark.banter._content_words`'s own
5+-char/stopword-filtered tokenizer, and every transformation-score
assertion is verified against `benchmark.joke_novelty.trigram_jaccard`
directly (mirrors `env/tests/test_rewards.py`'s "hand-computed, not
model-computed" discipline for `SelfRepetitionPenalty`). The embedding
OR-branch is tested with FAKE, hand-picked-cosine `embed_fn` vectors only
(same `_v(cos_to_ref)` / `_make_embed_fn` pattern as
`env/tests/test_incongruity_gate.py` / `env/tests/test_semantic_novelty.py`)
— NO model download, NO network call, anywhere in this file.

Run: python3 -m unittest discover benchmark/tests -v
"""

import math
import unittest
from pathlib import Path

from benchmark.callback_transform import (DEFAULT_EMBED_SIM_FLOOR,
                                          DEFAULT_MIN_GAP,
                                          DEFAULT_MIN_SHARED_CONTENT_WORDS,
                                          VERBATIM_FLOOR,
                                          CallbackTransformationReward,
                                          callback_transformation_score,
                                          find_callback_match)
from benchmark.joke_novelty import trigram_jaccard, trigrams
from benchmark.validate_callback_transform import (FIXTURE_PATH,
                                                    GOLD_CLASSES,
                                                    build_bars, load_fixture,
                                                    per_class_means,
                                                    score_new)

FIXTURE_FILE = Path(__file__).parent.parent / "fixtures" / "callback_transform_fixture.jsonl"


# --------------------------------------------------------- fake embed_fn


def _v(cos_to_ref):
    """A 2D unit vector whose cosine similarity to the fixed reference
    vector (1.0, 0.0) is exactly `cos_to_ref` -- identical helper to
    `env/tests/test_incongruity_gate.py`'s own `_v`, so every embedding
    test case below is chosen directly as a target cosine number rather
    than hand-worked trig, while staying genuinely unit-normalized (the
    `embed_fn` contract `find_callback_match` documents)."""
    return (cos_to_ref, math.sqrt(max(0.0, 1.0 - cos_to_ref ** 2)))


def _make_embed_fn(vector_map, default=(0.0, 0.0)):
    """Fake embedder: exact-text dict lookup, `default` for anything
    unrecognized (matching `env/tests/test_semantic_novelty.py`'s
    `_make_embed_fn` convention -- a ZERO vector default, deliberately
    NOT `_v(...)`-shaped, so an un-listed filler turn can never spuriously
    collide with a real `_v(cos_to_ref)` vector's cosine similarity).
    Batched: called once with the WHOLE `turns` list, as
    `find_callback_match` itself does."""
    def _embed(texts):
        return [vector_map.get(t, default) for t in texts]
    return _embed


# ------------------------------------------------------- content-word gate


class TestFindCallbackMatchContentWords(unittest.TestCase):
    def test_two_shared_content_words_fires(self):
        turns = [
            "My neighbor's golden retriever got loose in the garden again.",
            "Traffic was backed up on the highway this week.",
            "The weekend farmers market brought back the good honey stand.",
            "I forgot my umbrella and got soaked on the walk home.",
            "That retriever has basically taken over the whole garden now.",
        ]
        r = find_callback_match(turns, 4)
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertEqual(r["detection_reasons"], ["content_words"])
        self.assertEqual(sorted(r["shared_content_words"]),
                         ["garden", "retriever"])

    def test_single_shared_content_word_does_not_fire(self):
        # Exactly ONE 5+-char content word shared ("garden") -- the OLD
        # detect_callback's own bag-of-words gate (>=1 word) would fire
        # here; the fix under test requires >= 2.
        turns = [
            "My neighbor's golden retriever got loose in the garden again.",
            "filler one", "filler two", "filler three",
            "The garden club meeting got rescheduled because of the storm.",
        ]
        r = find_callback_match(turns, 4)
        self.assertIsNone(r["matched_turn_idx"])
        self.assertEqual(r["detection_reasons"], [])
        self.assertEqual(r["shared_content_words"], [])

    def test_short_words_never_count_even_if_repeated(self):
        # "cat"/"dog" are < 5 chars -- excluded by _content_words'
        # length filter regardless of how many times they repeat.
        turns = ["I saw a cat and a dog today.", "f1", "f2", "f3",
                "That cat and dog moment was hilarious."]
        r = find_callback_match(turns, 4)
        self.assertIsNone(r["matched_turn_idx"])

    def test_too_recent_turn_does_not_qualify_as_candidate(self):
        # Two shared words, but the shared turn is only 2 turns back
        # (min_gap=3 requires >= 3) -- topical continuity, not a callback.
        turns = [
            "filler zero",
            "My neighbor's golden retriever got loose in the garden again.",
            "filler two",
            "That retriever has basically taken over the whole garden now.",
        ]
        r = find_callback_match(turns, 3, min_gap=DEFAULT_MIN_GAP)
        self.assertIsNone(r["matched_turn_idx"])

    def test_exactly_min_gap_qualifies(self):
        turns = [
            "My neighbor's golden retriever got loose in the garden again.",
            "filler one", "filler two",
            "That retriever has basically taken over the whole garden now.",
        ]
        # current_turn_idx=3, candidate j=0 -> gap = 3 - 0 = 3 = min_gap: qualifies.
        r = find_callback_match(turns, 3, min_gap=3)
        self.assertEqual(r["matched_turn_idx"], 0)

    def test_nearest_qualifying_turn_preferred(self):
        # Two candidates both satisfy the gate (turn 0 and turn 1); the
        # NEAREST (turn 1, higher index) must be chosen, mirroring
        # detect_callback's own "closer setup preferred" rule.
        turns = [
            "My neighbor's golden retriever got loose in the garden again.",
            "That same retriever also destroyed the garden fence last week.",
            "filler two", "filler three",
            "Anyway the retriever and the garden situation continues.",
        ]
        r = find_callback_match(turns, 4, min_gap=3)
        self.assertEqual(r["matched_turn_idx"], 1)

    def test_not_enough_earlier_turns_returns_no_match(self):
        r = find_callback_match(["only one turn"], 0)
        self.assertIsNone(r["matched_turn_idx"])

    def test_current_turn_idx_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            find_callback_match(["a", "b"], 5)

    def test_min_shared_content_words_below_one_raises(self):
        with self.assertRaises(ValueError):
            find_callback_match(["a", "b", "c", "d"], 3,
                                min_shared_content_words=0)


# --------------------------------------------------- false-positive words


class TestFalsePositiveWordsExcluded(unittest.TestCase):
    """banter.py's detect_callback docstring names 'morning' and 'today'
    EXACTLY as its own documented, accepted false-positive source. These
    tests lock in that the improved gate does not inherit that gap."""

    def test_morning_and_today_alone_do_not_satisfy_the_gate(self):
        turns = [
            "Something happened this morning that ruined my whole day today.",
            "filler one", "filler two", "filler three",
            "Nothing much going on this morning, same as today I guess.",
        ]
        # Naive _content_words (without the false-positive exclusion)
        # would share {"morning"} -- still only 1 word, insufficient on
        # its own either way, but this test specifically pins down that
        # 'morning'/'today' are excluded from the gate's word set, not
        # merely coincidentally under-threshold.
        r = find_callback_match(turns, 4)
        self.assertIsNone(r["matched_turn_idx"])
        self.assertEqual(r["shared_content_words"], [])

    def test_morning_today_plus_one_real_word_does_not_reach_two(self):
        # Shares 'morning', 'today', AND one real content word
        # ('printer') -- with the two false-positive words excluded,
        # only 'printer' remains: 1 shared word, gate must NOT fire.
        turns = [
            "The printer jammed once this morning before lunch today.",
            "filler one", "filler two", "filler three",
            "Printer troubles resurfaced this morning, same story today.",
        ]
        r = find_callback_match(turns, 4)
        self.assertIsNone(r["matched_turn_idx"])

    def test_two_real_words_beyond_the_false_positives_does_fire(self):
        turns = [
            "The printer jammed again this morning right before the deadline today.",
            "filler one", "filler two", "filler three",
            "That printer and its deadline conspiracy continue this morning.",
        ]
        r = find_callback_match(turns, 4)
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertEqual(sorted(r["shared_content_words"]),
                         ["deadline", "printer"])
        self.assertNotIn("morning", r["shared_content_words"])
        self.assertNotIn("today", r["shared_content_words"])


# ------------------------------------------------------------ embedding gate


class TestFindCallbackMatchEmbedding(unittest.TestCase):
    def test_embedding_alone_fires_above_floor_with_no_shared_words(self):
        turns = ["alpha completely unrelated wording", "f1", "f2", "f3",
                "beta totally different phrasing entirely"]
        embed_fn = _make_embed_fn({turns[0]: _v(1.0), turns[4]: _v(0.9)})
        r = find_callback_match(turns, 4, embed_fn=embed_fn,
                                embed_sim_floor=DEFAULT_EMBED_SIM_FLOOR)
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertEqual(r["detection_reasons"], ["embedding"])
        self.assertEqual(r["shared_content_words"], [])
        self.assertAlmostEqual(r["embed_similarity"], 0.9, places=6)

    def test_embedding_below_floor_does_not_fire(self):
        turns = ["alpha completely unrelated wording", "f1", "f2", "f3",
                "beta totally different phrasing entirely"]
        embed_fn = _make_embed_fn({turns[0]: _v(1.0), turns[4]: _v(0.3)})
        r = find_callback_match(turns, 4, embed_fn=embed_fn,
                                embed_sim_floor=0.6)
        self.assertIsNone(r["matched_turn_idx"])

    def test_embedding_exactly_at_floor_fires(self):
        turns = ["alpha", "f1", "f2", "f3", "beta"]
        embed_fn = _make_embed_fn({turns[0]: _v(1.0), turns[4]: _v(0.6)})
        r = find_callback_match(turns, 4, embed_fn=embed_fn,
                                embed_sim_floor=0.6)
        self.assertEqual(r["matched_turn_idx"], 0)

    def test_no_embed_fn_means_word_gate_only_no_crash(self):
        turns = ["alpha unrelated", "f1", "f2", "f3", "beta different"]
        r = find_callback_match(turns, 4, embed_fn=None)
        self.assertIsNone(r["matched_turn_idx"])
        self.assertIsNone(r["embed_similarity"])

    def test_content_word_tier_preferred_when_both_tiers_qualify_same_turn(self):
        # Turn 0 satisfies BOTH tiers at once -- reasons should list both.
        turns = [
            "My neighbor's golden retriever got loose in the garden again.",
            "f1", "f2", "f3",
            "That retriever has basically taken over the whole garden now.",
        ]
        embed_fn = _make_embed_fn({turns[0]: _v(1.0), turns[4]: _v(0.95)})
        r = find_callback_match(turns, 4, embed_fn=embed_fn, embed_sim_floor=0.6)
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertEqual(set(r["detection_reasons"]),
                         {"content_words", "embedding"})

    def test_embedding_batches_all_turns_in_one_call(self):
        calls = []
        turns = ["alpha", "f1", "f2", "f3", "beta"]

        def embed_fn(texts):
            calls.append(list(texts))
            return [_v(1.0) if t == "alpha" else _v(0.9) if t == "beta"
                   else _v(0.0) for t in texts]

        find_callback_match(turns, 4, embed_fn=embed_fn, embed_sim_floor=0.6)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], turns)


# --------------------------------------------------------- transformation


class TestCallbackTransformationScore(unittest.TestCase):
    def test_no_gate_match_scores_zero(self):
        turns = ["alpha unrelated", "f1", "f2", "f3", "beta different"]
        r = callback_transformation_score(turns, 4)
        self.assertIsNone(r["matched_turn_idx"])
        self.assertIsNone(r["trigram_similarity"])
        self.assertEqual(r["score"], 0.0)

    def test_verbatim_repeat_scores_exactly_zero(self):
        line = "My neighbor's golden retriever got loose in the garden again."
        turns = [line, "f1", "f2", "f3", line]
        r = callback_transformation_score(turns, 4)
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertEqual(r["trigram_similarity"], 1.0)
        self.assertEqual(r["score"], 0.0)

    def test_low_similarity_gives_high_score(self):
        turn0 = "My neighbor's golden retriever got loose in the garden again."
        turn4 = "That retriever has quietly opened a full landscaping business in the garden now, apparently."
        turns = [turn0, "f1", "f2", "f3", turn4]
        r = callback_transformation_score(turns, 4)
        sim = trigram_jaccard(trigrams(turn0), trigrams(turn4))
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertAlmostEqual(r["trigram_similarity"], sim, places=9)
        self.assertLess(sim, VERBATIM_FLOOR)
        self.assertAlmostEqual(r["score"], 1.0 - sim, places=9)
        self.assertGreater(r["score"], 0.7)

    def test_verbatim_floor_boundary_exactly_0_8_scores_zero(self):
        # Hand-construct two texts whose trigram-Jaccard is exactly 0.8.
        # base: 11 distinct 5-char tokens -> 9 trigrams (windows starting
        # at each of positions 0..8). Changing ONLY the LAST token means
        # that token appears in exactly ONE trigram (the final window),
        # so each side contributes exactly 1 trigram unique to itself:
        # intersection = 9 - 1 = 8, union = 9 + 1 = 10, jaccard = 8/10 = 0.8.
        base = "aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhhh iiiii jjjjj kkkkk"
        turn0 = base
        turn4 = "aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhhh iiiii jjjjj lllll"
        sim = trigram_jaccard(trigrams(turn0), trigrams(turn4))
        self.assertEqual(sim, 0.8)  # sanity-check the hand construction
        turns = [turn0, "f1", "f2", "f3", turn4]
        r = callback_transformation_score(turns, 4, embed_fn=None)
        # word gate: 10 shared 5-char content words (no stopwords among
        # them) -- gate fires regardless of the trigram question.
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertEqual(r["trigram_similarity"], 0.8)
        self.assertEqual(r["score"], 0.0)

    def test_just_below_verbatim_floor_scores_nonzero(self):
        base = "aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhhh"
        turn0 = base
        turn4 = "aaaaa bbbbb ccccc ddddd eeeee iiiii jjjjj kkkkk"
        sim = trigram_jaccard(trigrams(turn0), trigrams(turn4))
        self.assertLess(sim, VERBATIM_FLOOR)
        turns = [turn0, "f1", "f2", "f3", turn4]
        r = callback_transformation_score(turns, 4)
        self.assertEqual(r["matched_turn_idx"], 0)
        self.assertAlmostEqual(r["score"], 1.0 - sim, places=9)
        self.assertGreater(r["score"], 0.0)

    def test_custom_verbatim_floor_is_respected(self):
        turn0 = "aaaaa bbbbb ccccc ddddd eeeee"
        turn4 = "aaaaa bbbbb ccccc ddddd fffff"
        sim = trigram_jaccard(trigrams(turn0), trigrams(turn4))
        turns = [turn0, "f1", "f2", "f3", turn4]
        # Force the floor low enough that this pair now counts as
        # "verbatim enough" -> score forced to 0 even though sim < 0.8.
        r = callback_transformation_score(turns, 4, verbatim_floor=sim)
        self.assertEqual(r["score"], 0.0)


# ------------------------------------------------- reward-term stub class


class TestCallbackTransformationRewardStub(unittest.TestCase):
    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            CallbackTransformationReward(weight=-0.1)

    def test_zero_weight_allowed(self):
        CallbackTransformationReward(weight=0.0)  # must not raise

    def test_call_multiplies_weight_by_score(self):
        line = "My neighbor's golden retriever got loose in the garden again."
        turn4 = "That retriever has quietly opened a full landscaping business in the garden now, apparently."
        turns = [line, "f1", "f2", "f3", turn4]
        reward = CallbackTransformationReward(weight=0.5)
        expected = 0.5 * callback_transformation_score(turns, 4)["score"]
        self.assertAlmostEqual(reward(turns, 4), expected, places=9)

    def test_call_returns_zero_when_gate_does_not_fire(self):
        turns = ["alpha unrelated", "f1", "f2", "f3", "beta different"]
        reward = CallbackTransformationReward(weight=0.5)
        self.assertEqual(reward(turns, 4), 0.0)

    def test_call_returns_zero_for_verbatim_repeat_despite_weight(self):
        line = "My neighbor's golden retriever got loose in the garden again."
        turns = [line, "f1", "f2", "f3", line]
        reward = CallbackTransformationReward(weight=0.9)
        self.assertEqual(reward(turns, 4), 0.0)

    def test_stub_has_trl_style_dunder_name(self):
        # Not TRL-wired (see class docstring), but shares the __name__
        # convention every reward term in env/rewards.py carries, for
        # when it eventually IS wired.
        reward = CallbackTransformationReward(weight=0.0)
        self.assertEqual(reward.__name__, "callback_transformation_reward")


# --------------------------------------------------------- fixture integrity


class TestFixtureIntegrity(unittest.TestCase):
    def test_fixture_file_exists(self):
        self.assertTrue(FIXTURE_FILE.exists(), FIXTURE_FILE)

    def test_forty_items_total(self):
        items = load_fixture()
        self.assertEqual(len(items), 40)

    def test_five_classes_eight_each(self):
        items = load_fixture()
        from collections import Counter
        counts = Counter(it["gold_class"] for it in items)
        self.assertEqual(set(counts), set(GOLD_CLASSES))
        for cls in GOLD_CLASSES:
            self.assertEqual(counts[cls], 8, cls)

    def test_unique_ids(self):
        items = load_fixture()
        ids = [it["id"] for it in items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_schema_has_required_fields(self):
        items = load_fixture()
        for it in items:
            for field in ("id", "gold_class", "turns", "current_turn_idx", "notes"):
                self.assertIn(field, it, it.get("id"))
            self.assertIsInstance(it["turns"], list)
            self.assertGreater(len(it["turns"]), 0)
            self.assertTrue(0 <= it["current_turn_idx"] < len(it["turns"]))

    def test_fixture_path_matches_validate_module_constant(self):
        self.assertEqual(FIXTURE_PATH.resolve(), FIXTURE_FILE.resolve())


class TestFixtureRegisteredBars(unittest.TestCase):
    """Locks in EXP-016's three registered bars against the ACTUAL
    committed fixture + real (non-fake) scoring functions -- a
    regression test for "quality here IS the experiment" (EXP-016's own
    framing): if a future fixture edit breaks a bar, this test fails
    immediately rather than silently drifting from
    experiment-runs/2026-07-22-exp016-callback/report.json's recorded
    result."""

    def test_all_three_registered_bars_pass_on_the_committed_fixture(self):
        items = load_fixture()
        new_scores = {i: r["score"] for i, r in score_new(items).items()}
        per_class = per_class_means(items, new_scores)
        bars, margin, pooled_mean = build_bars(per_class)
        for b in bars:
            self.assertTrue(b["passed"], b)
        self.assertGreaterEqual(margin, 0.50)
        self.assertLessEqual(per_class["coincidental_word_reuse"]["mean"], 0.10)
        self.assertEqual(per_class["no_callback"]["mean"], 0.0)

    def test_verbatim_repeat_class_is_exactly_zero(self):
        items = load_fixture()
        new_scores = {i: r["score"] for i, r in score_new(items).items()}
        verbatim_ids = [it["id"] for it in items
                       if it["gold_class"] == "verbatim_repeat"]
        for i in verbatim_ids:
            self.assertEqual(new_scores[i], 0.0, i)


if __name__ == "__main__":
    unittest.main()

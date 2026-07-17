"""Unit tests for env/rewards.py -- fake judge callables and a tiny
in-repo corpus fixture only, NO network or CLI calls.
Run: python3 -m unittest discover -s env/tests -v
"""

import tempfile
import unittest
import warnings
from pathlib import Path

from benchmark.joke_novelty import load_templates
from benchmark.joke_novelty import trigram_jaccard as jn_trigram_jaccard
from benchmark.joke_novelty import trigrams as jn_trigrams
from env.rewards import (_SELF_REPETITION_BOUNDARY_FLOOR, ComprehensibilityReward,
                         CorpusNoveltyPenalty, IntraGroupDiversityReward,
                         JudgePreferenceReward, RewardConfig,
                         SelfRepetitionPenalty, _jaccard, _ngrams, _normalize,
                         combined_reward, reward_stack)

FIXTURE_CORPUS_DIR = Path(__file__).parent / "fixtures" / "corpus"

# Two completions with completely disjoint vocabulary -- makes trigram
# Jaccard exactly 0 against each other AND against the fixture corpus, so
# self-repetition / corpus-novelty / diversity all reduce to clean,
# hand-computable numbers instead of a messy partial-overlap ramp.
JOKE_A = "Bananas wear tiny hats at the office party tonight."
JOKE_B = "Xylophones dream about quantum lizards during winter storms."


def _completions(*texts):
    return list(texts)


def _sim(text_a: str, text_b: str, n: int = 3) -> float:
    """Recompute trigram-Jaccard directly via env.rewards' own primitives
    -- used by the "locked-in ramp" tests below to derive an expected
    value from the ACTUAL similarity of a realistic near-clone edit,
    rather than a guessed number."""
    return _jaccard(_ngrams(_normalize(text_a), n), _ngrams(_normalize(text_b), n))


class TestRewardConfigDefaults(unittest.TestCase):
    def test_defaults_match_readme_table(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.judge_weight, 1.0)
        self.assertEqual(cfg.corpus_novelty_weight, -1.5)
        self.assertEqual(cfg.self_repetition_weight, -1.0)
        self.assertEqual(cfg.intra_group_diversity_weight, 0.5)
        self.assertEqual(cfg.comprehensibility_weight, 0.3)
        self.assertIsNone(cfg.judge)
        self.assertIsNone(cfg.joke_corpus_dir)
        self.assertIsNone(cfg.group_size)


class TestRewardConfigSignGuards(unittest.TestCase):
    """A flipped sign silently turns a penalty into a bonus (or vice
    versa) with no other symptom until someone reads completions -- these
    lock the __post_init__ guards in place, in both directions."""

    def test_positive_corpus_novelty_weight_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(corpus_novelty_weight=1.5)

    def test_positive_self_repetition_weight_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(self_repetition_weight=1.0)

    def test_negative_intra_group_diversity_weight_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(intra_group_diversity_weight=-0.5)

    def test_negative_comprehensibility_weight_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(comprehensibility_weight=-0.3)

    def test_negative_judge_weight_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(judge_weight=-1.0)

    def test_zero_weights_are_valid_both_directions(self):
        # Zero disables a term without flipping its sign -- must not raise.
        RewardConfig(corpus_novelty_weight=0.0, self_repetition_weight=0.0,
                    intra_group_diversity_weight=0.0,
                    comprehensibility_weight=0.0, judge_weight=0.0)

    def test_correctly_signed_custom_weights_are_valid(self):
        RewardConfig(corpus_novelty_weight=-3.0, self_repetition_weight=-2.0,
                    intra_group_diversity_weight=1.0,
                    comprehensibility_weight=0.6, judge_weight=2.0)


class TestJudgePreferenceReward(unittest.TestCase):
    def test_no_judge_is_zero_and_warns_once_not_per_call(self):
        term = JudgePreferenceReward()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r1 = term(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
            r2 = term(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
        self.assertEqual(r1, [0.0, 0.0])
        self.assertEqual(r2, [0.0, 0.0])
        run_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 1)

    def test_fake_judge_scaled_by_weight(self):
        term = JudgePreferenceReward(judge=lambda p, c: 0.5, weight=2.0)
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [1.0])

    def test_out_of_range_judge_score_raises(self):
        term = JudgePreferenceReward(judge=lambda p, c: 7.0)  # raw 1-10 scale, unnormalized
        with self.assertRaises(ValueError):
            term(prompts=[None], completions=_completions(JOKE_A))

    def test_prompt_passed_through_unmodified(self):
        seen = []

        def judge(p, c):
            seen.append(p)
            return 0.5

        term = JudgePreferenceReward(judge=judge)
        prompt = [{"role": "user", "content": "tell me a joke"}]
        term(prompts=[prompt], completions=_completions(JOKE_A))
        self.assertEqual(seen, [prompt])


class TestCorpusNoveltyPenalty(unittest.TestCase):
    def test_none_corpus_dir_raises(self):
        with self.assertRaises(ValueError):
            CorpusNoveltyPenalty(corpus_dir=None)

    def test_missing_corpus_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            CorpusNoveltyPenalty(corpus_dir="/nonexistent/path/for/sure/xyz")

    def test_empty_corpus_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                CorpusNoveltyPenalty(corpus_dir=d)

    def test_exact_match_gets_full_penalty(self):
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        exact_joke = ("Why did the chicken cross the road? To get to the "
                     "other side.")
        rewards = penalty(prompts=[None], completions=_completions(exact_joke))
        self.assertEqual(rewards, [-1.5])

    def test_template_similarity_ramps_between_threshold_and_full(self):
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                       threshold=0.35)
        near_template = ("Why don't scientists trust electrons? Because "
                        "they make up everything.")
        templates = load_templates(FIXTURE_CORPUS_DIR)
        best_sim = max(jn_trigram_jaccard(jn_trigrams(near_template),
                                          t["_trigrams"])
                      for t in templates)
        expected_severity = (best_sim - 0.35) / (1.0 - 0.35)
        expected = -1.5 * expected_severity

        [reward] = penalty(prompts=[None],
                           completions=_completions(near_template))
        self.assertAlmostEqual(reward, expected, places=6)
        self.assertLess(reward, 0.0)
        self.assertGreater(reward, -1.5)  # ramped, not the full-match cliff

    def test_unrelated_text_below_threshold_is_zero(self):
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        rewards = penalty(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [0.0])

    def test_two_word_template_reskin_fully_evades_and_is_locked_in(self):
        """Documented, NOT fixed tonight (see CorpusNoveltyPenalty's
        docstring "KNOWN, UNFIXED LIMITATION"): substituting 2 content
        words in the fixture template drops trigram-Jaccard below
        `threshold` and evades this term entirely. Locking in the exact
        evasion as a regression test, per the audit's request, rather
        than silently letting future n-gram tweaks paper over it."""
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                       threshold=0.35)
        # Template: "Why don't scientists trust atoms? Because they make
        # up everything." -- two words swapped (atoms->protons,
        # everything->anything).
        reskinned = ("Why don't scientists trust protons? Because they "
                    "make up anything.")
        templates = load_templates(FIXTURE_CORPUS_DIR)
        best_sim = max(jn_trigram_jaccard(jn_trigrams(reskinned),
                                          t["_trigrams"])
                      for t in templates)
        self.assertLess(best_sim, 0.35,
                        "fixture template text changed? this test's whole "
                        "point is a similarity that falls BELOW threshold")
        [reward] = penalty(prompts=[None], completions=_completions(reskinned))
        self.assertEqual(reward, 0.0)  # full evasion, exactly as documented


class TestSelfRepetitionPenalty(unittest.TestCase):
    def test_first_occurrence_is_zero(self):
        term = SelfRepetitionPenalty()
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [0.0])

    def test_exact_repeat_gets_full_penalty(self):
        term = SelfRepetitionPenalty(window=10, threshold=0.5, weight=-1.0)
        term(prompts=[None], completions=_completions(JOKE_A))  # seed history
        [reward] = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(reward, -1.0)

    def test_reset_clears_history(self):
        term = SelfRepetitionPenalty(window=10, threshold=0.5, weight=-1.0)
        term(prompts=[None], completions=_completions(JOKE_A))
        term.reset()
        [reward] = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(reward, 0.0)

    def test_disjoint_vocabulary_completions_no_penalty(self):
        term = SelfRepetitionPenalty()
        rewards = term(prompts=[None, None],
                      completions=_completions(JOKE_A, JOKE_B))
        self.assertEqual(rewards, [0.0, 0.0])

    def test_repeated_cyrillic_joke_triggers_penalty(self):
        # Audit finding: the old ASCII-only _WORD_RE tokenized non-Latin
        # text to an EMPTY set, so a repeated non-English joke was
        # invisible to this term (silently zero penalty, forever). A
        # repeated Cyrillic joke must trigger the penalty exactly like a
        # repeated English one.
        joke = "Почему курица перешла дорогу? Чтобы попасть на другую сторону."
        term = SelfRepetitionPenalty(window=10, threshold=0.5, weight=-1.0)
        term(prompts=[None], completions=_completions(joke))
        [reward] = term(prompts=[None], completions=_completions(joke))
        self.assertEqual(reward, -1.0)

    def test_boundary_at_exact_threshold_now_penalizes(self):
        # Historically `sim <= threshold` fell into the zero-penalty
        # branch, so a near-clone landing EXACTLY on the threshold
        # evaded entirely. Force that exact boundary deliberately (using
        # JOKE_A's own computed self-similarity against a near-clone as
        # the configured threshold) rather than hoping natural English
        # coincidentally hits it.
        near_clone = "Bananas wear tiny hats at the office deck tonight."
        sim = _sim(JOKE_A, near_clone)
        term = SelfRepetitionPenalty(window=10, threshold=sim, weight=-1.0)
        term(prompts=[None], completions=_completions(JOKE_A))
        [reward] = term(prompts=[None], completions=_completions(near_clone))
        self.assertLess(reward, 0.0)  # boundary now penalizes, not zero
        self.assertAlmostEqual(reward, -1.0 * _SELF_REPETITION_BOUNDARY_FLOOR,
                               places=6)

    def test_one_word_changed_ramp_locked(self):
        base = "A cat sat on the warm mat all night"       # 9 tokens
        edited = "A cat sat on the warm mat all day"        # last word changed
        sim = _sim(base, edited)
        self.assertGreaterEqual(sim, 0.5,  # confirms this really is a near-clone
                                "fixture sentences changed? this locks in "
                                "a specific realistic 1-word edit's ramp value")

        term = SelfRepetitionPenalty(window=10, threshold=0.5, weight=-1.0)
        term(prompts=[None], completions=_completions(base))
        [reward] = term(prompts=[None], completions=_completions(edited))

        severity = (_SELF_REPETITION_BOUNDARY_FLOOR
                   + (1 - _SELF_REPETITION_BOUNDARY_FLOOR) * (sim - 0.5) / 0.5)
        self.assertAlmostEqual(reward, -1.0 * severity, places=6)

    def test_two_words_changed_ramp_locked_and_lower_than_one_word(self):
        base = "A cat sat on the warm mat all night"        # 9 tokens
        edited = "A cat sat on the warm mat one day"         # 2 words changed
        sim = _sim(base, edited)

        term = SelfRepetitionPenalty(window=10, threshold=0.5, weight=-1.0)
        term(prompts=[None], completions=_completions(base))
        [reward] = term(prompts=[None], completions=_completions(edited))

        if sim < 0.5:
            expected = 0.0
        else:
            severity = (_SELF_REPETITION_BOUNDARY_FLOOR
                       + (1 - _SELF_REPETITION_BOUNDARY_FLOOR)
                       * (sim - 0.5) / 0.5)
            expected = -1.0 * severity
        self.assertAlmostEqual(reward, expected, places=6)

        # Lock in the RAMP'S DIRECTION too, not just the two individual
        # values: this is a SIMILARITY penalty, so changing MORE words
        # (lower similarity) must score WEAKER (less negative), not
        # stronger -- the ramp gets easier to evade as an edit drifts
        # further from the original, all the way to full evasion once
        # similarity drops below `threshold` (that endpoint is exactly
        # `corpus_novelty_penalty`'s documented, accepted "2-word reskin
        # fully evades" limitation -- this test locks in that the SAME
        # shape holds here, one step before full evasion, not that it's
        # been fixed).
        one_word_edited = "A cat sat on the warm mat all day"
        term_one = SelfRepetitionPenalty(window=10, threshold=0.5, weight=-1.0)
        term_one(prompts=[None], completions=_completions(base))
        [one_word_reward] = term_one(prompts=[None],
                                     completions=_completions(one_word_edited))
        self.assertGreaterEqual(reward, one_word_reward)  # weaker penalty


class TestIntraGroupDiversityReward(unittest.TestCase):
    def test_none_group_size_raises(self):
        with self.assertRaises(ValueError):
            IntraGroupDiversityReward(group_size=None)

    def test_non_divisible_batch_raises(self):
        term = IntraGroupDiversityReward(group_size=2)
        with self.assertRaises(ValueError):
            term(prompts=[None] * 3, completions=_completions(JOKE_A, JOKE_B, JOKE_A))

    def test_identical_completions_in_group_score_zero(self):
        term = IntraGroupDiversityReward(group_size=2, weight=0.5)
        rewards = term(prompts=[None, None],
                      completions=_completions(JOKE_A, JOKE_A))
        self.assertEqual(rewards, [0.0, 0.0])

    def test_disjoint_completions_score_full_weight(self):
        term = IntraGroupDiversityReward(group_size=2, weight=0.5)
        rewards = term(prompts=[None, None],
                      completions=_completions(JOKE_A, JOKE_B))
        # completely disjoint trigrams -> jaccard 0 -> distance 1.0 -> full weight
        self.assertAlmostEqual(rewards[0], 0.5, places=6)
        self.assertAlmostEqual(rewards[1], 0.5, places=6)

    def test_singleton_groups_have_no_signal(self):
        term = IntraGroupDiversityReward(group_size=1, weight=0.5)
        rewards = term(prompts=[None, None],
                      completions=_completions(JOKE_A, JOKE_B))
        self.assertEqual(rewards, [0.0, 0.0])

    # ---------------------------------------------- BLOCKER regression

    def test_all_empty_group_scores_zero_not_maximally_diverse(self):
        # Exploit verified by the audit: previously scored
        # [weight, weight, weight, weight] -- maximum reward for
        # producing NOTHING, because an empty-vs-empty _jaccard
        # comparison returned 0.0 -> distance 1.0 (max "diversity").
        term = IntraGroupDiversityReward(group_size=4, weight=0.5)
        rewards = term(prompts=[None] * 4, completions=_completions("", "", "", ""))
        self.assertEqual(rewards, [0.0, 0.0, 0.0, 0.0])

    def test_all_emoji_group_scores_zero(self):
        # Emoji are not \w characters even under the new Unicode-aware
        # tokenizer -- an emoji-only completion must still be degenerate.
        term = IntraGroupDiversityReward(group_size=3, weight=0.5)
        rewards = term(prompts=[None] * 3,
                      completions=_completions("😂😂😂", "🎉🎉", "🤡"))
        self.assertEqual(rewards, [0.0, 0.0, 0.0])

    def test_mixed_group_empty_scores_zero_and_clone_scores_drop(self):
        # The exact exploit the audit demonstrated: a group of 3
        # near-clone cat jokes + 1 empty string used to score the EMPTY
        # completion HIGHER than any real joke (0.5 vs 0.2333) because
        # pairing a clone against the empty string inflated its distance
        # to 1.0. Fixed behavior: the empty completion gets a hard 0.0,
        # and it must be EXCLUDED from the clones' own pairwise
        # computation -- so recompute what the clones "should" score
        # using ONLY each other, and assert that's what they get (a
        # STRICTLY LOWER number than before the fix, since the 1.0-
        # distance pairing against the empty member is gone).
        clone_1 = "Bananas wear tiny hats at the office party tonight."
        clone_2 = "Bananas wear tiny hats at the office desk tonight."
        clone_3 = "Bananas wear tiny hats at the office chair tonight."
        term = IntraGroupDiversityReward(group_size=4, weight=0.5)
        rewards = term(prompts=[None] * 4,
                      completions=_completions(clone_1, clone_2, clone_3, ""))

        self.assertEqual(rewards[3], 0.0)  # the empty completion: hard zero

        # Expected clone scores, computed among clones ONLY (indices 0,1,2):
        grams = [_ngrams(_normalize(t), 3) for t in (clone_1, clone_2, clone_3)]
        for i in range(3):
            dists = [1.0 - _jaccard(grams[i], grams[j])
                    for j in range(3) if j != i]
            expected_i = 0.5 * (sum(dists) / len(dists))
            self.assertAlmostEqual(rewards[i], expected_i, places=6)

        # And the regression check that actually matters: before the
        # fix, each clone's score was inflated by an extra 1.0-distance
        # pairing against the empty member. Recomputing what that
        # (broken) 4-way average would have been confirms the fix
        # strictly LOWERS the clones' scores, not just changes them.
        for i in range(3):
            dists_incl_empty = [1.0 - _jaccard(grams[i], grams[j])
                                for j in range(3) if j != i] + [1.0]
            broken_score = 0.5 * (sum(dists_incl_empty) / len(dists_incl_empty))
            self.assertLess(rewards[i], broken_score)

    def test_two_non_degenerate_members_still_score_normally(self):
        # Exactly 2 non-degenerate members in a chunk of 3 (1 empty): the
        # "< 2 non-degenerate -> everyone 0.0" rule must NOT fire here --
        # 2 real attempts is enough signal.
        term = IntraGroupDiversityReward(group_size=3, weight=0.5)
        rewards = term(prompts=[None] * 3,
                      completions=_completions(JOKE_A, JOKE_B, ""))
        self.assertEqual(rewards[2], 0.0)  # degenerate: hard zero
        self.assertAlmostEqual(rewards[0], 0.5, places=6)  # disjoint -> full weight
        self.assertAlmostEqual(rewards[1], 0.5, places=6)

    # ------------------------------------------- prompt-equality guard

    def test_mismatched_prompts_within_chunk_raises(self):
        term = IntraGroupDiversityReward(group_size=2, weight=0.5)
        with self.assertRaises(ValueError):
            term(prompts=["prompt-1", "prompt-2"],
                completions=_completions(JOKE_A, JOKE_B))

    def test_matched_prompts_within_chunk_is_fine(self):
        term = IntraGroupDiversityReward(group_size=2, weight=0.5)
        rewards = term(prompts=["same-prompt", "same-prompt"],
                      completions=_completions(JOKE_A, JOKE_B))
        self.assertAlmostEqual(rewards[0], 0.5, places=6)

    def test_prompts_none_skips_the_equality_check(self):
        term = IntraGroupDiversityReward(group_size=2, weight=0.5)
        # prompts=None (the whole argument, not a list of Nones) --
        # comparable prompts weren't supplied at all; must not raise.
        rewards = term(prompts=None, completions=_completions(JOKE_A, JOKE_B))
        self.assertAlmostEqual(rewards[0], 0.5, places=6)

    def test_mismatched_prompts_error_names_chunk_index(self):
        term = IntraGroupDiversityReward(group_size=2, weight=0.5)
        with self.assertRaises(ValueError) as ctx:
            term(prompts=["p", "p", "q", "r"],
                completions=_completions(JOKE_A, JOKE_A, JOKE_B, JOKE_B))
        self.assertIn("chunk 1", str(ctx.exception))


class TestComprehensibilityReward(unittest.TestCase):
    def test_refusal_is_hard_zero_regardless_of_length(self):
        term = ComprehensibilityReward(weight=0.3)
        refusal = ("I can't help with that request, and I won't be "
                  "generating jokes about it either.")
        [reward] = term(prompts=[None], completions=_completions(refusal))
        self.assertEqual(reward, 0.0)

    def test_empty_string_and_wall_of_text_equally_lose_length_credit(self):
        term = ComprehensibilityReward(min_tokens=5, max_tokens=120, weight=0.3)
        wall_of_text = ("joke " * 300).strip()  # 300 identical tokens, no
                                                # trailing punctuation
        rewards = term(prompts=[None, None],
                      completions=_completions("", wall_of_text))
        self.assertEqual(rewards, [0.0, 0.0])

    def test_well_formed_completion_gets_full_credit(self):
        term = ComprehensibilityReward(min_tokens=5, max_tokens=120, weight=0.3)
        [reward] = term(prompts=[None], completions=_completions(JOKE_A))
        # length band (+0.5) + trailing punctuation (+0.25); JOKE_A's 9
        # tokens are all distinct so the uniqueness band (+0.25, requires
        # ratio <= 0.95) is NOT earned -> 0.75 of 1.0 -> 0.3 * 0.75
        self.assertAlmostEqual(reward, 0.3 * 0.75, places=6)


class TestRewardStackAndCombined(unittest.TestCase):
    def _config(self, **overrides):
        defaults = dict(judge=lambda p, c: 0.8,
                        joke_corpus_dir=FIXTURE_CORPUS_DIR,
                        group_size=2)
        defaults.update(overrides)
        return RewardConfig(**defaults)

    def test_reward_stack_requires_corpus_dir(self):
        cfg = self._config(joke_corpus_dir=None)
        with self.assertRaises(ValueError):
            reward_stack(cfg)

    def test_reward_stack_requires_group_size(self):
        cfg = self._config(group_size=None)
        with self.assertRaises(ValueError):
            reward_stack(cfg)

    def test_reward_stack_shape_and_names(self):
        funcs = reward_stack(self._config())
        self.assertEqual(len(funcs), 5)
        names = [f.__name__ for f in funcs]
        self.assertEqual(names, [
            "judge_preference", "corpus_novelty_penalty",
            "self_repetition_penalty", "intra_group_diversity",
            "comprehensibility",
        ])

    def test_custom_weight_plumbed_through(self):
        cfg = self._config(judge_weight=3.0)
        funcs = reward_stack(cfg)
        self.assertEqual(funcs[0].weight, 3.0)

    def test_reward_func_signature_accepts_extra_kwargs(self):
        # TRL calls every reward_func with prompts, completions, PLUS
        # whatever extra dataset columns exist -- every term must swallow
        # unrecognized kwargs without raising.
        funcs = reward_stack(self._config())
        for f in funcs:
            f(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B),
              some_dataset_column=["x", "y"], answer=["a", "b"])

    def test_combined_reward_hand_computed(self):
        cfg = self._config()
        fn = combined_reward(cfg)
        rewards = fn(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))

        # Hand-computed, term by term, for JOKE_A / JOKE_B (see module
        # docstring: completely disjoint vocabulary keeps every term's
        # arithmetic clean):
        #   judge_preference        = 1.0 * 0.8                = 0.8
        #   corpus_novelty_penalty  = 0.0   (unrelated to fixture corpus)
        #   self_repetition_penalty = 0.0   (first-ever occurrence; JOKE_A
        #                              and JOKE_B share zero trigrams)
        #   intra_group_diversity   = 0.5 * (1 - jaccard(A, B)) = 0.5 * 1.0 = 0.5
        #   comprehensibility       = 0.3 * 0.75                = 0.225
        #                              (length band + trailing punctuation;
        #                               NOT the uniqueness band -- both
        #                               sentences use every word exactly once)
        # total = 0.8 + 0.0 + 0.0 + 0.5 + 0.225 = 1.525
        expected = 0.8 + 0.0 + 0.0 + 0.5 + 0.225
        self.assertAlmostEqual(rewards[0], expected, places=6)
        self.assertAlmostEqual(rewards[1], expected, places=6)

    def test_combined_reward_name(self):
        fn = combined_reward(self._config())
        self.assertEqual(fn.__name__, "combined_reward")


class TestConversationalCompletionShape(unittest.TestCase):
    """TRL's documented completion contract has two shapes: a plain
    string, or a one-element `[{'content': str}]` list. Before this
    audit round, the dict shape had ZERO coverage despite being a real
    contract surface -- every term (and combined_reward) must produce
    IDENTICAL output regardless of which shape it's handed."""

    @staticmethod
    def _as_dicts(texts):
        return [[{"content": t}] for t in texts]

    def test_judge_preference_shape_equivalence(self):
        term = JudgePreferenceReward(judge=lambda p, c: 0.6)
        texts = [JOKE_A, JOKE_B]
        self.assertEqual(
            term(prompts=[None, None], completions=texts),
            term(prompts=[None, None], completions=self._as_dicts(texts)))

    def test_corpus_novelty_shape_equivalence(self):
        term = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR)
        texts = [JOKE_A]
        self.assertEqual(
            term(prompts=[None], completions=texts),
            term(prompts=[None], completions=self._as_dicts(texts)))

    def test_self_repetition_shape_equivalence(self):
        # Stateful -- use two FRESH instances so the second call isn't
        # penalized for "repeating" the first call's history.
        term_str = SelfRepetitionPenalty()
        term_dict = SelfRepetitionPenalty()
        texts = [JOKE_A, JOKE_A]
        self.assertEqual(
            term_str(prompts=[None, None], completions=texts),
            term_dict(prompts=[None, None], completions=self._as_dicts(texts)))

    def test_intra_group_diversity_shape_equivalence(self):
        term = IntraGroupDiversityReward(group_size=2)
        texts = [JOKE_A, JOKE_B]
        self.assertEqual(
            term(prompts=[None, None], completions=texts),
            term(prompts=[None, None], completions=self._as_dicts(texts)))

    def test_comprehensibility_shape_equivalence(self):
        term = ComprehensibilityReward()
        texts = [JOKE_A, JOKE_B]
        self.assertEqual(
            term(prompts=[None, None], completions=texts),
            term(prompts=[None, None], completions=self._as_dicts(texts)))

    def test_combined_reward_shape_equivalence(self):
        cfg = RewardConfig(judge=lambda p, c: 0.5,
                           joke_corpus_dir=FIXTURE_CORPUS_DIR, group_size=2)
        texts = [JOKE_A, JOKE_B]
        # Fresh combined_reward() per call -- reward_stack() builds brand
        # new (stateless-history) term instances each time it's invoked,
        # so this is a fair apples-to-apples comparison, not reusing one
        # SelfRepetitionPenalty's history across both shapes.
        r1 = combined_reward(cfg)(prompts=[None, None], completions=texts)
        r2 = combined_reward(cfg)(prompts=[None, None],
                                  completions=self._as_dicts(texts))
        for a, b in zip(r1, r2):
            self.assertAlmostEqual(a, b, places=6)


class TestEnvEpisodeCompletionsThroughRewardStack(unittest.TestCase):
    """Bridge check: jokes/replies a CascadeEnv or BanterEnv episode
    actually produces must be scoreable by env/rewards.py's terms in
    EITHER completion shape -- these two envs are the realistic source
    of the completions this reward stack scores in training."""

    def test_cascade_episode_jokes_scoreable_in_conversational_shape(self):
        from env.cascade_env import CascadeEnv
        from env.tests.test_cascade_env import fake_labeler

        env = CascadeEnv(labeler=fake_labeler, max_turns=3)
        env.reset()
        jokes = ["cat jokes are great", "dog jokes are also fine",
                "parrot jokes win"]
        for j in jokes:
            env.step(j)

        term = ComprehensibilityReward()
        str_rewards = term(prompts=[None] * 3, completions=jokes)
        dict_rewards = term(prompts=[None] * 3,
                           completions=[[{"content": j}] for j in jokes])
        self.assertEqual(str_rewards, dict_rewards)

    def test_banter_episode_replies_scoreable_in_conversational_shape(self):
        from env.banter_env import BanterEnv
        from env.tests.test_banter_env import fake_partner

        env = BanterEnv(partner_complete=fake_partner, max_turns=3)
        env.reset()
        replies = ["Sure, noted.", "Yeah that's rough.", "Anyway, moving on."]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no judge configured, expected
            for r in replies:
                env.step(r)

        term = ComprehensibilityReward()
        str_rewards = term(prompts=[None] * 3, completions=replies)
        dict_rewards = term(prompts=[None] * 3,
                           completions=[[{"content": r}] for r in replies])
        self.assertEqual(str_rewards, dict_rewards)


if __name__ == "__main__":
    unittest.main()

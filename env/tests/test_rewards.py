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
                         _window_token_spans, combined_reward, reward_stack)

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

    def test_nonpositive_incongruity_thresholds_raise(self):
        # Audit finding: a non-positive gate threshold silently converts
        # the strict-AND incongruity gate into an always-pass bonus.
        with self.assertRaises(ValueError):
            RewardConfig(incongruity_surprise_threshold=-1.0)
        with self.assertRaises(ValueError):
            RewardConfig(incongruity_surprise_threshold=0.0)
        with self.assertRaises(ValueError):
            RewardConfig(incongruity_drop_threshold=-1.0)
        with self.assertRaises(ValueError):
            RewardConfig(incongruity_drop_threshold=0.0)

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


class TestCorpusNoveltyPaddingDilution(unittest.TestCase):
    """The 2026-07-17 adversarial audit's padding/dilution exploit, as
    regression tests: a VERBATIM memorized joke inside ~5 repetitions of
    filler used to defeat BOTH n-gram tiers at once (exact-hash: the
    full text hashes differently; trigram: boilerplate dilutes Jaccard
    below threshold). Fixed by the windowed tier (default ON). Each
    exploit is asserted in BOTH modes -- `windowed=False` must still
    reproduce the evasion (locking in that the exploit was real and that
    the fix is the windows, not some fixture accident), and the default
    windowed mode must catch it at severity 1.0."""

    # The fixture template, verbatim (see fixtures/corpus/
    # chatgpt-25-templates.jsonl).
    TEMPLATE = ("Why don't scientists trust atoms? Because they make up "
               "everything.")

    # Five DIFFERENT filler sentences -- the audit's dilution needs
    # varied filler here because trigram sets deduplicate: repeating one
    # sentence 5x adds only ~8 distinct trigrams, which does NOT dilute
    # whole-text Jaccard below 0.35 against a 10-token template. Varied
    # filler is also the stronger (more attacker-realistic) form.
    FILLERS = ["Here is a little something to brighten your day.",
               "People often ask me for my best material.",
               "Comedy is all about timing and delivery, friends.",
               "Let me set the stage with some context first.",
               "Warm up the audience before the main event."]

    def test_windowed_is_the_default(self):
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR)
        self.assertTrue(penalty.windowed)

    def test_exploit_reproduced_with_windowing_off(self):
        # The auditor's exact reproduction: verbatim template after ~5
        # filler sentences pays ZERO penalty from both whole-text tiers.
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                       windowed=False)
        exploit = " ".join(self.FILLERS) + " " + self.TEMPLATE
        [reward] = penalty(prompts=[None], completions=_completions(exploit))
        self.assertEqual(reward, 0.0)  # full evasion, exactly as audited

    def test_verbatim_template_in_padding_caught_at_full_severity(self):
        # THE fix under test: the same exploit against the DEFAULT
        # construction must score severity 1.0 (full weight) -- the
        # stride-1, template-length window aligns exactly on the
        # embedded template (trigram Jaccard 1.0, and a window
        # exact-hash hit), and dilution cannot lower a max over windows.
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        exploit = " ".join(self.FILLERS) + " " + self.TEMPLATE
        [reward] = penalty(prompts=[None], completions=_completions(exploit))
        self.assertEqual(reward, -1.5)

    def test_padding_position_does_not_matter(self):
        # Prefix, suffix, and both-sides padding must all be caught --
        # the guarantee is positional (stride 1), not "the joke is at
        # the end".
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        pad = " ".join(self.FILLERS)
        variants = [pad + " " + self.TEMPLATE,
                    self.TEMPLATE + " " + pad,
                    " ".join(self.FILLERS[:3]) + " " + self.TEMPLATE + " "
                    + " ".join(self.FILLERS[3:])]
        rewards = penalty(prompts=[None] * 3, completions=variants)
        self.assertEqual(rewards, [-1.5, -1.5, -1.5])

    def test_verbatim_corpus_joke_in_padding_caught_by_window_hash(self):
        # The exact-hash tier's windowed form: a verbatim NON-template
        # corpus joke (shares no trigrams with the template, so the
        # trigram windows can't be what catches it) embedded in padding.
        # Its normalized length (12) falls inside the template-derived
        # window-size range {10, 11, 12} (template length 10 + slack 2),
        # so a stride-1 window hashes to exactly the memorized joke.
        with tempfile.TemporaryDirectory() as d:
            tmpl_path = Path(d) / "chatgpt-25-templates.jsonl"
            with open(tmpl_path, "w") as f:
                f.write('{"id": "t0", "text": "Quantum pigeons negotiate '
                        'treaties with sleepy volcanoes every autumn '
                        'morning."}\n')  # 10 normalized tokens
            jokes_path = Path(d) / "jokes.jsonl"
            with open(jokes_path, "w") as f:
                f.write('{"text": "My toaster filed a complaint about '
                        'cold bread union meetings last week."}\n')
                # ^ 12 normalized tokens, vocabulary disjoint from the
                # template above
            joke = ("My toaster filed a complaint about cold bread union "
                   "meetings last week.")
            exploit = " ".join(self.FILLERS) + " " + joke

            off = CorpusNoveltyPenalty(d, weight=-1.5, windowed=False)
            [evaded] = off(prompts=[None], completions=_completions(exploit))
            self.assertEqual(evaded, 0.0)  # exploit real against this
                                           # corpus too

            on = CorpusNoveltyPenalty(d, weight=-1.5)
            [caught] = on(prompts=[None], completions=_completions(exploit))
            self.assertEqual(caught, -1.5)

    def test_genuinely_novel_long_completion_not_newly_penalized(self):
        # False-positive guard: a long, genuinely novel completion (no
        # template-like span anywhere in it) must score 0.0 under the
        # default windowed mode, identical to the shipped whole-text
        # score -- windowing must not turn length itself into a penalty.
        novel = ("The lighthouse keeper counted seagulls every morning "
                "while his coffee went cold on the railing. Nobody in "
                "the village believed his tally, so he started "
                "photographing each bird with a disposable camera he "
                "kept in the lamp room, and the pharmacy developed the "
                "film weekly without ever asking questions about it.")
        on = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        off = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                   windowed=False)
        [r_on] = on(prompts=[None], completions=_completions(novel))
        [r_off] = off(prompts=[None], completions=_completions(novel))
        self.assertEqual(r_on, 0.0)
        self.assertEqual(r_on, r_off)

    def test_shipped_scores_preserved_on_non_adversarial_inputs(self):
        # Default-ON safety, locked in: for the existing fixture cases
        # (exact corpus hit, near-template ramp, unrelated text) the
        # windowed default must produce bit-identical rewards to
        # windowed=False -- the max over windows only diverges from the
        # whole-text score when a diluted memorized span is present.
        near_template = ("Why don't scientists trust electrons? Because "
                        "they make up everything.")
        exact_joke = ("Why did the chicken cross the road? To get to the "
                     "other side.")
        cases = [exact_joke, near_template, JOKE_A]
        on = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        off = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                   windowed=False)
        self.assertEqual(on(prompts=[None] * 3, completions=cases),
                         off(prompts=[None] * 3, completions=cases))

    def test_reskin_inside_padding_still_evades_ngram_windows(self):
        # Windowing makes scoring UNIFORM w.r.t. padding; it does not
        # close the 2-word-reskin gap (limitation #1, deliberately
        # preserved -- the SEMANTIC tier's job). A padded reskin must
        # score exactly what the bare reskin scores: 0.0. This also
        # locks in the upward-only window slack decision -- sub-template
        # windows would inflate Jaccard past threshold here (a 9-token
        # window over this reskin scores 4/11 > 0.35).
        reskinned = ("Why don't scientists trust protons? Because they "
                    "make up anything.")
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        exploit = " ".join(self.FILLERS) + " " + reskinned
        [reward] = penalty(prompts=[None], completions=_completions(exploit))
        self.assertEqual(reward, 0.0)

    def test_joke_beyond_max_scan_tokens_cap_evades_windows(self):
        # The documented cap security implication, locked in as a test
        # rather than left as prose: padding longer than max_scan_tokens
        # pushes the memorized joke past the window scan's frontier and
        # the window tiers cannot see it (whole-text tiers still apply,
        # but they are what dilution defeats). The 5 fillers total 42
        # normalized tokens, so a cap of 30 excludes the joke entirely;
        # the default cap (4096) catches the same completion.
        exploit = " ".join(self.FILLERS) + " " + self.TEMPLATE
        capped = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                      max_scan_tokens=30)
        [r_capped] = capped(prompts=[None], completions=_completions(exploit))
        self.assertEqual(r_capped, 0.0)  # evaded: joke lives past the cap

        default_cap = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        [r_default] = default_cap(prompts=[None],
                                  completions=_completions(exploit))
        self.assertEqual(r_default, -1.5)

    def test_degenerate_completions_stay_zero_with_windowing_on(self):
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        rewards = penalty(prompts=[None] * 3,
                          completions=_completions("", "   ", "😂😂😂"))
        self.assertEqual(rewards, [0.0, 0.0, 0.0])


class TestWindowTokenSpans(unittest.TestCase):
    """Direct unit tests for `_window_token_spans`, the boundary
    tokenizer the 2026-07-17 audit BLOCKER fix introduces (see
    `CorpusNoveltyPenalty`'s "PADDING/DILUTION, ROUND 2" docstring
    section). These pin down the primitive itself, independent of the
    reward-scoring regression tests below."""

    def _tokens(self, text):
        return [text[s:e] for s, e in _window_token_spans(text)]

    def test_punctuation_is_a_boundary_not_a_deletion(self):
        # THE bug: norm() deletes "." leaving "hereWhy" fused. The
        # boundary tokenizer must instead split at it.
        self.assertEqual(self._tokens("here.Why"), ["here", "Why"])

    def test_hyphen_is_a_boundary(self):
        self.assertEqual(self._tokens("here-Why"), ["here", "Why"])

    def test_zero_width_space_is_a_boundary(self):
        # U+200B is not in string.punctuation and not str.isspace() --
        # norm() neither deletes nor splits on it. Category 'Cf' catches it.
        self.assertEqual(self._tokens("here​Why"), ["here", "Why"])

    def test_other_zero_width_format_chars_are_boundaries(self):
        for ch in ("‌", "‍", "﻿", "⁠", "­"):
            self.assertEqual(self._tokens("a" + ch + "b"), ["a", "b"],
                             "char %r should be a boundary" % ch)

    def test_internal_apostrophe_splits_the_contraction(self):
        # By design (see docstring): boundary-tokens != norm()-tokens for
        # a contraction. The round-trip is restored by SLICING the
        # original text over a whole window and re-running norm(), not
        # by treating these tokens as the final comparison unit.
        self.assertEqual(self._tokens("don't"), ["don", "t"])

    def test_whitespace_including_newline_and_tab_are_boundaries(self):
        self.assertEqual(self._tokens("a b\nc\td"), ["a", "b", "c", "d"])

    def test_spans_slice_back_to_original_substrings_with_case(self):
        # Spans index into the ORIGINAL (un-lowered) text -- case and
        # internal punctuation of the token itself are preserved.
        text = "Warm up.Why don't scientists trust atoms?"
        spans = _window_token_spans(text)
        # "Why" through "atoms" is one contiguous run of tokens; slicing
        # from the start of "Why" to the end of "atoms" must reproduce
        # the substring verbatim, apostrophe and all.
        why_idx = [text[s:e] for s, e in spans].index("Why")
        atoms_idx = [text[s:e] for s, e in spans].index("atoms")
        start = spans[why_idx][0]
        end = spans[atoms_idx][1]
        self.assertEqual(text[start:end], "Why don't scientists trust atoms")

    def test_empty_and_all_boundary_text_yields_no_spans(self):
        self.assertEqual(_window_token_spans(""), [])
        self.assertEqual(_window_token_spans("... --- \n\t ​"), [])


class TestCorpusNoveltyPunctuationGluedPadding(unittest.TestCase):
    """2026-07-17 audit BLOCKER regression tests, exactly reproducing the
    auditor's table: `benchmark/joke_novelty.py`'s `norm()` DELETES
    punctuation instead of treating it as a boundary, so padding GLUED
    to a memorized joke by punctuation (or a zero-width character norm()
    doesn't even touch) used to fuse the joke's edge token into the
    padding's edge token -- evading the windowed exact-hash tier
    ENTIRELY for non-template corpus jokes (no fallback tier covers
    them), and weakening (or, for a short-enough template, fully
    defeating) the trigram tier for templates. `windowed=False` is
    asserted alongside each case to lock in that the exploit was REAL
    (not a fixture accident) and that the boundary-tokenizer fix (not
    some incidental change) is what closes it -- same convention as
    `TestCorpusNoveltyPaddingDilution` above, which only ever exercised
    whitespace joins and so never caught this."""

    TEMPLATE = TestCorpusNoveltyPaddingDilution.TEMPLATE
    FILLERS = TestCorpusNoveltyPaddingDilution.FILLERS

    # A NON-template verbatim corpus joke (fresh fixture below), so the
    # trigram tier -- which only knows the 25 templates -- structurally
    # CANNOT be what catches it: this isolates the exact-hash window
    # tier's own punctuation-glued-padding bug.
    NON_TEMPLATE_JOKE = ("My toaster filed a complaint about cold bread "
                        "union meetings last week.")  # 12 normalized tokens

    @classmethod
    def setUpClass(cls):
        # A fresh, disjoint-vocabulary template (10 normalized tokens, no
        # internal punctuation) so NON_TEMPLATE_JOKE's length (12) falls
        # inside the template-derived window-size range {10, 11, 12} --
        # same construction as
        # test_verbatim_corpus_joke_in_padding_caught_by_window_hash
        # above, reused here for the punctuation-glued join methods.
        cls._tmpdir = tempfile.TemporaryDirectory()
        d = Path(cls._tmpdir.name)
        (d / "chatgpt-25-templates.jsonl").write_text(
            '{"id": "t0", "text": "Quantum pigeons negotiate treaties '
            'with sleepy volcanoes every autumn morning."}\n')
        (d / "jokes.jsonl").write_text(
            '{"text": "%s"}\n' % cls.NON_TEMPLATE_JOKE)
        cls.corpus_dir = d

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _padding(self):
        return " ".join(self.FILLERS)

    def _assert_full_severity_on_off(self, text):
        on = CorpusNoveltyPenalty(self.corpus_dir, weight=-1.5)
        off = CorpusNoveltyPenalty(self.corpus_dir, weight=-1.5,
                                   windowed=False)
        [r_on] = on(prompts=[None], completions=_completions(text))
        [r_off] = off(prompts=[None], completions=_completions(text))
        self.assertEqual(r_on, -1.5)   # THE FIX: caught at full severity
        self.assertEqual(r_off, 0.0)   # locks in the exploit was real

    # -- the auditor's exact reproduction table, one join method each --

    def test_period_glued_padding(self):
        self._assert_full_severity_on_off(
            self._padding() + "." + self.NON_TEMPLATE_JOKE)

    def test_hyphen_glued_padding(self):
        self._assert_full_severity_on_off(
            self._padding() + "-" + self.NON_TEMPLATE_JOKE)

    def test_zero_width_space_glued_padding(self):
        # U+200B: not in string.punctuation, not str.isspace() -- the
        # auditor's own probe. The OLD norm()-only tokenization neither
        # deleted nor split on it, so this used to be worse than the
        # ASCII-punctuation joins (no seam of any kind after norm()).
        self._assert_full_severity_on_off(
            self._padding() + "​" + self.NON_TEMPLATE_JOKE)

    def test_pure_concatenation_no_added_separator(self):
        # No join character is INSERTED by this test at all -- direct
        # string concatenation. The padding's own last sentence
        # ("...before the main event.") already ends in ".", so the
        # exact same fusion mechanism fires with zero test-added glue,
        # confirming this isn't an artifact of a specific join character.
        self._assert_full_severity_on_off(
            self._padding() + self.NON_TEMPLATE_JOKE)

    def test_newline_glued_padding(self):
        # Whitespace -- was ALREADY correct before this fix (str.split()
        # inside norm() splits on any whitespace, \n included). Locked in
        # here as a regression test alongside the punctuation cases so
        # the "whitespace is fine, punctuation isn't" distinction is
        # explicit and tested, not just asserted in prose.
        self._assert_full_severity_on_off(
            self._padding() + "\n" + self.NON_TEMPLATE_JOKE)

    def test_tab_glued_padding(self):
        self._assert_full_severity_on_off(
            self._padding() + "\t" + self.NON_TEMPLATE_JOKE)

    # -- template-specific cases (trigram tier, not just exact-hash) --

    def test_template_sandwich_fused_both_edges(self):
        # Both edges glued by punctuation, no whitespace anywhere near
        # the embedded template -- the sandwich-fusion case the audit
        # flagged as weakening severity to (L-4)/L under the old code.
        on = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        off = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                   windowed=False)
        text = self._padding() + "." + self.TEMPLATE + "-" + self._padding()
        [r_on] = on(prompts=[None], completions=_completions(text))
        [r_off] = off(prompts=[None], completions=_completions(text))
        self.assertEqual(r_on, -1.5)
        self.assertEqual(r_off, 0.0)

    def test_template_ending_in_punctuation_round_trips_through_norm(self):
        # The template itself ends in "." (norm() deletes it). Glue the
        # NEXT word directly onto that trailing period with no added
        # separator -- exercises exactly the round-trip argument in
        # _window_token_spans's docstring: the boundary-tokenizer window
        # containing the template must, once sliced from the ORIGINAL
        # text and re-normalized, hash IDENTICALLY to norm(TEMPLATE),
        # even though the template's own trailing punctuation sits right
        # at the window's edge with no whitespace after it.
        on = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        off = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5,
                                   windowed=False)
        text = self.TEMPLATE + self._padding()  # "...everything." + "Here is..."
        [r_on] = on(prompts=[None], completions=_completions(text))
        [r_off] = off(prompts=[None], completions=_completions(text))
        self.assertEqual(r_on, -1.5)
        self.assertEqual(r_off, 0.0)

    def test_reskin_inside_punctuation_glued_padding_still_evades(self):
        # False-positive guard, mirroring
        # test_reskin_inside_padding_still_evades_ngram_windows above:
        # the fix must not turn the boundary tokenizer into a fuzzy
        # matcher. A 2-word reskin glued into padding by "." must still
        # evade completely (limitation #1, deliberately preserved).
        reskinned = ("Why don't scientists trust protons? Because they "
                    "make up anything.")
        penalty = CorpusNoveltyPenalty(FIXTURE_CORPUS_DIR, weight=-1.5)
        exploit = self._padding() + "." + reskinned
        [reward] = penalty(prompts=[None], completions=_completions(exploit))
        self.assertEqual(reward, 0.0)


class TestCorpusNoveltyUnicodeMarkGluedPadding(unittest.TestCase):
    """2026-07-18 audit BLOCKER regression tests: `_is_window_boundary_char`
    checked `isspace()` / `string.punctuation` / Unicode category 'Cf'
    (format/zero-width) but NOT 'Mn' (mark, nonspacing), 'Me' (mark,
    enclosing), or 'Cc' (control) -- so a combining/enclosing mark or a
    control character glued at the padding/joke seam was just as invisible
    to the boundary tokenizer as the zero-width characters
    `TestCorpusNoveltyPunctuationGluedPadding` above already closes, one
    Unicode-category short. Reproduced exactly as the audit describes: a
    bare U+0301 COMBINING ACUTE ACCENT glued between padding and a
    VERBATIM non-template corpus joke scored reward 0.0 -- full evasion of
    the windowed exact-hash tier -- before the `_is_window_boundary_char`
    fix (Mn/Me/Cc added alongside the existing Cf). NBSP (U+00A0) and
    ideographic space (U+3000) are included too, NOT because they were
    ever broken (both already satisfy `str.isspace()`) but to lock them in
    as passing regression cases alongside the newly-fixed ones, per the
    audit's request for one combined table."""

    TEMPLATE = TestCorpusNoveltyPaddingDilution.TEMPLATE
    FILLERS = TestCorpusNoveltyPaddingDilution.FILLERS
    NON_TEMPLATE_JOKE = TestCorpusNoveltyPunctuationGluedPadding.NON_TEMPLATE_JOKE

    @classmethod
    def setUpClass(cls):
        # Same fixture shape as TestCorpusNoveltyPunctuationGluedPadding:
        # a fresh, disjoint-vocabulary template (10 normalized tokens, no
        # internal punctuation) plus NON_TEMPLATE_JOKE (12 normalized
        # tokens) as a jokes.jsonl entry, so the joke's length falls
        # inside the template-derived window-size range {10, 11, 12} and
        # a stride-1 window can hash-match it exactly.
        cls._tmpdir = tempfile.TemporaryDirectory()
        d = Path(cls._tmpdir.name)
        (d / "chatgpt-25-templates.jsonl").write_text(
            '{"id": "t0", "text": "Quantum pigeons negotiate treaties '
            'with sleepy volcanoes every autumn morning."}\n')
        (d / "jokes.jsonl").write_text(
            '{"text": "%s"}\n' % cls.NON_TEMPLATE_JOKE)
        cls.corpus_dir = d

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _padding(self):
        return " ".join(self.FILLERS)

    def _assert_full_severity_on_off(self, join_char):
        text = self._padding() + join_char + self.NON_TEMPLATE_JOKE
        on = CorpusNoveltyPenalty(self.corpus_dir, weight=-1.5)
        off = CorpusNoveltyPenalty(self.corpus_dir, weight=-1.5,
                                   windowed=False)
        [r_on] = on(prompts=[None], completions=_completions(text))
        [r_off] = off(prompts=[None], completions=_completions(text))
        self.assertEqual(r_on, -1.5)   # THE FIX: caught at full severity
        self.assertEqual(r_off, 0.0)   # locks in the exploit was real

    # -- the audit's exact reproduction table, one join char each --

    def test_combining_acute_accent_glued_padding(self):
        # U+0301 COMBINING ACUTE ACCENT -- category Mn. The BLOCKER's
        # exact reproduction: neither isspace() nor string.punctuation
        # nor (pre-fix) any covered Unicode category.
        self._assert_full_severity_on_off("́")

    def test_enclosing_circle_backslash_glued_padding(self):
        # U+20E0 COMBINING ENCLOSING CIRCLE BACKSLASH -- category Me, the
        # "enclosing mark" half of the Zalgo-text family, same gap as Mn.
        self._assert_full_severity_on_off("⃠")

    def test_control_char_glued_padding(self):
        # U+0007 BEL -- category Cc, neither whitespace nor punctuation.
        self._assert_full_severity_on_off("")

    def test_nbsp_glued_padding(self):
        # U+00A0 NO-BREAK SPACE -- str.isspace() is already True for
        # this one, so it was never broken; locked in here as a passing
        # regression case alongside the newly-fixed ones above.
        self._assert_full_severity_on_off(" ")

    def test_ideographic_space_glued_padding(self):
        # U+3000 IDEOGRAPHIC SPACE -- also already str.isspace() == True,
        # same "already passing, lock it in" rationale as NBSP above.
        self._assert_full_severity_on_off("　")


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

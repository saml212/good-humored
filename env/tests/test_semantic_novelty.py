"""Unit tests for env/semantic_novelty.py -- fake `embed_fn` only, NO
model download / network / real sentence_transformers construction
anywhere in this file (see env/validate_semantic_novelty.py for the real-
model validation run that produces the calibrated DEFAULT_THRESHOLD).
Run: python3 -m unittest discover -s env/tests -v
"""

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import env.semantic_novelty as sn
from benchmark.joke_novelty import load_templates
from env.rewards import (RewardConfig, _normalize, combined_reward,
                         reward_stack)
from env.semantic_novelty import (SemanticNoveltyPenalty,
                                  SemanticNoveltyUnavailable,
                                  _reservoir_sample_jokes)

FIXTURE_CORPUS_DIR = Path(__file__).parent / "fixtures" / "corpus"


def _fixture_corpus_texts():
    """The exact ordered list SemanticNoveltyPenalty.__init__ builds from
    FIXTURE_CORPUS_DIR (template texts first, always kept, then
    jokes.jsonl lines in file order) -- RE-DERIVED from the same loaders
    the class itself uses, not hand-copied, so this stays correct if the
    fixture content ever changes (same discipline as test_rewards.py's
    `_sim()` helper)."""
    templates = load_templates(FIXTURE_CORPUS_DIR)
    texts = [t["text"] for t in templates]
    with open(FIXTURE_CORPUS_DIR / "jokes.jsonl") as f:
        texts += [json.loads(line)["text"] for line in f if line.strip()]
    return texts


def _make_embed_fn(vector_map, default=(0.0, 0.0)):
    """Fake embedder: exact-text dict lookup, `default` for anything
    unrecognized. Vectors are assumed pre-unit-normalized by the caller
    (this fixture's job) so cosine similarity reduces to a plain dot
    product -- matching SemanticNoveltyPenalty's documented embed_fn
    contract."""
    def _embed(texts):
        return [vector_map.get(t, default) for t in texts]
    return _embed


class TestReservoirSampleJokes(unittest.TestCase):
    """`_reservoir_sample_jokes` is the deterministic-sampling primitive
    the whole corpus-cap story depends on -- test it in isolation before
    trusting SemanticNoveltyPenalty's construction-time use of it."""

    @staticmethod
    def _write_jokes(tmpdir, n):
        path = Path(tmpdir) / "jokes.jsonl"
        with open(path, "w") as f:
            for i in range(n):
                f.write(json.dumps({"text": "joke number %d" % i}) + "\n")
        return path

    def test_cap_zero_samples_nothing_but_still_scans(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_jokes(d, 10)
            sample, n_scanned = _reservoir_sample_jokes(Path(d), cap=0, seed=1)
            self.assertEqual(sample, [])
            self.assertEqual(n_scanned, 10)

    def test_cap_above_population_returns_everything(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_jokes(d, 5)
            sample, n_scanned = _reservoir_sample_jokes(Path(d), cap=100, seed=1)
            self.assertEqual(n_scanned, 5)
            self.assertEqual(set(sample),
                            {"joke number %d" % i for i in range(5)})

    def test_same_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_jokes(d, 200)
            s1, _ = _reservoir_sample_jokes(Path(d), cap=20, seed=42)
            s2, _ = _reservoir_sample_jokes(Path(d), cap=20, seed=42)
            self.assertEqual(s1, s2)

    def test_different_seed_yields_different_sample(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_jokes(d, 200)
            s1, _ = _reservoir_sample_jokes(Path(d), cap=20, seed=1)
            s2, _ = _reservoir_sample_jokes(Path(d), cap=20, seed=2)
            self.assertNotEqual(s1, s2)

    def test_malformed_lines_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "jokes.jsonl"
            with open(path, "w") as f:
                f.write("not json at all\n")
                f.write(json.dumps({"no_text_field": True}) + "\n")
                f.write(json.dumps({"text": "a real joke"}) + "\n")
                f.write("\n")  # blank line
            sample, n_scanned = _reservoir_sample_jokes(Path(d), cap=10, seed=1)
            self.assertEqual(sample, ["a real joke"])
            self.assertEqual(n_scanned, 1)


class TestConstructionValidation(unittest.TestCase):
    """corpus_dir contract -- mirrors CorpusNoveltyPenalty's own
    None -> exists -> content checks, via embed_fn so none of this needs
    a real embedding backend."""

    _fake_embed = staticmethod(_make_embed_fn({}))

    def test_none_corpus_dir_raises(self):
        with self.assertRaises(ValueError):
            SemanticNoveltyPenalty(corpus_dir=None, embed_fn=self._fake_embed)

    def test_missing_corpus_dir_raises_without_touching_backend(self):
        with self.assertRaises(FileNotFoundError):
            SemanticNoveltyPenalty(corpus_dir="/nonexistent/path/for/sure/xyz",
                                   embed_fn=self._fake_embed)

    def test_empty_corpus_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                SemanticNoveltyPenalty(corpus_dir=d, embed_fn=self._fake_embed)

    def test_templates_always_kept_even_under_a_tight_cap(self):
        with tempfile.TemporaryDirectory() as d:
            tmpl_path = Path(d) / "chatgpt-25-templates.jsonl"
            with open(tmpl_path, "w") as f:
                for i in range(3):
                    f.write(json.dumps({"id": "t%d" % i,
                                       "text": "template %d" % i}) + "\n")
            jokes_path = Path(d) / "jokes.jsonl"
            with open(jokes_path, "w") as f:
                for i in range(20):
                    f.write(json.dumps({"text": "joke %d" % i}) + "\n")

            term = SemanticNoveltyPenalty(
                corpus_dir=d, embed_fn=_make_embed_fn({}, default=(1.0, 0.0)),
                corpus_cap=5, sample_seed=7)
            self.assertEqual(term.n_templates, 3)
            self.assertEqual(term.n_corpus_scanned, 20)
            # 3 templates (ALWAYS kept) + 2 reservoir-sampled jokes = 5,
            # even though the cap (5) is far below the 20 jokes scanned.
            self.assertEqual(term.n_corpus_embedded, 5)
            # Default reference is "templates" -- constructing without
            # passing `reference` must resolve to it, and DEFAULT_THRESHOLD
            # must be the resolved threshold (no explicit threshold passed
            # here either).
            self.assertEqual(term.reference, "templates")
            self.assertEqual(term.threshold, sn.DEFAULT_THRESHOLD)

    def test_invalid_reference_raises(self):
        with tempfile.TemporaryDirectory() as d:
            tmpl_path = Path(d) / "chatgpt-25-templates.jsonl"
            with open(tmpl_path, "w") as f:
                f.write(json.dumps({"id": "t0", "text": "template 0"}) + "\n")
            with self.assertRaises(ValueError):
                SemanticNoveltyPenalty(
                    corpus_dir=d, embed_fn=_make_embed_fn({}, default=(1.0, 0.0)),
                    reference="not-a-real-mode")

    def test_corpus_reference_without_explicit_threshold_raises(self):
        # reference="corpus" has NO calibrated default -- DEFAULT_THRESHOLD
        # is only valid for reference="templates" (see module docstring's
        # WARNING). Omitting threshold under reference="corpus" must raise,
        # never silently fall back to DEFAULT_THRESHOLD.
        with tempfile.TemporaryDirectory() as d:
            tmpl_path = Path(d) / "chatgpt-25-templates.jsonl"
            with open(tmpl_path, "w") as f:
                f.write(json.dumps({"id": "t0", "text": "template 0"}) + "\n")
            with self.assertRaises(ValueError):
                SemanticNoveltyPenalty(
                    corpus_dir=d, embed_fn=_make_embed_fn({}, default=(1.0, 0.0)),
                    reference="corpus")

    def test_corpus_reference_with_explicit_threshold_constructs_fine(self):
        with tempfile.TemporaryDirectory() as d:
            tmpl_path = Path(d) / "chatgpt-25-templates.jsonl"
            with open(tmpl_path, "w") as f:
                f.write(json.dumps({"id": "t0", "text": "template 0"}) + "\n")
            term = SemanticNoveltyPenalty(
                corpus_dir=d, embed_fn=_make_embed_fn({}, default=(1.0, 0.0)),
                reference="corpus", threshold=0.9)
            self.assertEqual(term.reference, "corpus")
            self.assertEqual(term.threshold, 0.9)

    def test_templates_reference_with_zero_templates_raises(self):
        # reference="templates" (the default) needs at least one template
        # to compare against -- a corpus_dir with jokes.jsonl but no
        # chatgpt-25-templates.jsonl entries must fail loudly at
        # construction, not defer to a crash inside __call__'s empty-array
        # similarity matmul.
        with tempfile.TemporaryDirectory() as d:
            jokes_path = Path(d) / "jokes.jsonl"
            with open(jokes_path, "w") as f:
                f.write(json.dumps({"text": "a joke with no templates around"}) + "\n")
            with self.assertRaises(FileNotFoundError):
                SemanticNoveltyPenalty(
                    corpus_dir=d, embed_fn=_make_embed_fn({}, default=(1.0, 0.0)))


class TestSemanticNoveltyPenaltyMath(unittest.TestCase):
    """Hand-computable cosine-similarity fixtures: FIXTURE_CORPUS_DIR (1
    template + 2 jokes, the same fixture test_rewards.py uses) embedded
    via a fake embed_fn that assigns each REAL corpus text a known unit
    vector, so every expected reward is exactly hand-derivable rather than
    guessed -- same discipline as test_rewards.py's `_sim()` helper."""

    CORPUS_VECS = None

    @classmethod
    def setUpClass(cls):
        texts = _fixture_corpus_texts()
        assert len(texts) == 3, (
            "fixture corpus shape changed (expected 1 template + 2 jokes) "
            "-- update this test's basis-vector assignment")
        basis = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        cls.CORPUS_VECS = dict(zip(texts, basis))

    def _term(self, query_vecs, threshold=0.5, weight=-1.5, reference="templates"):
        vecs = dict(self.CORPUS_VECS)
        vecs.update(query_vecs)
        return SemanticNoveltyPenalty(
            corpus_dir=FIXTURE_CORPUS_DIR, threshold=threshold, weight=weight,
            reference=reference, embed_fn=_make_embed_fn(vecs))

    def test_below_threshold_is_zero(self):
        # (0.3, 0): cosine 0.3 vs (1,0), 0 vs (0,1), -0.3 vs (-1,0) -> best=0.3 < 0.5
        term = self._term({"low sim completion": (0.3, 0.0)}, threshold=0.5)
        [reward] = term(prompts=[None], completions=["low sim completion"])
        self.assertEqual(reward, 0.0)

    def test_exactly_at_threshold_is_zero_matching_corpus_novelty_convention(self):
        # best cosine EXACTLY 0.5 -- CorpusNoveltyPenalty's own convention
        # is `best <= threshold -> 0.0` (no boundary floor, unlike
        # SelfRepetitionPenalty's fixed boundary-floor fix); this locks in
        # that this class matches THAT convention deliberately.
        term = self._term({"boundary completion": (0.5, 0.0)}, threshold=0.5)
        [reward] = term(prompts=[None], completions=["boundary completion"])
        self.assertEqual(reward, 0.0)

    def test_above_threshold_ramps_and_hand_computed_value_matches(self):
        term = self._term({"high sim completion": (0.8, 0.0)}, threshold=0.5,
                          weight=-1.5)
        [reward] = term(prompts=[None], completions=["high sim completion"])
        expected_severity = (0.8 - 0.5) / (1.0 - 0.5)  # 0.6
        self.assertAlmostEqual(reward, -1.5 * expected_severity, places=6)

    def test_higher_similarity_is_a_more_negative_reward(self):
        term = self._term({"mid": (0.7, 0.0), "hi": (0.95, 0.0)}, threshold=0.5)
        rewards = term(prompts=[None, None], completions=["mid", "hi"])
        self.assertLess(rewards[1], rewards[0])

    def test_custom_weight_scales_linearly(self):
        term = self._term({"q": (0.9, 0.0)}, threshold=0.5, weight=-3.0)
        [reward] = term(prompts=[None], completions=["q"])
        severity = (0.9 - 0.5) / 0.5
        self.assertAlmostEqual(reward, -3.0 * severity, places=6)

    def test_degenerate_empty_completion_is_zero_even_with_maximal_vector(self):
        # If "" were embedded at all, (1,0) is a PERFECT match to the
        # template -- proves the degenerate guard skips embedding
        # entirely rather than happening to score low by luck.
        term = self._term({"": (1.0, 0.0)}, threshold=0.5)
        [reward] = term(prompts=[None], completions=[""])
        self.assertEqual(reward, 0.0)

    def test_whitespace_and_emoji_only_completions_are_zero(self):
        term = self._term({"   ": (1.0, 0.0), "\U0001F602\U0001F602\U0001F602": (1.0, 0.0)},
                          threshold=0.5)
        rewards = term(prompts=[None, None],
                       completions=["   ", "\U0001F602\U0001F602\U0001F602"])
        self.assertEqual(rewards, [0.0, 0.0])

    def test_batch_scores_each_completion_independently(self):
        term = self._term({"a": (0.9, 0.0), "b": (0.1, 0.0)}, threshold=0.5)
        rewards = term(prompts=[None, None], completions=["a", "b"])
        self.assertLess(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)

    def test_conversational_completion_shape_equivalence(self):
        term = self._term({"x": (0.9, 0.0)}, threshold=0.5)
        str_reward = term(prompts=[None], completions=["x"])
        dict_reward = term(prompts=[None], completions=[[{"content": "x"}]])
        self.assertEqual(str_reward, dict_reward)


class TestReferenceModeSelection(unittest.TestCase):
    """The 2026-07-17 fix under test: default `__call__` must score ONLY
    against the 25-template reference set, fully ignoring the
    general-corpus rows that `reference="corpus"` opts into. Same
    fixture/basis-vector setup as `TestSemanticNoveltyPenaltyMath`
    (template -> (1,0), joke1 -> (0,1), joke2 -> (-1,0)) -- reused here
    (not imported from that class) so this class's intent stands alone."""

    CORPUS_VECS = None

    @classmethod
    def setUpClass(cls):
        texts = _fixture_corpus_texts()
        assert len(texts) == 3, (
            "fixture corpus shape changed (expected 1 template + 2 jokes) "
            "-- update this test's basis-vector assignment")
        basis = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        cls.CORPUS_VECS = dict(zip(texts, basis))

    def _term(self, query_vecs, threshold, reference, weight=-1.5):
        vecs = dict(self.CORPUS_VECS)
        vecs.update(query_vecs)
        return SemanticNoveltyPenalty(
            corpus_dir=FIXTURE_CORPUS_DIR, threshold=threshold, weight=weight,
            reference=reference, embed_fn=_make_embed_fn(vecs))

    def test_default_reference_attribute_is_templates(self):
        term = self._term({}, threshold=0.5, reference="templates")
        self.assertEqual(term.reference, "templates")

    def test_default_mode_ignores_a_completion_that_only_matches_a_corpus_joke(self):
        # (0.0, 1.0) is a PERFECT match (cosine 1.0) for joke1, and
        # orthogonal (cosine 0.0) to the template. Under the OLD
        # (pre-fix) behavior, which scored against templates + general
        # corpus unconditionally, this would have been flagged as a
        # near-verbatim match (best=1.0) and penalized hard. At default
        # settings (reference="templates"), this completion must score
        # exactly 0.0 -- it doesn't resemble the memorized TEMPLATE at
        # all, which is the only thing this term is supposed to check by
        # default.
        term = self._term({"corpus-only match": (0.0, 1.0)}, threshold=0.5,
                          reference="templates")
        [reward] = term(prompts=[None], completions=["corpus-only match"])
        self.assertEqual(reward, 0.0)

    def test_corpus_mode_penalizes_the_same_completion_templates_mode_ignores(self):
        # Same completion, same fixture, same threshold -- ONLY
        # `reference` differs. reference="corpus" must see the perfect
        # match against joke1 (a general-corpus row) and penalize it,
        # proving the two modes genuinely score against different
        # reference sets rather than `reference` being a no-op.
        term = self._term({"corpus-only match": (0.0, 1.0)}, threshold=0.5,
                          reference="corpus")
        [reward] = term(prompts=[None], completions=["corpus-only match"])
        self.assertLess(reward, 0.0)

    def test_templates_mode_still_penalizes_a_true_template_match(self):
        # Sanity check: reference="templates" is not simply "never
        # penalize" -- a completion that DOES match the template must
        # still be caught.
        term = self._term({"template match": (0.9, 0.0)}, threshold=0.5,
                          reference="templates")
        [reward] = term(prompts=[None], completions=["template match"])
        self.assertLess(reward, 0.0)


def _make_dilution_embed_fn(joke_vocab, log=None):
    """Fake embedder that MODELS MEAN-POOLING DILUTION -- the exact
    mechanism the 2026-07-17 audit exploited -- instead of exact-text
    lookup: embeds any text as the unit vector of (a, b) where `a` =
    token count from `joke_vocab` and `b` = every other token. A text
    that is all joke-vocab embeds at (1, 0) (so the fixture template is
    its own reference axis), pure filler embeds at (0, 1), and a mix
    lands in between exactly the way filler dilutes a mean-pooled
    embedding: 20 reps of 6-token filler around a 10-token joke gives
    cosine 10/sqrt(10^2 + 120^2) ~= 0.083 vs the template axis --
    below DEFAULT_THRESHOLD, reproducing the audited whole-text evasion
    hand-computably. `log`, if supplied, is a list that records every
    batch of texts passed to the embedder (call counting + degenerate-
    window assertions)."""
    def _embed(texts):
        if log is not None:
            log.append(list(texts))
        out = []
        for t in texts:
            toks = _normalize(t)
            a = sum(1 for tok in toks if tok in joke_vocab)
            b = len(toks) - a
            n = (a * a + b * b) ** 0.5
            out.append((a / n, b / n) if n else (0.0, 0.0))
        return out
    return _embed


class TestWindowedSemanticDilution(unittest.TestCase):
    """The 2026-07-17 padding/dilution exploit against THIS tier, as
    regression tests: ~20 repetitions of filler around a verbatim
    template dilutes the whole-text mean-pooled embedding below
    DEFAULT_THRESHOLD (audited against the real model; reproduced here
    via `_make_dilution_embed_fn`, which models the same mechanism).
    `windowed=True` (opt-in, default OFF pending EXP-011 -- see module
    docstring) must catch it: some ladder window fully contains the
    embedded span with bounded in-window filler, and dilution cannot
    lower the max over windows.

    Fixture geometry, hand-derivable throughout: the fixture template
    ("Why don't scientists trust atoms? Because they make up
    everything.") is 11 BOUNDARY-tokens (not 10 whitespace words --
    "don't"'s internal apostrophe is itself a boundary character under
    `_window_token_spans`, see that function's docstring and the
    2026-07-17 audit BLOCKER fix note on the class docstring), so the
    ladder is [(11, 5, 15), (19, 9, 27)] (cover, stride, width) with
    window_growth=8 -- asserted below so every hand computation is
    guarded, not guessed. The content-based hand computations below
    (cosine similarity, filler-vs-template token split) are UNCHANGED
    from before this fix: window CONTENT is sliced from the original
    text and still contains "don't" intact, so `_normalize` (which,
    unlike `_window_token_spans`, keeps a contraction as ONE token) still
    counts exactly 10 template words in the captured span -- only the
    ladder's own units (boundary-tokens, not whitespace-tokens) changed."""

    FILLER = "here is some filler text padding"  # 6 tokens, none in the
                                                 # template's vocabulary

    @classmethod
    def setUpClass(cls):
        [tmpl] = load_templates(FIXTURE_CORPUS_DIR)
        cls.TEMPLATE = tmpl["text"]  # re-derived, not hand-copied

    def _term(self, log=None, **kwargs):
        vocab = set(_normalize(self.TEMPLATE))
        return SemanticNoveltyPenalty(
            corpus_dir=FIXTURE_CORPUS_DIR,
            embed_fn=_make_dilution_embed_fn(vocab, log=log), **kwargs)

    def _exploit(self, reps=20):
        return (self.FILLER + " ") * reps + self.TEMPLATE

    def test_windowed_is_opt_in_and_default_off(self):
        # The shipped, EXP-009-validated whole-text behavior must remain
        # the default until EXP-011 re-validates the threshold for
        # windowed scoring (see module docstring).
        self.assertFalse(self._term().windowed)
        self.assertTrue(self._term(windowed=True).windowed)

    def test_window_ladder_matches_hand_derivation(self):
        term = self._term(windowed=True)
        self.assertEqual(term._window_levels, [(11, 5, 15), (19, 9, 27)])

    def test_exploit_reproduced_whole_text_dilution_evades_default_mode(self):
        # The auditor's reproduction: 20 filler reps + verbatim template
        # scores cosine 10/sqrt(100 + 120^2) ~= 0.083 < 0.38 whole-text
        # -> zero penalty in (default) non-windowed mode.
        term = self._term()
        [reward] = term(prompts=[None], completions=[self._exploit(reps=20)])
        self.assertEqual(reward, 0.0)

    def test_windowed_catches_twenty_rep_dilution_hand_computed(self):
        # The fix under test. Best window: the level-0 tail window
        # (width 15 BOUNDARY-tokens, ending at the last token) = 4
        # filler boundary-tokens + all 11 of the template's own
        # boundary-tokens ("don't" -> "don", "t"). Sliced from the
        # ORIGINAL text and re-`_normalize`d for this hand computation,
        # "don't" is intact again -- 4 filler + all 10 template WORDS ->
        # cosine 10/sqrt(10^2 + 4^2) = 10/sqrt(116) ~= 0.928, far above
        # threshold despite 92% whole-text dilution, UNCHANGED by this
        # fix (only the ladder's own units changed -- see class
        # docstring).
        term = self._term(windowed=True)
        [reward] = term(prompts=[None], completions=[self._exploit(reps=20)])
        sim = 10.0 / (116.0 ** 0.5)
        expected = -1.5 * (sim - sn.DEFAULT_THRESHOLD) / (1.0 - sn.DEFAULT_THRESHOLD)
        self.assertAlmostEqual(reward, expected, places=6)
        self.assertLess(reward, -1.0)  # not a borderline catch

    def test_more_padding_does_not_weaken_the_windowed_catch(self):
        # Dilution's whole premise was "more filler -> less penalty".
        # Max-over-windows must be invariant to extra padding (until the
        # cap frontier): 60 reps must score exactly what 20 reps does.
        term = self._term(windowed=True)
        rewards = term(prompts=[None, None],
                       completions=[self._exploit(reps=20),
                                    self._exploit(reps=60)])
        self.assertAlmostEqual(rewards[0], rewards[1], places=6)

    def test_punctuation_glued_padding_scores_identically_to_spaced(self):
        # 2026-07-17 audit BLOCKER fix, same class as
        # env.rewards.CorpusNoveltyPenalty's: padding joined to the
        # template with NO whitespace at all (only a ".", one per rep,
        # directly glued) used to fuse the last filler word into the
        # template's first word under plain `text.split()` windowing.
        # Boundary-tokenized windowing (`_window_token_spans`) must
        # score this IDENTICALLY to the whitespace-joined exploit --
        # scoring is now invariant to the padding join character, not
        # just to how much padding there is.
        term = self._term(windowed=True)
        glued = (self.FILLER + ".") * 20 + self.TEMPLATE
        spaced = self._exploit(reps=20)
        rewards = term(prompts=[None, None], completions=[glued, spaced])
        self.assertAlmostEqual(rewards[0], rewards[1], places=6)
        self.assertLess(rewards[0], -1.0)  # not a borderline catch

    def test_one_embed_call_per_completion_when_windowed(self):
        # Performance contract: windowed mode batches ALL of a
        # completion's windows (plus the whole text) into ONE embed_fn
        # call -- never one call per window -- and degenerate
        # completions are skipped without any call at all.
        log = []
        term = self._term(log=log, windowed=True)
        n_construction = len(log)
        term(prompts=[None] * 3,
             completions=[self._exploit(reps=20), "a short novel quip", ""])
        scoring_calls = log[n_construction:]
        self.assertEqual(len(scoring_calls), 2)  # 2 non-degenerate only
        # First candidate of each call is the whole completion text
        # (windowed scores are always >= the shipped whole-text score).
        self.assertEqual(scoring_calls[0][0], self._exploit(reps=20))
        self.assertEqual(scoring_calls[1][0], "a short novel quip")

    def test_non_windowed_default_keeps_single_batch_call(self):
        # The shipped path's one-call-per-BATCH shape must be untouched
        # by this feature (it is the validated EXP-009 behavior).
        log = []
        term = self._term(log=log)
        n_construction = len(log)
        term(prompts=[None, None],
             completions=["a short novel quip", "another novel line"])
        self.assertEqual(len(log) - n_construction, 1)

    def test_genuinely_novel_long_completion_not_newly_penalized(self):
        # False-positive guard: 130 words of non-template vocabulary --
        # every window embeds at (0, 1), orthogonal to the template axis
        # -- must score 0.0 in BOTH modes. Windowing must not turn
        # length into a penalty.
        novel = ("the lighthouse keeper counted seagulls while coffee "
                "went cold on the railing nobody believed his tally ") * 8
        on = self._term(windowed=True)
        off = self._term()
        [r_on] = on(prompts=[None], completions=[novel])
        [r_off] = off(prompts=[None], completions=[novel])
        self.assertEqual(r_on, 0.0)
        self.assertEqual(r_on, r_off)

    def test_short_completion_windowed_equals_non_windowed(self):
        # A completion shorter than every window width degenerates to
        # whole-text-only scoring -- windowed and non-windowed must be
        # bit-identical (shipped-behavior preservation on the inputs
        # this stack actually rewards).
        text = "scientists trust weird tiny hats"  # 5 words < width 15
        on = self._term(windowed=True)
        off = self._term()
        self.assertEqual(on(prompts=[None], completions=[text]),
                         off(prompts=[None], completions=[text]))

    def test_degenerate_windows_skipped_and_joke_amid_emoji_caught(self):
        # Degenerate-WINDOW semantics (mirrors the degenerate-completion
        # convention): emoji-only windows must be skipped before
        # embedding -- never handed to embed_fn -- while windows that
        # do contain the embedded template still catch it. 30 emoji
        # tokens + the verbatim template: the level-0 tail window is 4
        # emoji + 10 template words, whose _normalize drops the emoji
        # entirely -> pure template -> cosine 1.0 -> full weight.
        log = []
        term = self._term(log=log, windowed=True)
        n_construction = len(log)
        exploit = ("😂 " * 30) + self.TEMPLATE
        [reward] = term(prompts=[None], completions=[exploit])
        self.assertEqual(reward, -1.5)
        for batch in log[n_construction:]:
            for text in batch:
                self.assertTrue(_normalize(text),
                                "degenerate window reached embed_fn: %r" % text)

    def test_cap_frontier_evades_and_warns_once(self):
        # The documented max_windows security implication, locked in as
        # a test: with the cap squeezed to 4 (budget 2 per ladder
        # level), coverage is a ~20-35-word prefix per level, so a
        # template pushed 240 words deep sits past every frontier and
        # the window tier cannot see it (whole-text is diluted -> 0.0).
        # The cap must warn EXACTLY once per instance, not per call.
        term = self._term(windowed=True, max_windows=4)
        deep = self._exploit(reps=40)  # joke at words 240..249
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            [r1] = term(prompts=[None], completions=[deep])
            [r2] = term(prompts=[None], completions=[deep])
        self.assertEqual(r1, 0.0)  # evaded: past the coverage frontier
        self.assertEqual(r2, 0.0)
        cap_warnings = [w for w in caught
                        if issubclass(w.category, RuntimeWarning)
                        and "max_windows" in str(w.message)]
        self.assertEqual(len(cap_warnings), 1)

    def test_windowed_without_templates_raises(self):
        # Windowed mode's ladder is anchored on template lengths -- a
        # corpus with no templates (only reachable via
        # reference="corpus") has no length anchor and must refuse
        # loudly at construction, not silently score without windows.
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "jokes.jsonl", "w") as f:
                f.write(json.dumps({"text": "a joke with no templates"}) + "\n")
            with self.assertRaises(ValueError):
                SemanticNoveltyPenalty(
                    corpus_dir=d, embed_fn=_make_embed_fn({}, default=(1.0, 0.0)),
                    reference="corpus", threshold=0.9, windowed=True)


class TestDegradedMode(unittest.TestCase):
    """`allow_degraded` is a deliberate, loud escape hatch -- NOT a silent
    fallback. Simulate `sentence_transformers`/`numpy` being uninstalled
    via `sys.modules` sentinels (the standard technique for forcing
    ImportError in tests) rather than actually uninstalling anything."""

    _UNAVAILABLE = {"sentence_transformers": None, "numpy": None}

    def test_unavailable_backend_raises_loudly_by_default(self):
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with self.assertRaises(SemanticNoveltyUnavailable):
                SemanticNoveltyPenalty(corpus_dir=FIXTURE_CORPUS_DIR)

    def test_semantic_novelty_unavailable_is_an_import_error(self):
        self.assertTrue(issubclass(SemanticNoveltyUnavailable, ImportError))

    def test_allow_degraded_warns_exactly_once_and_flags_degraded(self):
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                term = SemanticNoveltyPenalty(corpus_dir=FIXTURE_CORPUS_DIR,
                                              allow_degraded=True)
        self.assertTrue(term.degraded)
        run_warnings = [w for w in caught
                        if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 1)

    def test_degraded_instance_always_returns_zero(self):
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                term = SemanticNoveltyPenalty(corpus_dir=FIXTURE_CORPUS_DIR,
                                              allow_degraded=True)
        rewards = term(prompts=[None, None],
                       completions=["anything at all", "even a perfect reskin"])
        self.assertEqual(rewards, [0.0, 0.0])

    def test_nonexistent_corpus_dir_still_raises_even_when_degraded_allowed(self):
        # allow_degraded excuses a missing BACKEND, not a wrong PATH --
        # directory existence is a cheap check done unconditionally,
        # before backend availability is even examined (see __init__'s
        # check ordering), so a flat-out bad path is still an immediate,
        # loud config error regardless of allow_degraded.
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with self.assertRaises(FileNotFoundError):
                SemanticNoveltyPenalty(
                    corpus_dir="/nonexistent/path/for/sure/xyz",
                    allow_degraded=True)

    def test_degraded_construction_skips_expensive_corpus_content_loading(self):
        # Directory existence IS still validated (see test above) -- but
        # once degraded, the expensive content scan/reservoir-sample/
        # embedding work is skipped entirely, so an EXISTING but EMPTY
        # corpus dir (which raises "nothing to compare against" in normal
        # mode -- see test_empty_corpus_dir_raises) must NOT raise here.
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(sys.modules, self._UNAVAILABLE):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    term = SemanticNoveltyPenalty(corpus_dir=d,
                                                  allow_degraded=True)
        self.assertTrue(term.degraded)


class TestRewardConfigWiring(unittest.TestCase):
    """RewardConfig.semantic_novelty_weight -- the minimal, optional-tier
    diff on env/rewards.py. `reward_stack()`'s 6th-term wiring is tested
    via a fake `SemanticNoveltyPenalty` (monkeypatched onto the module
    object reward_stack() lazily imports from) so this needs no real
    embedding backend either."""

    def _config(self, **overrides):
        defaults = dict(judge=lambda p, c: 0.8,
                        joke_corpus_dir=FIXTURE_CORPUS_DIR, group_size=2)
        defaults.update(overrides)
        return RewardConfig(**defaults)

    def test_default_weight_is_zero_and_off(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.semantic_novelty_weight, 0.0)

    def test_positive_weight_raises_sign_guard(self):
        with self.assertRaises(ValueError):
            RewardConfig(semantic_novelty_weight=1.5)

    def test_zero_weight_is_valid(self):
        RewardConfig(semantic_novelty_weight=0.0)  # must not raise

    def test_negative_weight_is_valid(self):
        RewardConfig(semantic_novelty_weight=-1.5)  # must not raise

    def test_default_config_reward_stack_has_five_terms_no_semantic(self):
        funcs = reward_stack(self._config())
        names = [f.__name__ for f in funcs]
        self.assertEqual(len(funcs), 5)
        self.assertNotIn("semantic_novelty_penalty", names)

    def test_nonzero_weight_appends_sixth_term(self):
        class FakePenalty:
            __name__ = "semantic_novelty_penalty"

            def __init__(self, corpus_dir, weight):
                self.corpus_dir = corpus_dir
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [0.0] * len(completions)

        cfg = self._config(semantic_novelty_weight=-1.5)
        with mock.patch.object(sn, "SemanticNoveltyPenalty", FakePenalty):
            funcs = reward_stack(cfg)
        self.assertEqual(len(funcs), 6)
        self.assertEqual(funcs[-1].__name__, "semantic_novelty_penalty")
        self.assertEqual(funcs[-1].weight, -1.5)
        self.assertEqual(funcs[-1].corpus_dir, FIXTURE_CORPUS_DIR)

    def test_combined_reward_includes_semantic_term_when_enabled(self):
        class FakePenalty:
            __name__ = "semantic_novelty_penalty"

            def __init__(self, corpus_dir, weight):
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [self.weight] * len(completions)

        cfg = self._config(semantic_novelty_weight=-1.5)
        with mock.patch.object(sn, "SemanticNoveltyPenalty", FakePenalty):
            fn = combined_reward(cfg)
            rewards = fn(
                prompts=[None, None],
                completions=["Bananas wear tiny hats at the office party tonight.",
                             "Xylophones dream about quantum lizards during winter storms."])
        # Same hand-computed baseline as test_rewards.py's
        # test_combined_reward_hand_computed (0.8 + 0 + 0 + 0.5 + 0.225),
        # plus this fake term's flat -1.5 contribution.
        expected = 0.8 + 0.0 + 0.0 + 0.5 + 0.225 + (-1.5)
        self.assertAlmostEqual(rewards[0], expected, places=6)
        self.assertAlmostEqual(rewards[1], expected, places=6)


if __name__ == "__main__":
    unittest.main()

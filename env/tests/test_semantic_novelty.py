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
from env.rewards import RewardConfig, combined_reward, reward_stack
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

"""Unit tests for env/incongruity_gate.py -- fake `predictor`/`embed_fn`
callables only, NO model download / network / CLI calls anywhere in this
file (see env/tests/fixtures/incongruity_gate_fixture.jsonl and
docs/THEORY-MAP.md §12.2 for the real, pre-registered validation design
EXP-014 will run against a real predictor + embedding backend -- that run
is NOT this file's job; this file locks in the class's own two-gate
mechanics, the strict-AND/flat-bonus semantics, and config guards with
hand-picked fake responses/vectors).
Run: python3 -m unittest discover -s env/tests -v
"""

import json
import math
import re
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import env.incongruity_gate as incongruity_gate_module
from env.incongruity_gate import (PREDICT_COLD_PROMPT, PREDICT_PRIMED_PROMPT,
                                  SPLIT_PROMPT, TwoStageIncongruityGate)
from env.rewards import RewardConfig, combined_reward, reward_stack
from env.semantic_novelty import SemanticNoveltyUnavailable

FIXTURE_CORPUS_DIR = Path(__file__).parent / "fixtures" / "corpus"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "incongruity_gate_fixture.jsonl"

JOKE_A = "Bananas wear tiny hats at the office party tonight."
JOKE_B = "Xylophones dream about quantum lizards during winter storms."


def _completions(*texts):
    return list(texts)


# --------------------------------------------------------- fake predictor

_SPLIT_JOKE_RE = re.compile(r"Joke: (.*?)\n\nOutput exactly two lines:", re.S)
_SETUP_RE = re.compile(r"Setup: (.*?)\n\n(?:Predict|This is)", re.S)


def _make_predictor(split_map, cold_map=None, primed_map=None):
    """Fake predictor(prompt) -> str, routed by which of the three
    §12.2 prompt templates `prompt` was rendered from (SPLIT_PROMPT has
    'Joke: ...', PREDICT_COLD_PROMPT is the only one containing
    'UNSURPRISING', PREDICT_PRIMED_PROMPT is the only one containing
    'clever twist' -- mirrors env/tests/test_banter_env.py's regex-based
    fake_partner convention).

    `split_map` keys on the full completion text, values are either
    `(setup, punchline)` (a well-formed split) or the literal string
    "NO_SPLIT" (mirrors a real predictor's documented sentinel).
    `cold_map`/`primed_map` key on the SETUP text extracted from a
    PREDICT_COLD_PROMPT/PREDICT_PRIMED_PROMPT rendering.
    """
    cold_map = cold_map or {}
    primed_map = primed_map or {}
    calls = []

    def predictor(prompt):
        calls.append(prompt)
        m = _SPLIT_JOKE_RE.search(prompt)
        if m:
            completion = m.group(1).strip()
            entry = split_map.get(completion, "NO_SPLIT")
            if entry == "NO_SPLIT":
                return "NO_SPLIT"
            setup, punchline = entry
            return "SETUP: %s\nPUNCHLINE: %s" % (setup, punchline)
        if "UNSURPRISING" in prompt:
            setup = _SETUP_RE.search(prompt).group(1).strip()
            return cold_map[setup]
        if "clever twist" in prompt:
            setup = _SETUP_RE.search(prompt).group(1).strip()
            return primed_map[setup]
        raise AssertionError("unrecognized predictor prompt: %r" % prompt)

    predictor.calls = calls
    return predictor


# ----------------------------------------------------------- fake embed_fn


def _v(cos_to_ref):
    """A 2D unit vector whose cosine similarity to the fixed reference
    vector (1.0, 0.0) is exactly `cos_to_ref` -- lets every distance in
    these tests be chosen directly as a number (1 - cos_to_ref) instead of
    hand-worked trig, while staying genuinely unit-normalized (the
    embed_fn contract SemanticNoveltyPenalty/TwoStageIncongruityGate both
    assume)."""
    return (cos_to_ref, math.sqrt(max(0.0, 1.0 - cos_to_ref ** 2)))


def _make_embed_fn(vector_map):
    def embed(texts):
        return [vector_map[t] for t in texts]
    return embed


# ------------------------------------------------------- _split unit tests


class TestSplit(unittest.TestCase):
    def _gate(self, predictor):
        return TwoStageIncongruityGate(predictor=predictor,
                                       embed_fn=_make_embed_fn({}))

    def test_well_formed_split_parses_both_lines(self):
        predictor = lambda p: "SETUP: the setup\nPUNCHLINE: the punchline"
        gate = self._gate(predictor)
        setup, punchline = gate._split("irrelevant completion text")
        self.assertEqual(setup, "the setup")
        self.assertEqual(punchline, "the punchline")

    def test_no_split_sentinel_returns_none_none(self):
        predictor = lambda p: "NO_SPLIT"
        gate = self._gate(predictor)
        setup, punchline = gate._split("a single-clause pun")
        self.assertIsNone(setup)
        self.assertIsNone(punchline)

    def test_missing_punchline_line_returns_none_none(self):
        predictor = lambda p: "SETUP: the setup\n(no punchline line at all)"
        gate = self._gate(predictor)
        setup, punchline = gate._split("x")
        self.assertIsNone(setup)
        self.assertIsNone(punchline)

    def test_missing_setup_line_returns_none_none(self):
        predictor = lambda p: "PUNCHLINE: the punchline\n(no setup line)"
        gate = self._gate(predictor)
        setup, punchline = gate._split("x")
        self.assertIsNone(setup)
        self.assertIsNone(punchline)

    def test_empty_predictor_output_returns_none_none(self):
        predictor = lambda p: ""
        gate = self._gate(predictor)
        setup, punchline = gate._split("x")
        self.assertIsNone(setup)
        self.assertIsNone(punchline)

    def test_whitespace_around_values_is_stripped(self):
        predictor = lambda p: "SETUP:   the setup   \nPUNCHLINE:   the punchline  "
        gate = self._gate(predictor)
        setup, punchline = gate._split("x")
        self.assertEqual(setup, "the setup")
        self.assertEqual(punchline, "the punchline")

    def test_split_prompt_formats_completion_in(self):
        seen = []
        predictor = lambda p: (seen.append(p),
                               "SETUP: s\nPUNCHLINE: p")[1]
        gate = self._gate(predictor)
        gate._split("MY COMPLETION TEXT")
        self.assertIn("MY COMPLETION TEXT", seen[0])
        self.assertEqual(seen[0], SPLIT_PROMPT.format(completion="MY COMPLETION TEXT"))


# ----------------------------------------------------- _distance unit tests


class TestDistance(unittest.TestCase):
    def test_identical_vectors_zero_distance(self):
        gate = TwoStageIncongruityGate(
            predictor=lambda p: "", embed_fn=_make_embed_fn({"a": (1.0, 0.0), "b": (1.0, 0.0)}))
        self.assertAlmostEqual(gate._distance("a", "b"), 0.0, places=6)

    def test_orthogonal_vectors_distance_one(self):
        gate = TwoStageIncongruityGate(
            predictor=lambda p: "", embed_fn=_make_embed_fn({"a": (1.0, 0.0), "b": (0.0, 1.0)}))
        self.assertAlmostEqual(gate._distance("a", "b"), 1.0, places=6)

    def test_partial_similarity_hand_computed(self):
        gate = TwoStageIncongruityGate(
            predictor=lambda p: "",
            embed_fn=_make_embed_fn({"a": (1.0, 0.0), "b": _v(0.4)}))
        self.assertAlmostEqual(gate._distance("a", "b"), 0.6, places=6)


# ---------------------------------------------------- full gate semantics


class TestGateSemantics(unittest.TestCase):
    """The four scenarios docs/THEORY-MAP.md §12.2's gaming analysis and
    validation design distinguish: genuine resolution (both gates pass),
    unsurprising setup (gate 1 alone fails -- the `boring_expected`
    shape), surprising-but-never-resolves (gate 1 passes, gate 2 sees NO
    drop at all -- the `setup_nonsequitur` shape), and surprising-with-an-
    insufficient-drop (gate 1 passes, gate 2's drop exists but doesn't
    clear `drop_threshold` -- distinct from the no-drop case, both are
    real, distinguishable gate-2 failure modes)."""

    RESOLVES = "completion: genuinely resolves"
    BORING = "completion: unsurprising setup"
    NO_DROP = "completion: surprising but never resolves"
    SMALL_DROP = "completion: surprising with insufficient drop"
    NO_SPLIT = "completion: no setup/punchline structure at all"

    SPLIT_MAP = {
        RESOLVES: ("setup-resolves", "punchline-resolves"),
        BORING: ("setup-boring", "punchline-boring"),
        NO_DROP: ("setup-nodrop", "punchline-nodrop"),
        SMALL_DROP: ("setup-smalldrop", "punchline-smalldrop"),
        NO_SPLIT: "NO_SPLIT",
    }
    COLD_MAP = {
        "setup-resolves": "cold-resolves",     # far from punchline -> surprising
        "setup-boring": "cold-boring",         # close to punchline -> NOT surprising
        "setup-nodrop": "cold-nodrop",         # far
        "setup-smalldrop": "cold-smalldrop",   # far
    }
    PRIMED_MAP = {
        "setup-resolves": "primed-resolves",   # much closer than cold -> resolves
        "setup-boring": "primed-boring",       # irrelevant, gate 1 already failed
        "setup-nodrop": "primed-nodrop",       # SAME distance as cold -> no drop
        "setup-smalldrop": "primed-smalldrop", # closer, but drop < drop_threshold
    }
    VECTOR_MAP = {
        "punchline-resolves": (1.0, 0.0),
        "cold-resolves": _v(0.0),              # distance 1.0
        "primed-resolves": _v(0.95),           # distance 0.05

        "punchline-boring": (1.0, 0.0),
        "cold-boring": _v(0.9),                # distance 0.1 (< 0.5 threshold)
        "primed-boring": _v(0.0),

        "punchline-nodrop": (1.0, 0.0),
        "cold-nodrop": _v(0.0),                # distance 1.0
        "primed-nodrop": _v(0.0),              # distance 1.0 -- no drop at all

        "punchline-smalldrop": (1.0, 0.0),
        "cold-smalldrop": _v(0.4),             # distance 0.6
        "primed-smalldrop": _v(0.5),           # distance 0.5 -- drop = 0.1 < 0.15
    }

    def _gate(self, weight=1.0, surprise_threshold=0.5, drop_threshold=0.15):
        predictor = _make_predictor(self.SPLIT_MAP, self.COLD_MAP, self.PRIMED_MAP)
        embed_fn = _make_embed_fn(self.VECTOR_MAP)
        gate = TwoStageIncongruityGate(
            predictor=predictor, embed_fn=embed_fn, weight=weight,
            surprise_threshold=surprise_threshold, drop_threshold=drop_threshold)
        gate._predictor_calls = predictor.calls
        return gate

    def test_genuine_resolution_earns_the_flat_bonus(self):
        gate = self._gate(weight=1.0)
        rewards = gate(prompts=[None], completions=_completions(self.RESOLVES))
        self.assertEqual(rewards, [1.0])

    def test_unsurprising_setup_fails_gate_one_scores_zero(self):
        gate = self._gate(weight=1.0)
        rewards = gate(prompts=[None], completions=_completions(self.BORING))
        self.assertEqual(rewards, [0.0])

    def test_surprising_with_no_resolution_drop_scores_zero(self):
        gate = self._gate(weight=1.0)
        rewards = gate(prompts=[None], completions=_completions(self.NO_DROP))
        self.assertEqual(rewards, [0.0])

    def test_surprising_with_insufficient_drop_scores_zero(self):
        gate = self._gate(weight=1.0)
        rewards = gate(prompts=[None], completions=_completions(self.SMALL_DROP))
        self.assertEqual(rewards, [0.0])

    def test_no_split_completion_scores_zero(self):
        gate = self._gate(weight=1.0)
        rewards = gate(prompts=[None], completions=_completions(self.NO_SPLIT))
        self.assertEqual(rewards, [0.0])

    def test_no_split_completion_never_calls_predictor_for_cold_or_primed(self):
        # A can't-apply completion should short-circuit after the split
        # call -- exactly 1 predictor call, not 3.
        gate = self._gate(weight=1.0)
        gate(prompts=[None], completions=_completions(self.NO_SPLIT))
        self.assertEqual(len(gate._predictor_calls), 1)

    def test_full_call_uses_exactly_three_predictor_calls(self):
        # split + cold-predict + primed-predict, per §12.2's stated cost.
        gate = self._gate(weight=1.0)
        gate(prompts=[None], completions=_completions(self.RESOLVES))
        self.assertEqual(len(gate._predictor_calls), 3)

    def test_flat_bonus_is_never_scaled_by_surprisal_magnitude(self):
        # §12.2's single most load-bearing constraint: a MUCH more
        # surprising cold guess (larger d_cold) must NOT earn a larger
        # reward than a barely-surprising one that still clears both
        # gates -- this is a GATE, not distance_cold as a scalar to
        # maximize. Construct two passing completions with different
        # d_cold values (1.0 vs a lower-but-still-passing value) and
        # confirm both score the SAME flat weight.
        split_map = dict(self.SPLIT_MAP)
        split_map["barely surprising, still resolves"] = ("setup-barely", "punchline-barely")
        cold_map = dict(self.COLD_MAP, **{"setup-barely": "cold-barely"})
        primed_map = dict(self.PRIMED_MAP, **{"setup-barely": "primed-barely"})
        vector_map = dict(self.VECTOR_MAP)
        vector_map["punchline-barely"] = (1.0, 0.0)
        vector_map["cold-barely"] = _v(0.45)     # distance 0.55 -- barely clears 0.5
        vector_map["primed-barely"] = _v(0.95)   # distance 0.05 -- clears drop_threshold

        predictor = _make_predictor(split_map, cold_map, primed_map)
        gate = TwoStageIncongruityGate(predictor=predictor,
                                       embed_fn=_make_embed_fn(vector_map),
                                       weight=1.0)
        rewards = gate(prompts=[None, None], completions=_completions(
            self.RESOLVES, "barely surprising, still resolves"))
        self.assertEqual(rewards, [1.0, 1.0])  # identical flat bonus, not scaled

    def test_weight_scales_the_flat_bonus_not_the_gate_margin(self):
        gate = self._gate(weight=3.0)
        rewards = gate(prompts=[None], completions=_completions(self.RESOLVES))
        self.assertEqual(rewards, [3.0])

    def test_zero_weight_is_inert_even_when_both_gates_pass(self):
        gate = self._gate(weight=0.0)
        rewards = gate(prompts=[None], completions=_completions(self.RESOLVES))
        self.assertEqual(rewards, [0.0])

    def test_batch_scores_each_completion_independently(self):
        gate = self._gate(weight=1.0)
        rewards = gate(prompts=[None] * 4, completions=_completions(
            self.RESOLVES, self.BORING, self.NO_DROP, self.SMALL_DROP))
        self.assertEqual(rewards, [1.0, 0.0, 0.0, 0.0])

    def test_reward_func_signature_accepts_extra_kwargs(self):
        gate = self._gate(weight=1.0)
        gate(prompts=[None], completions=_completions(self.RESOLVES),
             some_dataset_column=["x"], answer=["a"])

    def test_dunder_name_is_two_stage_incongruity_gate(self):
        # Instance-level access -- see test_bvt_gate.py's identical test
        # for why `TwoStageIncongruityGate.__name__` (the CLASS object)
        # is NOT the right thing to assert here: it resolves through
        # `type`'s own data descriptor to the real class name instead.
        gate = self._gate(weight=1.0)
        self.assertEqual(gate.__name__, "two_stage_incongruity_gate")


class TestPromptTemplates(unittest.TestCase):
    """SPLIT_PROMPT/PREDICT_COLD_PROMPT/PREDICT_PRIMED_PROMPT are verbatim
    from docs/THEORY-MAP.md §12.2 -- lock the NO_SPLIT sentinel
    instruction and each prompt's placeholder in place so they can't
    silently drift out during an edit."""

    def test_split_prompt_has_completion_placeholder_and_no_split_sentinel(self):
        self.assertIn("{completion}", SPLIT_PROMPT)
        self.assertIn("NO_SPLIT", SPLIT_PROMPT)

    def test_cold_prompt_asks_for_unsurprising_continuation(self):
        self.assertIn("{setup}", PREDICT_COLD_PROMPT)
        self.assertIn("UNSURPRISING", PREDICT_COLD_PROMPT)
        self.assertNotIn("twist", PREDICT_COLD_PROMPT.lower())

    def test_primed_prompt_asks_for_the_twist(self):
        self.assertIn("{setup}", PREDICT_PRIMED_PROMPT)
        self.assertIn("clever twist", PREDICT_PRIMED_PROMPT)


class TestNoPredictorConfigured(unittest.TestCase):
    """§12.2's given __call__ pseudocode checks `self.degraded or
    self.predictor is None` and returns 0.0 with NO warnings.warn call --
    unlike JudgePreferenceReward/BVTGateReward's warn-once-per-instance
    precedent. Implemented exactly as specified (silent 0.0); this test
    locks that literal behavior in and documents the asymmetry rather than
    silently adding a warning the spec's pseudocode didn't call for."""

    def test_no_predictor_is_silently_zero(self):
        gate = TwoStageIncongruityGate(embed_fn=_make_embed_fn({}))
        rewards = gate(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
        self.assertEqual(rewards, [0.0, 0.0])

    def test_no_predictor_emits_no_warning_matching_spec_pseudocode_literally(self):
        gate = TwoStageIncongruityGate(embed_fn=_make_embed_fn({}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gate(prompts=[None], completions=_completions(JOKE_A))
        run_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 0)


class TestDegradedMode(unittest.TestCase):
    """`allow_degraded` is a deliberate, loud escape hatch -- same
    discipline as SemanticNoveltyPenalty (this class reuses that
    exception type, not a new one). Simulate sentence_transformers/numpy
    being uninstalled via sys.modules sentinels, same technique
    test_semantic_novelty.py's TestDegradedMode uses."""

    _UNAVAILABLE = {"sentence_transformers": None, "numpy": None}

    def test_unavailable_backend_raises_semantic_novelty_unavailable(self):
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with self.assertRaises(SemanticNoveltyUnavailable):
                TwoStageIncongruityGate(predictor=lambda p: "NO_SPLIT")

    def test_unavailable_backend_raises_even_with_no_predictor(self):
        # Backend failure is a distinct, more fundamental configuration
        # problem than "no predictor" -- construction must not silently
        # skip the backend check just because predictor is also unset.
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with self.assertRaises(SemanticNoveltyUnavailable):
                TwoStageIncongruityGate()

    def test_allow_degraded_warns_exactly_once_and_flags_degraded(self):
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gate = TwoStageIncongruityGate(predictor=lambda p: "NO_SPLIT",
                                               allow_degraded=True)
        self.assertTrue(gate.degraded)
        run_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 1)

    def test_degraded_instance_always_returns_zero(self):
        with mock.patch.dict(sys.modules, self._UNAVAILABLE):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gate = TwoStageIncongruityGate(predictor=lambda p: "NO_SPLIT",
                                               allow_degraded=True)
        rewards = gate(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
        self.assertEqual(rewards, [0.0, 0.0])


class TestRewardConfigWiring(unittest.TestCase):
    """RewardConfig.two_stage_incongruity_weight/incongruity_predictor --
    reward_stack()'s wiring, tested via a fake TwoStageIncongruityGate
    (monkeypatched onto the module object reward_stack() lazily imports
    from). Mirrors test_semantic_novelty.py's TestRewardConfigWiring and
    test_bvt_gate.py's."""

    def _config(self, **overrides):
        defaults = dict(judge=lambda p, c: 0.8,
                        joke_corpus_dir=FIXTURE_CORPUS_DIR, group_size=2)
        defaults.update(overrides)
        return RewardConfig(**defaults)

    def test_default_weight_is_zero_and_off(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.two_stage_incongruity_weight, 0.0)
        self.assertIsNone(cfg.incongruity_predictor)
        self.assertEqual(cfg.incongruity_surprise_threshold, 0.5)
        self.assertEqual(cfg.incongruity_drop_threshold, 0.15)

    def test_negative_weight_raises_sign_guard(self):
        with self.assertRaises(ValueError):
            RewardConfig(two_stage_incongruity_weight=-1.0)

    def test_zero_weight_is_valid(self):
        RewardConfig(two_stage_incongruity_weight=0.0)  # must not raise

    def test_positive_weight_is_valid(self):
        RewardConfig(two_stage_incongruity_weight=1.0)  # must not raise

    def test_default_config_reward_stack_has_five_terms_no_incongruity_gate(self):
        funcs = reward_stack(self._config())
        names = [f.__name__ for f in funcs]
        self.assertEqual(len(funcs), 5)
        self.assertNotIn("two_stage_incongruity_gate", names)

    def test_nonzero_weight_appends_a_term(self):
        class FakeGate:
            __name__ = "two_stage_incongruity_gate"

            def __init__(self, predictor, weight, surprise_threshold, drop_threshold):
                self.predictor = predictor
                self.weight = weight
                self.surprise_threshold = surprise_threshold
                self.drop_threshold = drop_threshold

            def __call__(self, prompts, completions, **kwargs):
                return [0.0] * len(completions)

        pred = lambda p: "NO_SPLIT"
        cfg = self._config(two_stage_incongruity_weight=1.0,
                           incongruity_predictor=pred,
                           incongruity_surprise_threshold=0.6,
                           incongruity_drop_threshold=0.2)
        with mock.patch.object(incongruity_gate_module, "TwoStageIncongruityGate", FakeGate):
            funcs = reward_stack(cfg)
        self.assertEqual(len(funcs), 6)
        self.assertEqual(funcs[-1].__name__, "two_stage_incongruity_gate")
        self.assertEqual(funcs[-1].weight, 1.0)
        self.assertIs(funcs[-1].predictor, pred)
        self.assertEqual(funcs[-1].surprise_threshold, 0.6)
        self.assertEqual(funcs[-1].drop_threshold, 0.2)

    def test_combined_reward_includes_incongruity_term_when_enabled(self):
        class FakeGate:
            __name__ = "two_stage_incongruity_gate"

            def __init__(self, predictor, weight, surprise_threshold, drop_threshold):
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [self.weight] * len(completions)

        cfg = self._config(two_stage_incongruity_weight=0.4,
                           incongruity_predictor=lambda p: "NO_SPLIT")
        with mock.patch.object(incongruity_gate_module, "TwoStageIncongruityGate", FakeGate):
            fn = combined_reward(cfg)
            rewards = fn(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
        expected = (0.8 + 0.0 + 0.0 + 0.5 + 0.225) + 0.4
        self.assertAlmostEqual(rewards[0], expected, places=6)
        self.assertAlmostEqual(rewards[1], expected, places=6)


class TestFullOptionalStackOrdering(unittest.TestCase):
    """All three optional tiers (semantic_novelty, bvt_gate,
    two_stage_incongruity) stacked at once -- reward_stack()'s own
    docstring claims "up to 8 terms" and a fixed append order
    (semantic novelty, then bvt gate, then incongruity gate); this is the
    one place that combined claim is exercised directly, with every
    optional module faked so it needs neither a real embedding backend
    nor real judges."""

    def _config(self, **overrides):
        defaults = dict(judge=lambda p, c: 0.8,
                        joke_corpus_dir=FIXTURE_CORPUS_DIR, group_size=2,
                        semantic_novelty_weight=-1.5,
                        bvt_gate_weight=1.0,
                        violation_judge=lambda p, c: 0.5,
                        benign_judge=lambda p, c: 0.5,
                        two_stage_incongruity_weight=0.7,
                        incongruity_predictor=lambda p: "NO_SPLIT")
        defaults.update(overrides)
        return RewardConfig(**defaults)

    def test_eight_terms_in_fixed_order(self):
        class FakeSemantic:
            __name__ = "semantic_novelty_penalty"

            def __init__(self, corpus_dir, weight):
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [0.0] * len(completions)

        class FakeBVT:
            __name__ = "bvt_gate_reward"

            def __init__(self, violation_judge, benign_judge, weight):
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [0.0] * len(completions)

        class FakeIncongruity:
            __name__ = "two_stage_incongruity_gate"

            def __init__(self, predictor, weight, surprise_threshold, drop_threshold):
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [0.0] * len(completions)

        import env.semantic_novelty as sn
        import env.bvt_gate as bg

        with mock.patch.object(sn, "SemanticNoveltyPenalty", FakeSemantic), \
             mock.patch.object(bg, "BVTGateReward", FakeBVT), \
             mock.patch.object(incongruity_gate_module, "TwoStageIncongruityGate",
                               FakeIncongruity):
            funcs = reward_stack(self._config())

        self.assertEqual(len(funcs), 8)
        names = [f.__name__ for f in funcs]
        self.assertEqual(names, [
            "judge_preference", "corpus_novelty_penalty",
            "self_repetition_penalty", "intra_group_diversity",
            "comprehensibility", "semantic_novelty_penalty",
            "bvt_gate_reward", "two_stage_incongruity_gate",
        ])


class TestFixtureIntegrity(unittest.TestCase):
    """env/tests/fixtures/incongruity_gate_fixture.jsonl -- the 40-item
    (real_joke=12, setup_nonsequitur=12, boring_expected=8,
    vague_abstract_gaming_probe=8) fixture docs/THEORY-MAP.md §12.2
    registers for EXP-014. Locks the fixture's shape/integrity; does not
    run the actual (real-predictor) validation."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH) as f:
            cls.items = [json.loads(line) for line in f if line.strip()]

    def test_total_item_count_is_forty(self):
        self.assertEqual(len(self.items), 40)

    def test_class_counts_match_spec(self):
        from collections import Counter
        counts = Counter(item["gold_class"] for item in self.items)
        self.assertEqual(counts["real_joke"], 12)
        self.assertEqual(counts["setup_nonsequitur"], 12)
        self.assertEqual(counts["boring_expected"], 8)
        self.assertEqual(counts["vague_abstract_gaming_probe"], 8)

    def test_ids_are_unique(self):
        ids = [item["id"] for item in self.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_item_has_setup_and_punchline(self):
        for item in self.items:
            self.assertTrue(item["setup"].strip(), msg=item["id"])
            self.assertTrue(item["punchline"].strip(), msg=item["id"])

    def test_setup_nonsequitur_shares_setups_with_real_joke(self):
        real_joke_setups = {item["id"]: item["setup"] for item in self.items
                            if item["gold_class"] == "real_joke"}
        nonseq = [item for item in self.items
                 if item["gold_class"] == "setup_nonsequitur"]
        self.assertEqual(len(nonseq), 12)
        for item in nonseq:
            self.assertIn(item["base_id"], real_joke_setups)
            self.assertEqual(item["setup"], real_joke_setups[item["base_id"]])

    def test_setup_nonsequitur_punchlines_differ_from_real_joke_punchlines(self):
        real_joke_punchlines = {item["id"]: item["punchline"] for item in self.items
                                if item["gold_class"] == "real_joke"}
        for item in self.items:
            if item["gold_class"] != "setup_nonsequitur":
                continue
            self.assertNotEqual(item["punchline"], real_joke_punchlines[item["base_id"]])

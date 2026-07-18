"""Unit tests for env/bvt_gate.py -- fake violation/benign judge callables
only, NO network or CLI calls (see env/tests/fixtures/bvt_gate_fixture.jsonl
and docs/THEORY-MAP.md §12.1 for the real, pre-registered validation design
EXP-013 will run against real judges -- that run is NOT this file's job;
this file locks in the class's own product-semantics/config-guard
mechanics with hand-picked fake scores).
Run: python3 -m unittest discover -s env/tests -v
"""

import json
import unittest
import warnings
from pathlib import Path
from unittest import mock

import env.bvt_gate as bvt_gate_module
from env.bvt_gate import BENIGN_PROMPT, VIOLATION_PROMPT, BVTGateReward
from env.rewards import RewardConfig, combined_reward, reward_stack

FIXTURE_CORPUS_DIR = Path(__file__).parent / "fixtures" / "corpus"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bvt_gate_fixture.jsonl"

JOKE_A = "Bananas wear tiny hats at the office party tonight."
JOKE_B = "Xylophones dream about quantum lizards during winter storms."


def _completions(*texts):
    return list(texts)


def _const_judge(value):
    """A fake judge that returns `value` regardless of (prompt, completion)."""
    return lambda p, c: value


class TestBVTGateReward(unittest.TestCase):
    def test_no_judges_configured_is_zero_and_warns_once_not_per_call(self):
        term = BVTGateReward()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r1 = term(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
            r2 = term(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
        self.assertEqual(r1, [0.0, 0.0])
        self.assertEqual(r2, [0.0, 0.0])
        run_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 1)

    def test_only_violation_judge_configured_is_still_zero_and_warns(self):
        term = BVTGateReward(violation_judge=_const_judge(0.9))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [0.0])
        run_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 1)

    def test_only_benign_judge_configured_is_still_zero_and_warns(self):
        term = BVTGateReward(benign_judge=_const_judge(0.9))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [0.0])
        run_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 1)

    def test_product_semantics_violation_only_is_zero(self):
        # violation high, benign ~0 -- the product gates it to ~0, the
        # entire point of the multiplicative structure (§12.1: an additive
        # sum would let high violation partially compensate).
        term = BVTGateReward(violation_judge=_const_judge(0.9),
                             benign_judge=_const_judge(0.0), weight=1.0)
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [0.0])

    def test_product_semantics_benign_only_is_zero(self):
        term = BVTGateReward(violation_judge=_const_judge(0.0),
                             benign_judge=_const_judge(0.9), weight=1.0)
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [0.0])

    def test_product_semantics_both_high_is_high(self):
        term = BVTGateReward(violation_judge=_const_judge(0.8),
                             benign_judge=_const_judge(0.9), weight=1.0)
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertAlmostEqual(rewards[0], 0.72, places=6)

    def test_weight_scales_the_product(self):
        term = BVTGateReward(violation_judge=_const_judge(0.5),
                             benign_judge=_const_judge(0.5), weight=2.0)
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertAlmostEqual(rewards[0], 0.5, places=6)  # 2.0 * 0.5 * 0.5

    def test_zero_weight_is_inert_even_with_both_judges_configured(self):
        term = BVTGateReward(violation_judge=_const_judge(1.0),
                             benign_judge=_const_judge(1.0), weight=0.0)
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertEqual(rewards, [0.0])

    def test_mid_range_scores_ramp_multiplicatively_not_additively(self):
        # Sanity check that this is genuinely v*b, not (v+b)/2 or similar --
        # an additive stand-in would give 0.35 here, not 0.06.
        term = BVTGateReward(violation_judge=_const_judge(0.2),
                             benign_judge=_const_judge(0.3), weight=1.0)
        rewards = term(prompts=[None], completions=_completions(JOKE_A))
        self.assertAlmostEqual(rewards[0], 0.06, places=6)

    def test_out_of_range_violation_judge_score_raises(self):
        term = BVTGateReward(violation_judge=_const_judge(7.0),  # raw 0-10, unnormalized
                             benign_judge=_const_judge(0.5))
        with self.assertRaises(ValueError):
            term(prompts=[None], completions=_completions(JOKE_A))

    def test_out_of_range_benign_judge_score_raises(self):
        term = BVTGateReward(violation_judge=_const_judge(0.5),
                             benign_judge=_const_judge(-0.1))
        with self.assertRaises(ValueError):
            term(prompts=[None], completions=_completions(JOKE_A))

    def test_prompt_passed_through_unmodified_to_both_judges(self):
        seen_violation, seen_benign = [], []

        def violation_judge(p, c):
            seen_violation.append(p)
            return 0.5

        def benign_judge(p, c):
            seen_benign.append(p)
            return 0.5

        term = BVTGateReward(violation_judge=violation_judge,
                             benign_judge=benign_judge)
        prompt = [{"role": "user", "content": "tell me a joke"}]
        term(prompts=[prompt], completions=_completions(JOKE_A))
        self.assertEqual(seen_violation, [prompt])
        self.assertEqual(seen_benign, [prompt])

    def test_batch_shape_matches_completions_length(self):
        term = BVTGateReward(violation_judge=_const_judge(0.4),
                             benign_judge=_const_judge(0.6), weight=1.0)
        rewards = term(prompts=[None, None, None],
                      completions=_completions(JOKE_A, JOKE_B, JOKE_A))
        self.assertEqual(len(rewards), 3)
        for r in rewards:
            self.assertAlmostEqual(r, 0.24, places=6)

    def test_each_completion_scored_independently(self):
        scores = {JOKE_A: (0.9, 0.9), JOKE_B: (0.1, 0.1)}
        term = BVTGateReward(
            violation_judge=lambda p, c: scores[c][0],
            benign_judge=lambda p, c: scores[c][1], weight=1.0)
        rewards = term(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
        self.assertAlmostEqual(rewards[0], 0.81, places=6)
        self.assertAlmostEqual(rewards[1], 0.01, places=6)

    def test_conversational_completion_shape_equivalence(self):
        term = BVTGateReward(violation_judge=_const_judge(0.5),
                             benign_judge=_const_judge(0.5), weight=1.0)
        plain = term(prompts=[None], completions=[JOKE_A])
        chat = term(prompts=[None], completions=[[{"content": JOKE_A}]])
        self.assertEqual(plain, chat)

    def test_reward_func_signature_accepts_extra_kwargs(self):
        # TRL calls every reward_func with prompts, completions, PLUS
        # whatever extra dataset columns exist.
        term = BVTGateReward(violation_judge=_const_judge(0.5),
                             benign_judge=_const_judge(0.5), weight=1.0)
        term(prompts=[None], completions=_completions(JOKE_A),
             some_dataset_column=["x"], answer=["a"])

    def test_dunder_name_is_bvt_gate_reward(self):
        # Instance-level access, not the class object itself: `Foo.__name__`
        # on the CLASS always resolves through `type`'s own data
        # descriptor to the real class name ("BVTGateReward"), but TRL
        # calls `.__name__` on an INSTANCE (every entry in reward_stack()'s
        # returned list), where the class-body override below correctly
        # wins -- same reason JudgePreferenceReward's `__name__` class
        # attribute works in practice despite this quirk.
        self.assertEqual(BVTGateReward().__name__, "bvt_gate_reward")


class TestPromptsContainDisclaimerGuard(unittest.TestCase):
    """VIOLATION_PROMPT/BENIGN_PROMPT are verbatim from docs/THEORY-MAP.md
    §12.1 -- these tests lock the load-bearing benign-washing guard
    instruction (BENIGN_PROMPT must tell the judge to discount a mere
    disclaimer) in place so it can't silently drift out during an edit."""

    def test_violation_prompt_has_completion_placeholder(self):
        self.assertIn("{completion}", VIOLATION_PROMPT)

    def test_benign_prompt_has_completion_placeholder(self):
        self.assertIn("{completion}", BENIGN_PROMPT)

    def test_benign_prompt_instructs_discounting_disclaimers(self):
        lowered = BENIGN_PROMPT.lower()
        self.assertIn("jk", lowered)
        self.assertIn("disclaimer", lowered)

    def test_violation_prompt_is_agnostic_to_funniness_and_safety(self):
        lowered = VIOLATION_PROMPT.lower()
        self.assertIn("regardless of whether", lowered)


class TestRewardConfigWiring(unittest.TestCase):
    """RewardConfig.bvt_gate_weight/violation_judge/benign_judge --
    reward_stack()'s wiring, tested via a fake BVTGateReward
    (monkeypatched onto the module object reward_stack() lazily imports
    from) so this needs no real judge either. Mirrors
    test_semantic_novelty.py's TestRewardConfigWiring exactly."""

    def _config(self, **overrides):
        defaults = dict(judge=lambda p, c: 0.8,
                        joke_corpus_dir=FIXTURE_CORPUS_DIR, group_size=2)
        defaults.update(overrides)
        return RewardConfig(**defaults)

    def test_default_weight_is_zero_and_off(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.bvt_gate_weight, 0.0)
        self.assertIsNone(cfg.violation_judge)
        self.assertIsNone(cfg.benign_judge)

    def test_negative_weight_raises_sign_guard(self):
        # bvt_gate_weight is a BONUS field (>= 0) -- a genuine dual
        # appraisal is something to reward, not penalize.
        with self.assertRaises(ValueError):
            RewardConfig(bvt_gate_weight=-1.0)

    def test_zero_weight_is_valid(self):
        RewardConfig(bvt_gate_weight=0.0)  # must not raise

    def test_positive_weight_is_valid(self):
        RewardConfig(bvt_gate_weight=1.0)  # must not raise

    def test_default_config_reward_stack_has_five_terms_no_bvt_gate(self):
        funcs = reward_stack(self._config())
        names = [f.__name__ for f in funcs]
        self.assertEqual(len(funcs), 5)
        self.assertNotIn("bvt_gate_reward", names)

    def test_nonzero_weight_appends_a_term(self):
        class FakeGate:
            __name__ = "bvt_gate_reward"

            def __init__(self, violation_judge, benign_judge, weight):
                self.violation_judge = violation_judge
                self.benign_judge = benign_judge
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [0.0] * len(completions)

        vj = _const_judge(0.5)
        bj = _const_judge(0.5)
        cfg = self._config(bvt_gate_weight=1.0, violation_judge=vj, benign_judge=bj)
        with mock.patch.object(bvt_gate_module, "BVTGateReward", FakeGate):
            funcs = reward_stack(cfg)
        self.assertEqual(len(funcs), 6)
        self.assertEqual(funcs[-1].__name__, "bvt_gate_reward")
        self.assertEqual(funcs[-1].weight, 1.0)
        self.assertIs(funcs[-1].violation_judge, vj)
        self.assertIs(funcs[-1].benign_judge, bj)

    def test_combined_reward_includes_bvt_term_when_enabled(self):
        class FakeGate:
            __name__ = "bvt_gate_reward"

            def __init__(self, violation_judge, benign_judge, weight):
                self.weight = weight

            def __call__(self, prompts, completions, **kwargs):
                return [self.weight] * len(completions)

        cfg = self._config(bvt_gate_weight=0.6, violation_judge=_const_judge(0.5),
                           benign_judge=_const_judge(0.5))
        with mock.patch.object(bvt_gate_module, "BVTGateReward", FakeGate):
            fn = combined_reward(cfg)
            rewards = fn(prompts=[None, None], completions=_completions(JOKE_A, JOKE_B))
        # Same 5-term baseline as test_rewards.py's
        # test_combined_reward_hand_computed, plus this term's flat 0.6.
        expected = (0.8 + 0.0 + 0.0 + 0.5 + 0.225) + 0.6
        self.assertAlmostEqual(rewards[0], expected, places=6)
        self.assertAlmostEqual(rewards[1], expected, places=6)


class TestFixtureIntegrity(unittest.TestCase):
    """env/tests/fixtures/bvt_gate_fixture.jsonl -- the 32-gold-item (4
    classes of 8) + 8-item gaming-probe fixture docs/THEORY-MAP.md §12.1
    registers for EXP-013. This file does not RUN that validation (no real
    judges here) -- it locks the fixture's own shape/integrity so a future
    edit can't silently drift the class layout the pre-registered bars
    depend on."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH) as f:
            cls.items = [json.loads(line) for line in f if line.strip()]

    def test_total_item_count_is_forty(self):
        self.assertEqual(len(self.items), 40)

    def test_class_counts_match_spec(self):
        from collections import Counter
        counts = Counter(item["gold_class"] for item in self.items)
        self.assertEqual(counts["both"], 8)
        self.assertEqual(counts["violation_only"], 8)
        self.assertEqual(counts["benign_only"], 8)
        self.assertEqual(counts["neither"], 8)
        self.assertEqual(counts["disclaimer_washed_violation"], 8)

    def test_ids_are_unique(self):
        ids = [item["id"] for item in self.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_item_has_nonempty_text(self):
        for item in self.items:
            self.assertTrue(item["text"].strip(), msg=item["id"])

    def test_disclaimer_washed_items_reference_a_real_violation_only_base(self):
        violation_only_ids = {item["id"] for item in self.items
                              if item["gold_class"] == "violation_only"}
        washed = [item for item in self.items
                 if item["gold_class"] == "disclaimer_washed_violation"]
        self.assertEqual(len(washed), 8)
        for item in washed:
            self.assertIn(item["base_id"], violation_only_ids)

    def test_disclaimer_washed_items_contain_a_disclaimer_marker(self):
        markers = ("jk", "kidding", "hypothetical", "fictional")
        for item in self.items:
            if item["gold_class"] != "disclaimer_washed_violation":
                continue
            lowered = item["text"].lower()
            self.assertTrue(any(m in lowered for m in markers), msg=item["id"])

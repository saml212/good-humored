"""Unit tests for env/banter_env.py -- fake partner/judge callables only,
NO network or CLI calls.
Run: python3 -m unittest discover -s env/tests -v
"""

import re
import unittest
import warnings

from benchmark.banter import SCENARIO_OPENERS
from env.banter_env import BanterEnv


def fake_partner(prompt):
    """Echoes the scenario blurb it was asked to remark on -- same fake
    used in benchmark/tests/test_banter.py, deterministic and cheap."""
    m = re.search(r"Situation: (.+)", prompt)
    situation = m.group(1) if m else "something"
    return "So, %s." % situation


def fake_judge_context_aware(prompt):
    """Scores 9 if the reply's embedded marker matches the context's LAST
    Friend line, else 2 -- makes the ablation delta mechanical, same
    pattern as benchmark/tests/test_banter.py's fake_judge_context_aware."""
    context_part, _, reply_part = prompt.partition("Reply to score:")
    ctx_friend_lines = [l for l in context_part.splitlines()
                        if l.startswith("Friend:")]
    last_friend = ctx_friend_lines[-1] if ctx_friend_lines else ""
    if last_friend and ("[%s]" % last_friend) in reply_part:
        return "9"
    return "2"


class TestBanterEnvConstruction(unittest.TestCase):
    def test_requires_partner_complete(self):
        with self.assertRaises(ValueError):
            BanterEnv(partner_complete=None)


class TestBanterEnvReset(unittest.TestCase):
    def test_reset_returns_first_scenario_opener(self):
        env = BanterEnv(partner_complete=fake_partner, seed_topic=2)
        opening = env.reset()
        self.assertEqual(opening, "So, %s." % SCENARIO_OPENERS[2])

    def test_reset_clears_prior_episode_state(self):
        env = BanterEnv(partner_complete=fake_partner, seed_topic=0, max_turns=3)
        env.reset()
        env.step("Sure, noted.")
        env.reset()
        self.assertEqual(env.turn, 0)
        self.assertFalse(env.done)
        self.assertEqual(len(env.messages), 1)  # just the fresh opener


class TestBanterEnvCallback(unittest.TestCase):
    def test_callback_bonus_fires_beyond_min_gap(self):
        env = BanterEnv(partner_complete=fake_partner, judge=None,
                        seed_topic=0, max_turns=5, min_gap=2,
                        callback_weight=0.5)
        env.reset()  # turn-0 partner line mentions "the barista got your coffee order wrong"

        _, r0, done0, info0 = env.step("Sure, noted.")
        self.assertIsNone(info0["callback"])
        self.assertEqual(r0, 0.0)  # not enough history yet (n=0 < min_gap)
        self.assertFalse(done0)

        _, r1, done1, info1 = env.step(
            "Yeah that sounds rough, hope the yard's okay now.")
        self.assertIsNone(info1["callback"])
        self.assertEqual(r1, 0.0)  # n=1 < min_gap=2, still not enough
        self.assertFalse(done1)

        _, r2, done2, info2 = env.step("Anyway, that barista thing still bugs me.")
        self.assertEqual(info2["callback"], "barista")
        self.assertEqual(r2, 0.5)  # callback bonus, no judge configured
        self.assertFalse(done2)


class TestBanterEnvAblationDelta(unittest.TestCase):
    def test_delta_computed_when_judge_and_swap_context_supplied(self):
        env = BanterEnv(partner_complete=fake_partner,
                        judge=fake_judge_context_aware, seed_topic=0,
                        max_turns=5, ablation_weight=1.0, callback_weight=0.5)
        partner_line0 = env.reset()
        reply0 = "Oh totally, re[Friend: %s]" % partner_line0
        swapped_context = "Friend: something totally different happened today."

        next_prompt, reward, done, info = env.step(
            reply0, swapped_context=swapped_context)

        self.assertEqual(info["ablation_delta"], 7)  # true=9, swapped=2, raw
        self.assertAlmostEqual(info["ablation_delta_normalized"], 7 / 9.0,
                               places=6)
        # Default is NORMALIZED (audit magnitude fix): reward uses
        # delta/9.0, NOT the raw ±9-scaled delta -- otherwise this single
        # turn's ablation term alone would swamp env/rewards.py's stack
        # (±0.3-1.5 terms) by 5-10x, the exact judge-dominance bug that
        # stack hard-fails on elsewhere.
        self.assertAlmostEqual(reward, 7 / 9.0, places=6)
        self.assertFalse(done)
        self.assertIsInstance(next_prompt, str)

    def test_raw_ablation_delta_flag_restores_old_magnitude(self):
        env = BanterEnv(partner_complete=fake_partner,
                        judge=fake_judge_context_aware, seed_topic=0,
                        max_turns=5, ablation_weight=1.0, callback_weight=0.5,
                        raw_ablation_delta=True)
        partner_line0 = env.reset()
        reply0 = "Oh totally, re[Friend: %s]" % partner_line0
        swapped_context = "Friend: something totally different happened today."

        _, reward, _, info = env.step(reply0, swapped_context=swapped_context)

        self.assertEqual(info["ablation_delta"], 7)  # raw value unaffected
        self.assertAlmostEqual(info["ablation_delta_normalized"], 7 / 9.0,
                               places=6)  # always reported regardless of flag
        self.assertEqual(reward, 7.0)  # but the REWARD uses the raw delta

    def test_no_swapped_context_skips_ablation_without_error(self):
        env = BanterEnv(partner_complete=fake_partner,
                        judge=fake_judge_context_aware, seed_topic=0,
                        max_turns=5)
        env.reset()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, reward, _, info = env.step("Oh totally, sure.")
        self.assertIsNone(info["ablation_delta"])
        self.assertIsNone(info["ablation_delta_normalized"])
        self.assertEqual(reward, 0.0)
        # judge WAS configured -- omitting swapped_context for one turn is
        # not the "no judge at all" case and must not warn.
        self.assertEqual(len(caught), 0)

    def test_no_judge_warns_once_not_per_step(self):
        env = BanterEnv(partner_complete=fake_partner, judge=None,
                        seed_topic=0, max_turns=5)
        env.reset()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env.step("Sure, noted.")
            env.step("Still fine, thanks.")
        run_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(run_warnings), 1)


class TestBanterEnvRefusalAndErrors(unittest.TestCase):
    def test_refusal_terminates_episode(self):
        env = BanterEnv(partner_complete=fake_partner, seed_topic=0, max_turns=5)
        env.reset()
        next_prompt, reward, done, info = env.step(
            "I'm sorry, but I can't joke about that.")
        self.assertIsNone(next_prompt)
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertTrue(info["refusal"])

    def test_step_before_reset_raises(self):
        env = BanterEnv(partner_complete=fake_partner)
        with self.assertRaises(RuntimeError):
            env.step("hello")

    def test_step_after_done_raises(self):
        env = BanterEnv(partner_complete=fake_partner, max_turns=1)
        env.reset()
        _, _, done, _ = env.step("Sure, noted.")
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            env.step("one more")

    def test_max_turns_terminates_with_no_next_prompt(self):
        env = BanterEnv(partner_complete=fake_partner, max_turns=2)
        env.reset()
        next_prompt_1, _, done_1, _ = env.step("Sure, noted.")
        self.assertFalse(done_1)
        self.assertIsInstance(next_prompt_1, str)
        next_prompt_2, _, done_2, _ = env.step("Okay, noted again.")
        self.assertTrue(done_2)
        self.assertIsNone(next_prompt_2)


if __name__ == "__main__":
    unittest.main()

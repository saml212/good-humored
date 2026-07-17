"""Unit tests for env/cascade_env.py -- fake labeler callables only,
NO network or CLI calls.
Run: python3 -m unittest discover -s env/tests -v
"""

import re
import unittest

from benchmark.rejector import UNPARSEABLE
from env.cascade_env import CascadeEnv


def fake_labeler(prompt):
    """Extracts the joke text label_topic embedded between <joke> tags and
    returns its first word as the "topic" -- a deterministic stand-in for
    a real labeling model, entirely driven by the joke text CascadeEnv is
    fed (so a test can control the topic path just by choosing jokes).

    Takes the LAST <joke>...</joke> match, not the first: LABEL_PROMPT's
    instructions literally contain the string "<joke>" before the
    few-shot examples, and the few-shot examples themselves are each
    wrapped in their own <joke></joke> pair -- the actual joke under test
    is always the FINAL <joke>...</joke> occurrence in the formatted
    prompt."""
    matches = re.findall(r"<joke>(.*?)</joke>", prompt, re.DOTALL)
    joke_text = (matches[-1] if matches else prompt).strip()
    words = joke_text.split()
    return words[0] if words else ""


def fake_labeler_always_unparseable(prompt):
    """Always returns a >4-word 'label' so label_topic exhausts its two
    attempts and falls back to the UNPARSEABLE sentinel."""
    return "this label has way too many words in it"


class TestCascadeEnvEpisode(unittest.TestCase):
    def test_full_episode_new_then_repeat(self):
        env = CascadeEnv(labeler=fake_labeler, max_turns=10)
        prompt = env.reset()
        self.assertIsInstance(prompt, str)

        jokes = ["cat wants to be pet", "dog runs in the park",
                "parrot talks a lot", "cat again repeats the topic"]
        expected_rewards = [1.0, 1.0, 1.0, 0.0]
        expected_topics = ["cat", "dog", "parrot", "cat"]

        for joke, exp_reward, exp_topic in zip(jokes, expected_rewards,
                                                expected_topics):
            next_prompt, reward, done, info = env.step(joke)
            self.assertEqual(reward, exp_reward)
            self.assertEqual(info["topic"], exp_topic)
            self.assertFalse(info["refusal"])
            self.assertFalse(done)
            self.assertIsInstance(next_prompt, str)

        self.assertEqual(env.path, ["cat", "dog", "parrot", "cat"])

    def test_refusal_terminates_without_needing_a_labeler(self):
        # No labeler injected at all -- a refusal must never need one.
        env = CascadeEnv(labeler=None, max_turns=10)
        env.reset()
        next_prompt, reward, done, info = env.step(
            "I can't think of any more jokes right now.")
        self.assertIsNone(next_prompt)
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertTrue(info["refusal"])

    def test_missing_labeler_raises_on_non_refusal_turn(self):
        env = CascadeEnv(labeler=None, max_turns=10)
        env.reset()
        with self.assertRaises(RuntimeError):
            env.step("a perfectly ordinary joke about cats")

    def test_step_before_reset_raises(self):
        env = CascadeEnv(labeler=fake_labeler, max_turns=10)
        with self.assertRaises(RuntimeError):
            env.step("cat joke")

    def test_step_after_done_raises(self):
        env = CascadeEnv(labeler=fake_labeler, max_turns=1)
        env.reset()
        _, _, done, _ = env.step("cat joke")
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            env.step("dog joke")

    def test_max_turns_terminates_with_no_next_prompt(self):
        env = CascadeEnv(labeler=fake_labeler, max_turns=2)
        env.reset()
        next_prompt_1, _, done_1, _ = env.step("cat joke")
        self.assertFalse(done_1)
        self.assertIsInstance(next_prompt_1, str)
        next_prompt_2, _, done_2, _ = env.step("dog joke")
        self.assertTrue(done_2)
        self.assertIsNone(next_prompt_2)

    def test_unparseable_label_never_counts_as_new_even_on_first_turn(self):
        env = CascadeEnv(labeler=fake_labeler_always_unparseable, max_turns=10)
        env.reset()
        _, reward, done, info = env.step("literally anything")
        self.assertEqual(info["topic"], UNPARSEABLE)
        self.assertEqual(reward, 0.0)  # NOT 1.0 -- the anti-exploit guard
        self.assertFalse(done)

    def test_reset_clears_prior_episode_state(self):
        env = CascadeEnv(labeler=fake_labeler, max_turns=10)
        env.reset()
        env.step("cat joke")
        env.step("dog joke")
        env.reset()
        self.assertEqual(env.path, [])
        self.assertEqual(env.turn, 0)
        self.assertFalse(env.done)


if __name__ == "__main__":
    unittest.main()

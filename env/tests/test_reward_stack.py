"""Unit tests for the GRPO reward stack arithmetic.

The gate is stubbed (BandGate is certified separately); these tests
pin the STACK's behavior: multiplicative structure, hard screens,
self-rep crush, taste normalization, and the offline (no-reaction)
path that verl smoke tests will use.
"""

import unittest

from env.reward_stack import session_reward, screens_pass, REACTION_FLOOR


class StubGate:
    def __init__(self, floor_pass=True):
        self.floor_pass = floor_pass

    def score(self, context, text):
        return {"floor_pass": self.floor_pass, "anchor_sim": 0.5,
                "context_overlap": 0.1}


def make_session(policy_texts, partner_text="let's sort the shelf"):
    turns = []
    for i, p in enumerate(policy_texts):
        turns.append({"role": "partner", "turn": i, "text": partner_text,
                      "provocation": None})
        turns.append({"role": "policy", "turn": i, "text": p,
                      "provocation": None})
    return {"session_id": 1, "task": "t", "turns": turns}


class TestRewardStack(unittest.TestCase):
    def test_clean_session_offline(self):
        s = make_session(["a witty reply here", "another fresh line entirely"])
        r = session_reward(s, StubGate(), reaction_fn=None)
        self.assertEqual(r["n_policy_turns"], 2)
        self.assertEqual(r["taste"], 0.0)          # offline path
        self.assertEqual(r["floor_rate"], 1.0)
        self.assertGreater(r["total"], 0.9)        # low self-rep, no taste

    def test_floor_failure_zeroes(self):
        s = make_session(["anything"])
        r = session_reward(s, StubGate(floor_pass=False))
        self.assertEqual(r["total"], 0.0)

    def test_parrot_crushed(self):
        s = make_session(["the same exact words again",
                          "the same exact words again"])
        r = session_reward(s, StubGate())
        self.assertEqual(r["max_self_rep"], 1.0)
        self.assertEqual(r["total"], 0.0)          # (1 - 1.0) factor

    def test_cjk_screen_zeroes_total(self):
        s = make_session(["fine reply", "without惦记ing this"])
        r = session_reward(s, StubGate())
        self.assertLess(r["screen_rate"], 1.0)
        self.assertEqual(r["total"], 0.0)          # hard product defect

    def test_asterisk_screen_zeroes_total(self):
        s = make_session(["*grabs the tape and laughs*"])
        r = session_reward(s, StubGate())
        self.assertEqual(r["total"], 0.0)

    def test_taste_normalization(self):
        s = make_session(["reply one", "distinct reply two"])
        # audience at floor -> taste 0; ceiling (L=0) -> taste 1
        r_floor = session_reward(s, StubGate(),
                                 reaction_fn=lambda m: REACTION_FLOOR)
        r_ceil = session_reward(s, StubGate(), reaction_fn=lambda m: 0.0)
        self.assertEqual(r_floor["taste"], 0.0)
        self.assertEqual(r_ceil["taste"], 1.0)
        self.assertGreater(r_ceil["total"], r_floor["total"])
        # modest weight: ceiling taste multiplies total by exactly 1.5
        self.assertAlmostEqual(r_ceil["total"] / r_floor["total"], 1.5,
                               places=3)

    def test_empty_session(self):
        r = session_reward({"turns": []}, StubGate())
        self.assertEqual(r["total"], 0.0)

    def test_screens_pass_plain(self):
        self.assertTrue(screens_pass("a perfectly normal chat line"))


if __name__ == "__main__":
    unittest.main()


from env.reward_stack import smooth_session_reward


class TestSmoothObjective(unittest.TestCase):
    def test_no_cliff_on_screen(self):
        clean = make_session(["fine reply", "another distinct line"])
        dirty = make_session(["fine reply", "*grabs tape* okay"])
        rc = smooth_session_reward(clean, StubGate())
        rd = smooth_session_reward(dirty, StubGate())
        self.assertGreater(rc["total"], rd["total"])   # penalized...
        self.assertGreater(rd["total"], -0.5)          # ...not annihilated

    def test_additive_bounded(self):
        s = make_session(["a", "b", "c"])
        r = smooth_session_reward(s, StubGate(), reaction_fn=lambda m: 0.0)
        self.assertLessEqual(r["total"], 0.9 + 1e-9)   # max 0.4+0.2+0.3
        self.assertEqual(r["n_policy_turns"], 3)

    def test_empty(self):
        self.assertEqual(smooth_session_reward({"turns": []},
                                               StubGate())["total"], 0.0)

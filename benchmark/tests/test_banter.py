"""Unit tests for banter.py (Track 2) — fake model/partner/judge
callables only, no network or CLI calls. Run:
  python3 -m unittest discover benchmark/tests -v
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from benchmark.banter import (SCENARIO_OPENERS, context_ablation_score,
                              detect_callback, run_banter_episode,
                              score_episode, swap_partner)

# ------------------------------------------------------------- fakes


def fake_partner(prompt):
    """Echoes the scenario blurb it was asked to remark on — deterministic
    and cheap, no need to actually sound natural for these tests."""
    m = re.search(r"Situation: (.+)", prompt)
    situation = m.group(1) if m else "something"
    return "So, %s." % situation


def fake_model_echo(prompt):
    """Encodes the exact Friend line it just saw into its reply, so a
    fake judge can mechanically check whether a reply's embedded
    reference matches the context it is being scored against."""
    friend_lines = [l for l in prompt.splitlines() if l.startswith("Friend:")]
    last = friend_lines[-1] if friend_lines else "Friend: nothing"
    return "Oh totally, re[%s]" % last


def fake_judge_context_aware(prompt):
    """Scores 9 if the reply's embedded reference matches the context's
    LAST Friend line, else 2 — makes the ablation delta mechanical and
    predictable for a fake_model_echo reply."""
    context_part, _, reply_part = prompt.partition("Reply to score:")
    ctx_friend_lines = [l for l in context_part.splitlines()
                        if l.startswith("Friend:")]
    last_friend = ctx_friend_lines[-1] if ctx_friend_lines else ""
    if last_friend and ("[%s]" % last_friend) in reply_part:
        return "9"
    return "2"


# ------------------------------------------------------- episode running


class TestRunBanterEpisode(unittest.TestCase):
    def test_structure_and_scenario_rotation(self):
        ep = run_banter_episode(fake_model_echo, fake_partner,
                                n_turns=3, seed_topic=1)
        self.assertEqual(ep["n_turns"], 3)
        self.assertEqual(ep["seed_topic"], 1)
        self.assertEqual(len(ep["exchanges"]), 3)
        self.assertEqual(len(ep["messages"]), 6)  # partner+model per turn

        expected_scenarios = [
            SCENARIO_OPENERS[(1 + i) % len(SCENARIO_OPENERS)]
            for i in range(3)]
        self.assertEqual([e["scenario"] for e in ep["exchanges"]],
                         expected_scenarios)

        # strict partner/model alternation, partner first
        roles = [m["role"] for m in ep["messages"]]
        self.assertEqual(roles, ["partner", "model"] * 3)

    def test_each_exchange_has_required_fields(self):
        ep = run_banter_episode(fake_model_echo, fake_partner,
                                n_turns=2, seed_topic=0)
        for exch in ep["exchanges"]:
            self.assertIn("turn", exch)
            self.assertIn("scenario", exch)
            self.assertIn("partner", exch)
            self.assertIn("reply", exch)
            self.assertIn("ts", exch)

    def test_log_written_matches_exchanges(self):
        d = tempfile.mkdtemp(prefix="banter-test-")
        log_path = Path(d) / "turns.jsonl"
        ep = run_banter_episode(fake_model_echo, fake_partner,
                                n_turns=2, seed_topic=0, log_path=log_path)
        with open(log_path) as f:
            logged = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(logged), 2)
        self.assertEqual([r["scenario"] for r in logged],
                         [e["scenario"] for e in ep["exchanges"]])
        self.assertEqual([r["reply"] for r in logged],
                         [e["reply"] for e in ep["exchanges"]])


class TestSwapPartner(unittest.TestCase):
    def test_deterministic_rotation(self):
        self.assertEqual(swap_partner(0, 3), 1)
        self.assertEqual(swap_partner(1, 3), 2)
        self.assertEqual(swap_partner(2, 3), 0)  # wraps

    def test_requires_at_least_two_episodes(self):
        with self.assertRaises(ValueError):
            swap_partner(0, 1)


# --------------------------------------------------- context ablation


class TestContextAblationScore(unittest.TestCase):
    def test_delta_reflects_context_marker(self):
        def judge(prompt):
            if "TRUE_MARK" in prompt:
                return "8 - fits great"
            if "SWAP_MARK" in prompt:
                return "3 - odd fit here"
            return "5"

        out = context_ablation_score(
            "some reply", "TRUE_MARK conversation context",
            "SWAP_MARK conversation context", judge)
        self.assertEqual(out["true_score"], 8)
        self.assertEqual(out["swapped_score"], 3)
        self.assertEqual(out["delta"], 5)
        self.assertIn("judge_prompt_version", out)

    def test_canned_joke_zero_delta(self):
        # A context-blind judge (or a canned joke that reads the same
        # either way) should show delta ~ 0.
        def flat_judge(_prompt):
            return "6"

        out = context_ablation_score("reply", "context A", "context B",
                                     flat_judge)
        self.assertEqual(out["delta"], 0)

    def test_parse_failure_sentinel(self):
        def bad_judge(_prompt):
            return "I refuse to give a number."

        out = context_ablation_score("reply", "ctx a", "ctx b", bad_judge)
        self.assertIsNone(out["true_score"])
        self.assertIsNone(out["swapped_score"])
        self.assertIsNone(out["delta"])

    def test_retry_recovers_from_one_bad_parse(self):
        calls = {"n": 0}

        def flaky_then_ok(prompt):
            calls["n"] += 1
            if "TRUE_MARK" in prompt:
                # fails to parse on first call, succeeds on the retry
                return "no score today" if calls["n"] == 1 else "9"
            return "4"  # swapped context: parses immediately

        out = context_ablation_score("reply", "TRUE_MARK ctx", "SWAP ctx",
                                     flaky_then_ok)
        self.assertEqual(out["true_score"], 9)
        self.assertEqual(out["swapped_score"], 4)
        self.assertEqual(out["delta"], 5)
        self.assertEqual(calls["n"], 3)  # 2 for true (fail+retry) + 1 swapped

    def test_hundred_not_misparsed_as_ten(self):
        def judge(_prompt):
            return "100"

        out = context_ablation_score("reply", "ctx a", "ctx b", judge)
        # "100" must not be misread as "10" (or "1"); both calls fail to
        # parse a valid 1-10 score, twice each (retry), sentinel None.
        self.assertIsNone(out["true_score"])
        self.assertIsNone(out["swapped_score"])


# -------------------------------------------------------- callbacks


class TestDetectCallback(unittest.TestCase):
    def test_positive_callback_beyond_gap(self):
        earlier = [
            "My neighbor's dog escaped again today.",          # old
            "Coffee tasted burnt again this morning.",         # old
            "The weather has been miserable lately.",          # intervening
            "Traffic was brutal on the highway this morning.",  # intervening
        ]
        reply = "Anyway, hope that neighbor situation got sorted out."
        self.assertEqual(detect_callback(reply, earlier, min_gap=3),
                         "neighbor")

    def test_no_callback_when_word_too_recent(self):
        # 'neighbor' only appears inside the intervening window (2 turns
        # before reply), not far enough back to count under min_gap=3.
        earlier = [
            "Something else entirely happened.",   # old
            "My neighbor's dog escaped today.",      # intervening
            "The weather has been miserable.",       # intervening
        ]
        reply = "Anyway, hope the neighbor thing worked out."
        self.assertIsNone(detect_callback(reply, earlier, min_gap=3))

    def test_no_callback_when_refreshed_by_intervening_turn(self):
        # 'neighbor' IS old enough, but it was also re-mentioned in an
        # intervening turn — that's topical continuity, not a callback.
        earlier = [
            "My neighbor's dog escaped again today.",           # old
            "Coffee tasted burnt again this morning.",          # old
            "Speaking of which, my neighbor called about it.",  # intervening
            "Traffic was brutal on the highway this morning.",  # intervening
        ]
        reply = "Anyway, hope that neighbor situation got sorted out."
        self.assertIsNone(detect_callback(reply, earlier, min_gap=3))

    def test_no_callback_when_nothing_shared(self):
        earlier = ["Coffee this morning was fine.", "Traffic was awful.",
                  "Weather report was boring."]
        reply = "Totally unrelated statement about nothing in particular."
        self.assertIsNone(detect_callback(reply, earlier, min_gap=3))

    def test_short_words_excluded(self):
        # 'cat'/'dog' are < 5 chars and must not register even though
        # they literally repeat — the crudeness is length-gated on purpose.
        earlier = [
            "I saw a cat and a dog today.",
            "Something else entirely happened.",
            "Weather report was boring.",
            "Traffic again this morning.",
        ]
        reply = "That cat and dog moment was hilarious."
        self.assertIsNone(detect_callback(reply, earlier, min_gap=3))

    def test_not_enough_history_returns_none(self):
        self.assertIsNone(detect_callback("reply", ["one turn only"],
                                          min_gap=3))


# --------------------------------------------------------- score_episode


class TestScoreEpisode(unittest.TestCase):
    def test_integration_summary_shape(self):
        ep_a = run_banter_episode(fake_model_echo, fake_partner,
                                  n_turns=3, seed_topic=0)
        ep_b = run_banter_episode(fake_model_echo, fake_partner,
                                  n_turns=3, seed_topic=4)
        out = score_episode(ep_a, ep_b, fake_judge_context_aware, min_gap=1)

        self.assertEqual(out["summary"]["n_turns"], 3)
        self.assertEqual(len(out["per_turn"]), 3)
        self.assertEqual(out["summary"]["n_unparseable"], 0)
        # fake_model_echo always encodes the TRUE friend line, so scoring
        # against the true context should mechanically beat the swapped
        # one (different seed_topic => different scenario at every turn).
        self.assertEqual(out["summary"]["mean_delta"], 7.0)
        for turn in out["per_turn"]:
            self.assertEqual(turn["delta"], 7)

    def test_swap_episode_shorter_clamps_instead_of_erroring(self):
        ep_a = run_banter_episode(fake_model_echo, fake_partner,
                                  n_turns=3, seed_topic=0)
        ep_b = run_banter_episode(fake_model_echo, fake_partner,
                                  n_turns=1, seed_topic=4)
        out = score_episode(ep_a, ep_b, fake_judge_context_aware)
        self.assertEqual(len(out["per_turn"]), 3)  # no crash on the short pair

    def test_callback_count_surfaces_in_summary(self):
        def partner_script(prompt):
            # ignore the scripted scenario; force a specific word to
            # reappear so the callback fires deterministically.
            if "Conversation so far" not in prompt:
                return "My neighbor's fence collapsed overnight."
            return "Anyway, unrelated update about the weather."

        def model_callback_reply(prompt):
            n_friend = len([l for l in prompt.splitlines()
                           if l.startswith("Friend:")])
            if n_friend >= 4:
                return "Still thinking about that neighbor situation."
            return "Sure, noted."

        ep = run_banter_episode(model_callback_reply, partner_script,
                                n_turns=4, seed_topic=0)
        out = score_episode(ep, ep, lambda _p: "5", min_gap=2)
        self.assertGreaterEqual(out["summary"]["n_callbacks"], 1)


if __name__ == "__main__":
    unittest.main()

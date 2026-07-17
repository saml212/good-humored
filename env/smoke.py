#!/usr/bin/env python3
"""Zero-dependency smoke test for env/ (CLAUDE.md hard rule: smoke test
every model/environment before training). Pure stdlib, no network, no
CLI calls, no GPU -- this is the "does the wiring work at all" check to
run BEFORE ever pointing this package at a real model.

Walks:
  1. the reward stack (env/rewards.py) against a tiny fake GRPO group,
     printing every term's reward plus the combined sum;
  2. one full CascadeEnv (Track 1) episode against a fake labeler;
  3. one full BanterEnv (Track 2) episode against a fake partner + judge;

and prints per-term/per-turn rewards for each, so a human can eyeball
that the numbers look like what the docstrings promise before spending
any GPU time.

Run: python3 env/smoke.py
"""

import re
import sys
import warnings
from pathlib import Path

# Make the repo root importable regardless of how this script is invoked
# (`python3 env/smoke.py` from repo root, or from anywhere else).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.banter import SCENARIO_OPENERS  # noqa: E402
from benchmark.rejector import UNPARSEABLE  # noqa: E402
from env.banter_env import BanterEnv  # noqa: E402
from env.cascade_env import CascadeEnv  # noqa: E402
from env.rewards import RewardConfig, combined_reward, reward_stack  # noqa: E402

FIXTURE_CORPUS_DIR = _REPO_ROOT / "env" / "tests" / "fixtures" / "corpus"


def _section(title: str) -> None:
    print("\n=== %s ===" % title)


# --------------------------------------------------------- reward stack


def fake_judge(prompt, completion: str) -> float:
    """A trivially cheap stand-in judge: rewards completions ending in a
    question mark slightly more, everything else a flat mid-range score.
    Real training must plug in an actual preference/judge model here --
    see env/rewards.py's JudgePreferenceReward docstring."""
    return 0.7 if completion.strip().endswith("?") else 0.5


def smoke_reward_stack() -> None:
    _section("reward stack (env/rewards.py)")
    config = RewardConfig(
        judge=fake_judge,
        joke_corpus_dir=FIXTURE_CORPUS_DIR,
        group_size=2,
    )
    completions = [
        "Why did the bicycle fall over? It was two-tired!",
        "My houseplant filed a complaint about my parenting skills.",
    ]
    funcs = reward_stack(config)
    per_term = {}
    for f in funcs:
        vals = f(prompts=[None, None], completions=completions)
        per_term[f.__name__] = vals
        print("  %-24s %s" % (f.__name__, ["%.4f" % v for v in vals]))

    combined = combined_reward(config)
    totals = combined(prompts=[None, None], completions=completions)
    print("  %-24s %s" % ("combined_reward", ["%.4f" % v for v in totals]))

    expected_totals = [sum(per_term[name][i] for name in per_term)
                      for i in range(len(completions))]
    assert all(abs(a - b) < 1e-9 for a, b in zip(totals, expected_totals)), (
        "combined_reward did not equal the sum of its own per-term outputs")
    print("  OK: combined_reward == sum(per-term rewards)")


# ---------------------------------------------------------- cascade env


def fake_labeler(prompt: str) -> str:
    """Extracts the joke text from label_topic's <joke> wrapper and
    returns its first word as the topic -- see
    env/tests/test_cascade_env.py for the same pattern under test. Takes
    the LAST <joke>...</joke> match (the few-shot examples in
    LABEL_PROMPT are each wrapped in their own <joke></joke> pair too;
    the joke under test is always the final occurrence)."""
    matches = re.findall(r"<joke>(.*?)</joke>", prompt, re.DOTALL)
    joke_text = (matches[-1] if matches else prompt).strip()
    words = joke_text.split()
    return words[0] if words else ""


def smoke_cascade_env() -> None:
    _section("CascadeEnv (Track 1) -- one fake episode")
    env = CascadeEnv(labeler=fake_labeler, max_turns=4)
    prompt = env.reset()
    print("  opening prompt: %r" % prompt)

    fake_jokes = [
        "cat jokes are the best kind of joke honestly",
        "dog jokes come in a close second place",
        "cat jokes again because the model got stuck",
        "parrot jokes are underrated as a genre",
    ]
    total_reward = 0.0
    for joke in fake_jokes:
        prompt, reward, done, info = env.step(joke)
        total_reward += reward
        print("  turn %d: topic=%-8s reward=%.1f refusal=%s done=%s"
             % (info["turn"], info["topic"], reward, info["refusal"], done))
        if done:
            break
    print("  path: %s" % env.path)
    print("  total reward: %.1f" % total_reward)
    assert env.path[0] == "cat" and env.path[2] == "cat", "fixture jokes changed?"


# ----------------------------------------------------------- banter env


def fake_partner(prompt: str) -> str:
    m = re.search(r"Situation: (.+)", prompt)
    situation = m.group(1) if m else "something"
    return "So, %s." % situation


def fake_banter_judge(prompt: str) -> str:
    """Same mechanical context-aware fake as
    env/tests/test_banter_env.py: scores 9 if the reply's embedded
    marker matches the shown context's last Friend line, else 2."""
    context_part, _, reply_part = prompt.partition("Reply to score:")
    ctx_friend_lines = [l for l in context_part.splitlines()
                        if l.startswith("Friend:")]
    last_friend = ctx_friend_lines[-1] if ctx_friend_lines else ""
    if last_friend and ("[%s]" % last_friend) in reply_part:
        return "9"
    return "2"


def smoke_banter_env() -> None:
    _section("BanterEnv (Track 2) -- one fake episode")
    env = BanterEnv(partner_complete=fake_partner, judge=fake_banter_judge,
                    seed_topic=0, max_turns=4, min_gap=2)
    prompt = env.reset()
    print("  opening partner line: %r" % prompt)

    total_reward = 0.0
    for turn in range(4):
        # A cheap fake policy: echo the partner's line back so the fake
        # judge can mechanically detect true-context vs swapped-context.
        reply = "Oh totally, re[Friend: %s]" % prompt
        # No sibling rollout to pull a real swap from in this smoke
        # script -- pass a deliberately unrelated swapped context so the
        # ablation delta has something nonzero to show.
        swapped_context = "Friend: a completely unrelated thing happened."
        prompt, reward, done, info = env.step(reply, swapped_context=swapped_context)
        total_reward += reward
        print("  turn %d: callback=%s ablation_delta=%s reward=%.2f done=%s"
             % (info["turn"], info["callback"], info["ablation_delta"],
                reward, done))
        if done:
            break
    print("  total reward: %.2f" % total_reward)


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # this smoke run intentionally
                                         # exercises the pluggable-default
                                         # warnings elsewhere; keep this
                                         # script's own output readable.
        smoke_reward_stack()
        smoke_cascade_env()
        smoke_banter_env()
    print("\nSMOKE TEST PASSED -- env/ wiring is sound end to end.")


if __name__ == "__main__":
    main()

"""Track 1 (the rejection cascade, docs/BENCHMARK.md §1) as a trainable
RL environment.

`benchmark/cascade.py`'s `run_cascade()` drives a whole episode in one
call against two `complete(prompt) -> str` callables -- the right shape
for a benchmark sweep, but not for a GRPO/PPO-style training loop, which
needs a `reset()`/`step()` episode boundary that lines up with the
policy's own per-turn generation call. `CascadeEnv` is that step-wise
wrapper around the SAME rejector/labeling machinery
(`benchmark/rejector.py`, `benchmark/metrics.py`) -- nothing is
duplicated, only reshaped into an env API. `env/rewards.py`'s reward
stack (in particular `judge_preference`, "is it funny") is a separate,
orthogonal concern layered on top by whatever trains against this env;
this environment's own reward is Track 1's native diagnostic signal --
topic-path novelty -- not joke quality.

Per-turn reward (deliberately simple, matching what Track 1 actually
measures -- docs/BENCHMARK.md is explicit that "the jokes are not the
measurement, the trajectory is"):

  +1.0  the turn's topic (raw `benchmark.metrics.normalize_label` string,
        via `benchmark.rejector.label_topic` -- NOT the semantic
        `LabelSpace` canonicalization in `benchmark/label_space.py`; raw
        matching is this project's own "primary" scoring convention,
        `docs/TRANSFER-PLAN.md` §5.3) has not appeared earlier in this
        episode's path.
   0.0  a repeat of an already-seen topic.
   0.0  the label came back as `benchmark.rejector.UNPARSEABLE`. This is
        a deliberate addition beyond the literal per-turn spec ("+1 new,
        0 repeat"): an unparseable label is not a real topic, and without
        this guard a model could win a free +1.0 on the FIRST turn its
        joke fails to label, every episode, at zero cost -- a
        reward-hacking vector structurally identical to the ones
        CLAUDE.md already documents for this project, just cheaper to
        find. Treating the sentinel as "always a repeat of itself"
        removes the exploit without needing a new turn-terminate rule.
   0.0  refusal (`benchmark.metrics.looks_like_refusal` on the joke text
        itself, checked BEFORE any labeler call) -- the episode
        terminates immediately; the labeler is never invoked for a turn
        that's already a refusal, so `CascadeEnv` can legitimately run
        pure-refusal episodes end-to-end with no labeler ever configured
        (an edge case, not a recommended mode).

Deterministic and network-free by construction. The ONLY external call
this environment can make is through the injected `labeler` callable,
which the caller supplies completely -- there is NO default labeler.
`CascadeEnv` must never silently fall back to a CLI or API call (README.md
/ CLAUDE.md: no hidden network calls anywhere in this package); calling
`step()` on a real (non-refusal) turn before a labeler is injected raises
immediately, with an actionable message, rather than trying anything
else. No wall-clock time or randomness appears anywhere in the turn logic
-- `reset(seed=...)` accepts a seed purely for gym-style API parity (see
its docstring); nothing in this environment actually consults it, because
the topic path is a pure function of the labeler's outputs and the
policy's jokes.
"""

from typing import Callable, Dict, List, Optional, Tuple

from benchmark.metrics import looks_like_refusal
from benchmark.rejector import (OPENING_PROMPT, UNPARSEABLE, label_topic,
                                rejection_message)

# (next_prompt, reward, done, info) -- next_prompt is None once done.
StepResult = Tuple[Optional[str], float, bool, Dict]


class CascadeEnv:
    """One rejection-cascade episode, step-wise.

    MUTABLE, ONE EPISODE AT A TIME: this instance holds per-episode state
    (`path`, `turn`) -- use one `CascadeEnv` per sequential episode, never
    share a single instance across concurrent rollouts in a GRPO group
    (that would interleave unrelated episodes' topics into one `path`);
    create one instance per rollout instead.

    Args:
        labeler: `complete(prompt) -> str` callable passed straight
            through to `benchmark.rejector.label_topic`. MUST be injected
            before any non-refusal `step()` call; there is no default
            (see module docstring). In training this is typically the
            same cheap judge/labeler model the rest of the run uses; in
            tests, a fake.
        max_turns: episode length cap. `docs/BENCHMARK.md` runs the
            benchmark cascade at ~50 turns; that's this default too.

    Usage:
        env = CascadeEnv(labeler=my_complete_fn, max_turns=50)
        prompt = env.reset()
        while True:
            joke = policy(prompt)
            prompt, reward, done, info = env.step(joke)
            if done:
                break
    """

    def __init__(self, labeler: Optional[Callable[[str], str]] = None,
                max_turns: int = 50):
        self.labeler = labeler
        self.max_turns = max_turns
        self.path: List[str] = []
        self.turn = 0
        self.done = True  # must reset() before the first step()
        self._seed: Optional[int] = None

    def reset(self, seed: Optional[int] = None) -> str:
        """Start a new episode. Returns the opening prompt
        (`benchmark.rejector.OPENING_PROMPT`).

        `seed` is accepted and stored for gym-style API parity only --
        nothing in this environment is stochastic (the topic path is a
        deterministic function of the labeler's outputs), so it is never
        actually consulted. This is documented behavior, not a
        placeholder bug.
        """
        self._seed = seed
        self.path = []
        self.turn = 0
        self.done = False
        return OPENING_PROMPT

    def step(self, joke_text: str) -> StepResult:
        """Advance one turn given the policy's joke for the current
        prompt. Returns `(next_prompt, reward, done, info)`;
        `next_prompt` is `None` once `done` is `True` (nothing left to
        say after a refusal or the final turn).
        """
        if self.done:
            raise RuntimeError(
                "CascadeEnv.step() called after the episode finished -- "
                "call reset() to start a new episode.")

        if looks_like_refusal(joke_text):
            self.done = True
            info = {"turn": self.turn, "topic": None, "refusal": True,
                    "path": list(self.path)}
            return None, 0.0, True, info

        if self.labeler is None:
            raise RuntimeError(
                "CascadeEnv requires a labeler callable injected at "
                "construction (CascadeEnv(labeler=...)) to score a "
                "non-refusal turn; it never falls back to a CLI or "
                "network call. Inject the training-time labeler (e.g. the "
                "same judge model used elsewhere in the run, or a fake in "
                "tests).")

        topic = label_topic(joke_text, self.labeler)
        is_new = topic != UNPARSEABLE and topic not in self.path
        reward = 1.0 if is_new else 0.0

        prior_path = list(self.path)
        self.path.append(topic)
        self.turn += 1
        self.done = self.turn >= self.max_turns

        next_prompt = (None if self.done
                      else rejection_message(topic, prior_path))
        info = {"turn": self.turn - 1, "topic": topic, "refusal": False,
               "path": list(self.path), "is_new_topic": is_new}
        return next_prompt, reward, self.done, info

"""Track 2 (contextual banter, docs/BENCHMARK.md §1b) as a trainable RL
environment.

`benchmark/banter.py`'s `run_banter_episode()` drives a whole episode
against two `complete()` callables and scores it only AFTERWARD with
`score_episode()`; `BanterEnv` reshapes the SAME machinery
(`SCENARIO_OPENERS`, `detect_callback`, `context_ablation_score`, and the
private prompt-formatting helpers `_partner_prompt`/`_format_turns`) into
a step-wise `reset()`/`step()` episode boundary, mirroring
`cascade_env.CascadeEnv`, so a GRPO/PPO-style training loop can score (and
feed) the policy's reply turn by turn instead of only at the end of a
whole logged episode.

Deliberate reuse-over-duplication choice, worth flagging explicitly:
this module imports `benchmark.banter`'s underscore-prefixed
`_partner_prompt` and `_format_turns`. Python does not enforce privacy on
these -- they're conventionally internal, not actually hidden -- and
reimplementing them here would risk a silent format drift between this
env's prompts and `benchmark/banter.py`'s, which would quietly break the
swap-pairing/context-ablation design's parity (the whole point of
`context_ablation_score` is that both context strings are built by the
IDENTICAL formatting function). Importing them is the correct reuse
decision, not a shortcut.

Reward composition per turn -- two pluggable terms, both default to a
harmless zero + one-time warning rather than ever making a hidden network
call:

  callback bonus (`callback_weight`, default +0.5) -- flat bonus when
      `benchmark.banter.detect_callback()` fires on the reply against the
      partner's turns so far. +0.5 is chosen to sit in the same order of
      magnitude as `env/rewards.py`'s `intra_group_diversity` weight -- a
      structural nudge, not a dominant signal. IMPORTANT: this weight is
      NOT specified anywhere in README.md's reward table (that table
      covers the main single-turn GRPO stack, not Track 2's banter
      reward) -- it is this module's own default, flagged here explicitly
      rather than silently picked, the same way `docs/TRANSFER-PLAN.md`
      §4.2 flags the main stack's judge-weight gap.

  context-ablation delta (`ablation_weight`, default +1.0) --
      `docs/BENCHMARK.md` §1b's central design: the reward is the DELTA
      between two judge calls under an IDENTICAL rubric
      (`benchmark.banter.JUDGE_PROMPT`), differing only in which
      conversation context is shown -- NEVER an absolute judge score.
      That is preserved exactly here: `BanterEnv` never exposes or
      rewards a raw judge score, only `context_ablation_score()`'s delta.
      Residual risk, carried over verbatim from the design doc rather
      than silently dropped: subtraction cancels a judge's constant
      scale/leniency/length bias, but NOT a bias that is itself
      context-dependent -- e.g. a judge that scores any reply echoing a
      context keyword higher, independent of true responsiveness. A
      policy could learn to sprinkle context-echoing words into an
      otherwise context-blind reply and inflate delta without actually
      being in-context. This is the same judge-hacking family
      `.claude/skills/humor-rl/SKILL.md` documents for the main stack.

      MAGNITUDE FIX (adversarial-audit finding): `JUDGE_PROMPT` scores on
      a 1-10 scale, so a raw delta ranges over roughly ±9 -- next to
      `env/rewards.py`'s stack, whose terms live in ±0.3-1.5, an
      unnormalized ±9 delta would numerically dominate any composition
      of the two by 5-10x. That is EXACTLY the judge-dominance failure
      `env/rewards.py`'s `JudgePreferenceReward` hard-fails on elsewhere
      in this package (see its `[0, 1]` contract) -- it would be
      inconsistent to enforce that discipline in one module and ship the
      other module with the same bug on day one. `BanterEnv` is new
      tonight with no downstream callers yet, so the default here is
      NORMALIZED: the raw delta is divided by `_JUDGE_SCORE_RANGE` (9.0)
      to land in [-1, 1] before `ablation_weight` is applied, putting it
      in the same order of magnitude as the Track 1 stack's terms so a
      future composition of the two doesn't silently reproduce judge
      dominance. Pass `raw_ablation_delta=True` to opt back into the raw
      ±9-scaled delta (e.g. to match a specific published number). `info`
      always reports BOTH `ablation_delta` (the raw, human-interpretable
      -9..+9 delta from `context_ablation_score`) and
      `ablation_delta_normalized` (the same value / 9.0) regardless of
      which one the reward actually used, so a training log never loses
      the raw number even when the reward is normalized.

      NOTE the two different "judge" contracts in this package:
      `env/rewards.py`'s `judge(prompt, completion) -> float in [0, 1]`
      is a normalized scorer (a GRPO reward_func needs a bounded scalar
      directly). This module's `judge` is instead a raw text-completion
      callable, `judge_complete(prompt) -> str`, because it's reusing
      `benchmark.banter.context_ablation_score`'s existing 1-10
      rubric-parsing machinery UNMODIFIED. Do not pass one where the
      other is expected.

The swapped context the delta needs is NOT generated internally --
`BanterEnv` owns exactly one conversation, so it has no "other episode"
to pull a swap from. `step()` takes `swapped_context` as an explicit,
caller-supplied string argument; the natural source in a real training
loop is a sibling GRPO-group rollout's conversation at the same turn
index (the online analogue of `benchmark.banter.swap_partner`'s fixed
rotation across a pilot's fixed episode set). If `swapped_context` is
omitted for a turn, the ablation term simply contributes 0.0 for that
turn (skipped, not faked with a made-up context) -- the callback bonus is
computed regardless.
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple

from benchmark.banter import (SCENARIO_OPENERS, _format_turns,
                              _partner_prompt, context_ablation_score,
                              detect_callback)
from benchmark.metrics import looks_like_refusal

StepResult = Tuple[Optional[str], float, bool, Dict]

# JUDGE_PROMPT (benchmark/banter.py) scores 1-10, so a delta between two
# such scores ranges over roughly ±9. Dividing by this constant maps the
# default normalized delta into [-1, 1] -- see the module docstring's
# "MAGNITUDE FIX" note for why that matters.
_JUDGE_SCORE_RANGE = 9.0


class BanterEnv:
    """One contextual-banter episode, step-wise.

    MUTABLE, ONE EPISODE AT A TIME: this instance holds per-episode state
    (`messages`, `_partner_history`, `turn`) -- use one `BanterEnv` per
    sequential episode, never share a single instance across concurrent
    rollouts in a GRPO group (that would interleave unrelated
    conversations' turns into one `_partner_history`/`messages`); create
    one instance per rollout instead.

    Args:
        partner_complete: `complete(prompt) -> str` generating the
            partner's side of the conversation. REQUIRED at construction
            (unlike `judge` below) -- without it there is no conversation
            to `reset()` into at all; this is the interlocutor, not a
            grading dependency that can legitimately be absent.
        judge: `judge_complete(prompt) -> str` raw completion callable,
            reused as `benchmark.banter.context_ablation_score`'s scoring
            model. Optional; `None` (the default) means the
            context-ablation term is always skipped (warns once) while
            the callback bonus still works. NEVER a hidden network call.
        seed_topic: starting index into `SCENARIO_OPENERS`
            (`benchmark.banter`'s fixed, deterministic-order scenario
            list); `(seed_topic + turn) % len(SCENARIO_OPENERS)` picks
            each turn's scenario, exactly matching
            `run_banter_episode`'s rotation rule.
        max_turns: episode length cap. Defaults to
            `len(SCENARIO_OPENERS)` (8) so a default episode visits every
            scripted scenario once.
        callback_weight / ablation_weight: see module docstring.
        min_gap: passed straight through to `detect_callback`.
        raw_ablation_delta: if `True`, use the raw (±9-scaled) judge
            delta instead of the [-1, 1]-normalized default -- see module
            docstring's "MAGNITUDE FIX" note. Default `False`.

    Usage:
        env = BanterEnv(partner_complete=partner_fn, judge=judge_fn)
        prompt = env.reset()
        while True:
            reply = policy(prompt)
            # swapped_context from a sibling rollout, if you have one:
            prompt, reward, done, info = env.step(reply, swapped_context)
            if done:
                break
    """

    def __init__(
        self,
        partner_complete: Callable[[str], str],
        judge: Optional[Callable[[str], str]] = None,
        seed_topic: int = 0,
        max_turns: Optional[int] = None,
        callback_weight: float = 0.5,
        ablation_weight: float = 1.0,
        min_gap: int = 3,
        raw_ablation_delta: bool = False,
    ):
        if partner_complete is None:
            raise ValueError(
                "BanterEnv requires a partner_complete callable -- it "
                "never auto-calls a CLI or API to generate the other side "
                "of the conversation.")
        self.partner_complete = partner_complete
        self.judge = judge
        self.seed_topic = seed_topic
        self.max_turns = (max_turns if max_turns is not None
                          else len(SCENARIO_OPENERS))
        self.callback_weight = callback_weight
        self.ablation_weight = ablation_weight
        self.min_gap = min_gap
        self.raw_ablation_delta = raw_ablation_delta

        self.messages: List[Dict[str, str]] = []
        self._partner_history: List[str] = []
        self.turn = 0
        self.done = True  # must reset() before the first step()
        self._seed: Optional[int] = None
        self._warned_no_judge = False

    def _next_scenario(self) -> str:
        n = len(SCENARIO_OPENERS)
        return SCENARIO_OPENERS[(self.seed_topic + self.turn) % n]

    def reset(self, seed: Optional[int] = None) -> str:
        """Start a new episode: generate the partner's opening remark and
        return it as the first prompt for the policy to reply to.

        `seed` is accepted for gym-API parity only, same as
        `cascade_env.CascadeEnv.reset` -- this environment's own turn
        logic is deterministic; `partner_complete`/`judge` may or may not
        be, but that is entirely outside this environment's control.
        """
        self._seed = seed
        self.messages = []
        self._partner_history = []
        self.turn = 0
        self.done = False

        scenario = self._next_scenario()
        partner_line = self.partner_complete(
            _partner_prompt(scenario, self.messages))
        self.messages.append({"role": "partner", "content": partner_line})
        return partner_line

    def step(self, reply_text: str,
            swapped_context: Optional[str] = None) -> StepResult:
        """Advance one turn given the policy's reply to the current
        partner line. Returns `(next_prompt, reward, done, info)`.

        `swapped_context` is an optional preformatted conversation string
        (see module docstring for where a real training loop would source
        one) used ONLY for this turn's context-ablation delta; omit it to
        skip that term for the turn.
        """
        if self.done:
            raise RuntimeError(
                "BanterEnv.step() called after the episode finished -- "
                "call reset() to start a new episode.")

        self.messages.append({"role": "model", "content": reply_text})
        current_partner_line = self.messages[-2]["content"]

        if looks_like_refusal(reply_text):
            self.done = True
            self._partner_history.append(current_partner_line)
            info = {"turn": self.turn, "callback": None,
                   "ablation_delta": None, "ablation_delta_normalized": None,
                   "refusal": True,
                   "reward_breakdown": {"callback": 0.0, "ablation": 0.0}}
            return None, 0.0, True, info

        # Context up to and including the partner's line for this turn,
        # excluding the reply being scored -- matches
        # benchmark.banter._context_up_to's definition exactly.
        true_context = _format_turns(self.messages[:-1])

        callback = detect_callback(reply_text, self._partner_history,
                                   min_gap=self.min_gap)
        callback_term = self.callback_weight if callback is not None else 0.0

        ablation_delta = None
        ablation_delta_normalized = None
        ablation_term = 0.0
        if self.judge is None:
            if not self._warned_no_judge:
                warnings.warn(
                    "BanterEnv: no judge configured -- the context-"
                    "ablation delta term contributes 0.0 every turn "
                    "(callback bonus is unaffected). Pass judge=<a raw "
                    "completion callable> to score whether replies are "
                    "actually in-context rather than canned "
                    "(docs/BENCHMARK.md §1b). Never auto-calls a "
                    "network model. This warning fires once per BanterEnv "
                    "instance, not once per step().",
                    RuntimeWarning, stacklevel=2)
                self._warned_no_judge = True
        elif swapped_context is None:
            pass  # no swap partner supplied this turn -- skip, don't fake one
        else:
            ablation = context_ablation_score(
                reply_text, true_context, swapped_context, self.judge)
            if ablation["delta"] is not None:
                ablation_delta = ablation["delta"]
                ablation_delta_normalized = ablation_delta / _JUDGE_SCORE_RANGE
                used_delta = (ablation_delta if self.raw_ablation_delta
                             else ablation_delta_normalized)
                ablation_term = self.ablation_weight * used_delta

        reward = callback_term + ablation_term

        self._partner_history.append(current_partner_line)
        self.turn += 1
        self.done = self.turn >= self.max_turns

        info = {
            "turn": self.turn - 1, "callback": callback,
            "ablation_delta": ablation_delta,
            "ablation_delta_normalized": ablation_delta_normalized,
            "refusal": False,
            "reward_breakdown": {"callback": callback_term,
                                 "ablation": ablation_term},
        }

        next_prompt = None
        if not self.done:
            scenario = self._next_scenario()
            partner_line = self.partner_complete(
                _partner_prompt(scenario, self.messages))
            self.messages.append({"role": "partner", "content": partner_line})
            next_prompt = partner_line

        return next_prompt, reward, self.done, info

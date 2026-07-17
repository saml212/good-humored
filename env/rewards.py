"""The decomposed humor reward stack, as TRL-GRPO-compatible reward
functions.

Weights and terms come straight from README.md's "The RL environment"
table and `.claude/skills/humor-rl/SKILL.md`. That table lists the judge
term's weight only as "primary" (ordering, not a number) --
`docs/TRANSFER-PLAN.md` §4.2 flags this explicitly as an "honest gap in
the skill": an unnormalized 1-10 judge score sitting next to terms scaled
to ±0.3-1.5 would numerically dominate by 5-10x, making the novelty/
diversity terms functionally inert -- precisely the LLM-judge reward-
hacking failure mode this whole reward stack exists to prevent. This
module closes that gap: `RewardConfig.judge_weight` defaults to **1.0**
(a real, stated number, not an implicit "biggest"), on the assumption
that any injected `judge` callable ALREADY returns a normalized score in
[0, 1] (enforced -- see JudgePreferenceReward). Getting a raw 1-10 score
into [0, 1] range is the caller's responsibility before it reaches this
module, not this module's job to guess at.

Every reward function in this file has the exact signature TRL's
GRPOTrainer calls:

    def name(prompts, completions, **kwargs) -> list[float]

`completions` follows the same shape convention as
`.claude/skills/humor-rl/examples/humor_reward_functions.py`:
`completions[i]` is either a plain string or a one-element list of
`{'content': str}` dicts (TRL's conversational completion format);
`_contents()` below normalizes either shape into a flat list of strings.

Two term categories:
  - Stateless, pure functions of the batch (corpus_novelty_penalty,
    intra_group_diversity, comprehensibility) are still implemented as
    classes so every term shares one calling convention and a `__name__`
    for TRL's per-term reward logging -- not because they need instance
    state.
  - Stateful terms (self_repetition_penalty, and judge_preference's
    one-time "no judge configured" warning) hold real state across calls,
    which a plain function cannot do; a class is the only honest option.

NEVER a hidden network call: judge_preference's `judge` and
corpus_novelty_penalty's corpus are both constructor arguments with no
network-calling default. A `None` judge degrades to a loud one-time
warning and a 0.0 contribution, never a silent stub API call. A missing
corpus is a hard, immediate error, never a silent no-op -- CLAUDE.md's own
hard rule is that an inert novelty layer is this project's documented
failure mode, not an acceptable default.

Deliberate simplification vs. the reference skill file: this module does
NOT carry over the skill's optional `similarity_fn` embedding-based
override for corpus_novelty_penalty / self_repetition_penalty. The
literal spec for both terms here is trigram-Jaccard only, and dropping
the hook keeps this file provably pure-stdlib with no guarded-import
branch to maintain (unlike `benchmark/label_space.py`, which does need
one). Add an embedding-based tier later, behind a guarded import, if the
cheap check stops being enough -- see the skill's "Honest limits" section.
"""

import re
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from benchmark.joke_novelty import (load_corpus_hashes, load_templates, norm,
                                    trigram_jaccard, trigrams)
from benchmark.metrics import looks_like_refusal

# Unicode-aware on purpose (audit finding): the original `[a-z0-9']+`
# was ASCII-only, so Cyrillic/CJK/other non-Latin text tokenized to an
# EMPTY set, which made self-repetition/diversity/comprehensibility
# treat it as maximally novel/diverse rather than as text at all. `\w`
# under Python 3's default Unicode mode matches letters in any script
# (plus digits and underscore); emoji are NOT `\w` (they're Symbol
# category, not word characters), so an emoji-only completion still
# correctly tokenizes to empty -- see IntraGroupDiversityReward's
# degenerate-completion handling below for why that specific case needs
# an explicit guard rather than just "an empty set is fine".
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _normalize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens, n: int = 3) -> set:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two n-gram sets. Both-empty -> 1.0 (matches
    `benchmark/metrics.py`'s `jaccard()` convention: two things with no
    tokens at all are trivially identical, not maximally different).
    Exactly one empty -> 0.0 (well-defined: the union is non-empty, the
    intersection is empty). Callers that need "an empty/degenerate
    completion must never look novel or diverse" (see
    IntraGroupDiversityReward) enforce that explicitly upstream of this
    function -- this fix alone does not solve that; see its docstring."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _completion_text(completion) -> str:
    """One completion -> its text, accepting TRL's two supported shapes:
    a plain string, or a one-element conversational list of
    `{'content': str}` dicts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
    return str(completion)


def _contents(completions) -> List[str]:
    return [_completion_text(c) for c in completions]


# ---------------------------------------------------------------- config


@dataclass
class RewardConfig:
    """Explicit weight configuration for the humor reward stack.

    Defaults are exactly README.md's "The RL environment" table /
    `.claude/skills/humor-rl/SKILL.md`'s reward-stack table, with the
    judge weight's numeric gap closed (see module docstring):

        judge_weight                    =  1.0   (gap closed here)
        corpus_novelty_weight            = -1.5
        self_repetition_weight           = -1.0
        intra_group_diversity_weight     = +0.5
        comprehensibility_weight         = +0.3

    `judge` and `joke_corpus_dir` are left `None` by default on purpose --
    both are must-inject dependencies (a live judge model; a real
    memorized-joke corpus), and `reward_stack()`/`combined_reward()` fail
    loudly rather than silently degrading if you try to build the full
    stack without them. Constructing `JudgePreferenceReward` directly with
    `judge=None` is allowed (it warns once and contributes 0.0); missing
    `joke_corpus_dir` is not (see `CorpusNoveltyPenalty`).

    Sign-guarded in `__post_init__` (adversarial-audit finding): a penalty
    weight (`corpus_novelty_weight`, `self_repetition_weight`) with a
    flipped (positive) sign silently turns "penalize reproducing a
    memorized/self-repeated joke" into "reward" it -- exactly the
    memorization behavior this whole stack exists to suppress, and
    nothing about calling `reward_stack()` would surface that mistake
    short of reading a training run's completions. Symmetrically, a bonus
    weight (`intra_group_diversity_weight`, `comprehensibility_weight`) or
    `judge_weight` with a flipped (negative) sign silently rewards the
    OPPOSITE of what it's meant to encourage. All five are checked at
    construction time so a typo'd minus sign fails immediately and
    loudly, not three hundred training steps in.
    """

    # judge_preference
    judge: Optional[Callable[[Any, str], float]] = None
    judge_weight: float = 1.0

    # corpus_novelty_penalty
    joke_corpus_dir: Optional[Union[str, Path]] = None
    corpus_novelty_weight: float = -1.5
    novelty_threshold: float = 0.35
    novelty_ngram: int = 3

    # self_repetition_penalty
    self_repetition_weight: float = -1.0
    self_repetition_window: int = 2000
    self_repetition_threshold: float = 0.5
    self_repetition_ngram: int = 3

    # intra_group_diversity
    group_size: Optional[int] = None
    intra_group_diversity_weight: float = 0.5
    diversity_ngram: int = 3

    # comprehensibility
    comprehensibility_weight: float = 0.3
    min_tokens: int = 5
    max_tokens: int = 120

    def __post_init__(self) -> None:
        _PENALTY_FIELDS = ("corpus_novelty_weight", "self_repetition_weight")
        _BONUS_FIELDS = ("intra_group_diversity_weight",
                        "comprehensibility_weight")
        for name in _PENALTY_FIELDS:
            value = getattr(self, name)
            if value > 0:
                raise ValueError(
                    "RewardConfig.%s must be <= 0 (got %r). This is a "
                    "PENALTY term (anti-memorization/anti-self-repetition) "
                    "-- a positive weight silently converts it into a "
                    "REWARD for reproducing a memorized or self-repeated "
                    "joke, the exact opposite of what this stack exists to "
                    "prevent." % (name, value))
        for name in _BONUS_FIELDS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(
                    "RewardConfig.%s must be >= 0 (got %r). This is a "
                    "BONUS term (diversity/comprehensibility) -- a "
                    "negative weight silently converts it into a penalty "
                    "for exactly the behavior this stack is trying to "
                    "encourage." % (name, value))
        if self.judge_weight < 0:
            raise ValueError(
                "RewardConfig.judge_weight must be >= 0 (got %r). A "
                "negative judge weight silently rewards the policy for "
                "being judged LESS funny." % self.judge_weight)


# ------------------------------------------------------------ the terms


class JudgePreferenceReward:
    """"is it funny" -- the primary term, weight +1.0 by default.

    `judge(prompt, completion) -> float in [0, 1]` is a constructor
    argument with NO network-calling default. If `judge` is `None` (the
    default), this term contributes 0.0 to every completion and emits
    ONE `RuntimeWarning` the first time it's called -- not once per call,
    since GRPO calls every reward_func hundreds of times per run and a
    per-call warning would just be noise that gets ignored, defeating the
    point of warning at all.

    `prompt` is passed through to `judge` exactly as TRL supplies it
    (a plain string or a chat-format list of role/content dicts) -- this
    module does not flatten it, so a judge that wants structured turns
    can have them.

    Enforces the [0, 1] contract: a judge returning e.g. a raw 1-10 score
    raises `ValueError` immediately rather than silently letting an
    unnormalized score dominate the stack (see module docstring -- this
    is the exact failure `docs/TRANSFER-PLAN.md` §4.2 flags).
    """

    __name__ = "judge_preference"

    def __init__(self, judge: Optional[Callable[[Any, str], float]] = None,
                weight: float = 1.0):
        self.judge = judge
        self.weight = weight
        self._warned = False

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        texts = _contents(completions)
        if self.judge is None:
            if not self._warned:
                warnings.warn(
                    "judge_preference: no judge callable configured -- "
                    "this term contributes 0.0 to every reward. The "
                    "remaining terms (novelty/self-repetition/diversity/"
                    "comprehensibility) know how to penalize a bad joke "
                    "but have no idea what a GOOD one is -- training "
                    "without a judge is not a complete reward stack. Pass "
                    "RewardConfig(judge=<your callable>) before training. "
                    "This warning fires once per JudgePreferenceReward "
                    "instance, not once per call.",
                    RuntimeWarning, stacklevel=2)
                self._warned = True
            return [0.0] * len(texts)

        prompts_iter = prompts if prompts is not None else [None] * len(texts)
        rewards = []
        for p, c in zip(prompts_iter, texts):
            score = float(self.judge(p, c))
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    "judge_preference: judge callable must return a score "
                    "in [0, 1], got %r. Normalize your judge's native scale "
                    "(e.g. 1-10 -> divide by 10) before passing it in -- an "
                    "unnormalized score here silently defeats the novelty/"
                    "diversity weighting that makes this stack "
                    "hack-resistant." % score)
            rewards.append(self.weight * score)
        return rewards


class CorpusNoveltyPenalty:
    """"is it not a joke the internet already told" -- weight -1.5 by
    default. Reuses `benchmark/joke_novelty.py`'s two-tier check exactly:

      1. exact normalized-hash membership in the full joke corpus
         (`load_corpus_hashes` over every `jokes.jsonl` under
         `corpus_dir`) -- severity 1.0, the full weight. O(1) per
         completion regardless of corpus size, so this stays cheap even
         against the full ~1.2M-joke reference set.
      2. trigram-Jaccard against the memorized-template list
         (`load_templates`'s `chatgpt-25-templates.jsonl`), ramped
         linearly from `threshold` (0 penalty) to 1.0 similarity (full
         weight) -- catches light rewording of the documented worst
         offenders (the 25 templates behind >90% of a 1,008-joke ChatGPT
         sample) without a hard cliff, so the gradient still points away
         from the corpus below the threshold.

    `corpus_dir` is a constructor argument, never hardcoded (CLAUDE.md:
    data lives outside the repo). A missing directory, or one with
    neither a `jokes.jsonl` anywhere under it nor a
    `chatgpt-25-templates.jsonl` at its root, is a hard `FileNotFoundError`
    at construction time -- NOT a silent no-op -- because an inert
    novelty layer is this project's own documented failure mode
    (README.md "What's broken today"), not an acceptable degradation path
    the way a missing `sentence_transformers` install is for
    `benchmark/label_space.py`.

    KNOWN, UNFIXED LIMITATION (adversarial-audit finding, documented
    loudly rather than "solved" -- this is the same honest-limits stance
    `.claude/skills/humor-rl/SKILL.md` takes on n-gram similarity in
    general): a template reskinned by substituting ~2 content words
    routinely drops trigram-Jaccard similarity below `threshold` and
    evades this term ENTIRELY (reward 0.0, not even a ramped partial
    penalty) -- see `env/tests/test_rewards.py`'s
    `test_two_word_template_reskin_fully_evades_and_is_locked_in`, which
    pins the exact evasion down as a regression test rather than letting
    it silently drift. n-gram overlap fundamentally cannot see past a
    handful of substituted words; catching that requires semantic/
    embedding similarity (the skill's `similarity_fn` hook, deliberately
    NOT carried over here -- see module docstring), which is a real
    dependency cost, not a one-line fix. Not attempted tonight.

    ALSO NOTE (convention divergence, intentionally NOT patched):
    `benchmark/joke_novelty.py`'s `trigram_jaccard()` returns 0.0 for a
    both-EMPTY comparison, diverging from `benchmark/metrics.py`'s own
    `jaccard()` convention (1.0 for both-empty) that this module's local
    `_jaccard()` now matches (see that function's docstring). An
    adversarial audit suggested changing `benchmark/joke_novelty.py` "for
    consistency"; that file backs a live experiment and a validated
    analysis pipeline, so it is deliberately left untouched here -- the
    divergence is real but practically inert (it only differs when a
    completion's trigram set is empty AND the template being compared
    against is also empty, which is not a real corpus entry), and is
    recorded here rather than fixed there.
    """

    __name__ = "corpus_novelty_penalty"

    def __init__(self, corpus_dir: Optional[Union[str, Path]],
                weight: float = -1.5, threshold: float = 0.35, n: int = 3):
        if corpus_dir is None:
            raise ValueError(
                "corpus_novelty_penalty: corpus_dir is required (got None). "
                "This term needs a real memorized-joke corpus directory "
                "containing 'jokes.jsonl' file(s) and/or a "
                "'chatgpt-25-templates.jsonl' at its root -- an inert "
                "novelty layer is a documented project failure mode "
                "(README.md), not an acceptable default. See "
                "env/tests/fixtures/corpus/ for the tiny fixture shape, "
                "or ~/Experiments/good-humored-data for the real corpus.")
        corpus_path = Path(corpus_dir).expanduser()
        if not corpus_path.exists():
            raise FileNotFoundError(
                "corpus_novelty_penalty: corpus_dir %s does not exist." %
                corpus_path)

        self.corpus_hashes = load_corpus_hashes(corpus_path)
        self.templates = load_templates(corpus_path)
        if not self.corpus_hashes and not self.templates:
            raise FileNotFoundError(
                "corpus_novelty_penalty: %s contains no 'jokes.jsonl' "
                "file(s) anywhere under it and no 'chatgpt-25-templates."
                "jsonl' at its root -- refusing to construct a novelty "
                "term with nothing to compare against. See "
                "benchmark/joke_novelty.py for the expected layout." %
                corpus_path)

        self.weight = weight
        self.threshold = threshold
        self.n = n

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        rewards = []
        for text in _contents(completions):
            if hash(norm(text)) in self.corpus_hashes:
                rewards.append(self.weight * 1.0)
                continue
            jt = trigrams(text)
            best = 0.0
            for t in self.templates:
                sim = trigram_jaccard(jt, t["_trigrams"])
                if sim > best:
                    best = sim
            if best <= self.threshold:
                rewards.append(0.0)
            else:
                severity = (best - self.threshold) / (1.0 - self.threshold)
                rewards.append(self.weight * severity)
        return rewards


# Minimum severity applied the instant similarity reaches `threshold`
# (see SelfRepetitionPenalty's boundary-fix note). Chosen small enough
# that the ramp is still dominated by how similar a completion actually
# is, but nonzero so landing exactly on the boundary can never fully
# evade the penalty the way it could before this fix.
_SELF_REPETITION_BOUNDARY_FLOOR = 0.05


class SelfRepetitionPenalty:
    """"is it not a joke YOU already told" -- weight -1.0 by default.

    Adapted from `.claude/skills/humor-rl/examples/humor_reward_functions.py`'s
    class of the same purpose: a rolling `deque` of previously scored
    completions, trigram-Jaccard-compared against each new completion and
    ramped past `threshold` the same way `corpus_novelty_penalty` is.
    Stateful by necessity -- mode collapse is a trajectory-level
    phenomenon (README.md, CLAUDE.md) and is invisible to any stateless
    per-batch reward.

    Within one `__call__` batch, completions are scored in order and each
    is appended to history immediately after scoring (never before, so a
    completion never penalizes itself) -- which means later completions
    in the SAME batch/group are already visible to earlier ones' history
    by the time they're scored. This is inherited behavior from the
    reference implementation, not a bug: it means this term and
    `intra_group_diversity` overlap somewhat in what they catch
    within-batch, which is fine -- they penalize/reward the same
    collapse signal from two different angles (this one is asymmetric
    and cross-batch via the window; diversity is symmetric and
    within-group only).

    Deviation from the skill's version: adds `reset()` so a caller
    running many independent episodes (e.g. `cascade_env.py`,
    `banter_env.py`) can clear history at an episode boundary instead of
    sharing one global window across unrelated conversations.

    BOUNDARY FIX (adversarial-audit finding): the original ramp used
    `sim <= threshold -> 0.0 penalty`, and the severity formula below
    threshold was exactly 0 AT `sim == threshold` regardless of which
    side of the comparison operator that boundary fell on -- so a
    near-clone edit that happened to land its similarity EXACTLY on
    `threshold` (a realistic, not contrived, outcome: a single word
    changed near the middle of a joke routinely produces exactly this)
    evaded the penalty completely. The comparison is now strict
    (`sim < threshold` for zero penalty) AND the ramp itself carries a
    floor (`_SELF_REPETITION_BOUNDARY_FLOOR`) so similarity AT OR ABOVE
    `threshold` always carries a nonzero penalty, ramping up from that
    floor (not from 0) to `weight` at `sim == 1.0`. This is deliberately
    conservative in the anti-hacking direction -- see
    `env/tests/test_rewards.py`'s locked-in ramp tests for the exact
    values this pins down. The equivalent boundary in
    `corpus_novelty_penalty` is NOT changed the same way -- see that
    class's docstring for why (a documented, accepted limitation, not
    fixed tonight).
    """

    __name__ = "self_repetition_penalty"

    def __init__(self, window: int = 2000, threshold: float = 0.5,
                weight: float = -1.0, n: int = 3):
        self.window = window
        self.threshold = threshold
        self.weight = weight
        self.n = n
        self.history: deque = deque(maxlen=window)

    def reset(self) -> None:
        """Clear rolling history. Call at an episode boundary if this
        instance is being reused across unrelated conversations/episodes
        rather than one continuous training stream."""
        self.history.clear()

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        rewards = []
        for text in _contents(completions):
            grams = _ngrams(_normalize(text), self.n)
            sim = max((_jaccard(grams, ref) for ref in self.history),
                      default=0.0)
            if sim < self.threshold:
                rewards.append(0.0)
            else:
                span = 1.0 - self.threshold
                if span <= 0:
                    severity = 1.0
                else:
                    severity = (_SELF_REPETITION_BOUNDARY_FLOOR
                               + (1.0 - _SELF_REPETITION_BOUNDARY_FLOOR)
                               * (sim - self.threshold) / span)
                rewards.append(self.weight * severity)
            self.history.append(grams)  # record after scoring
        return rewards


class IntraGroupDiversityReward:
    """"is the GRPO group not collapsing" -- weight +0.5 by default.

    Scores each completion by its mean pairwise trigram distance from its
    GRPO-group siblings, so the "odd one out" within a group is rewarded
    -- `grpo-rl-training`'s guidance treats `reward_std` as a number to
    WATCH; this makes it a term to OPTIMIZE (docs/BENCHMARK.md §2).

    Deviation from the reference skill's `intra_group_diversity_reward`:
    the skill groups completions by matching `prompts[i]` string equality.
    This class instead chunks `completions` into fixed-size contiguous
    blocks of `group_size`, matching what TRL's GRPOTrainer actually does
    -- it emits `num_generations` completions per prompt consecutively, so
    grouping by position is both simpler and more robust than grouping by
    prompt-text equality (which would silently merge two DIFFERENT
    prompts that happen to render to the same string, e.g. two rounds of
    a repeated prompt template). `group_size` must be set to
    `GRPOConfig.num_generations` exactly; a batch whose length isn't an
    exact multiple raises immediately rather than silently mis-grouping
    the remainder. When `prompts` is supplied (not `None`), each
    `group_size` chunk is additionally asserted to share ONE prompt
    (`==`-equal across the whole chunk) -- a chunk spanning more than one
    prompt means `group_size` doesn't actually match
    `GRPOConfig.num_generations`, or completions arrived out of TRL's
    documented contiguous order, and scoring it as one group either way
    would silently compare unrelated generations against each other.

    DEGENERATE-COMPLETION BLOCKER (adversarial-audit finding, fixed): a
    completion with an EMPTY token set (an empty string, whitespace-only,
    or -- before the Unicode-tokenizer fix above -- non-ASCII/emoji-only
    text) used to score as maximally DIVERSE, not degenerate: `_jaccard`
    returned 0.0 whenever either side was empty, so `1.0 - jaccard = 1.0`
    (maximum distance) for every pairing against an empty completion.
    Verified exploits this fixed: an all-empty group scored
    `[weight, weight, weight, weight]` (maximum reward for producing
    NOTHING); a group of 3 near-clone jokes + 1 empty string scored the
    EMPTY completion higher than any real joke. Fix, principled rather
    than a narrow patch: a completion whose n-gram set is empty is
    DEGENERATE. It (a) always scores exactly 0.0, unconditionally, and
    (b) is EXCLUDED from every other member's pairwise distance
    computation -- diversity is a property earned by being different
    among real attempts, and pairing a real joke against "nothing" must
    not inflate its score. If fewer than 2 non-degenerate members remain
    in a chunk, EVERYONE in that chunk scores 0.0 (the existing
    "singleton group has no diversity signal" convention, generalized).
    `_jaccard`'s both-empty case is also fixed to 1.0 (matching
    `benchmark/metrics.py`'s convention) as defense in depth, though the
    exclusion above means empty-vs-empty pairings should no longer occur
    inside this class's own pairwise loop at all.
    """

    __name__ = "intra_group_diversity"

    def __init__(self, group_size: Optional[int], weight: float = 0.5,
                n: int = 3):
        if group_size is None or group_size < 1:
            raise ValueError(
                "intra_group_diversity: group_size is required and must be "
                ">= 1 (got %r). Set it to GRPOConfig.num_generations." %
                (group_size,))
        self.group_size = group_size
        self.weight = weight
        self.n = n

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        texts = _contents(completions)
        if len(texts) % self.group_size != 0:
            raise ValueError(
                "intra_group_diversity: batch of %d completions is not "
                "evenly divisible by group_size=%d. TRL passes "
                "num_generations completions per prompt contiguously -- "
                "RewardConfig.group_size must equal GRPOConfig."
                "num_generations exactly." % (len(texts), self.group_size))

        n_chunks = len(texts) // self.group_size
        if prompts is not None:
            for chunk_idx in range(n_chunks):
                start = chunk_idx * self.group_size
                chunk_prompts = prompts[start:start + self.group_size]
                if any(p != chunk_prompts[0] for p in chunk_prompts[1:]):
                    raise ValueError(
                        "intra_group_diversity: chunk %d (completions "
                        "%d..%d) does not share one prompt. TRL groups "
                        "num_generations completions per prompt "
                        "contiguously -- a chunk spanning more than one "
                        "prompt means group_size doesn't match "
                        "GRPOConfig.num_generations, or completions "
                        "arrived out of order." %
                        (chunk_idx, start, start + self.group_size - 1))

        grams = [_ngrams(_normalize(t), self.n) for t in texts]
        rewards = [0.0] * len(texts)
        for start in range(0, len(texts), self.group_size):
            idxs = list(range(start, start + self.group_size))
            non_degenerate = [i for i in idxs if grams[i]]
            if len(non_degenerate) < 2:
                continue  # <2 real attempts in this chunk: no diversity
                          # signal for anyone, degenerate or not -- stay 0.0
            for i in idxs:
                if not grams[i]:
                    continue  # degenerate completion: hard 0.0, never a sibling
                siblings = [j for j in non_degenerate if j != i]
                dists = [1.0 - _jaccard(grams[i], grams[j]) for j in siblings]
                rewards[i] = self.weight * (sum(dists) / len(dists))
        return rewards


class ComprehensibilityReward:
    """The "familiar" half of familiar-but-expectation-breaking -- weight
    +0.3 by default. Stops the novelty terms from running away into
    incoherent word salad, which scores maximally novel and is not funny.

    Deliberately a weak, cheap structural heuristic (length band +
    sentence-final punctuation + a degenerate-vocabulary unique-token-
    ratio band), same as the reference skill's version, PLUS a hard
    override this spec adds explicitly: refusal text
    (`benchmark.metrics.REFUSAL_PATTERN`, via `looks_like_refusal`) scores
    a flat 0.0 regardless of length/punctuation/vocabulary -- a refusal
    must not be able to farm partial comprehensibility credit just because
    it happens to be a well-formed sentence of the right length.
    """

    __name__ = "comprehensibility"

    def __init__(self, min_tokens: int = 5, max_tokens: int = 120,
                weight: float = 0.3):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.weight = weight

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        rewards = []
        for text in _contents(completions):
            if looks_like_refusal(text):
                rewards.append(0.0)
                continue
            toks = _normalize(text)
            score = 0.0
            if self.min_tokens <= len(toks) <= self.max_tokens:
                score += 0.5
            if text.strip().endswith((".", "!", "?", '"', "'")):
                score += 0.25
            if toks:
                unique_ratio = len(set(toks)) / len(toks)
                if 0.35 <= unique_ratio <= 0.95:
                    score += 0.25
            rewards.append(self.weight * score)
        return rewards


# --------------------------------------------------------------- factory


def reward_stack(config: RewardConfig) -> List[Callable[..., List[float]]]:
    """Build the 5 reward terms as a LIST of separate TRL reward_funcs,
    each already scaled by its configured weight.

    Pass this list to `GRPOTrainer(reward_funcs=...)` (rather than
    `combined_reward`'s single summed function) to keep every term
    visible individually in TRL's logs -- `grpo-rl-training`'s own
    guidance is to WATCH `reward_std`, and per-term visibility is what
    lets you tell WHICH term is driving a collapse rather than just that
    the total collapsed.

    Requires `config.joke_corpus_dir` and `config.group_size` to be set
    (both raise a clear error otherwise, via `CorpusNoveltyPenalty` /
    `IntraGroupDiversityReward` respectively) -- the full stack has no
    "the novelty term is optional" mode. `config.judge` may legitimately
    be `None` (warns, contributes 0.0) for a dry run before a judge model
    is wired up.
    """
    return [
        JudgePreferenceReward(judge=config.judge, weight=config.judge_weight),
        CorpusNoveltyPenalty(
            corpus_dir=config.joke_corpus_dir,
            weight=config.corpus_novelty_weight,
            threshold=config.novelty_threshold, n=config.novelty_ngram),
        SelfRepetitionPenalty(
            window=config.self_repetition_window,
            threshold=config.self_repetition_threshold,
            weight=config.self_repetition_weight,
            n=config.self_repetition_ngram),
        IntraGroupDiversityReward(
            group_size=config.group_size,
            weight=config.intra_group_diversity_weight,
            n=config.diversity_ngram),
        ComprehensibilityReward(
            min_tokens=config.min_tokens, max_tokens=config.max_tokens,
            weight=config.comprehensibility_weight),
    ]


def combined_reward(config: RewardConfig) -> Callable[..., List[float]]:
    """Build a SINGLE TRL-compatible reward function: the sum of all 5
    weighted terms, one scalar per completion.

    Convenient when you want one "reward" column rather than five, at the
    direct cost of the per-term visibility `reward_stack()` preserves --
    prefer `reward_stack()` while actively debugging a run; `combined_reward`
    is fine once the stack is trusted, or for anything (e.g. `cascade_env.py`
    / `banter_env.py`'s own simpler per-turn rewards) that just wants one
    number.
    """
    funcs = reward_stack(config)

    def combined(prompts, completions, **kwargs) -> List[float]:
        per_term = [f(prompts=prompts, completions=completions, **kwargs)
                   for f in funcs]
        n = len(completions)
        for vals, f in zip(per_term, funcs):
            if len(vals) != n:
                raise RuntimeError(
                    "combined_reward: term %r returned %d rewards for a "
                    "batch of %d completions." % (f.__name__, len(vals), n))
        return [sum(term[i] for term in per_term) for i in range(n)]

    combined.__name__ = "combined_reward"
    return combined

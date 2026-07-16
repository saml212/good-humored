"""
Humor-specific reward functions for GRPO training.

Drop-in composable with grpo-rl-training/examples/reward_functions_library.py:
same signature convention, fn(completions, **kwargs) -> List[float], where
each completion is [{'content': str}].

These target the two documented failure modes of humor RL:

  1. Mode collapse onto memorized jokes. TRL's no_repetition_penalty only sees
     repetition WITHIN a single completion. It cannot tell that a completion is
     a joke the model already told last batch, or one it memorized in
     pretraining. Those are the failures that actually kill humor training — a
     joke heard twice is dead.

  2. LLM-judge reward hacking. Published GRPO-with-judge runs collapsed to
     regurgitating classic jokes; a judge scores those highly because they ARE
     funny. The judge is not wrong, it is incomplete.

So these are composed WITH a preference/judge signal, never instead of it.
Their job is to make memorized output score badly no matter how funny the judge
finds it — which is what makes the judge hard to hack.

Default paths are stdlib-only (token-set Jaccard). Embedding-based similarity is
strictly better and is supported by passing a `similarity_fn`; it is not the
default because it adds a model dependency to every reward call.
"""

import re
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence

_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int = 3) -> set:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _contents(completions) -> List[str]:
    return [c[0]["content"] for c in completions]


def _max_similarity(text: str, corpus_grams: Sequence[set], n: int, similarity_fn) -> float:
    """Highest similarity between `text` and anything in the corpus."""
    if similarity_fn is not None:
        return max((similarity_fn(text, ref) for ref in corpus_grams), default=0.0)
    grams = _ngrams(_normalize(text), n)
    return max((_jaccard(grams, ref) for ref in corpus_grams), default=0.0)


def corpus_novelty_penalty(
    completions,
    joke_corpus: Sequence[str],
    threshold: float = 0.35,
    weight: float = -1.5,
    n: int = 3,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    **kwargs,
) -> List[float]:
    """Penalize completions that reproduce a known joke.

    `joke_corpus` is the memorized-joke reference set — the jokes the model must
    not simply retell (scrape from the pretraining-era joke datasets, plus the
    classic-joke lists that judge-hacked runs converged onto).

    Scales the penalty with similarity above `threshold` rather than applying a
    cliff, so the gradient points away from the corpus instead of just fencing
    it off at one point.

    Weight is deliberately larger in magnitude than a typical format reward: a
    memorized joke should not be able to win on judge score alone.
    """
    if similarity_fn is None:
        refs = [_ngrams(_normalize(j), n) for j in joke_corpus]
    else:
        refs = list(joke_corpus)

    rewards = []
    for text in _contents(completions):
        sim = _max_similarity(text, refs, n, similarity_fn)
        if sim <= threshold:
            rewards.append(0.0)
        else:
            # Linear ramp: threshold -> 0 penalty, 1.0 -> full weight.
            severity = (sim - threshold) / (1.0 - threshold)
            rewards.append(weight * severity)
    return rewards


class SelfRepetitionPenalty:
    """Penalize the model retelling its OWN recent jokes across batches.

    Stateful by necessity: mode collapse is a trajectory-level phenomenon and is
    invisible to any stateless per-batch reward. Holds a rolling window of
    recently generated completions and penalizes near-duplicates of them.

    `window` bounds memory and also bounds how long a joke stays "burned" — set
    it to roughly the number of completions in an epoch if you want a joke
    retired for an epoch.

    Usage:
        self_rep = SelfRepetitionPenalty(window=2000)
        trainer = GRPOTrainer(reward_funcs=[..., self_rep], ...)
    """

    __name__ = "self_repetition_penalty"  # TRL reads this for logging

    def __init__(
        self,
        window: int = 2000,
        threshold: float = 0.5,
        weight: float = -1.0,
        n: int = 3,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.history: deque = deque(maxlen=window)
        self.threshold = threshold
        self.weight = weight
        self.n = n
        self.similarity_fn = similarity_fn

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for text in _contents(completions):
            sim = _max_similarity(text, list(self.history), self.n, self.similarity_fn)
            if sim <= self.threshold:
                rewards.append(0.0)
            else:
                severity = (sim - self.threshold) / (1.0 - self.threshold)
                rewards.append(self.weight * severity)
            # Record after scoring, so a completion never penalizes itself.
            self.history.append(
                text if self.similarity_fn is not None else _ngrams(_normalize(text), self.n)
            )
        return rewards


def intra_group_diversity_reward(
    completions,
    prompts=None,
    weight: float = 0.5,
    n: int = 3,
    **kwargs,
) -> List[float]:
    """Reward completions that differ from their GRPO group siblings.

    GRPO samples several completions per prompt and learns from within-group
    comparison. If the group collapses to one joke, the comparison carries no
    signal and reward_std goes to ~0 — the documented mode-collapse signature.
    grpo-rl-training treats reward_std as a metric to WATCH; this makes it a
    term to OPTIMIZE.

    Scores each completion by mean pairwise distance from its siblings, so a
    completion is rewarded for being the odd one out. Groups by prompt when
    `prompts` is supplied; otherwise treats the whole batch as one group.
    """
    texts = _contents(completions)
    grams = [_ngrams(_normalize(t), n) for t in texts]

    groups: Dict[str, List[int]] = {}
    for i, _ in enumerate(texts):
        key = str(prompts[i]) if prompts is not None else "_batch"
        groups.setdefault(key, []).append(i)

    rewards = [0.0] * len(texts)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue  # No siblings: no diversity signal, stay neutral.
        for i in idxs:
            dists = [1.0 - _jaccard(grams[i], grams[j]) for j in idxs if j != i]
            rewards[i] = weight * (sum(dists) / len(dists))
    return rewards


def comprehensibility_reward(
    completions,
    min_tokens: int = 5,
    max_tokens: int = 120,
    weight: float = 0.3,
    **kwargs,
) -> List[float]:
    """The 'familiar' half of familiar-but-expectation-breaking.

    Novelty rewards alone push toward incoherent word salad, which is maximally
    novel and not funny. This is the counterweight: a cheap structural floor.

    Deliberately a weak heuristic — it checks that output is well-formed and
    reasonably sized, NOT that it is comprehensible in any deep sense. It exists
    to stop the novelty terms from running away, and should carry a small weight
    relative to them. Real comprehensibility needs the judge or a human.
    """
    rewards = []
    for text in _contents(completions):
        toks = _normalize(text)
        score = 0.0
        if min_tokens <= len(toks) <= max_tokens:
            score += 0.5
        if text.strip().endswith((".", "!", "?", '"', "'")):
            score += 0.25
        # Degenerate-vocabulary check: word salad tends to never reuse a word.
        if toks:
            unique_ratio = len(set(toks)) / len(toks)
            if 0.35 <= unique_ratio <= 0.95:
                score += 0.25
        rewards.append(weight * score)
    return rewards


def make_humor_reward_stack(
    joke_corpus: Sequence[str],
    judge_reward: Callable,
    self_repetition_window: int = 2000,
) -> List[Callable]:
    """Compose the full stack in the order the failure modes demand.

    The judge supplies "is it funny"; everything else supplies "is it actually
    yours, and new". Passing a judge is required, not optional — the novelty
    terms alone have no idea what a joke is.

    Verify before you trust: train a few hundred steps and read the completions.
    If reward is climbing while the jokes are getting worse, the judge is being
    hacked and the novelty weights are too low.
    """
    self_rep = SelfRepetitionPenalty(window=self_repetition_window)

    def corpus_novelty(completions, **kwargs):
        return corpus_novelty_penalty(completions, joke_corpus=joke_corpus, **kwargs)

    corpus_novelty.__name__ = "corpus_novelty_penalty"

    return [
        judge_reward,
        corpus_novelty,
        self_rep,
        intra_group_diversity_reward,
        comprehensibility_reward,
    ]

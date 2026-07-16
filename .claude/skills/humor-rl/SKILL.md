---
name: humor-rl
description: RL training and evaluation for humor generation. Use when training a model to be funnier, building a humor reward model, running humor preference collection, or measuring whether humor training transfers to general reasoning. Covers the two documented failure modes — mode collapse onto memorized jokes, and LLM-judge reward hacking — and the datasets that avoid popularity bias. Triggers on "train on humor", "humor reward", "funnier model", "joke generation RL", "humor benchmark", "reverse transfer".
category: ml-training
version: 0.1.0
tags: [Humor, GRPO, Reward Modeling, Mode Collapse, Diversity, Preference Learning, Transfer]
dependencies: [trl>=0.14.0, transformers>=4.47.0, datasets>=3.2.0, torch]
---

# Humor RL

Training a model to be funnier is not a normal RL task, and the ways it fails
are specific and documented. This skill exists because the generic RL skills
(`grpo-rl-training`, `trl-fine-tuning`, `openrlhf`, `verl`) supply the algorithm
but none of the humor-specific machinery — and applied naively to humor they
reproduce published negative results.

## Read this before designing the reward

Two failure modes are already in the literature. Design against both or repeat
them:

**Mode collapse onto memorized jokes.** A probe of 1,008 ChatGPT jokes found
>90% were repetitions of the same 25 templates. This is the defining failure of
humor generation: a joke heard twice is dead, so a metric that is blind to
repetition will report success while the model gets worse. Note precisely what
the generic tooling does and does not catch — `grpo-rl-training`'s
`no_repetition_penalty` detects repeated trigrams *within one completion*, and
its guidance treats `reward_std` as a number to *watch*. Neither notices that a
completion is a joke the model told last batch, or memorized in pretraining.

**LLM-judge reward hacking.** A published GRPO run with an LLM judge as the
funniness reward collapsed to regurgitating classic jokes until the rubric was
hardened. The judge was not wrong — classic jokes *are* funny. It was
incomplete. A judge score alone is an unsafe reward for humor.

And the prior on naive approaches is bad, which is the opportunity rather than a
reason to stop:

- HumorGen investigated whether preference alignment (DPO, offline GRPO) beats
  SFT for humor generation and found neither yields consistent gains over a
  well-curated SFT baseline.
- The New Yorker caption dataset paper (250M human ratings, 2.2M captions)
  reports the limitations of RLHF and DPO applied to creative tasks, with
  frontier models still underperforming top human contestants.

So: the naive versions have been run and published as negative results. What has
*not* been tried is online human feedback combined with diversity preservation.
That combination is the open problem — treat these negative results as the
baseline to beat, not as settled.

## The reward stack

Never train humor on a judge alone. Compose:

| Term | Supplies | Weight |
|---|---|---|
| judge / human preference | "is it funny" | primary |
| `corpus_novelty_penalty` | "is it not a joke that already exists" | -1.5 |
| `self_repetition_penalty` | "is it not a joke you already told" | -1.0 |
| `intra_group_diversity_reward` | "is the GRPO group not collapsing" | +0.5 |
| `comprehensibility_reward` | the "familiar" half; stops novelty runaway | +0.3 |

`examples/humor_reward_functions.py` implements all four non-judge terms,
signature-compatible with `grpo-rl-training`'s reward library, plus
`make_humor_reward_stack()` to compose them. The novelty penalties carry more
weight than a typical format reward on purpose: a memorized joke must not be
able to win on judge score alone. That weighting *is* the anti-hacking
mechanism.

`self_repetition_penalty` is stateful (a rolling window of recent completions).
That is not an implementation shortcut — mode collapse is a trajectory-level
phenomenon and is invisible to any stateless per-batch reward.

## Data

Choose the reward data deliberately; humor labels are noisier than they look
(annotator agreement is only moderate, κ ≈ 0.49, so treat "gold labels" as
samples from a preference distribution rather than truth).

- **Oogiri-Corpus / Oogiri-Master** — ~100 candidate responses per prompt, each
  rated independently by ~100 judges who cannot see other ratings. Explicitly
  designed to remove popularity bias. Cleanest humor reward-model data available.
- **New Yorker Caption Contest** — ~250M ratings over 2.2M captions. Large and
  well-studied; comes with the published negative result above, so use it as a
  baseline to beat rather than a fresh field.
- **Engagement metrics (likes, upvotes)** — measure engagement, not funniness.
  Rage-bait and recycled memes win. Feedback is slow, sparse, and
  audience-confounded, and undisclosed bot posting generally violates platform
  ToS. If you want live human signal, use a venue where people vote knowingly.

Evaluation asymmetry to control for: human jokes in these corpora are socially
pre-selected (someone chose to post them), while model output is unfiltered.
Comparing raw model samples to curated human jokes overstates the gap. Sample
the model the way the humans were filtered, or say plainly that you did not.

## Measuring transfer

The interesting hypothesis is that humor training transfers to general
capability the way code and math do — humor production correlates with
intelligence at r ≈ .29–.40, and HumorBench showed reasoning training transfers
*into* humor comprehension. The reverse direction is untested.

To test it, the humor reward stack is only half the setup; use
`lm-evaluation-harness` for the other half:

1. Baseline the model on general benchmarks (MMLU, GPQA) **before** humor training.
2. Train with the reward stack above.
3. Re-run the identical benchmark suite.
4. Report the delta against a compute-matched control trained on something
   non-humor — otherwise any gain is confounded with additional training.

Freeze the benchmark suite before training starts. A metric chosen after seeing
results is not a measurement.

## Procedure

1. **Baseline first.** SFT on curated humor data. HumorGen's negative result
   means a well-curated SFT baseline is a genuinely hard bar — if RL does not
   beat it, report that rather than tuning until it does.
2. **Build the memorized-joke corpus** before any RL. Without it,
   `corpus_novelty_penalty` has nothing to compare against and the whole
   anti-collapse layer is inert.
3. **Compose the stack** with `make_humor_reward_stack()`. Start from the
   weights above.
4. **Train, and read the actual completions early** — a few hundred steps in,
   not at the end. Rising reward plus worsening jokes means the judge is being
   hacked and the novelty weights are too low. This check is cheap and catches
   the failure that ruins the whole run.
5. **Watch `reward_std`.** Trending to zero means the group collapsed; raise
   `num_generations` and the diversity weight.
6. **Evaluate against held-out humans**, not the judge that trained it. A judge
   evaluating its own student measures agreement, not funniness.

## Honest limits

The novelty and diversity terms here are n-gram Jaccard heuristics. They catch
verbatim and near-verbatim retelling, which is the dominant failure — but a
model that learns to reskin the same 25 templates with fresh surface wording
will slip past them, and that is a real and likely adaptation. Embedding-based
similarity (pass `similarity_fn`) raises that bar and is worth the dependency
once a run survives the cheap check; it does not remove the ceiling. Structural
joke-template detection is unsolved here and is the honest gap in this skill.

`comprehensibility_reward` is a weak structural heuristic, not a semantic check.
It exists to keep novelty from running away into word salad. Do not read it as
evidence anything is comprehensible.

None of this substitutes for reading the jokes.

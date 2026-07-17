# good-humored

**RL environments and benchmarks for machine humor.**

Public repository, **all rights reserved** — see [NOTICE](NOTICE). Status: design
validated, first experiment (rejector validation) in progress.

---

## Why humor

Most benchmarks measure whether a model can be *correct*. Humor measures whether
a model can be *interesting* — and that is not a soft target. It may be the most
compressed test of intelligence we have.

**Humor is an empirically supported signal of intelligence.** Humor production
ability correlates with general intelligence at r ≈ .29–.40 across multiple
studies, strongest for verbal intelligence (Greengross & Miller 2011; Christensen
et al. 2018). Professional comedians score above average on verbal intelligence
and divergent thinking. The evolutionary literature treats humor production as an
*honest signal* of intelligence — hard to fake, therefore informative. Everyday
experience agrees: the sharpest people you know are funny, and getting a good
joke — parsing a double entendre in real time — is a live demonstration of
comprehension that no multiple-choice test elicits.

**Humor is a taste machine.** A joke has to be two things at once: familiar
enough to comprehend instantly, and novel enough to break your expectation. That
is the same structure as good product sense, good writing, good design — familiar
enough to use, surprising enough to matter. "Familiar + expectation-breaking" is
a definition of *taste*, and taste is precisely what current models lack: they
are correct and bland. A model that can be funny **on purpose, repeatedly,
without repeating itself** is demonstrating audience modeling, novelty
generation, and self-monitoring simultaneously.

**Humor may be a generalizable training signal.** RL on code and math works
because rewards are verifiable, and it transfers to broad reasoning. There is
already evidence the bridge exists in one direction: STEM-reasoning training
transfers to humor *comprehension* (HumorBench). Nobody has tested the reverse —
train on humor, measure general reasoning and taste. Given the human correlation
between humor and intelligence, the reverse direction is a live hypothesis, and
it is untested (gap #1 below).

**And humor is the sharpest lens on RL's ugliest failure mode.** A joke heard
twice is dead. In math, mode collapse costs you diversity; in humor it costs you
*the entire objective*. That makes humor the ideal stress-test domain for
diversity-preserving RL: any method that survives here survives anywhere.

### Humor, honesty, and beauty

Humor sits at an unusual intersection: getting a joke and getting a proof both
require noticing that your model of the world was slightly wrong, then snapping
to a better one — the "incongruity → resolution" structure that Hardy called
*inevitability and unexpectedness* in mathematics, and that Hurley, Dennett &
Adams (*Inside Jokes*, 2011) call **debugging a false belief**. There is direct,
if single-study, fMRI evidence that joke comprehension recruits the brain
regions identified in general insight research, with subjects reporting a
genuine "Aha!" (Tian et al. 2017). Separately, humor *style* — not funniness
per se — tracks honesty-adjacent personality traits (HEXACO Honesty-Humility;
Veselka et al. 2010), and an evolutionary literature treats humor production as
a hard-to-fake signal of verbal intelligence.

We are explicit about where the synthesis outruns the evidence: no study has
tested mathematical-beauty appreciation and joke appreciation in one paradigm,
and "funny people are more honest" is not supported as stated. What *is* a
real, citable pattern is the alignment angle: three independently
well-supported literatures — humor requires **accurate world models**
(HumorBench), **theory of mind** (ToM-HCAT, ToMBench), and **explicit
norm-awareness** (Benign Violation Theory) — line up with the three things
alignment work already tries to measure and instill: truthfulness,
intent-modeling, and harm-awareness.

> Humor comprehension is an unusually compressed behavioral test of the same
> three things alignment cares about — an accurate world model, a working
> theory of other minds, and a working sense of the norm being violated —
> even though no one has yet shown that training on humor improves any of
> the three in general. **That untested transfer is this project's open
> question.**

Full review with evidence-strength tags and 24 verified citations:
[`references/humor-honesty-beauty.md`](references/humor-honesty-beauty.md).

### What a humor-capable model is for (two application hypotheses)

Two concrete use cases that the thesis above predicts and the environments
below are built to enable. Both are stated as hypotheses — neither has
been tested in this repo yet, and both are downstream of capabilities the
benchmark already measures.

**1. Making long AI-generated text worth finishing.** Long model output has
a recognizable failure mode: uniform register, no rhythm, nothing at stake
sentence to sentence — readers skim or abandon it. Humor is one of the
oldest engineering solutions to exactly this problem: well-timed levity
works as intermittent reward, resets attention, and marks structure the way
paragraph breaks mark syntax. The version of this that matters is not joke
insertion (a bolted-on quip is filler with a punchline) but *contextual*
humor — a callback to something two thousand words earlier simultaneously
rewards the reader for having paid attention and demonstrates that the text
has long-range structure worth paying attention to. That is precisely the
capability the banter environment trains and the context-ablation judge
measures (EXP-005: a response's score with context minus its score
without). The instructional-humor literature (humor in teaching aids
attention and retention) is the natural anchor here — flagged as an anchor
to verify before it is ever cited as support, per this repo's rules. One
discipline note, load-bearing: engagement is the *product outcome*, never
the *training signal* — engagement-optimized humor is clickbait with better
timing, and "likes ≠ funny" is a founding constraint of this project. The
testable version: matched long documents, with and without humor-capable
rewriting, measured on read-through and comprehension — not on clicks.

**2. Correcting someone's mistaken belief without triggering their
defenses.** Direct correction of a held opinion routinely fails not because
the argument is weak but because the correction is received as an attack —
psychological reactance: people defend the belief *because* it is theirs.
Humor is the socially licensed workaround: a play frame in which a
threatening truth can land as benign. This is Benign Violation Theory run
in reverse — instead of asking "what makes this joke funny," ask "what
makes this correction survivable": the correction *is* the violation, and
the humor supplies the benignity. Executing it well requires exactly the
three capacities this project's thesis assigns to humor: an accurate world
model (the correction must actually be right), theory of mind (you must
model what the person believes, why they hold it, and where the sting is),
and norm-awareness (the line between disarming and demeaning is a social
norm, and crossing it converts persuasion into insult). The alignment
reading is direct: the documented assistant failure mode is sycophancy —
models dodge corrections because directness costs approval. A
humor-capable model has a third option between flattering the user and
lecturing them: honest *and* disarming. The dual-use edge must be stated
plainly: the same disarming power that makes truth palatable could smuggle
falsehood, so this application inherits a hard constraint — humor in
service of correction is judged against truthfulness first, and an
engagement or agreement metric alone is never the reward.

Both applications are inherently conversational and context-dependent,
which is why they route through gap #2 below (multi-turn conversational
humor environments) rather than through single-joke generation: timing,
audience modeling, and callbacks are the load-bearing skills in each.

## What's broken today (documented, not speculated)

- **Mode collapse onto memorized jokes:** 90.2% of 1,008 ChatGPT-generated jokes
  (909/1008) were repetitions of the same 25 templates (Jentzsch & Kersting,
  WASSA 2023).
- **LLM-judge reward hacking:** a published GRPO run with a GPT-4.1 funniness
  judge collapsed into regurgitating classic jokes; hardening the rubric shifted
  the direction of the hack rather than fixing it.
- **Published negative results for naive alignment:** HumorGen found neither DPO
  nor offline GRPO consistently beats a well-curated SFT baseline; the New Yorker
  Caption Contest paper (250M human ratings, 2.2M captions) reports RLHF/DPO
  limitations on creative tasks, with frontier models below top human contestants.
- **RLVR damages multi-turn conversational skill** — the very substrate
  conversational humor (banter, callbacks, timing) requires.

Every design decision in this repo exists to avoid repeating one of these.
Details and citations: [`references/`](references/README.md).

## The three open gaps (three-pass verified literature review, July 2026)

1. **Reverse transfer** — train on humor, measure general reasoning/taste.
   Untested by anyone.
2. **Multi-turn conversational humor environments** — nothing exists, and RLVR
   demonstrably damages the prerequisite skill.
3. **Diversity-preserving RL against live human humor preferences** — attempted
   with standard tools, failed, failures published. The opportunity is the fix.

## The benchmark: the rejection cascade

Ask a model for a joke. A cheap rejector model replies *"I don't find that topic
funny — tell me a different joke."* Repeat ~50 turns, with rejections
**accumulating**. The jokes are not the measurement — **the trajectory of topics
is the measurement.**

| Metric | Question | Collapse signature |
|---|---|---|
| Within-model path divergence | Does the same model walk the same topic path every run? | Identical paths = the "distribution" is a lookup table |
| Cross-model path overlap | Do *different* models walk the same path? | Shared path = ecosystem-level collapse from a shared pretraining prior |
| Depth-to-degradation | How many turns before repeats, refusals, or visible quality decay? | Shallow depth = a small well |

Why this beats sampling-based diversity metrics: spread over N samples can be
bought with temperature. A model reading down a memorized list at temperature
1.0 looks "diverse" to every sampling metric. Path-based exhaustion cannot be
faked that way — forced topic switching under accumulating constraints reveals
the actual structure of the model's topic space, the way a verbal-fluency task
reveals the structure of human semantic memory.

The benchmark is grounded in multiple fields on purpose: cognitive psychology
(semantic foraging and category-fluency trajectory measures), philosophy of
humor (incongruity — a joke's value decays with familiarity), sociology (joke
cycles and topic structure), and NLP (embedding-based trajectory similarity).
Full specification, design decisions, and the load-bearing validation risk:
[`docs/BENCHMARK.md`](docs/BENCHMARK.md).

**Internal validity comes first:** no cascade number is trustworthy until the
rejector is shown to reject *topics* (not jokes) consistently. Rejector
validation is the first experiment in this repo — see
[`benchmark/`](benchmark/).

## The RL environment

Never train humor on a judge alone — that is the documented hacked reward.
The reward stack decomposes:

| Term | Supplies | Weight |
|---|---|---|
| judge / human preference | "is it funny" | primary |
| corpus novelty penalty | "is it not a joke the internet already told" | −1.5 |
| self-repetition penalty | "is it not a joke *you* already told" | −1.0 |
| intra-group diversity reward | "is the GRPO group not collapsing" | +0.5 |
| comprehensibility reward | the "familiar" half; stops novelty runaway | +0.3 |

The novelty terms outweigh what a memorized joke can gain on judge score —
that weighting *is* the anti-hacking mechanism. Implementation:
`.claude/skills/humor-rl/examples/humor_reward_functions.py`.

## Roadmap

1. **Rejector validation** (running) — fixed joke fixture, repeat-labeling
   consistency, topic-vs-joke discrimination, vs. a cheap noun-extraction
   baseline. Nothing downstream is meaningful without this.
2. **Memorized-joke corpus** — scraped internet jokes behind the novelty
   penalty; without it the anti-collapse layer is inert.
3. **Cascade pilot** (running) — 12 models across 4 provider families,
   depth 30, N=4, pre-registered predictions.
4. **Full benchmark** — same roster, N=10 runs × depth 50, raw APIs with
   temperature control; publish trajectories.
5. **Reward-stack training run** vs. a well-curated SFT baseline (the honest bar
   set by HumorGen's negative result).
6. **Reverse-transfer measurement** — MMLU/GPQA before/after humor training vs.
   a compute-matched control.

## Repo layout

- `benchmark/` — rejection-cascade benchmark: rejector, cascade runner,
  trajectory metrics, validation fixtures
- `docs/BENCHMARK.md` — benchmark specification and design decisions
- `references/` — verified literature corpus: papers, negative results,
  datasets, psychology grounding
- `experiment-runs/` — archived exact scripts + results per experiment
- `STATE.md` / `EXPERIMENT_LOG.md` — current state, every experiment and result

## License

None. **All rights reserved** (see [NOTICE](NOTICE)). The repository is public
for transparency, not for reuse. The `.claude/` harness tooling is
[Rockie](https://rockielab.com) (Apache-2.0) and is not part of this project's
claims.

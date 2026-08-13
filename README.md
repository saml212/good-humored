# good-humored

**RL environments and benchmarks for machine humor.**

Public repository, **all rights reserved** — see [NOTICE](NOTICE).

**Status:** the conversational-humor environment is built, instrumented, and
characterized (~2.9M banked sessions), and the training campaign has reached an
honest, fully-documented resting point. The headline results: (1) a redesigned
smooth objective produced the program's **first valid positive result** — RL
reliably teaches defect-cleanup and topical grounding (certified held-out
delta +0.02–0.06 across four measurements, guards clean); (2) the *taste*
dimension — making the frozen audience actually laugh more — resisted
training even when it dominated the objective, and a four-stage redesign
waterfall (brainstorm → literature → adversarial attack → validation) then
killed every proposed replacement construct *by measurement* before another
GPU-hour was spent. Along the way the verification machinery caught a silent
merge bug that had voided two earlier "null" evals, a laughter-bait reward
exploit measured at +7.3 logits for two tokens, a partner-echo lottery
channel in the audience judge, and a word-shuffle attack that beats any
embedding-based surprise gate. The story of how each of these was caught —
and why the negatives are as load-bearing as the positive — is most of what
this README is about.

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

## What we built: a banter environment, not a joke generator

Early on this project pivoted away from standalone joke generation (see the
rejection-cascade work preserved in `benchmark/` and `docs/BENCHMARK.md`)
toward what actually matters for a conversational model: **wit in context** —
banter, comebacks, callbacks, timing. The environment is deliberately simple
to describe:

- A **policy model** chats with a coworker while they do a mundane office
  task together (move desks, clean the fridge, write the meeting agenda).
- The **policy's prompt is neutral**. It is never told to be funny. That is
  the load-bearing design decision: the dataset's value is what wit a model
  produces *unprompted*, and how it responds when an opening appears.
- A frozen **partner model** (much larger) drives the conversation and, at
  seeded random turns, is directed to provoke: tease the policy, crack a
  joke, swear in frustration, make an observation. Every session's schedule
  is reproducible from its seed.
- A frozen **audience model** watches and, at each policy turn, we read the
  probability mass it puts on laughing next — not a judge being *asked* to
  rate funniness (the documented hacked reward), but a spontaneous reaction.

Why a strong partner? We measured it: with the policy held constant and only
the partner swapped for a weaker model, session quality dropped, audience
reaction dropped, and the policy slid into yes-and sycophancy (a controlled
same-seed A/B, 500 paired sessions). **The partner is part of the
environment's spec, not an implementation detail.**

Why provocations? Also measured: being *teased* elicits the best comebacks —
post-mock turns draw roughly two logits more audience laughter than
unprovoked turns, consistently across models — and raising the provocation
rate measurably reduces the policy's yes-and rate. The provocation scheduler
is an adjustable anti-sycophancy dial.

## How we verify everything (and why)

This project's working assumption is that in open-ended domains, **most
exciting results are artifacts**, and the only defense is machinery that
makes it hard to fool yourself. Every rule below exists because it caught
something real here:

- **Pre-registration with pinned predictions.** Every experiment states its
  hypothesis, its predicted number, its success bar, and — critically — its
  *failure consequence* before the data arrives. When our reaction-logprob
  signal scored ρ=0.122 against a pinned bar of 0.15, the pre-signed
  consequence executed (demotion to diagnostic) instead of a debate.
- **Blind, multi-author fixtures.** Every instrument certified on
  same-author test data later failed a blind set. The anchoring gate that
  survives is the one certified cross-author, blind.
- **The human read is the quality bar, never the metric.** Scores rank
  candidates; a person reads the top transcripts. Reads caught what metrics
  could not: a roleplay-register drift infecting 27% of turns, a Chinese
  token leak that scaled 20× with temperature, and a case where the metric
  ranked an objectively weaker transcript first because its lane had 50× the
  sample size to get lucky in. Each read-discovery then became a standing
  automated counter — reads *discover* defect classes; counters establish
  prevalence.
- **Adversarial pre-run audits.** A separate agent reviews every experiment
  before it runs. The audit of our reward-model trainer caught that the
  baseline we planned to compare against was computed on a different
  population — the comparison would have been silently meaningless.
- **Matched-seed A/B evaluation.** Training claims are judged only on
  held-out sessions where base and trained models face *identical* seeds —
  same tasks, same provocation schedules, same partner behavior — so every
  session is its own control. Fresh seed spaces are never reused.
- **Instrument-parity gates.** Before believing any training curve, we now
  verify (a) the training-time reward and the eval-time reward score the
  same transcripts identically, and (b) the training rollout pathway
  produces text the clean pipeline scores the same (n=200 matched pairs).
  Our first RL run failed to have either, and its entire curve turned out
  to be uninterpretable.
- **Statistical power before mechanism claims.** Three successive n=30
  comparisons produced three contradictory explanations of the same
  phenomenon (SE ≈ the effect size). The rule now: compute the n the effect
  demands, then conclude. We publicly retracted two of our own diagnoses on
  this basis — the retractions are in the experiment log, dated.

## Results so far — the honest version

**The environment works.** ~2.9M banked sessions. From neutral prompts, the
material at the top is genuinely funny: multi-character office universes
(the coworker who renamed the Wi-Fi "FBI Surveillance Van #4" and is managed
via a fake "Cable Integrity and Compliance Officer" badge), transformed
callbacks that pay off eight turns later, a policy that types a flirtation
*into the expense-report form* to commit to a bit. We catalogued the move
classes — and their weak shadows (echo-affirmation, yes-and drift) — because
a reward model eventually needs to tell them apart.

**Characterization findings** (each from the config table, most confirmed by
targeted A/B): partner quality gates everything; teasing is the best wit
elicitor; provocation density suppresses sycophancy; temperature response is
model-specific; strong instruction-tuned models yes-and *more* than weak
ones (trained-in conversational risk-aversion — the thing RL is supposed to
fix); model "house-style" motifs survive every prompt intervention and will
only move with training.

**A trained reward model certified — then honestly disqualified.** We
trained a caption-ranking model on ~600k New Yorker Caption Contest ratings:
0.395 mean within-contest correlation on fully held-out contests,
sign-correct on 77 of 77. A real instrument. Then we checked whether it
transfers to banter turns before wiring it into training — it
*anti-correlates* (−0.09): it detects decontextualized zinger-ness, which is
a different thing from conversational wit. It now serves as an independent
cross-check, not a reward. The check that caught this cost an hour; wiring
it blind would have trained a model to produce captions instead of banter.

**Three training runs, three nulls — and that is the finding.**
1. *GRPO v1* (LoRA rank 32, 200 steps): training reward rose +0.07; held-out
   A/B showed nothing. Post-mortem found the training prompts were
   malformed (a template-construction bug) — the policy had adapted to its
   own corrupted context. The curve was never measuring skill.
2. *GRPO v2* (pathway verified by a 200-pair parity gate first): the curve
   rose again, late and real (~3 standard errors). Held-out A/B: +0.008,
   t=0.41 — nothing transferred. The mechanism signature: max reward flat
   all run while the mean rose — the policy was concentrating onto its
   existing good modes, not acquiring new competence.
3. *SFT distillation* of the bank's certified top 9.3% (38,823 sessions):
   delta −0.012. No mode collapse (diversity guards all held) and no gain.
   The model absorbed its own best behavior and came out unchanged.

**Then we diagnosed it, quantitatively.** Scoring 8 rollouts each for 64
identical contexts: **94% of reward variance is within-context** — same
prompt, same partner, wildly different scores — driven by the reward's hard
cliffs (one flagged turn zeroes a session), multiplicative gates, and
max-terms. Instruments that are excellent for *ranking and curation* were
drowning the gradient in discontinuity noise. The project's deepest lesson,
now a design rule: **a certified measurement is not automatically a
trainable objective.** Validity and trainability are different properties.

**The fix worked — with a twist that voided the earlier nulls.** A smoothed
training objective — per-turn, additive, bounded, no cliffs — cut
same-context noise five-fold while agreeing with the certified metric at
ρ=0.77 (train smooth, judge certified — the verdict cannot be gamed by the
training signal). While staging its eval, a pre-serve weight check caught
something worse than a bug: the merge tool had **never folded the LoRA
adapters** — the two earlier "null" A/Bs had compared the base model against
itself. Their ±0.01 deltas were retroactively reframed as blind measurements
of the eval instrument's noise floor, and the smooth-objective run (RL-C)
became the program's **first valid trained-arm A/B — and its first positive
result**: certified delta +0.059 (t=2.9), later honestly restated as a
+0.02–0.06 range once a day-shift experiment showed between-day sampling
variance rivals the effect sizes. Decomposed: two-thirds fewer
product-defect zeros, one-third topical grounding. The audience's laughter
did not move.

**Then the taste frontier fought back, and we mapped exactly why.** A
follow-up run (RL-D) made the laughter term dominate the objective — 0.6
weight, warm-started, laughter-bait channel closed after measuring the
exploit at +7.3 logits for appending "Haha!" — and the held-out taste delta
was +0.003 (t=0.6): decisively flat. The post-mortem measurements are the
finding: the audience roleplays the conversation partner, so
partner-emitted laughter primes its reaction (+0.10/turn, causally probed —
a lottery the policy cannot steer); 61% of all reactions sit exactly at the
construct's floor (the median turn carries zero gradient); and
policy-attributable observable variance is ~1%. A four-stage redesign
waterfall then killed every replacement candidate by measurement — the
cheap-feature surrogate (R²=0.08, its best feature was the bait channel),
the contrastive baseline (floor-censored 95.7%), and an
incongruity-theoretic gate that word-shuffled gibberish defeats, because
bag-of-words embeddings keep their topical identity when syntax dies.
Nothing survived to registration, and no GPU-hours were spent finding that
out the slow way.

**Why publish nulls?** Because the alternative is the literature this
project was built against: exciting curves, no held-out verification, and
mode collapse discovered by users. An environment plus a harness that
*reliably distinguishes real learning from its imitations* — demonstrated on
its own failures, including two evals it voided itself — is the asset.
Sixty-three evidence-backed lessons from building it are in
[`docs/ENV-BUILDING-TAKEAWAYS.md`](docs/ENV-BUILDING-TAKEAWAYS.md).

## Repo layout

- `env/` — the banter environment: rollout engine, certified reward stack +
  smooth training objective (with tests), scoring, curation, report card,
  demo-pack generator, verl training bridge, box-side keepers
- `rm/` — reward-model training and transfer evaluation (NYCC emulator)
- `benchmark/` — instruments and validators, incl. the earlier
  rejection-cascade benchmark work
- `data_adapters/` — licensed-dataset loaders (license firewall enforced)
- `references/` — verified literature corpus with evidence-strength tags
- `experiment-runs/` — archived exact scripts per experiment
- `STATE.md` — current state; `EXPERIMENT_LOG.md` — every experiment,
  registration, result, retraction, and close-out, in order
- `docs/ENV-BUILDING-TAKEAWAYS.md` — the field manual distilled from all of it

## License

None. **All rights reserved** (see [NOTICE](NOTICE)). The repository is public
for transparency, not for reuse. The `.claude/` harness tooling is
[Rockie](https://rockielab.com) (Apache-2.0) and is not part of this project's
claims.

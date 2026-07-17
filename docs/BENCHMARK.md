# Humor benchmark — specification

Status: **design, not built.** Owner: Sam Larson. Written 2026-07-16.

The benchmark is the first thing to build. Everything downstream (reward model,
RL environment, transfer measurement) depends on being able to measure whether a
model is actually funny and actually diverse — and nothing published does the
second part well.

---

## 1. The rejection cascade (the novel contribution)

**The idea, in Sam's framing:** ask a model for a joke. A cheap model responds
"I don't find that topic funny, can you make a different joke?" Repeat ~50 times.
Then measure **how similar the path of topics is — across runs of one model, and
across different models.**

The jokes are not the measurement. **The trajectory is the measurement.**

### Why this is different from everything published

Every existing diversity metric samples N outputs and measures spread. Spread can
be bought with temperature; a model reading down a memorized list at temperature
1.0 looks "diverse" to any sampling-based metric.

Path-based exhaustion cannot be faked that way. If a model walks
`cat → dog → parrot` on *every* run, its "distribution" is a lookup table, and
the cascade makes that visible in a way sampling structurally cannot.

### Related work (attack-agent verified 2026-07-16 — full analysis in `references/related-work-cascade.md`)

The idea survives, but three papers are mandatory related work:

- **Denial Prompting / NEOCODER** (arXiv:2407.09007, NAACL 2025) — the direct
  structural ancestor: iteratively denies a *technique* in code generation.
  T≈5 turns, objective constraints, scalar score — not a semantic path.
- **MUTATE / "Beyond One Path"** (arXiv:2605.28465) — closest: forces path
  divergence via a failure memory in a text-adventure env. But rejection there
  is *objective task failure*; ours is subjective, content-agnostic rejection
  on an open creative task. It never runs the cross-model comparison.
- **NoveltyBench §4.3** (arXiv:2504.05228) — "generate another" in-context,
  8 turns, set-cardinality metric (Distinct-k), no rejection framing, no path.

Cross-model *single-shot* homogeneity is already published (2501.19361;
2604.08757 humor-specific). So the claim is not "first to show LLMs converge on
humor" — it is: **first to force subjective, content-agnostic rejection on an
open social task at ~50-turn depth and treat the resulting topic *sequence* —
compared across runs and across models — as the measured object.** The
cross-model trajectory comparison exists nowhere.

Metric lineage (see `references/trajectory-grounding.md`): cluster/switch
statistics port Troyer et al. (1997) verbal-fluency scoring; the patch-walk
framing follows Hills, Jones & Todd (2012) optimal semantic foraging; category
fluency has been run on LLMs unforced and single-model (2405.06714) — the
cascade is the forced, adversarial, multi-model version.

### The three metrics

| Metric | Question | Collapse signature |
|---|---|---|
| **Within-model path divergence** (across runs) | Does the same model take the same path every time? | Low divergence = deterministic list, not a distribution |
| **Cross-model path overlap** | Do *different* models walk the same path? | High overlap = shared pretraining prior; collapse at the ecosystem level |
| **Depth-to-degradation** | How many turns until it repeats, refuses, or gets visibly worse? | Shallow depth = a small well |

Cross-model overlap is the most interesting of the three. A finding that GPT,
Claude, and Llama all walk `cat → dog → parrot` would say something about the
industry's shared humor prior, not about any one model.

### Design decisions (made)

- **The rejector labels the topic explicitly.** A trajectory must be a readable
  list of topic labels, not a vibe. This is what makes paths comparable.
- **Rejections accumulate.** Turn 50 carries all 49 prior rejections. A sliding
  window lets the model loop back to `cat` and hides the collapse we're measuring.
- **~50 turns, N runs per model, M models.** No GPU — API calls plus a cheap
  rejector (Haiku-tier).

### The load-bearing risk — validate the rejector FIRST

The rejector decides whether this measures anything at all:

1. **It must reject the TOPIC, not the joke.** If it rejects a joke, the model
   rewords the same joke and the cascade measures nothing.
2. **Its own topic prior contaminates the result.** A rejector that labels
   inconsistently makes us measure *the rejector*, not the model under test.

**First code written must be rejector validation** against a fixed transcript.
No cascade numbers are trustworthy until the rejector is shown to be consistent
and topic-focused. This is not optional scaffolding — it is the experiment's
internal validity.

---

## 1b. Track 2 — contextual banter (design sketch, 2026-07-16)

Sam: "we aren't just telling jokes — humor is contextualized." The cascade
(Track 1) is the sterile diagnostic; Track 2 measures and trains
*conversational* humor — gap #2, where nothing exists and RLVR damages the
prerequisite skill.

- **Episode = conversation, not a joke slot.** Multi-turn episodes; reward
  per-utterance plus trajectory-level bonuses.
- **Callback bonus:** reusing an entity/theme from ≥k turns earlier in a novel
  payoff. Detectable cheaply (entity overlap + novelty check vs. the earlier
  mention), and it is the most mechanical marker of real banter.
- **Context-ablation scoring (the quantifiable "contextualized" metric):**
  score(reply | true conversation) − score(reply | randomly swapped
  conversation). A canned joke scores identically under both — Δ≈0 exposes it.
  Contextual humor degrades under swap. This turns "humor is contextualized"
  into a number, and doubles as an anti-canned-joke reward term for training.
- Data that fits this shape: Oogiri (prompt-conditioned responses), LOL Arena /
  Bad Cards (contextual prompt-response with human votes).

Track 2 depends on Track 1's rejector/labeling machinery being validated, and
on the judge infrastructure. Build order unchanged: rejector validation →
cascade → banter.

### Track 2 implementation notes (`benchmark/banter.py`, 2026-07-17)

- **The judge is an LLM.** Unlike the cascade's rejector — which only labels
  a topic word, a narrow, low-stakes judgment — context-ablation scoring asks
  a judge model for a 1-10 funniness/fit rating. That means Track 2 inherits
  the LLM-judge reward-hacking risk that Track 1's topic-labeling design
  mostly avoids (`.claude/skills/humor-rl/SKILL.md`).
  **Mitigation:** the reported metric is never the raw judge score, it is the
  *delta* between two calls to the same judge with the identical rubric
  (`JUDGE_PROMPT`), differing only in which context block is shown. Any
  judge tendency that is constant across contexts — a scale bias ("always
  says 7"), a leniency bias, a length preference — cancels in the
  subtraction. A judge-hackable *policy* therefore can't win by pushing the
  absolute score up; it has to specifically make the true-context score
  beat the swapped-context score.
  **Residual risk, stated plainly:** this does NOT cancel a judge whose bias
  is itself context-dependent — e.g. a judge that scores any reply
  mentioning "traffic" higher when the context also mentions "traffic",
  independent of whether the reply is actually responsive. A model that
  learns to sprinkle context-echoing keywords into an otherwise
  context-blind reply could inflate delta without truly being in-context.
  This is exactly the kind of failure mode `.claude/skills/humor-rl/SKILL.md`
  warns judge-alone rewards produce; a real Track 2 reward stack should not
  use context-ablation delta as the sole signal, the same way Track 1
  should not use judge score alone.

- **Swapped-context sampling rule.** The swap partner for episode `i` is
  episode `(i + 1) % n_episodes` (`swap_partner()`), wrapping the last
  episode back to the first. Fixed and index-based, not random: a random
  swap would make a pilot run non-reproducible from its own logs, which
  cuts against this repo's standing requirement that every number trace
  back to an exact script and dataset. The swap is taken **at the same
  turn index** — episode A's reply at turn 3 is scored against episode B's
  conversation truncated to its own turn 3 — so the swap changes *which
  conversation* the reply is dropped into without changing *how much*
  conversation has accumulated, keeping context length roughly matched
  between the true and swapped conditions.

## 2. Entropy-collapse penalization

Sam: entropy collapse should be penalized in the benchmark, not merely reported.
`grpo-rl-training` treats `reward_std` as a number to *watch*; the whole thesis
here is that for humor it must be a term to *optimize*. See
`.claude/skills/humor-rl/examples/humor_reward_functions.py::intra_group_diversity_reward`.

## 3. Internet-joke similarity penalty

Sam: find the jokes already told on the internet and penalize similarity to them.
This is what `corpus_novelty_penalty` implements — but it is inert without a real
scraped corpus behind it. **Building that corpus is a prerequisite, not a
detail.** Without it the anti-collapse layer does nothing.

Known target: >90% of 1,008 ChatGPT jokes were the same 25 templates
(Jentzsch & Kersting, WASSA 2023 — exact figure 909/1008 = 90.2%). Those 25
templates are the minimum viable memorized-joke corpus.

## 4. Grounding

Sam: the benchmark should draw on **psychology, philosophy, sociology, NLP, and
quantifiable metrics** — not vibes. Relevant, verified, in `references/`:

- **Benign Violation Theory** (McGraw & Warren) and **incongruity-resolution**
  (Suls) converge on a dual-appraisal structure — a candidate reward architecture,
  not just a taxonomy. See `references/psychology.md`.
- **Humor production ↔ intelligence**, r ≈ .29–.40 (Greengross & Miller 2011;
  Christensen et al. 2018 — primary full text confirmed). This is the basis of
  the transfer thesis.
- **Annotator agreement on funniness is κ = 0.41** (Sun et al. 2022,
  ExPUNations). NOT 0.49 — that figure is agreement on pun-word *semantic
  validity*, a narrower judgment. Cite 0.41. Labels are noisier than commonly
  claimed, so treat "gold labels" as samples from a preference distribution.
- **Oogiri-Master / Oogiri-Corpus** — ~100 candidates per prompt × ~100
  independent raters, explicitly designed to remove popularity bias. Cleanest
  humor reward-model data in existence. **Oogiri-GO is downloadable now.**

---

## 5. What exists and what does not

**Usable today:** DARLING (Meta — mode collapse), RAGEN/StarPO (multi-turn),
verl, NYCC (250M ratings / 2.2M captions), Oogiri-GO. See
`references/code-and-models.md`.

**Not usable:** every core humor-generation paper — HumorBench, HumorGen,
HumorRank, CLoST, IRS — has **zero released code, data, or weights.** There is no
baseline to fork. We build.

**Published negative results to beat, not repeat** (`references/negative-results.md`):

- HumorGen: neither DPO nor offline-GRPO beats a well-curated SFT baseline.
- NYCC paper: RLHF/DPO limitations on creative tasks; SFT regressed *below*
  zero-shot; frontier models still underperform top human contestants.
- Two distinct LLM-judge reward-hacking collapses, including one where hardening
  the rubric *shifted the direction* of the hack rather than resolving it.
- RLVR documented to *damage* multi-turn conversation (ICPO, UFO papers) —
  which is the skill the cascade benchmark requires. This cuts against the grain
  of current training and is a publication hook.

## 6. The three open gaps this serves

1. **Reverse transfer** — train on humor, measure MMLU/GPQA. Untested by anyone.
   HumorBench confirmed only the forward direction (STEM reasoning → humor
   comprehension).
2. **Multi-turn conversational humor environments** — nothing exists. The
   cascade benchmark is inherently multi-turn.
3. **Diversity-preserving RL against live human preferences** — attempted with
   standard tools, failed, failures published.

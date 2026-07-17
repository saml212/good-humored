# The Rejection Cascade: A Path-Based Benchmark for Diversity Collapse in LLM Humor Generation

**Status:** draft skeleton, pilot in flight. Target: NeurIPS/ICLR Datasets & Benchmarks track.
Numbers marked `[RESULT]` are placeholders — do not fill until the pilot (EXP-004,
`experiment-runs/2026-07-17-cascade-pilot/`) completes and the adversarial audit
signs off on scoring. Sections marked `[PENDING]` are structurally drafted,
content-empty by design.

---

## 1. Abstract

Diversity evaluations for language models sample N outputs and measure spread —
but spread can be bought with temperature: a model reading down a memorized
list at temperature 1.0 looks diverse to any sampling-based metric. We
introduce the **rejection cascade**, a benchmark that instead forces a model
off its default answer through ~30–50 turns of accumulating, subjective,
content-agnostic rejection ("I don't find that topic funny — try another"),
and measures the resulting **topic trajectory** rather than the jokes
themselves. Path-based exhaustion under accumulating denial cannot be faked by
sampling temperature. We ground three metric families — within-model path
divergence, cross-model path overlap, and depth-to-degradation — in the
verbal-fluency and semantic-foraging literatures, and report an
instrument-validation arc showing raw topic labels, not embedding-clustered
ones, are the safer primary metric. Pilot: 12 models, 4 provider families,
depth 30, N=4. `[RESULT: cross-model Jaccard overlap]`,
`[RESULT: within-model run-to-run divergence]`, `[RESULT: fraction of models
degrading by turn 30]`.

*(Word count: ~150 — see report.)*

---

## 2. Introduction

Benchmarks for language-model diversity almost universally take the same
shape: sample N completions from one prompt, measure how different they are
from each other. This works when diversity is the thing being sampled. It
fails when diversity is the thing being *performed*. A model that has
memorized twenty-five joke templates and reads down the list at temperature
1.0 will look, to Distinct-k, self-BLEU, or any embedding-spread metric, like
a genuinely diverse generator — because those metrics measure the shape of a
single unconditioned sample, and a lookup table sampled at temperature can
have exactly the shape of a distribution. This is not a hypothetical failure
mode: 90.2% of 1,008 ChatGPT-generated jokes were, word for word, one of 25
recurring templates (Jentzsch & Kersting, 2023) — under a flat, repeated
"tell me a joke" prompt with no forcing function at all. Sampling-based
diversity metrics have no mechanism that would have caught this, because
nothing about the sampling procedure interrogates whether the model *could*
have produced something else.

Our thesis: a metric that forces the model off its first, second, and
twentieth answer — through **accumulating, subjective, content-agnostic
rejection** — cannot be gamed the same way. If a model's real generative
capacity for a topic is a short, ordered list (`cat → dog → parrot`), forcing
it past `cat` and `dog` on every independent run will expose the list,
because there is nowhere left for a memorized ranking to hide. Spread over
independent samples answers "how different are ten things drawn from the
same distribution." Path-based exhaustion under forced denial answers a
different, harder question: "how much distribution is actually there once
the first several answers are taken away." We call this *rejection cascade*
protocol, and the object we measure is the **sequence of topics** the model
is pushed through, not the jokes it tells along the way — compared across
independent runs of one model, and across different models.

We choose humor as the domain deliberately, not incidentally. Two properties
make collapse in humor a uniquely clean and uniquely consequential test case.
First, mode collapse *is* the failure, not a side effect that costs some
percentage of a diversity score: a joke heard twice is dead, so a benchmark
that cannot detect collapse in this domain cannot detect the thing that
matters most about it. Second, humor is an unusually compressed behavioral
probe of capacities alignment work already cares about, though we are careful
about how far this claim extends. Three independent literatures, each
individually well-supported, converge on the same three prerequisites for
"getting" and generating a joke: an accurate world model (HumorBench shows
STEM-reasoning training transfers into humor *comprehension*; Narad et al.,
2025), a working theory of mind (the ToM-HCAT humor-comprehension test and
LLM-side ToM benchmarks such as ToMBench), and explicit norm-awareness — the
capacity to represent both what "ought" to be true and that a stated
violation of it is not threatening (Benign Violation Theory; McGraw &
Warren, 2010). These three links are each independently established; what is
**not** established, and what we do not claim here, is that training a model
*on* humor causes general improvement in any of the three — that causal
direction is untested by anyone, HumorBench having confirmed only the
forward direction (reasoning capability transferring into humor
comprehension, not the reverse). We treat the alignment framing as
motivation for why humor is a domain worth measuring carefully, not as a
result this paper reports.

The rest of the paper: Section 3 places the cascade against the closest
published precedents, none of which we find to be a kill — but two (Denial
Prompting, MUTATE) are close enough that the differentiation must be made
explicit and load-bearing, not a polite footnote. Section 4 gives the
protocol, the metric family, and — as a first-class part of the method, not
an appendix — the four-experiment arc by which we validated the measurement
instrument (the rejector) before trusting any cascade number. Section 5
reports the pilot design and its known confounds; results are pending.
Section 6 is an honest limitations section, because several of those
limitations bound what any downstream claim in Section 5 can support.
Section 7 previews the reverse-transfer question this benchmark exists,
ultimately, to make askable.

---

## 3. Related Work

### 3.1 Humor generation and its documented failures

Every naive approach to LLM humor generation that has been tried and
published has a documented failure mode, and any new humor benchmark or
training method is answerable to these results, not just to the literature's
successes. **Mode collapse onto memorized jokes** is the load-bearing case:
Jentzsch & Kersting (2023, WASSA) found 909/1,008 (90.2%) of ChatGPT's
outputs under a flat repeated prompt were one of 25 pre-existing,
web-findable joke templates, with the top four templates alone covering more
than half of all output. **LLM-judge reward hacking** is separately
documented in the open: a GRPO run against a GPT-4.1 funniness judge
collapsed into classic-joke regurgitation under a naive 1–5 rubric, and
hardening the rubric did not remove the hack — it *relocated* it, first to
irrelevant appended "bonus jokes" the judge over-scored, then to bizarre
non-sequiturs under a different rubric (documented, non-peer-reviewed
write-up; treated here as a real, reproducible-in-spirit observation, not a
peer-reviewed result). **Preference optimization has been tried against
strong SFT baselines and has not won**: HumorGen (Ajayi & Mitra, 2026) reports
that neither DPO nor offline-GRPO beats a well-curated SFT baseline on their
own Humor Transfer Bench, with overlapping 95% confidence intervals across
two independent judge models, and states plainly that this reflects "a data
quality ceiling," not an optimization failure. The New Yorker Caption
Contest paper (Zhang et al., 2024, NeurIPS D&B) — 250M+ human ratings on
2.2M+ captions — reports the same qualitative pattern from the RLHF/DPO side:
SFT regressed *below* zero-shot, and frontier models still trail top human
contestants even though they beat the median. A follow-up (Zhou et al., 2025,
EMNLP Findings) shows the cheapest plausible fix for audience-dependent humor
— persona-conditioned prompting — has "minimal impact," and only small-scale
fine-tuning on real human preference judgments closes the gap. We treat all
of the above as the baseline any humor-diversity or humor-training claim must
clear, not as settled science to build on uncritically.

### 3.2 Diversity and novelty evaluation

Three papers converge closely enough on the cascade's mechanics that a
reviewer familiar with this literature would name them unprompted, and each
requires an explicit, load-bearing differentiation rather than a citation in
passing. **Denial Prompting / NEOCODER** (Lu et al., 2024/2025, NAACL,
arXiv:2407.09007) is the direct structural ancestor: it iteratively forbids a
code-generation solution from reusing an *atomic technique* it just used, for
T≈5 rounds, and scores the resulting trajectory of correctness ×
constraint-adherence — the "measure the path, not the endpoint" spirit is
identical. It differs on every axis that matters for our claim: the
constraint is a *technique* extracted programmatically from the model's own
last answer (an objective, mechanical denial), not a human-legible topic; the
task is competitive programming, with a correctness signal, not an open
social/creative one; and the depth is an order of magnitude shallower than
ours. **NoveltyBench** (Zhang et al., 2025, arXiv:2504.05228) §4.3 runs the
closest same-conversation "give me another" mechanic — up to 8 in-context
regenerations, all prior answers kept in context — but with no rejection or
denial framing at all (the model is simply asked to try again, never told an
answer is unacceptable), and its metric (Distinct-k, an equivalence-class
count over a DeBERTa-scored similarity function) is a set cardinality, not a
sequence. **MUTATE** (Park et al., 2026, arXiv:2605.28465) is the single
closest paper found in either direction: an escape-room agent accumulates a
per-object failure memory and is explicitly pushed toward alternative
approaches once a target's failure count crosses a threshold, with a metric
family (Path Discovery, Divergence Momentum) built on the same premise that
path-level and action-level divergence can fail independently. The
differentiating facts, all four load-bearing: (i) MUTATE's rejection is
*objective task failure* (the crafted item did not work); ours is a
subjective, content-agnostic verdict with no ground truth ("I don't find that
funny"); (ii) MUTATE's failure memory is *per-object*, reset per target;
ours is a single global list that accumulates monotonically across the
entire conversation; (iii) MUTATE's domain has no notion of a semantic
"topic" category to track; and (iv) MUTATE never runs a cross-model
comparison, despite reporting per-model results — the cross-model path-overlap
question, which we argue is the most interesting of our three metrics, exists
in none of the three papers above. The sharpest honest differentiation
sentence, quoted from our own pre-registration audit: prior iterative-forcing
methods constrain either a *technique* on a single conversation at ~5-step
depth with an objective correctness signal (Denial Prompting), or *actions*
via per-object failure memory in a deterministic puzzle environment without
cross-model comparison (MUTATE); we apply a subjective, content-agnostic
rejector to an open social-creative task for an order of magnitude more
turns, and treat the resulting topic sequence itself — compared both within
one model across runs and across different models — as the object of
measurement.

### 3.3 Cognitive-science lineage

The cascade's metric family is not invented from nothing; it ports a
methodology with forty years of prior use scoring *human* topic trajectories
under forced exhaustion. The verbal (category) fluency task — name as many
animals as you can in a fixed time — is scored, classically, into two
dissociable components: **clustering** (mean run-length of semantically
related consecutive items) and **switching** (count of transitions between
clusters) (Troyer, Moscovitch & Winocur, 1997). These two numbers dissociate
from raw item count in clinical populations (they distinguish frontal- from
temporal-lesion patients, and different dementia subtypes) precisely because
raw output volume can look healthy while the underlying search strategy is
degenerate — the same logic that motivates measuring topic sequences instead
of joke counts here. Hills, Jones & Todd (2012, *Psychological Review*) frame
the same task as **optimal foraging in semantic memory**: humans search
within a semantic "patch" until its local yield drops to the task's running
average yield (the Marginal Value Theorem, imported from foraging ecology),
then jump to a new patch — and patch-departure timing that better matches
this optimal-foraging rule predicts *higher* total recall. This gives our
depth-to-degradation metric a principled null model: a real generative
capacity should show patch-like clustering with foraging-consistent
departure timing; a memorized list forced into rejection should degrade
early and abruptly, the LLM analog of a forager that cannot find a next
patch. Category fluency has already been ported to language models directly
— Heineman, Koenen & Varma (2024, arXiv:2405.06714) model an LLM's
unforced, free-generation fluency sequence and score it against human
sequences via n-gram overlap — but this is single-model, single-category,
and unforced: no rejector, no adversarial exhaustion, no cross-model
comparison. The cascade is, relative to this line of work, the forced,
multi-model, humor-domain extension the foraging literature has not yet
run.

### 3.4 Cross-model homogeneity

Cross-model convergence in creative output is already a published, not a
novel, finding — our contribution cannot be "LLMs converge on humor," only
"they converge on the *same forced escape route* under identical adversarial
pressure," which is a different and stronger claim. Wenger & Kenett (2025,
arXiv:2501.19361) administer standardized creativity tests single-shot
across multiple LLMs and find LLM responses are far more similar to each
other than human responses are to each other. In the humor domain
specifically, Fettach et al. (2026, arXiv:2604.08757, "Cards Against LLMs")
have five frontier models *select* (not generate) the funniest response
across 9,894 Cards-Against-Humanity rounds and find models agree with each
other substantially more than they agree with humans. Both findings support
the mechanism we test — a shared pretraining prior producing shared humor
output — but both are single-shot and unforced. Neither asks whether that
convergence survives, or intensifies, once every model is pushed down the
same ~30-turn adversarial denial gauntlet; that is the question path
overlap is built to answer.

---

## 4. Method

### 4.1 The cascade protocol

A model under test is asked for a joke. A separate, cheap rejector model
responds with a content-agnostic verdict — "I don't find that topic funny,
tell me a different joke" — and **explicitly labels the topic it is
rejecting** (e.g., `cat`), so that the trajectory is a readable list of topic
labels rather than a vibe. Rejections **accumulate**: turn *k* carries the
topic labels of all *k*−1 prior rejections, not a sliding window. This design
choice is load-bearing, not incidental — a sliding window would let a model
loop back to an early topic (`cat`) once it aged out of the window, hiding
exactly the exhaustion the benchmark exists to detect. The cascade runs to a
fixed depth (pilot: 30 turns) for N independent runs per model (pilot: N=4),
with no GPU required — API or CLI calls to the model under test plus a
Haiku-tier rejector.

### 4.2 Three metric families and their psychological lineage

**Within-model path divergence** (across independent runs of the same model)
asks whether the model takes the same topic path every time it is run.
Near-identical paths across runs are the direct LLM analog of what would be
a pathological result in the human clustering/switching paradigm: a healthy
forager's run-to-run search pattern varies with the specific patches it
happens to enter first, so a rejection-cascade "distribution" that is
actually a fixed list should reveal itself as near-zero run-to-run
divergence, not high spread.

**Cross-model path overlap** asks whether *different* models, forced down
the same rejection gauntlet, walk the same escape route. This is the metric
we consider most diagnostic and the one with no precedent in either the
diversity-eval or the foraging literature (Sections 3.2–3.4): high overlap
under forced adversarial pressure is a stronger and more specific claim than
single-shot output similarity, because it holds even when the sampling
distributions of the underlying models are, by construction, being pushed
away from their unconditioned mode.

**Depth-to-degradation** asks how many turns elapse before a model repeats
an already-rejected topic, refuses outright, or produces visibly
lower-quality output. This is our port of Marginal Value Theorem
patch-departure timing (Hills, Jones & Todd, 2012): a shallow
depth-to-degradation is the signature of a forager (model) that cannot find
a next patch — a small well, not a distribution.

### 4.3 Instrument validation as first-class methodology

No cascade number is trustworthy until the rejector — the instrument that
labels every trajectory — is shown to reject *topics*, not jokes, and to
label consistently enough that a trajectory comparison measures the model
under test rather than the rejector's own labeling noise. We treat the
validation of this instrument as a methods contribution in its own right,
and report the full arc, including its two dead ends, rather than only the
calibrated result — a benchmark paper that reports only its final instrument
hides exactly the failure modes a reviewer most needs to check.

**EXP-001 (rejector v1).** Pre-registered bars: ARI ≥ 0.80 against a
32-item hand-built fixture (10 topic groups × original/reworded/same-topic
variants, plus 2 deliberately ambiguous traps), reworded-pair label
invariance ≥ 0.90, and a built-in disproof attempt — a crude keyword
(most-frequent-non-stopword) baseline run on the identical fixture. Result:
failed both absolute bars (ARI 0.620, invariance 0.600) but beat the
baseline decisively (baseline ARI 0.271; predicted delta +0.35, actual
+0.349 — the quantitative prior closed correctly even though the instrument
did not yet pass). Diagnosed failure modes were specific and benign:
synonym scatter (`fitness`/`exercise`/`gym` all topically correct but
string-distinct, splitting the ARI partition), one prompt-format parse bug
(a joke containing an internal colon broke the label delimiter), and two
fixture items with defensibly opinionated gold labels. Critically, **zero**
labels showed punchline-mechanism contamination — the topic-vs-joke
discrimination the whole benchmark depends on held even in the failing
version.

**EXP-002 (rejector v2).** A revised label prompt (delimited joke input,
explicit "most generic common noun" instruction, two added
generalize-upward few-shot examples) lifted ARI to 0.837 (bar 0.80, met;
predicted +0.23, actual +0.217) but left reworded-pair invariance at 0.800
against a 0.90 bar. The two residual invariance misses were `cat`/`pet` and
`health`/`medicine` — semantically identical labels failing on string
equality alone.

**EXP-003 (semantic re-scoring — negative, twice).** Rather than a third
prompt iteration, we asked whether scoring EXP-002's existing labels under a
semantic-equivalence layer (embedding similarity, all-MiniLM-L6-v2, threshold
calibrated on a separate 64-pair fixture) would close the invariance gap at
zero additional API cost. Union-find clustering lifted invariance to 0.900
but a **hypernym** (`pet`) bridged `cat` and `dog` into a single false-merged
cluster — because both exceeded the similarity threshold against `pet`
independently — crashing ARI from 0.837 to 0.659. This is the one failure
mode the benchmark cannot tolerate: a false merge manufactures collapse
evidence, inflating apparent topic-overlap where none exists. A
complete-linkage rewrite, which makes hub-chaining structurally impossible,
correctly split `cat` and `dog` but then assigned `pet` to whichever had
higher pairwise similarity, producing a different partial error (ARI 0.697,
invariance back to 0.800). Conclusion: `pet` is a hypernym relative to `cat`
and `dog`, not a synonym, and no flat clustering method places a hypernym
correctly. We keep complete linkage as strictly safer than union-find, but
demote the semantic layer to a *reported-alongside* view — never the primary
metric.

**EXP-003b (bigger model as rejector — negative).** We tested whether a
larger model (Sonnet) makes a more consistent rejector than Haiku.
Prediction: +0.06 ARI. Result: Sonnet is a **worse** instrument (ARI 0.633
vs. Haiku's 0.837; invariance 0.700 vs. 0.800) — a large negative surprise
against the registered prior. A richer model labels with a richer
vocabulary, which increases granularity variance, the opposite of what a
measurement instrument needs. Haiku is retained not despite being the
cheaper model, but because its narrower vocabulary is a better-behaved
label space.

**Instrument decision (pilot grade).** Haiku + LABEL_PROMPT v2, scored on
raw label strings (not the semantic layer), is the cascade's instrument.
It passes the ARI bar, decisively beats the keyword baseline, and shows zero
punchline-mechanism contamination across every validation run — the
load-bearing property. It does **not** pass the pre-registered invariance
bar (0.800 vs. 0.90), and we record this as an unmet criterion rather than
relaxing the bar post hoc. We proceed to pilot anyway because the residual
failure — granularity jitter on near-synonym pairs — has a known and
*conservative* bias direction for the benchmark's central claim: label noise
splits topics that should be counted as one, which can only make a model
look *more* diverse than it is, never less. Any collapse (low-overlap,
low-divergence) finding from the pilot survives this noise as a lower bound;
any diversity (high-divergence, low-overlap) finding must be reported with
the caveat that it may be partly a labeling-noise artifact. The paper-grade
fix on the roadmap — constrained-vocabulary, two-pass labeling — is
discussed in Section 6.

### 4.4 Memorized-joke novelty check

Any generation eval in this domain must check output against a
memorized-joke corpus, because mode collapse onto memorized jokes is the
single best-documented failure mode in the literature (Section 3.1): the
25 templates covering 90.2% of Jentzsch & Kersting's (2023) ChatGPT sample
are the minimum viable corpus for this check. This mechanism is a design
requirement, not yet a completed feature: it is inert without a real scraped
corpus of internet jokes behind it, and building that corpus (target scale:
~3.1M jokes across the sources catalogued in `references/corpus-sources.md`)
is a stated prerequisite for the cascade's benchmark results to be reported
as paper-grade, not an implementation detail to defer silently.

---

## 5. Experiments

### 5.1 Pilot setup

Twelve models across four access lanes: four via the `claude` subscription
CLI (haiku, sonnet, opus, fable), three via the `codex` CLI (sol, mini,
5.4), four via direct provider APIs (deepseek, qwen, glm, kimi), and one via
a direct API added as a fourth lane once a working key became available
(grok). Depth 30, N=4 runs per model, rejector = the Haiku instrument
validated in Section 4.3 at pilot grade (known limitation: invariance 0.800,
conservative bias direction as argued above). Primary metrics are computed
on raw labels; the semantic (complete-linkage) label space is reported
alongside but is never primary, a constraint enforced in code after an
adversarial pre-launch audit caught an earlier version of the scoring path
computing the primary metric from canonicalized (semantic-merged) labels —
which, because merges only ever *reduce* apparent topic count, could only
have inflated the headline collapse finding. Other pre-launch audit fixes:
API-key-fragment scrubbing in error paths, CLI error-text capture, and an
explicit `UNPARSEABLE` sentinel for rejector outputs that fail the expected
label format rather than silently coercing them.

**Predicted deltas (registered before launch):** cross-model mean
topic-set Jaccard overlap (raw labels) ≈ 0.35; within-model mean set
Jaccard across runs (raw labels), averaged over models, ≈ 0.55 (i.e., models
are predicted to repeat themselves across runs more than they match each
other); at least a third of the roster degrading (repeating an
already-rejected topic, or refusing) by turn 30.

### 5.2 Wrapper-confound limitation and planned ablation

The four access lanes are not methodologically equivalent, and this is a
confound in the current pilot, not a detail. The `claude` and `codex` lanes
run through subscription CLI wrappers with no exposed temperature control,
and multi-turn context is encoded as a transcript re-injected into a single
prompt rather than a native multi-turn message array; the `api` and `grok`
lanes call provider APIs directly, with temperature control and native
multi-turn state. A model's apparent topic-path behavior is therefore
partly a function of *which lane it happens to run in*, not purely a
function of the model. The planned ablation is to run at least one model
through both a CLI-wrapper path and a direct-API path where both are
available, holding the model fixed, and measure whether path-divergence and
overlap metrics shift with the access method alone — quantifying the
wrapper confound directly before any cross-family comparison in the full
roster is treated as a claim about the *models* rather than about the
*wrappers*.

### 5.3 Results

`[PENDING]` — cascade pilot (`experiment-runs/2026-07-17-cascade-pilot/`) is
in flight; per-lane run counts are uneven as of this draft (e.g., the
`grok` lane has one completed run of four). Tables to fill once complete
and audited:

- **Table 1:** within-model mean set Jaccard across N=4 runs, per model
  (raw labels; semantic-layer values reported alongside).
- **Table 2:** cross-model 12×12 path-overlap matrix (raw labels).
- **Table 3:** depth-to-degradation per model (turns to first repeat or
  refusal; censored at 30 if none).
- **Table 4:** fraction of roster degrading by turn 30, vs. the
  pre-registered ≥1/3 threshold.

No result numbers are asserted in this draft.

---

## 6. Discussion and Limitations

**CLI wrapper confound.** As above (5.2) — three of four lanes lack
temperature control and encode multi-turn state non-natively. This bounds
every cross-family claim in the current pilot until the planned ablation
runs.

**No temperature control in the pilot.** Because the CLI-driven lanes
expose no temperature parameter, we cannot currently ask "does the cascade
finding hold across temperature settings" for those models — only for the
directly-API-called ones. A paper-grade version needs either uniform direct
API access across the whole roster or an explicit, reported temperature
sweep on the subset where it is possible.

**N=4 pilot scale.** Four runs per model is a protocol shakeout, not a
statistically powered comparison. Within-model divergence estimated from
four runs has wide, unreported uncertainty; we treat the pilot's purpose as
validating the protocol and establishing effect *direction*, not as
supplying paper-grade numbers.

**Rejector granularity jitter.** The validated instrument misses its own
pre-registered invariance bar (0.800 vs. 0.90; Section 4.3). We have argued
the residual noise's bias direction is conservative for collapse claims —
noise splits topics, inflating apparent diversity — but this argument has
not itself been independently stress-tested (e.g., by injecting known
synthetic collapse into a fixture and confirming the instrument does not
under-report it). That check belongs on the roadmap alongside the labeling
fix below.

**Free-vocabulary labeling roadmap.** The core unresolved problem (Section
4.3, EXP-003) is that free-vocabulary labels admit hypernym relationships
(`pet` over `cat`/`dog`) that no flat clustering method — semantic or
string — handles correctly. The planned paper-grade fix is a constrained,
two-pass labeling scheme: an unconstrained first-pass label, then a second
pass that maps the free label onto a fixed, pre-specified topic vocabulary
(or the nearest entry in one), removing both the synonym-scatter failure of
raw strings and the hub-bridging failure of embedding clustering by
construction, at the cost of imposing a vocabulary a reviewer could
reasonably contest as a research-design choice in itself.

---

## 7. Future Work: The Reverse-Transfer Agenda

The cascade benchmark exists, ultimately, in service of a question it does
not itself answer: does training a model to be funnier — including training
against a cascade-shaped diversity signal rather than a naive judge-only
reward — transfer back into general reasoning or taste, the direction
opposite the one HumorBench has confirmed (STEM-reasoning training
transferring into humor comprehension)? Given the human literature's
robust, if one-research-program-heavy, correlation between humor production
ability and general/verbal intelligence (r ≈ .29–.40; Greengross & Miller,
2011; Christensen et al., 2018), and given that no published work has run
the reverse-transfer direction in either humans or models, this is a clean,
answerable, and — as far as this project's literature review has
established — unclaimed question. It depends on this benchmark (or a
successor with a validated instrument and a working novelty-penalty corpus)
existing first, as the pre-registered, collapse-resistant measurement
against which any claimed "humor training" would need to be shown not to
have just re-discovered the twenty-five templates. We treat it as a
separate paper, not a section of this one.

---

*End of draft skeleton.*

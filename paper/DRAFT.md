# The Rejection Cascade: A Path-Based Benchmark for Diversity Collapse in LLM Humor Generation

**Status:** pilot complete, adversarially audited, and hostile-review corrected
(2026-07-17). Target: NeurIPS/ICLR Datasets & Benchmarks track. The eleven-model
cascade pilot (EXP-004, `experiment-runs/2026-07-17-cascade-pilot/`) has
concluded, grounded by the instrument-validation arc in Section 4 (EXP-001
through EXP-003b on the rejector, EXP-006 through EXP-009 on noise bias,
temperature-fakeability, and novelty detection; EXP-005 validates a separate,
second instrument not used in this pilot's numbers). Every number in this
draft traces to `docs/FINDINGS.md` and `experiment-runs/2026-07-17-
cascade-pilot/stats_inference.json`; scale, wrapper-confound, and instrument-
precision limitations are reported in Section 6, not hidden. **Framing note
(hostile-review verdict, carried into this draft):** the citable core of this
pilot is narrower than a first read of Section 5 suggests — the two
pre-registered misses (Section 5.3.1), the family-level degradation contrast
in its two robustness-surviving forms (haiku dropped, meta-register labels
excluded — Section 5.3.3), and grok's now-triangulated profile (path-level
self-similarity, exact-match memorization, and template-trigram memorization
all independently pointing the same way — Section 5.3.2). The four-fingerprint
taxonomy below is reported as a **descriptive summary that generates
registered predictions for a replication roster**, not as four independently
confirmed findings in its own right; only the family contrast and grok's
profile currently carry statistical weight beyond description.

---

## 1. Abstract

Diversity evaluations for language models sample N outputs and measure
spread, but spread can be bought with temperature: a model reading down a
memorized list at temperature 1.0 looks diverse to any sampling-based
metric. We introduce the rejection cascade, a benchmark that forces a
model off its default answer through accumulating, content-agnostic
rejection ("I do not find that topic funny, tell me a different joke"),
and measures the resulting topic trajectory rather than the jokes
themselves. In an eleven-model pilot across four provider families (depth
30, N=2 to 4), cross-model topic overlap is 0.113, below the entire range
of a pooled-frequency null built from the pilot's topic vocabulary;
within-model overlap across runs of one model is 0.208. Both invert the
pre-registered prediction that models share a pool and repeat themselves
more than they diverge from each other — these two misses are this
pilot's cleanest claims, being the only ones registered before any data
existed. A disclosed post-hoc contrast separates two labs by 13.17 turns
of degradation depth (p = 0.0002), robust to dropping the dual-role
rejector model (11.42 turns, p = 0.0005) and, in a further robustness
check, to excluding meta-register labels ("comedy," "joke") from repeat
detection (8.79 and 7.75 turns respectively, both p < 0.002) — the
strongest exploratory result this pilot supports. One model, grok,
triangulates a distinct profile across three independent signals: the
highest within-model path self-similarity in the roster, the highest
exact-match memorization rate (40.9%), and the highest template-trigram
memorization rate (20.7%) — a fixed, heavily memorized retrieval
repertoire, not a wide distribution. Nine of eleven models degrade by
turn 30; we describe this as four per-lab fingerprints (near-uniform
constraint collapse with one within-family outlier; adherence with
uneven memorization; adherence through a small, heavily memorized
repertoire; fast degradation with low memorization among open-weight
models), reported as a descriptive typology that generates registered
predictions for a replication roster, not as four independently
confirmed results. Five further experiments validate the instruments:
temperature-unfakeability, a constrained-vocabulary labeler, and a
semantic novelty check. We report every limitation the pilot carries,
including a wrapper confound bounded by direct transcript evidence, an
uncorrected pairwise battery, and a corrected dual-tier memorization
analysis that surfaces a style confound in exact-match scoring.

*(Word count: 376 (recomputed directly, not estimated) — grew from the
pilot draft's 230 to state the hostile-review reframing and the
meta-register/dual-tier robustness checks explicitly. Left untrimmed
deliberately: this fix wave's brief is to fold in verified findings
honestly, not to protect a word-count target at the expense of dropping
one of them; a submission pass would need to cut this back down, and
that trim is flagged here as outstanding work, not hidden by a stale
count.)*

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
an appendix — the experiment arc by which we validated the measurement
instruments (the rejector, the labeler, and the temperature-fakeability
claim) before trusting any cascade number. Section 5 reports the pilot
design, its known confounds, and the resulting eleven-model fingerprint
results. Section 6 states every limitation plainly, because several of
those limitations bound what any downstream claim in Section 5 can
support. Section 7 previews the reverse-transfer question this benchmark
exists, ultimately, to make askable.

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
responds with a content-agnostic verdict — "I do not find that topic
funny, tell me a different joke" — and **explicitly labels the topic it
is rejecting** (e.g., `cat`), so that the trajectory is a readable list of topic
labels rather than a vibe. Rejections **accumulate**: turn *k* carries the
topic labels of all *k*−1 prior rejections, not a sliding window. This design
choice is load-bearing, not incidental — a sliding window would let a model
loop back to an early topic (`cat`) once it aged out of the window, hiding
exactly the exhaustion the benchmark exists to detect. The cascade runs to a
fixed depth (pilot: 30 turns) for N independent runs per model (pilot
target: N=4; realized N=2-4, Section 5.1), with no GPU required — API or
CLI calls to the model under test plus a Haiku-tier rejector.

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

**The roadmap fix has since landed (EXP-008).** A closed, 110-entry topic
vocabulary (LABEL_PROMPT v3) that deliberately omits the hypernym/synonym
pairs responsible for EXP-003's failure (`pet` sitting above `cat` and
`dog`; `medicine` above `health`) clears every bar the instrument has ever
been held to, including the one v2 missed: reworded-pair invariance
**1.000** (bar 0.90), ARI-vs-gold **0.924** (bar 0.80), repeat consistency
**0.958**. The registered swing risk — that species-level granularity would
let a straddling fixture item (a flamingo joke) scatter across
`bird`/`animal`/`marriage` — did not materialize; majority labels are
canonical across the fixture, including that item. Constrained-vocabulary
labeling (v3) is the paper-grade instrument. The pilot reported in Section 5
ran on v2, for the reasons argued above; a v3 post-hoc relabel of the
pilot's already-collected jokes (zero additional API calls) is queued as a
robustness check (Section 6).

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

**EXP-009 adds a semantic tier against the same corpus-coverage ceiling.**
An n-gram-overlap check against the reference corpus misses a documented
evasion: a two-word reskin of a memorized template (changing `cat` to
`dog`) drives exact and near-exact overlap to zero while leaving the
joke's mechanism unchanged. An embedding-similarity tier (all-MiniLM-L6-v2
cosine, scored against the 25 templates) closes this gap: at a threshold
of 0.38 (calibrated for false-positive rate ≤0.05), paraphrase detection is
**1.000** at edit depths 1 and 2, **0.810** at depth 4, and **1.000** on a
hand-written full-paraphrase set, against 1.0/0.64/0.0/0.0 for the n-gram
baseline scored on the identical set. The first validation attempt
reported 0.0 detection at every depth and read as a negative result; a
fresh adversarial verification traced this to two validation-harness
artifacts, not a property of the tier itself — a held-out-set leak
(negatives independently resampled from the same pool feeding the
calibration corpus) and a reference-set mismatch (the semantic sweep was
calibrated against templates plus 50,000 general-corpus rows while the
n-gram baseline used templates alone, so the two tiers were not scored
against the same reference). Both are fixed in the reported numbers above,
and the general methodological point outlives this one experiment: a
detector-vs-baseline comparison is valid only when every tier scores
against an identical reference set, or an unrelated corpus-redundancy
signal gets folded into what looks like a paraphrase-detection number. One
exploit remains undefended by either tier: padding a verbatim memorized
joke behind roughly twenty filler sentences drives both the n-gram and the
semantic score to zero, because both tiers score complete outputs rather
than sliding windows. Neither tier is a sole defense against this dilution
exploit until a windowed scoring pass is built; it is carried forward as a
documented limitation (Section 6), not absorbed into the detection numbers
above. Because both tiers score against the same small template-plus-
hand-corpus reference rather than the ~3.1M-joke target corpus, the
corpus-coverage ceiling stated above applies to the semantic tier exactly
as it does to the n-gram tier.

### 4.5 Temperature-unfakeability as a methods contribution

The cascade's central methodological claim — that scoring topic-set
trajectories rather than sampled joke text cannot be gamed by raising
sampling temperature — had been asserted, not demonstrated, prior to
EXP-007. On `api:deepseek` (native API, temperature control, N=6 runs per
setting), raising temperature 0.2 to 1.2 moves surface-diversity
(`distinct_2`) by **+0.390** while moving within-model set Jaccard by only
**−0.012**, against a registered bar requiring the sampling-family delta to
exceed the path-family delta by at least 3×: the observed ratio clears the
bar by more than 10×. Temperature reorders which topic a model visits next
(prefix agreement across runs collapses from 0.933 to 0.000) but does not
expand the pool of topics it draws from.

A single-model result does not establish a methods claim, and the first
replication attempt is instructive about why. EXP-007b repeated the design
on `api:qwen` and found a near-zero surface-diversity delta (+0.006)
alongside a small path-family delta (+0.037) — numbers that, read
naively, resemble a failed replication. A manipulation check added after
the fact shows why the numbers carry no evidential weight: all 6 runs at
temperature 1.2 open with a byte-identical first joke, as do all 6 at
temperature 0.2 — the qwen endpoint silently ignores the temperature
parameter. The experiment supports neither the claim nor its denial,
because the manipulation never reached the model. EXP-007c replaced qwen
with glm-4.5-air after pre-probing confirmed its endpoint honors
temperature (a legal range of [0,1], with an HTTP 400 above it), and
registered a pass/fail manipulation-check gate before interpreting any
delta: at least 3 of 6 runs at the top of the range must produce distinct
first turns. glm cleared the gate at 6/6 (against qwen's 1/6) and
replicated the differentiator pattern on the honored-endpoint model:
`distinct_2` **+0.143** (near-exact match to a +0.15 prior), set Jaccard
**−0.037** (inside the ≤0.05 bound established by EXP-007), ratio **3.9×**
(bar ≥3×). The claim now stands on two honored-endpoint native-API models,
not one, and the registered manipulation-check gate — verify that the
sampling parameter reached the model, with a pass/fail bar, before
interpreting any delta computed from it — is a reusable methods
contribution for any future temperature ablation run against a
third-party API.

### 4.6 Labeler-noise bias is regime-dependent, not universally conservative

Every claim of the form "the cascade's collapse findings survive labeling
noise" rests on the direction of that noise's bias, asserted since EXP-002
but not tested until EXP-006. The simulation reproduces the rejector's
known confusion profile (estimated from EXP-001/002/003b's raw
repeat-label logs: exact match 0.563, synonym-swap 0.149, generalize-up
0.276) over synthetic trajectories spanning five true cross-model-overlap
regimes, 2,000 seeded repetitions per regime. The bias flips sign across
the range: at high true overlap (0.39–1.00), noise net-understates overlap
(bias −0.11 to −0.47) — the original conservative-for-collapse argument
holds exactly where a collapse claim would be made. At low true overlap
(0.00–0.07), generalize-up merges dominate and noise net-overstates
overlap (bias +0.02 to +0.05) — the opposite direction. Because the
pilot's observed cross-model regime (0.10–0.22) sits in this
low-to-moderate band, Section 5.3's headline cross-model number should be
read as an upper bound on the true value: real disjointness is at least as
strong as reported, and the earlier blanket claim that labeling noise is
conservative is superseded by the regime-dependent result above. The
constrained-vocabulary labeler (Section 4.3, EXP-008) removes most of the
generalize-up channel by construction; re-estimating this simulation's
noise rates under v3 before any paper-grade collapse claim is queued work
(Section 6).

### 4.7 A second validated instrument, not used in this pilot's numbers

The cascade's rejector is not the only measurement instrument this project
subjected to a pre-registered validation arc. A Haiku-tier judge scoring
banter replies by context-ablation (the delta between a reply scored with
and without its preceding turn) separates genuinely contextual replies
from verbatim canned jokes with an effect size of **6.17** (bar 3.0) and
resists surface keyword overlap (echo-resistance r = **0.224**, bar ≤0.5).
This instrument belongs to a separate, multi-turn conversational-humor
environment and contributes no numbers to Section 5; it is reported here
because it was validated the same week under the same registered-bars
discipline, and because its one carried-forward caveat — generic, on-topic
pleasantry already earns roughly two-thirds of a genuinely contextual
reply's delta — is the kind of instrument-level scrutiny this paper argues
a humor benchmark cannot skip.

---

## 5. Experiments

### 5.1 Pilot setup

Twelve models were registered across four access lanes: four via the
`claude` subscription CLI (haiku, sonnet, opus, fable), three via the
`codex` CLI (sol = gpt-5.6-sol, mini = gpt-5.4-mini, 5.4 = gpt-5.4), four
via direct provider APIs (deepseek = deepseek-chat, qwen =
qwen-plus-2025-07-28, glm = glm-4.5-air, kimi = kimi-k2.5), and grok
(grok-4.5) via a fifth direct-API lane added once a working key became
available. Depth 30, N=4 runs per model (registered target),
rejector = the Haiku instrument validated in Section 4.3 at pilot grade
(known limitation at launch: invariance 0.800, conservative bias direction
as argued above; superseded post-pilot by the paper-grade v3 instrument,
Section 4.3). Primary metrics are computed
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

**kimi-k2.5 is dropped from the path-level roster reported in Section 5.3.**
It is a reasoning model whose internal `reasoning_content` token burn grows
with the cascade's accumulating rejection list; three escalating
`max_tokens` budgets (400, 2048, 4096) each failed to complete a single
depth-30 run, the last against a pre-committed "4096 or drop" rule. Zero
complete cascade runs exist for kimi at any budget tried. Its memorization
rate (40.0%, [28.6%, 52.6%], 24/60 jokes, Section 6) is retained from
partial-run scraps because the novelty check does not require a complete
cascade, but it is reported as a flag alongside the four per-lab
fingerprints, never folded into them. The path-level roster is therefore
**eleven models**; grok's retry lane (read-timeout raised from 120s to
300s after the original lane's failures) completed all four runs and is
the eleventh.

### 5.2 Wrapper-confound limitation and planned ablation

The four access lanes are not methodologically equivalent, and this is a
confound in the current pilot, not a detail. The `claude` and `codex` lanes
run through subscription CLI wrappers with no exposed temperature control;
the `api` and `grok` lanes call provider APIs directly, with temperature
control. **Multi-turn encoding, corrected (2026-07-17 hostile-review fix
wave): every lane, not only `claude`/`codex`, encodes conversation history
as a transcript re-injected into a single prompt.** An earlier draft of
this section stated the `api`/`grok` lanes carry "native multi-turn
state" — this is FALSE, verified directly against
`benchmark/cascade.py`'s `run_cascade`, which calls
`model_complete(transcript_prompt(messages))` for every lane's model
uniformly (`benchmark/providers.py`'s `transcript_prompt`), and against
`benchmark/providers.py`'s `make_openai_compat`, whose request body is
`{"messages": [{"role": "user", "content": prompt}]}` — a single
flattened string, the same one every other lane receives, not a native
multi-turn message array. This strengthens, not weakens, the
cross-lane comparison: encoding is uniform across all twelve models, so
the CLI-wrapper confound below is specifically the absence of
temperature control and native message-array state in the `claude`/
`codex` lanes, not a difference in how conversation history is presented
to the model. A model's apparent topic-path behavior is therefore
partly a function of *which lane it happens to run in*, not purely a
function of the model. The planned ablation is to run at least one model
through both a CLI-wrapper path and a direct-API path where both are
available, holding the model fixed, and measure whether path-divergence and
overlap metrics shift with the access method alone — quantifying the
wrapper confound directly before any cross-family comparison in the full
roster is treated as a claim about the *models* rather than about the
*wrappers*. EXP-007 and EXP-007c (Section 4.5) demonstrate
temperature-unfakeability only on the two lanes where temperature is
controllable (`api:deepseek`, `api:glm`); that result grounds the choice
of set-Jaccard as the primary metric but does not itself remove the
wrapper confound from the `claude`/`codex` lanes' numbers, which remains
open until the ablation above runs. Two additional, independently
verified pieces of evidence bound (without eliminating) that confound:
`lane-claude/turns-haiku-r01.jsonl` spends 25 of its 30 turns in
CLI-assistant persona ("I'm Claude Code, built to help with software
engineering tasks..."), and opening-turn topics leak the wrapper (fable
opens with a `programming` joke in all 5 of its 5 non-empty attempts;
multiple `codex` aliases open with near-identical "I told my computer I
needed a break..." jokes; no `api`-lane model opens with either topic in
any run). Family-level claims in Section 5.3 are accordingly claims about
*model+wrapper deployment stacks*, not about the underlying models in
isolation, pending the same-model-both-lanes ablation.

### 5.3 Results

#### 5.3.1 The pre-registered shared-pool hypothesis fails

The predicted deltas of Section 5.1 do not hold, and both fail in the same
direction: models are more idiosyncratic, not more uniform, than
registered. Cross-model mean topic-set Jaccard (raw labels, eleven models)
is **0.113**, against a prediction of ≈0.35. Within-model mean set Jaccard
across N=4 runs (N=2 for `api:glm`) of the same model is **0.208**,
against a prediction of ≈0.55 — models were predicted to repeat themselves
across runs more than they match other models, and by a wide margin they
do the opposite.

The cross-model figure is not merely "not significantly above chance."
Table 2 compares it against a `pooled_frequency_baseline` null: 10,000
synthetic universes (`benchmark/run_stats_inference.py`, seed 0) in which
each of the eleven models draws independently from one shared,
frequency-weighted bag built from the pilot's own 281-topic vocabulary. The observed 0.113 sits below every one
of the 10,000 draws (null range 0.1189–0.1567); the one-sided p-value is
1.0 under the add-one convention, i.e. the real cross-model overlap is
lower than the lowest synthetic draw a genuinely shared pool would
produce. A second null (`label_shuffle`) is reported for transparency
only; it is the documented wrong null for a shared-pool claim (near-zero
power to detect a genuinely shared pool) and is not cited as evidence
(Section 4.3, `benchmark/stats.py`). Per the labeler-noise simulation
(Section 4.6), the pilot's observed regime (0.10–0.22) is one where noise
makes measured overlap an upper bound on the true value — cross-model
disjointness, if anything, is understated here.

**Table 1: Within-model topic-set Jaccard across N independent runs, by
model (raw labels).** The mean is a model-level average (n=11); ten models
contribute six within-run pairs each (N=4), `api:glm` contributes one
(N=2).

| Family | Model | N | Within-model Jaccard |
|---|---|---|---|
| Anthropic | haiku | 4 | 0.101 |
| Anthropic | sonnet | 4 | 0.117 |
| Anthropic | opus | 4 | 0.061 |
| Anthropic | fable | 4 | 0.250 |
| OpenAI | codex:mini | 4 | 0.250 |
| OpenAI | codex:sol | 4 | 0.247 |
| OpenAI | codex:5.4 | 4 | 0.281 |
| — | grok | 4 | **0.443** |
| Open-weights | deepseek | 4 | 0.151 |
| Open-weights | qwen | 4 | 0.233 |
| Open-weights | glm | 2 | 0.150 |
| **Mean (11 models)** | | | **0.208** [95% CI 0.1507, 0.2706] |

grok's self-Jaccard (0.443) is close to double the next-highest model
(codex:5.4, 0.281); it is one point among eleven equally-weighted models
in the mean, so the model-level headline moves with it directly — which
is why the per-model column, not the mean alone, carries the finding.

**Table 2: Cross-model overlap headline and its null test.**

| Statistic | Value |
|---|---|
| Observed mean cross-model Jaccard (raw labels, 11 models) | **0.1126** |
| `pooled_frequency_baseline` null: range over 10,000 draws (281 topics) | [0.1189, 0.1567] |
| `pooled_frequency_baseline` p-value (one-sided; authoritative for this claim) | 1.0 (below entire null range) |
| Bootstrap CI, model-pair level (n=55 pairs) | [0.0983, 0.1296] |
| Bootstrap CI, run-pair level (n=800 pairs; non-independent observations, likely too narrow) | [0.1072, 0.1178] |

#### 5.3.2 Four per-lab fingerprints

**Framing, stated once here rather than re-qualified after every claim
below:** at N=2-4 runs/model, this taxonomy is a **descriptive summary**
of what this pilot's roster did, useful for generating specific,
testable, pre-registered predictions for a replication roster — it is
not four independently confirmed findings. Within it, two things do
carry statistical weight beyond description and are cited as such
throughout this paper: the Anthropic-vs-OpenAI family degradation
contrast (robust to dropping the dual-role rejector model and to
excluding meta-register labels, Section 5.3.3) and grok's triangulated
profile (three independent signals — path self-similarity, exact-match
memorization, template-trigram memorization — converging on the same
retrieval-repertoire story, this section and Table 3). Per-model
memorization percentages and single-run degradation depths, on their
own, are descriptive at this N and should be read that way.

**Anthropic: constraint collapse, with one within-family outlier.** haiku,
sonnet, and opus degrade — repeat an already-rejected topic — in 12 of 12
runs, every run, no exceptions (Table 3). fable breaks this pattern: 1 of
4 runs degrades, matching the 0-or-1-per-4 adherence rate of the OpenAI
family rather than its own. Memorization is near-zero for opus, sonnet,
and fable (pooled 3.1%, 12/389 jokes) but not for haiku (25.8%, [18.8%,
34.3%]) — statistically indistinguishable from the OpenAI family's
heavy-recall tier and distinguishable from its own family (Fisher exact
p = 1.7×10⁻¹², Holm-corrected p = 3.8×10⁻¹²). Haiku is also the rejector
instrument that judges every run's degradation, including its own
(Section 4.3); the direction of any resulting bias on haiku's own numbers
is untested (Section 6).

**OpenAI: constraint adherence, heterogeneous memorization.** All three
OpenAI models resist degradation (0–1 of 4 runs each), but the family's
pooled memorization rate (67/360 = 18.6%) is not evenly distributed within
it: mini's own rate (7.5%) sits closer to Anthropic's low tier, while sol
(21.7%) and 5.4 (26.7%) carry the family average. Pooled OpenAI
memorization remains far above pooled open-weights (7/405 = 1.7%; Fisher
exact p = 1.5×10⁻¹⁶, Holm-corrected p = 5.9×10⁻¹⁶).

**grok: constraint adherence, fixed retrieval repertoire, top
memorization.** grok never degrades in 4 runs (matching OpenAI's adherence
pattern) but shows the highest within-model overlap of any model in the
roster (set Jaccard 0.443, Table 1) and the highest memorization rate
measured (40.9%, [34.3%, 47.9%], 81/198 jokes). Every pairwise
Fisher-exact contrast against grok clears Holm correction (vs. pooled
open-weights, p = 9.4×10⁻³⁷, Holm 5.7×10⁻³⁶; vs. pooled OpenAI,
p = 2.5×10⁻⁸, Holm 2.5×10⁻⁸; vs. Anthropic non-haiku, p = 1.6×10⁻³¹, Holm
8.0×10⁻³¹). The fixed repertoire and the heavy memorized-joke reliance
read as two views of one underlying behavior: the rejection constraint is
satisfied by retreating into a small, heavily-memorized set of topics, not
by generating new ones.

**Open-weights: fast degradation, low memorization.** deepseek degrades in
4 of 4 runs at a median depth of 8.5; qwen shares the pattern less
consistently (one run reaching turn 24); glm's 2 surviving runs (Section
6) both degrade. All three show memorization at or below 2.4% (Table 3).

**Table 3: Depth-to-degradation and dual-tier memorization rate, by
model.** Depths are turns to first repeat of a rejected topic or outright
refusal; a run reaching turn 30 without either is right-censored at 30
for every statistic that touches depth (marked "—" below). Memorization
intervals are Wilson score 95% confidence intervals; per-model joke
counts range 60-198 (`stats_inference.json`,
`memorization_proportions`/`dual_tier_memorization`). **Template-trigram
rate is a new column (2026-07-17 hostile-review fix wave)**: trigram
Jaccard against the 25 ChatGPT templates alone (≥0.5 counts a hit),
already computed by `benchmark/joke_novelty.py` but not previously
reported alongside the exact-match tier; see the corrected corpus
description and the framing-prefix style confound in Section 6.

| Family | Model | N | Degradation depths (turn) | Degraded / N | Exact-match rate [95% CI] | Template-trigram rate |
|---|---|---|---|---|---|---|
| Anthropic | haiku | 4 | 22, 7, 7, 7 | 4/4 | 25.8% [18.8%, 34.3%] | 5.0% |
| Anthropic | sonnet | 4 | 20, 11, 10, 14 | 4/4 | 0.8% [0.1%, 4.6%] | 2.5% |
| Anthropic | opus | 4 | 13, 11, 13, 13 | 4/4 | 3.3% [1.3%, 8.3%] | 0.8% |
| Anthropic | fable | 4 | 18, —, —, — | 1/4 | 4.7% [2.3%, 9.4%] | 2.7% |
| OpenAI | codex:mini | 4 | —, —, —, — | 0/4 | 7.5% [4.0%, 13.6%] | 2.5% |
| OpenAI | codex:sol | 4 | 26, —, —, — | 1/4 | 21.7% [15.2%, 29.9%] | 7.5% |
| OpenAI | codex:5.4 | 4 | 24, —, —, — | 1/4 | 26.7% [19.6%, 35.2%] | 10.0% |
| — | grok | 4 | —, —, —, — | 0/4 | 40.9% [34.3%, 47.9%] | **20.7%** |
| Open-weights | deepseek | 4 | 11, 9, 6, 8 | 4/4 | 0.8% [0.1%, 4.6%] | 3.3% |
| Open-weights | qwen | 4 | 11, 8, 24, 9 | 4/4 | 1.7% [0.5%, 5.9%] | 10.0% |
| Open-weights | glm | 2 | 21, 14 | 2/2 | 2.4% [0.9%, 6.1%] | 2.4% |

Grok remains the clear outlier on both tiers (next-highest template rate:
10.0%, tied between qwen and codex:5.4), but the "open-weights barely
memorize" reading is tier-specific: qwen's and deepseek's template rates
are 6x and 4x their respective exact-match rates. Part of the mechanism
is a style confound in the exact-match tier (a whole-string hash match):
sonnet prefixes 74.2% of jokes with framing prose before the punchline,
against grok's 0% and deepseek's independent 53.3% — see Section 6.

**Table 3b: Degradation-event decomposition (new).** Each run's FIRST
degradation event, re-derived directly from raw per-turn logs and
classified as a genuine topic repeat, a meta-register-label repeat
(`comedy`/`joke`/`humor`/`ai`/`software`/`laughter`), or a refusal
(`benchmark/metrics.py`'s conservative refusal regex). **Caveat, stated
plainly:** the refusal regex has confirmed false positives on
in-character creative escalation that happens to use casual "can't"/
"won't" phrasing as part of the joke itself, not as a genuine break in
character (e.g. a comedian-bit line like "I can't even say \[X\], because
\[X\] is banned too" is not a refusal); of haiku's three
refusal-classified runs, direct inspection finds only two contain a
genuine assistant-persona break, the third is in-character escalation.
Read this column as "a refusal-adjacent turn preceded the first topic
repeat," not as verified ground truth. This table also uses a different
depth convention than Table 3 above, disclosed for exactly this reason:
the pipeline that produced Table 3's published depths never considers
refusal turns at all (a latent gap, not a red-team finding, surfaced
while building this table), so a handful of this table's classifications
occur at an earlier turn than the corresponding published depth in
Table 3.

| Family | Model | Topic-repeat | Meta-register-repeat | Refusal | Survived |
|---|---|---|---|---|---|
| Anthropic | haiku | 0 | 1 | 3 | 0 |
| Anthropic | sonnet | 1 | 1 | 2 | 0 |
| Anthropic | opus | 0 | 3 | 1 | 0 |
| Anthropic | fable | 0 | 0 | 3 | 1 |
| OpenAI | codex:mini | 0 | 0 | 0 | 4 |
| OpenAI | codex:sol | 1 | 0 | 0 | 3 |
| OpenAI | codex:5.4 | 1 | 0 | 0 | 3 |
| — | grok | 0 | 0 | 0 | 4 |
| Open-weights | deepseek | 1 | 3 | 0 | 0 |
| Open-weights | qwen | 2 | 0 | 2 | 0 |
| Open-weights | glm | 1 | 1 | 0 | 0 |
| **Total (42 runs)** | | **7** | **9** | **11** | **15** |

#### 5.3.3 Degradation incidence and the family contrast

Nine of eleven models show at least one degrading run; `codex:mini` and
`grok` are the only two that never degrade across all four of their runs
(Table 4), against a pre-registered threshold of ≥1/3 of the roster (met).
An earlier count against the original ten-model pilot (before fable's and
grok's fill runs landed) reported "8/10 models degrade"; recounting
against the completed eleven-model roster under the same rule (≥1 run
degrades) gives 9/11 — neither figure reproduces the other exactly under
the most natural reading of "degrades," and this is stated here rather
than silently reconciled in the underlying experiment log.

**Table 4: Degradation incidence against the pre-registered threshold.**

| Statistic | Value |
|---|---|
| Models with ≥1 degrading run | 9 / 11 (81.8%) |
| Pre-registered threshold | ≥ 1/3 of roster (36.7%) |
| Threshold met | Yes |
| Never-degrading models | codex:mini, grok |

The best-supported single contrast for the family-level pattern pools
Anthropic's 16 runs (haiku, sonnet, opus, fable) against OpenAI's 12
(mini, sol, 5.4): mean difference **−13.17 turns**, Monte Carlo
permutation p = **0.0002** (10,000 draws), Cliff's delta **−0.781
(large)**. This contrast was chosen after the pilot's results made the
family-level pattern visible; it is a **disclosed post-hoc contrast, not
pre-registered in EXP-004**, and is reported as strong exploratory
evidence, not confirmatory evidence — a genuinely pre-registered
replication would be required to earn that label. Both the mean
difference and the effect size weakened relative to an earlier
ten-model snapshot of this same pilot (−15.17 turns, delta −0.917) once
fable's three additional surviving runs landed, which is the expected
direction for adding a model that partially breaks the family pattern.

Because haiku is both a model under test and the rejector instrument that
judges every run's degradation including its own, a robustness check
drops haiku from the Anthropic pool: sonnet, opus, and fable (12 runs)
against OpenAI (12 runs) gives mean difference **−11.42 turns**,
p = **0.0005**, Cliff's delta **−0.708 (large)**. The effect shrinks
somewhat without haiku but remains large and significant: the dual-role
confound does not carry the family-level result.

**Meta-register-exclusion robustness (new).** Table 3b shows 9 of the 13
raw Anthropic degradation events are meta-register repeats or worse
confounded with a refusal; isolating the 11 that are pure meta-register
repeats (Section 6), recomputing both family contrasts above with these
six labels treated as never-matching for repeat detection — re-derived
independently from raw per-turn logs, verified against the published
depths before trusting the result — gives mean difference **−8.79 turns**
(family) and **−7.75 turns** (no-haiku), both still large and significant
(p = 0.0004 and p = 0.0018 respectively). The contrast survives because
sonnet and opus also repeat ordinary everyday topics (`appliance`,
`organization`) that no OpenAI model repeats. A censoring-free companion
test (Fisher exact on degraded-vs-survived run counts, avoiding the
depth=30 censoring convention entirely) agrees at every level: raw
Anthropic-all 13/16 vs. OpenAI 2/12 (p = 0.0016), raw non-haiku 9/12 vs.
2/12 (p = 0.0123), meta-excluded Anthropic-all 11/16 vs. 2/12
(p = 0.0093), meta-excluded non-haiku 8/12 vs. 2/12 (p = 0.0361). Under
meta-exclusion, Table 4's incidence count also shifts: haiku degrades in
3 of 4 runs (not 4 of 4), and fable degrades in 0 of 4 (not 1 of 4,
consistent with fable's one counted degradation being itself
comedy-mediated) — joining `codex:mini` and `api:grok` in the
never-degrades set under this stricter definition.

An exploratory dyadic battery over all C(11,2) = 55 model pairs (exact
permutation, feasible for every pair at this sample size) finds raw
p-values as low as 0.0286, with several pairs showing perfect separation
(Cliff's delta = ±1.0). After Holm correction across the 55 comparisons,
every corrected p-value is 1.0. This is not an error: the achievable
p-value floor for a two-sided, swap-symmetric statistic at N=4-per-group
is 2/70 ≈ 0.0286 (the combinatorial 1/70 ≈ 0.0143 floor is unreachable by
such a statistic), and Holm's correction multiplies that floor by 55,
which alone exceeds 1. (An earlier snapshot of this pilot, before fable
and grok's fill runs completed the ten-model roster to eleven, reported
this as a 45-pair battery, C(10,2); the roster is now C(11,2) = 55 pairs,
and the conclusion is unchanged.) None of the 55 pairwise comparisons
should be read as individually significant; the two family-level
contrasts above are the citable results, and only as strong exploratory
evidence, never as confirmatory.

---

## 6. Discussion and Limitations

**CLI-wrapper confound, bounded, not eliminated — and visible directly in
the transcripts, not only argued from lane design.** Three of four access
lanes (`claude`, `codex`) run through subscription CLIs with no exposed
temperature control (Section 5.2). Multi-turn encoding is uniform across
ALL FOUR lanes (transcript-in-prompt everywhere, corrected in Section 5.2
— this is not a `claude`/`codex`-specific property, and stating it as one
in an earlier draft of this section was a factual error), which
*strengthens* rather than weakens the comparability of the cross-lane
family contrasts below: the confound specific to `claude`/`codex` is the
absence of temperature control and native message-array state, isolated
from any encoding difference. Two pieces of direct transcript evidence
bound the remaining confound: `lane-claude/turns-haiku-r01.jsonl` spends
25 of its 30 turns in CLI-assistant persona ("I'm Claude Code, built to
help with software engineering tasks. I'm not going to roleplay as a
comedian..."), and opening-turn topics leak the wrapper (fable opens with
a `programming` joke in all 5 of its 5 non-empty attempts across both
lanes; multiple `codex` aliases open with near-identical "I told my
computer I needed a break..." jokes; no `api`-lane model opens with
either topic in any run). EXP-007 and EXP-007c (Section 4.5) demonstrate
temperature-unfakeability only on the two lanes where temperature is
controllable (`api:deepseek`, `api:glm`); that result grounds the choice
of set-Jaccard as the primary metric but does not itself remove the
wrapper confound from the `claude`/`codex` lanes' numbers, and it does
not by itself explain the 13-turn Anthropic/OpenAI depth gap either —
EXP-007's own temperature sweep moves deepseek's median degradation depth
by at most ~5.5 turns, non-monotonically, across temp 0.2→0.7→1.2. Every
cross-family comparison in Section 5.3, including the Anthropic-vs-OpenAI
family contrast, is therefore best read as a claim about *model+wrapper
deployment stacks*, bounded by this confound until the planned
same-model, both-lanes ablation (Section 5.2) runs — the single
score-moving experiment this paper's own hostile review identified
(estimated cost: ~$5 in API spend).

**Meta-register labels mediate most of the Anthropic degradation
pattern, but the family contrast survives excluding them.** Eleven of the
13 raw Anthropic degradation events are repeats of a meta-register label
(`comedy`, `joke`, `humor`, `ai`, `software`, `laughter` — a model joking
about joke-telling, or breaking into "as an AI" register, under rejection
pressure); opus's uncannily consistent 13,11,13,13 depths are `comedy`
every time. Recomputing the family contrast (Table 3, Section 5.3.3) with
these six labels treated as never-matching for repeat detection, re-derived
independently from raw per-turn logs and verified against the published
depths before trusting the result: the contrast survives at **−8.79
turns** (p = 0.0004) and, with haiku dropped, **−7.75 turns** (p = 0.0018)
— it survives because sonnet and opus also repeat ordinary everyday
topics (`appliance`, `organization`) that no OpenAI model repeats. A
censoring-free companion test (Fisher exact on degraded-vs-survived run
counts, sidestepping the depth=30 censoring convention entirely) agrees:
raw 13/16 vs. 2/12 (p = 0.0016), meta-excluded 11/16 vs. 2/12 (p = 0.0093),
and the corresponding no-haiku variants (p = 0.0123, p = 0.0361). Incidence
under meta-exclusion also corrects two of Table 3's counts: haiku
degrades in 3 of 4 runs, not 4 of 4 (one of its four "degradations" was
purely a repeated meta-register label), and fable degrades in 0 of 4, not
1 of 4 (its one counted degradation was itself comedy-mediated) — placing
fable alongside `codex:mini` and `api:grok` in the never-degrades set
under this stricter definition.

**N = 2–4 runs per model.** Ten of the eleven path-level models have N=4
runs; `api:glm` has N=2. The combinatorial permutation floor for any two
N=4-run groups is 1/70 ≈ 0.0143 (practically achievable ≈2/70 ≈ 0.0286,
Section 5.3.3); any pair involving glm is coarser still. Every confidence
interval and p-value in Section 5.3 should be read at this precision — the
width of the interval is part of the finding, not a defect of the method
to apologize for. glm's two surviving runs were, in addition, produced
under a mixed generation-config protocol: `max_tokens` was raised from 400
to 2048 mid-experiment after early attempts silently exhausted the entire
budget on internal reasoning tokens and returned empty completions;
glm's effective sample is smaller and less uniform than its raw run count
implies. glm's N=2 also excludes, by an explicit and now-stated rule, a
third, complete, degrading run: one lane produced exactly one successful
glm cascade (a full 30-turn run, degrading at turn 16) alongside one
failure, one short of the pipeline's own ">= 2 successful runs per lane"
gate for writing a per-model entry at all — a batching artifact, not a
principled exclusion, left uncorrected here because fixing it would
require re-deriving the frozen pilot dataset itself.

**Depth capped at 30 turns.** Degradation depths and "survived" counts in
Table 3 are relative to this cap, not an absolute ceiling. A model
recorded as surviving to turn 30 could still degrade at turn 45; the cap
is a pilot-scale design choice, not evidence of an unbounded generative
capacity.

**The pilot is labeled with the v2 free-vocabulary labeler, not the
paper-grade v3.** EXP-008's constrained-vocabulary instrument (Section 4.3)
was validated after the pilot in Section 5.3 had already run. The core
problem v2 leaves unresolved is that free-vocabulary labels admit hypernym
relationships (`pet` over `cat`/`dog`, `medicine` over `health`) that no
flat clustering method, semantic or string, handles correctly (Section
4.3, EXP-003). v2's known failure mode — synonym and hypernym jitter on
pairs like these — has an argued conservative bias direction for collapse
claims (label noise splits topics, which can only make a model look more
diverse than it is), which is why proceeding at pilot grade was
defensible. But every Jaccard number in Section 5.3 could shift once the
queued v3 post-hoc relabel of the pilot's already-collected jokes (zero
new API calls) lands; the direction of that shift is not knowable a
priori for every model, only the aggregate bias direction argued in
Section 4.6.

**Memorization is a corpus-coverage lower bound, for every model, not
selectively — and the exact-match reference corpus was misdescribed in
an earlier draft, corrected here (2026-07-17 hostile-review fix wave).**
The reference behind Table 3's exact-match memorization percentages is
the **~1.2M-joke corpus** (commercial-safe + research-only jokes,
overwhelmingly Reddit-derived), not "the 25 ChatGPT joke templates plus a
small hand-built corpus" as stated previously — that description belongs
to the *separate* template-trigram tier (below), which scores against
the 25 Jentzsch & Kersting templates alone. Verified directly against
`benchmark/joke_novelty.py`'s corpus-loading code: the exact-match tier
hashes every `jokes.jsonl` file under the corpus directory, and the
25-template file (`chatgpt-25-templates.jsonl`) is named and loaded
separately, never folded into that hash set. It cannot contain every
joke any model has memorized regardless; every exact-match percentage
reported understates true memorization reliance. The semantic novelty
tier introduced in Section 4.4 is unaffected by this correction — it
scores by design against the templates-only reference (its `reference`
argument defaults to `"templates"`), so the original ceiling statement
for that tier was accurate as written.

**Dual-tier memorization and a style confound (new).** Table 3 reports
the exact-match tier only. `benchmark/joke_novelty.py` also computes a
template-trigram tier (Jaccard similarity against the 25 templates,
≥0.5 counts a hit) that was collected but never reported alongside it.
Both tiers together change two of Table 3's cleanest-looking contrasts:
grok remains the clear outlier on both (40.9% exact / 20.7% template,
vs. the next-highest template rate at 10.0%), but the "open-weights
barely memorize" reading is tier-specific — qwen's template rate (10.0%)
is 6x its exact-match rate (1.7%) and ties codex:5.4 exactly, and
deepseek's template rate (3.3%) is 4x its exact rate (0.8%). Part of the
mechanism is a style confound in the exact-match tier itself, which is a
whole-normalized-string hash match: sonnet prefixes 74.2% of its jokes
with framing prose before the punchline ("Alright, no science. This
one's about my bank account..."), against 0% for grok, whose jokes open
cold. This is not vendor-specific — deepseek independently shows a 53.3%
framing-prefix rate — and it means part of the exact-match tier's
40.9%-vs-0.8% grok-vs-sonnet gap is a delivery-format artifact, not a
pure memorization-depth signal. The template-trigram tier, computed over
trigram sets rather than whole-string equality, is comparatively robust
to this prefix dilution.

**Single rejector, and haiku's dual role.** EXP-003b showed that a larger
model (sonnet) is a *worse* rejector instrument than haiku (ARI 0.633 vs.
0.837) — richer vocabulary increases label-granularity variance rather
than reducing it — but only two rejector models have ever been tried, and
rejector-model generality beyond haiku is untested. Compounding this,
haiku is both the rejector instrument that labels every trajectory and
judges every run's degradation, and a model under test in the cascade
roster (Section 5.3): its own degradation depths and its own memorization
rate are judged by itself, and the direction of any resulting bias is
untested. The robustness check that drops haiku from the Anthropic pool
(Section 5.3.3) shows the family-level degradation contrast does not
depend on this confound, but the confound is not otherwise fixable
without either a second, independent rejector or removing haiku from the
roster entirely — both queued as follow-up work.

**kimi has zero complete cascade runs and is dropped from the path-level
roster** (Section 5.1). Three escalating `max_tokens` budgets all failed
against the reasoning-content burn that scales with cascade depth, and the
diagnosis suggests this will not resolve without a design change
(streaming with reasoning-budget control, or a non-reasoning kimi
variant). Its memorization rate is retained as a flag from partial-run
scraps, uncorroborated by any path-level data, and is never counted among
the four per-lab fingerprints in Section 5.3.

**The exploratory dyadic battery does not survive correction.** Of the 55
pairwise model comparisons underlying Table 3, several show perfect
separation before correction, but every Holm-corrected p-value is 1.0
(Section 5.3.3). Only the two family-level contrasts, both disclosed as
post-hoc, should be cited from this pilot, and only as strong exploratory
evidence — never as confirmatory evidence, which would require a
genuinely pre-registered replication.

**What this pilot supports claiming, stated once, plainly.** A hostile
review of this draft's full claim chain (reproducing every published
contrast from raw lanes before attacking any of them) returned a verdict
of weak reject as originally framed — not because any individual number
was wrong, but because the four-fingerprint taxonomy and the headline
prose around it claimed more uniformity and more confirmatory weight than
N=2-4 runs/model can support. We accept that verdict and reframe
accordingly rather than defend the original framing: **the honest,
stronger version of this paper is "two robustness-surviving contrasts, a
validated instrument chain, grok's triangulated profile, and a registered
replication design,"** not "four confirmed per-lab behavioral fingerprints."
Concretely, this paper's citable core is (i) the two pre-registered misses
(Section 5.3.1) — the only claims registered before any data existed; (ii)
the Anthropic-vs-OpenAI family degradation contrast, which survives both
the haiku dual-role check and, newly, meta-register-label exclusion
(Section 5.3.3); (iii) grok's profile, triangulated across three
independent signals rather than resting on any one (Section 5.3.2, Table
3); and (iv) the instrument-validation arc itself (Section 4), which is a
methods contribution independent of what the pilot's numbers turn out to
mean. The four-fingerprint taxonomy remains in this paper because it is a
useful, honestly-labeled generator of specific predictions for the
replication roster below — not because it stands as four separate results
in its own right. The one experiment that would move this paper's score
the most is the same-model-both-lanes wrapper ablation (Section 5.2,
~\$5 in API spend): it is not run in this zero-API-cost revision, and is
registered here as the decisive next step rather than deferred silently.

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
established — unclaimed question. It depends on this benchmark — now
carrying a paper-grade rejector instrument (Section 4.3, EXP-008) and a
working, if corpus-incomplete, novelty-penalty check (Section 4.4,
EXP-009) — existing first, as the pre-registered, collapse-resistant
measurement against which any claimed "humor training" would need to be
shown not to have re-discovered the twenty-five templates alone. We treat
it as a separate paper, not a section of this one.

---

*End of draft skeleton.*

# ATTACK: Rejection-Cascade Humor Benchmark — Novelty Kill Attempt

Target idea: ask a model for a joke; a cheap rejector says "not funny, try a
different topic"; repeat ~50 turns with rejections ACCUMULATING. Measure the
TRAJECTORY of topics (not the jokes): (a) within-model across-run path
divergence, (b) cross-model path overlap, (c) depth-to-degradation. Claim:
sampling-diversity metrics (temperature, distinct-k) can be gamed; path-based
exhaustion under accumulating rejection cannot.

All arXiv IDs below verified via `curl -s -o /dev/null -w '%{http_code}' -L
https://arxiv.org/abs/<id>` → 200.

---

## 1. Denial Prompting / NEOCODER — arXiv:2407.09007

**Citation:** Lu, Wang, Li, Jiang, Khudanpur, Jiang, Khashabi. "Benchmarking
Language Model Creativity: A Case Study on Code Generation." NAACL 2025
(submitted Jul 2024, rev. Feb 2025). arXiv:2407.09007.

**Protocol (verified from HTML full text):** Algorithm 1. For t = 1..T
(T≈5 typical): (1) generate solution y_t conditioned on problem x plus all
prior constraints τ_1..τ_{t-1}; (2) an LM detects the "atomic technique" used
in y_t (e.g., recursion, hashmap, for-loop); (3) that technique is added to
a growing constraint set 𝒞_t forbidding its reuse; (4) repeat with the newly
constrained problem. Output is a **trajectory of NeoGauge scores**
(convergent creativity = correctness × constraint adherence; divergent
creativity = fraction of techniques never seen in the human solution set),
not a single number. Comparison is cross-model (different target LLMs) and
state-aware (same model across constraint states), but **not** within-model
across-independent-run path divergence, and **not** cross-model path/topic
overlap.

**Verdict: WOUNDS.** This is the closest *structural* precedent —
iterative, single-conversation constraint accumulation forcing the model off
its default answer, repeated t times, with the state-by-state trajectory as
the artifact of interest (matches the "measure the path, not the endpoint"
spirit exactly). But: (1) domain is competitive programming, not humor/social
creativity; (2) what's denied is a *technique* the model itself just used,
extracted programmatically — not a *topic*, and not an arbitrary/subjective
"I don't find that funny"; (3) depth is T≈5, an order of magnitude shallower
than the proposed ~50; (4) the measured object is a scalar creativity score
per state, not a semantic trajectory (cat→dog→parrot) compared across runs
or models. Must cite as the direct ancestor and differentiate hard on domain,
rejection semantics (subjective vs. programmatic), depth, and the
path-as-object-of-study framing.

---

## 2. NoveltyBench — arXiv:2504.05228

**Citation:** Zhang, Diddee, Holm, Liu, Liu, Samuel, Wang, Ippolito.
"NoveltyBench: Evaluating Language Models for Humanlike Diversity." arXiv
(Apr 2025, rev. Aug 2025). 2504.05228.

**Protocol (verified from HTML v2):** Primary eval is single-turn: 10
independent parallel samples at temperature 1, scored with a "Distinct-k"
metric = count of equivalence classes among k samples, where equivalence is
decided by a fine-tuned DeBERTa classifier (79% acc., F1 0.811 vs. human
labels) trained on 1,000 annotated pairs. **Section 4.3 also runs
"in-context regeneration":** after each generation, the model is explicitly
asked to give a different answer, with all prior answers kept in context —
up to **8 generations** per prompt on the 100 curated prompts. No rejection
or denial semantics — the model isn't told its answer is unacceptable, just
asked to try again. No cross-model path overlap; no trajectory/path
divergence metric; equivalence classes are unordered, not sequenced.

**Verdict: WOUNDS.** This is the closest precedent for the *within-model,
same-conversation, repeated "give me another" mechanic* — but shallow (8
turns vs. ~50), no rejector/denial framing, no accumulating constraint
memory, and the metric is a set-cardinality (Distinct-k) rather than a
path/trajectory object. Must cite as the direct precedent for "regeneration
loop" and differentiate on depth, rejection semantics, and treating the
*sequence* (not the set) as the measurement target.

---

## 3. MUTATE — arXiv:2605.28465

**Citation:** Park, Baek, Park, Lee. "Beyond One Path: Evaluating and
Enhancing Divergent Thinking in Interactive LLM Agents." arXiv (May 2026).
2605.28465.

**Protocol (verified from HTML full text):** Interactive escape-room-style
text environment. Agent takes actions (`move`, `click`, `apply`, `craft`,
`input`) toward a fixed goal. A "Reflect" module builds a **target-indexed
failure memory**: failed actions are stored per-object with a repetition
count; once failures on a single target exceed a threshold, a
"Diverge-Narrowing" module forces the agent toward alternative approaches.
Episodes run until goal, step budget, or 20 consecutive no-op steps (baseline
avg. 21–45 steps; working memory window = 10 steps). Two metrics: **Path
Discovery** (count of distinct valid solutions found across 4 attempts) and
**Divergence Momentum** (LLM-judge originality/elaboration/groundedness
score on every off-path attempt, not just successes). Comparison is
**within-model only** (same model, 4 runs per scenario) — no cross-model
path overlap analysis is performed, despite reporting per-model results.

**Verdict: WOUNDS — the single closest paper found.** It independently
converges on almost the same idea: accumulating rejection memory forcing a
model down alternate paths, with the trajectory (not just the destination)
as an explicit scored object, and a metric explicitly designed so
"path-level and action-level divergence can fail independently." Critical
differences that keep the humor cascade novel: (1) rejection here is
**objective task failure** (the crafted item didn't work) — the proposed
idea uses a **subjective, content-agnostic rejector** ("I don't find that
topic funny") with no ground truth, which is a fundamentally different
elicitation pressure; (2) failure memory is **object-indexed**, not a single
global growing list — the proposed idea's constraint list is global and
monotonically growing across the whole 50-turn conversation; (3) domain is
puzzle-solving, not humor/social creativity, so there's no notion of
"topic" as a semantic category to track; (4) **no cross-model path overlap**
is measured — this is the paper's most exploitable gap for claim (b) of the
target idea; (5) no notion of "identical paths across independent runs of
the same model = lookup table," which is a distinct diagnostic MUTATE
doesn't attempt. This paper must be cited prominently and the differentiation
argued carefully — a reviewer who knows MUTATE will ask "how is this not
just MUTATE for jokes," and the honest answer is: subjective vs. objective
rejection, global vs. per-target memory, and the cross-model + across-run
path-identity diagnostics MUTATE never runs.

---

## 4. Creative Homogeneity Across LLMs — arXiv:2501.19361

**Citation:** Wenger, Kenett. "We're Different, We're the Same: Creative
Homogeneity Across LLMs." arXiv (Jan 2025). 2501.19361.

**Protocol (verified via fetch):** Standardized creativity tests
administered single-shot to multiple LLMs; population-level response
similarity compared across models vs. across humans. Finding: "LLM responses
are much more similar to other LLM responses than human responses are to
each other." No iterative/sequential elicitation; no rejection mechanism.

**Verdict: WOUNDS claim (b) specifically.** Cross-model output convergence
in creative tasks is already an established, published finding — the
target idea cannot claim to be first to show LLMs converge. The
differentiation has to be the *mechanism of measurement*: single-shot
population similarity vs. **path overlap under identical accumulating
rejection pressure**, which is a much stronger/more diagnostic claim
("even when forced down the same adversarial gauntlet, unrelated models
take the same escape route") than "their unprompted outputs happen to
overlap."

---

## 5. Cards Against LLMs — arXiv:2604.08757

**Citation:** Fettach, Bied, Toivonen, De Bie. "Cards Against LLMs:
Benchmarking Humor Alignment in Large Language Models." arXiv (Apr 2026).
2604.08757.

**Protocol (verified via fetch):** Cards Against Humanity format — models
*select* (not generate) the funniest response from 10 candidates across
9,894 rounds. Single-shot selection, not generation, not iterative. Finding:
"models agree with each other substantially more often than they agree with
humans" — systematic position bias and shared content preferences.

**Verdict: WOUNDS claim (b) in the humor domain specifically.** This is the
most on-domain cross-model-convergence-in-humor precedent found. It is a
*selection* task over a fixed candidate set, though, not open-ended
generation, and single-shot, not path-based. Must cite as direct evidence
the "shared pretraining priors → shared humor" mechanism is real and already
suspected in this exact domain — strengthens the motivation, doesn't kill
the path-based measurement novelty.

---

## 6. IDEAFix — arXiv:2606.00875

**Citation:** (authors per arXiv listing) "IDEAFix: Evaluation Framework
for Creative Defixation Prompting in LLMs." arXiv (May/Jun 2026). 2606.00875.

**Protocol (verified via fetch):** 14,350 prompts (81 briefs × 6 attribute
variants × 25 prompts), **explicitly single-shot** per the authors: "our
study focuses on a non-iterative setting, whereas many creativity methods
are typically applied through iterative, guided processes with several
steps of idea refinement." Homogenization measured via cosine distance
between models' answer sets and top-keyword overlap (Fig. 5b); finds
"persistent output homogenization across models."

**Verdict: MISSES the mechanism, WOUNDS the motivation.** Useful mainly as
a 2026 paper *explicitly flagging the iterative extension as unaddressed
future work* — i.e., a live acknowledgment from the field that the gap
being targeted is real and open as of mid-2026. Cite as motivating gap,
not as prior art to differentiate against.

---

## 7. Path-dependent category fluency in LLMs — arXiv:2405.06714

**Citation:** Heineman, Koenen, Varma. "Towards a Path Dependent Account of
Category Fluency." CogSci 2024 (May 2024). 2405.06714.

**Protocol (verified via fetch):** Applies the classic human verbal/semantic
fluency paradigm (name-as-many-animals) to an LLM (Llama 2 Chat 7B) as a
cognitive model, using per-step log-probability/entropy to detect "patch
switches," compared against human patch-foraging dynamics (marginal value
theorem). No adversarial rejection — free autoregressive generation. No
cross-model comparison in the abstract/available text.

**Verdict: WOUNDS the "semantic foraging" framing specifically.** This
confirms the human category-fluency paradigm has already been ported to
LLMs, including foraging-theoretic "patch depletion" language very close to
"depth-to-degradation." But it's unforced (no rejector), single-category
(animals), single-model, and not humor. The proposed idea's novelty here is
narrower than the abstract framing suggested: "verbal fluency applied to
LLMs" is done; "verbal fluency *under adversarial forced exhaustion*,
cross-model, for humor" is not.

---

## 8. Divergent Association Task on LLMs (multiple)

- Agnoli et al. framework, arXiv:2405.13012 / Nature Sci. Reports version —
  DAT + DSI + LZ-complexity battery comparing LLM and human divergent
  thinking; single-shot, no cross-model path analysis.
- GitHub `lechmazur/divergent` (verified via fetch) — LLMs generate 25
  words unrelated to an initial 50-word list; single-shot; reports
  **% repeated words as an implicit cross-model overlap metric**
  (e.g., o1-preview 0% vs. GPT-4o 23.68%).

**Verdict: MISSES.** Single-shot word generation, not sequential/rejection-
based; the lechmazur benchmark's repeated-word-% is a crude cross-model
overlap precedent worth a footnote, but doesn't touch trajectory or forced
exhaustion.

---

## 9. Mode-collapse-onto-memorized-jokes precedent — arXiv:2306.04563

**Citation:** Jentzsch, Kersting. "ChatGPT is fun, but it is not funny!
Humor is still challenging Large Language Models." arXiv (2023). 2306.04563.
(Already the basis of the CLAUDE.md "documented failure mode" claim —
independently reconfirmed here.)

**Protocol:** Same flat prompt ("tell me a joke") repeated 1,008 times to
ChatGPT-3.5; >90% of outputs were variations on 25 joke templates.

**Verdict: WOUNDS the motivating premise, not the mechanism.** No rejection,
no accumulating constraint, no trajectory measurement, single model. This is
evidence *for* why the cascade idea should work (mode collapse under
repeated identical elicitation is real) but is not a competing method — it's
the baseline the cascade idea should cite as "this is what happens with the
naive version of our probe, and it's why we need forced denial, not just
repetition, to see the shape of the distribution."

---

## 10. Diversity-metric-gaming argument — arXiv:2605.11128

**Citation:** "Sampling More, Getting Less: Calibration is the Diversity
Bottleneck in LLMs." arXiv (May 2026). 2605.11128.

**Protocol:** Decomposes diversity collapse into order-calibration and
shape-calibration miscalibration; argues temperature sampling shifts mass
toward *invalid* continuations before recovering enough *valid* diversity —
i.e., temperature is an unreliable diversity lever, though for a different
reason (validity trade-off) than "temperature can be gamed by a benchmark
designer."

**Verdict: WOUNDS (partially) the "temperature can be faked" claim** — it's
making an adjacent but distinct argument (temperature doesn't reliably buy
diversity because of calibration, not because it's easy to overfit a metric
to). Should be cited to support the general thesis that sampling-based
diversity metrics are unreliable, while being explicit that this paper's
reason is different from the proposed paper's reason (gameability vs.
calibration failure) — don't conflate the two in the write-up.

---

## Also checked, found non-threatening (MISSES)

- **CREATE** (arXiv:2603.09970) — "path" = static associative chains in a
  knowledge graph, scored for specificity/diversity; no iteration, no
  rejection, no sequential elicitation. Different sense of "path" entirely.
- **Generative Monoculture** (arXiv:2407.02209) — single-model output
  narrowing relative to training-data diversity (e.g., book reviews);
  not a cross-model or path-based claim.
- **Verbalized Sampling** (arXiv:2510.01171) — a mitigation technique
  (prompt for a distribution over responses) to fight mode collapse; not an
  eval, not sequential/rejection-based. Relevant as a technique the cascade
  benchmark could stress-test, not as competing prior art.
- **Measuring LLM Novelty as Frontier** (arXiv:2504.09389) — uses Denial
  Prompting as one baseline manipulation among several; confirms Denial
  Prompting is already a known, citable technique in this exact
  conversation, which raises (not lowers) the bar for differentiating from
  it explicitly in related work.
- Searches for "rejection cascade," "escalating rejection," "50 turns
  denial," "topic trajectory divergence LLM," and "20 questions repeated
  denial diversity" turned up no paper combining accumulating *subjective*
  rejection with topic-trajectory measurement, in any domain, at any
  reported depth close to ~50 turns. No KILL candidate surfaced.

---

## Overall Verdict

**No single paper kills this.** Nothing found measures topic-trajectory
divergence — within-model across independent runs, or cross-model — under
an *accumulating, subjective, content-agnostic* rejection loop of anything
close to 50 turns. But this is a **crowded, converging space**, and three
papers must be treated as required related work, not optional citations:

1. **Denial Prompting / NEOCODER** (2407.09007) — same iterative
   constraint-accumulation skeleton, different domain/depth/rejection
   semantics/measured object.
2. **MUTATE** (2605.28465) — closest overall: independently invented
   accumulating-failure-memory forcing path divergence, with an explicit
   path-vs-action divergence metric split. Differs on subjective-vs-
   objective rejection, global-vs-per-target memory, domain, and (crucially)
   never runs the cross-model comparison.
3. **NoveltyBench** (2504.05228) §4.3 in-context regeneration — same
   "keep asking for another, same conversation" mechanic at 1/6th the depth,
   no denial framing, set-based not sequence-based metric.

Cross-model humor/creative convergence is *already documented* at the
single-shot level (Creative Homogeneity Across LLMs, Cards Against LLMs,
IDEAFix) — the paper cannot claim "first to show LLMs converge on humor,"
only "first to show they converge on the *same forced escape route* under
identical adversarial pressure," which is a stronger and different claim.

The idea is defensible as a paper, but the related-work section is not
optional filler here — a reviewer who knows this literature (plausible for
an ACL/EMNLP/NAACL creativity-track reviewer given how recent and clustered
these citations are, several from 2026) will name Denial Prompting and
MUTATE unprompted. Pre-empting both explicitly, with the differentiation
table above, is required to survive review, not just polite.

**Sharpest honest differentiation sentence:**

> "Prior iterative-forcing methods either constrain *technique* on a
> single-conversation, ~5-step depth with an objective correctness signal
> (Denial Prompting) or constrain *actions* via per-object failure memory
> in a deterministic puzzle environment without cross-model comparison
> (MUTATE); we instead apply a subjective, content-agnostic rejector to an
> open social-creative task for an order of magnitude more turns, and are
> the first to treat the resulting *topic sequence itself* — compared both
> across independent runs of one model and across different models — as
> the object of measurement, showing that path-identity survives where
> sampling-based diversity metrics can be gamed by temperature alone."

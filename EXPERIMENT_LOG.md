# EXPERIMENT LOG

Every experiment, in order. Format: hypothesis → prediction → setup → result →
verdict. Negative results are data. Exact scripts archived in
`experiment-runs/`.

---

## EXP-001 — rejector-validation-v1 (2026-07-16)

**Status:** running (audit in progress before execution)

**Hypothesis (one sentence):** A Haiku-tier LLM rejector labels joke topics
consistently enough to serve as the cascade's measurement instrument —
operationalized as ARI vs. gold partition ≥ 0.80 and reworded-pair invariance
≥ 0.90 on a 32-item hand-built fixture — and beats a crude keyword baseline.

**Predicted deltas (registered before run):**
- rejector `ari_vs_gold` ≈ 0.90
- keyword baseline `ari_vs_gold` ≈ 0.55
- predicted delta (rejector − baseline) ≈ **+0.35**

**Compute on paper:** no training. 32 items × 3 repeats = 96 Haiku calls
(~300 tok in / ~10 tok out each ≈ 30K tokens total) + 0 GPU. Wall time
bound by CLI latency, ~5-10 min serial.

**Disproof attempt (checklist item 4):** built into the design — the
`keyword_baseline` (most frequent non-stopword) runs on identical fixtures.
If the LLM rejector doesn't clearly beat it, use the cheaper thing.

**Comparison design:** same fixtures, same repeat count, same scoring
function for both labelers. Fixture has known structure: 10 topic groups ×
(original / reworded / same-topic) + 2 ambiguous traps scored separately.

**Success criteria:** ARI ≥ 0.80 AND reworded invariance ≥ 0.90 AND beats
baseline on both. Failure → iterate the label prompt (bump
`LABEL_PROMPT_VERSION`, re-run) or reconsider the labeling design before any
cascade runs.

**Result:** FAIL on absolutes, PASS on relative. ARI 0.620 (bar: 0.80),
reworded invariance 0.600 (bar: 0.90), repeat consistency 0.688. Beats
baseline decisively: ARI 0.620 vs 0.271 → **actual delta +0.349 vs predicted
+0.35** (calibration closed; the prior was right, the instrument still isn't
good enough). Report: `experiment-runs/2026-07-16-rejector-validation/`.

**Verdict:** Instrument invalid as-is — but the failure modes are benign and
specific: (1) synonym scatter (`fitness/exercise/gym` all correct, different
words → splits the ARI partition); (2) one prompt parse failure (joke with an
internal colon broke the `Topic:` format — delimiter bug in LABEL_PROMPT v1);
(3) two fixture golds were opinionated (rejector consistently said `flamingo`
for the flamingo-impression marriage joke — defensible). **Zero
punchline-mechanism labels** — the topic-vs-joke discrimination the cascade
depends on held. Iterating: LABEL_PROMPT v2 (delimited joke, "most generic
common noun" instruction) → EXP-002. Design consequence for the cascade
proper: trajectory metrics need *semantic* label equivalence, not string
equality — `flying`≈`travel` must count as one topic. Caught before it could
contaminate any cascade number.

---

## EXP-002 — rejector-validation-v2 (2026-07-16)

**Status:** running

**Hypothesis:** LABEL_PROMPT v2 (delimited joke input; "one most-generic
common noun" output instruction; two added generalization few-shots) lifts
label canonicalization enough to pass the absolute bars: ARI ≥ 0.80,
reworded invariance ≥ 0.90.

**Predicted deltas (registered before run):** ARI 0.620 → ≈ 0.85
(**+0.23**); reworded invariance 0.600 → ≈ 0.90.

**Setup:** fixture repaired, not loosened: weather-a/b replaced (dual-topic —
cold setup, politician butt — same defect class audit-W6 caught in gym-c);
marriage-a/b kept unchanged as a fair test of v2's generalize-up instruction.
Same scoring, same model (haiku), repeats=3. Prompt v1 → v2; `politics`
singularization bug fixed.

**Result:** ARI 0.837 ✓ (bar 0.80, was 0.620 — predicted +0.23, actual +0.217,
calibration closed). Reworded invariance 0.800 ✗ (bar 0.90, was 0.600).
Repeat consistency 0.760 (was 0.688). 7/10 groups now perfectly canonical
across all repeats (`exercise×3`, `travel×3` — v1's synonym scatter is fixed).

**Verdict:** Partial pass. The two remaining invariance misses are `cat` vs
`pet` and `health` vs `medicine` — semantically identical labels failing
STRING equality. Also: v2's generalize-up instruction overshoots on edge cases
(flamingo→`animal`, consistently across the pair — invariant but mis-clustered
vs gold `marriage`), and `cats-c` (cat + cooking joke) is another dual-topic
fixture item of the audit-W6 class — my fixture authoring keeps producing
punchline-pivot jokes; systematic authoring flaw, noted. Decision: do NOT
iterate to prompt v3 (diminishing returns, granularity tension is inherent to
free-vocabulary labeling). Instead score EXP-002's raw labels under the
calibrated semantic label space (EXP-003, zero new API calls) — the same
equivalence machinery the cascade metrics need anyway. If invariance ≥ 0.90
under semantic equivalence, the instrument is valid with the documented caveat
that all downstream metrics use the same equivalence.

---

## EXP-003 — semantic re-score of EXP-002 labels (2026-07-17)

**Setup:** zero new API calls. LabelSpace (all-MiniLM-L6-v2, threshold 0.70
calibrated on a 64-pair fixture) over EXP-002's raw labels.

**Result — negative, twice, instructively:**
- Union-find clustering: invariance 0.900 ✓ but `pet` hub-bridged `cat`↔`dog`
  into ONE cluster (both ≥0.70 to `pet`) → ARI crashed 0.837→0.659. A false
  merge manufactures collapse evidence — the one bias this benchmark cannot
  afford.
- Complete-linkage rewrite (hub-chaining structurally impossible): cat/dog
  correctly split, but greedy linkage handed `pet` to `dog` (higher pairwise
  sim) → cats-b joined the dog cluster → ARI 0.697, invariance back to 0.800.

**Verdict:** `pet` is a HYPERNYM, not a synonym — no flat clustering can place
it. Semantic equivalence over free labels is fragile wherever hypernyms occur.
Complete linkage kept (strictly safer than union-find); semantic layer demoted
to a *reported-alongside* view, never the primary metric.

## EXP-003b — sonnet as rejector (2026-07-17)

**Hypothesis:** Sonnet holds label granularity consistently: raw ARI ≥ 0.85,
raw invariance ≥ 0.90. **Predicted delta vs haiku ARI: +0.06. Actual: −0.204.**

**Result:** Sonnet is WORSE as an instrument: ARI 0.633 (haiku: 0.837),
invariance 0.700 (haiku: 0.800), consistency 0.729 (haiku: 0.760).

**Verdict:** Negative result, kept loud: bigger ≠ better instrument. Richer
models label with richer vocabulary — more granularity variance, the opposite
of what a measurement instrument needs. Haiku stays.

---

## EXP-004 — cascade pilot, 10 models (2026-07-17, pre-registered BEFORE launch)

**Status:** awaiting adversarial audit GO.

**Hypothesis:** frontier and open-weight models share a substantially
overlapping joke-topic pool and walk overlapping escape paths under
accumulating rejection; a nontrivial fraction degrade (repeat an already
rejected topic, or refuse) within 30 turns.

**Setup:** depth 30, N=4 runs/model, rejector = claude:haiku (validated
EXP-002; known limitation: invariance 0.800, conservative bias direction).
Models (11): claude:haiku, claude:sonnet, claude:opus, claude:fable,
codex:sol (gpt-5.6-sol), codex:mini (gpt-5.4-mini), codex:5.4 (gpt-5.4),
api:deepseek (deepseek-chat), api:qwen (qwen-plus-2025-07-28), api:glm
(glm-4.5-air), api:kimi (kimi-k2.5 — k3 was 429-overloaded; added pre-launch
when Sam supplied a fresh key). PRIMARY metrics on RAW labels; semantic
(complete-linkage LabelSpace) reported alongside, never primary — enforced in
code after adversarial audit BLOCKER-1 caught run_pilot scoring canon paths as
primary (canon only merges → headline could only inflate). Audit's other fixes
applied pre-launch: key-fragment scrub in API error paths, CLI error-text
capture, rejector output-shape guard (UNPARSEABLE sentinel). Execution: three
parallel lanes by provider family (claude / codex / api), ~1–1.7 h wall-clock;
cross-model overlap computed post-hoc over merged raw paths.

**Predicted deltas (registered before run):**
- Cross-model mean topic-set Jaccard (raw): **≈ 0.35**
- Within-model mean set Jaccard across runs (raw), averaged over models:
  ≈ 0.55 (models repeat themselves more than they match each other)
- ≥ 4/10 models hit degradation (repeat/refusal) by turn 30
  _(amendment: roster grew 10 → 12 after registration — kimi key arrived,
  grok account arrived; threshold interpreted as ≥ 1/3 of roster, i.e. ≥ 4/12.
  Both additions predate any data from those models.)_

**Compute on paper:** 10 models × 4 runs × 30 turns × 2 calls ≈ 2,400 calls
(~1,200 subscription-CLI + ~480 API at pennies + 2,400 haiku rejector...
corrected: 1,200 model-under-test calls + 1,200 rejector calls). No GPU.
Serial ≈ 2–3.5 h; parallelized by model ≈ 1 h.

**Known validity limits (stated before data):** CLI providers have no
temperature control and encode multi-turn as transcript-in-prompt; per-model
run count N=4 is pilot-scale; the rejector's granularity jitter dilutes
overlap metrics (conservative for collapse claims). This is a PILOT — effect
directions and protocol shakeout, not paper numbers.

**Result:** _(pending)_

**Verdict:** _(pending)_

## EXP-005 — banter judge validation (2026-07-17, pre-registered BEFORE run)

**Status:** running.

**Hypothesis:** A Haiku-tier LLM judge's context-ablation delta cleanly
separates genuinely contextual banter replies from verbatim canned jokes and
from generic on-topic pleasantries, and is not primarily driven by surface
keyword overlap between reply and context.

**Setup:** 30-item hand-authored fixture (10 contextual / 10
generic_responsive / 10 canned — canned drawn verbatim from the 25 ChatGPT
templates), keyword-disjoint contexts verified programmatically; judge =
claude:haiku, repeats = 3, swap partner = `swap_partner(i, n)` over file
order. Runner: `benchmark/validate_banter_judge.py` (22 unit tests with
perfect/constant/echo fake judges lock in the metric mechanics).
~180 judge calls, no GPU.

**Success bars (registered):** separation(contextual − canned) ≥ 3.0;
separation(contextual − generic_responsive) ≥ 1.5; |canned mean_delta| ≤ 1.0;
repeat_delta_stdev_mean ≤ 1.5; keyword_echo_check.risk_detected == False
(generic_responsive Pearson r ≤ 0.5).

**Predicted values (registered):** contextual mean_delta ≈ +4.5;
generic_responsive ≈ +1.5; canned ≈ 0.0 ± 0.5;
separation(contextual−canned) ≈ +4.5 (calibration:
exp-005-banter-judge-v1); separation(contextual−generic_responsive) ≈ +3.0;
repeat_delta_stdev_mean ≈ 1.0–1.5; generic_responsive echo r ≈ 0.2–0.3
(widest error bars — this is the live residual risk from BENCHMARK.md §1b).

**Disproof check (checklist item 4):** compare against `keyword_overlap`
alone as a zero-LLM predictor of gold class from the same raw output — if
raw overlap separates the three classes as well as the judge delta does, the
judge isn't earning its keep.

**Result:** ALL FIVE registered bars PASS (180 calls, 0 unparseable,
27 min): separation(contextual−canned) **6.167** (bar 3.0, predicted 4.5 —
calibration closed); separation(contextual−generic) 2.067 (bar 1.5,
predicted 3.0); canned mean_delta −0.100 (bar |≤1.0|, predicted 0±0.5 —
dead on); repeat_delta_stdev 0.679 (bar ≤1.5); generic-class echo r =
**0.224** (bar ≤0.5, predicted 0.2–0.3 — dead on), risk_detected False.
Disproof: zero-LLM keyword overlap ranks the classes in the same order
(2.8 / 0.8 / 0.1 mean shared content words), so part of the *between-class*
signal co-varies with surface overlap — recorded, not hidden. But the delta
does what overlap can't: separates generic-responsive (4.00) from canned
(−0.10) where overlap is nearly degenerate (0.8 vs 0.1 words), and is
echo-resistant within class. Surprise vs prediction: generic_responsive
scored 4.00, far above the predicted 1.5 — the judge credits topical
responsiveness generously, making contextual-vs-generic (2.07) the
narrowest margin.

**Verdict:** Track 2 instrument VALIDATED at pilot grade. Context-ablation
delta with haiku judge is fit to score banter episodes. Caveat for
training use: because generic on-topic pleasantry already earns ~2/3 of a
genuinely contextual reply's delta, a Track 2 reward should not use delta
alone to push past pleasantry-humor — pair with callback bonus (already in
banter_env) and novelty terms.

---

## EXP-007 — temperature fakeability ablation (2026-07-17, pre-registered BEFORE launch)

**Status:** running.

**Hypothesis:** Raising sampling temperature markedly inflates surface-diversity
metrics (distinct-2, inverse trigram similarity) over the emitted jokes while
within-model path divergence over the same runs' topic sequences moves only
slightly — demonstrating that sampling diversity is temperature-buyable and
cascade path divergence is not (the benchmark's core differentiator claim,
until now asserted rhetorically).

**Setup:** api:deepseek (native API, temperature control), rejector = haiku
(never receives the override), temperatures {0.2, 0.7, 1.2}, N=6, depth 30.
3 parallel lanes → experiment-runs/2026-07-17-temp-fakeability/temp-*/.
540 deepseek calls (~$0.15–0.25) + 540 haiku rejector calls.
Machinery: `--temperature` on run_pilot (API providers only — get_provider
raises on CLI specs, keeping the wrapper confound un-muddied);
benchmark/sampling_diversity.py (distinct-n, pairwise trigram similarity);
26 new tests, all mocked, 212 total green.

**Predicted deltas 0.2 → 1.2 (registered):** distinct_1 +0.13 (0.55→0.68);
distinct_2 +0.21 (0.42→0.63, calibration: exp-007-temp-fakeability
distinct_2_delta); mean_pairwise_trigram_jaccard −0.09; within-model
set_jaccard −0.03 (calibration: path_divergence_set_jaccard_delta);
prefix_depth ~0; norm_edit_distance ~0.

**Success bar (registered):** |Δ sampling family| ≥ 3× |Δ path divergence| —
concretely |Δ distinct_2| ≥ 0.15 while |Δ set_jaccard| ≤ 0.05.

**Known validity limits:** single model (deepseek) — pattern should replicate
on a second API model before the claim generalizes; N=6 per temperature;
deepseek-chat deprecates 2026-07-24 (launched inside the window).

**Result:** BAR CLEARED at 32× the required separation (540 turns, zero
failures). Sampling family 0.2→1.2: distinct_2 0.268→0.658 (**Δ +0.390**,
predicted +0.21 — underpredicted); distinct_1 +0.085; pairwise trigram
jaccard 0.060→0.002 (jokes look near-perfectly "diverse" at temp 1.2).
Path family: topic-set jaccard 0.164→0.229→0.152 (**Δ −0.012**, predicted
−0.03; bound |Δ|≤0.05 ✓; non-monotonic, N=6 noise). Both calibrations
closed. Honest surprise: prefix_depth IS temperature-sensitive
(0.933→0.000) — near-greedy decoding walks near-identical ORDERINGS.

**Verdict:** The differentiator claim survives, and gets sharper:
temperature buys lexical/sample diversity (distinct-k) and shuffles the
ORDER a model walks its topic pool (prefix agreement), but does NOT expand
the pool itself (set jaccard flat). The set-level trajectory metric is the
temperature-unfakeable quantity; papers/pitch should say exactly that
rather than "path divergence" generically. Replication on a second API
model queued as follow-up before the claim generalizes beyond deepseek.

---

## EXP-008 — rejector-validation-v3 constrained vocabulary (2026-07-17, pre-registered BEFORE run)

**Status:** running.

**Hypothesis:** A closed 110-entry topic vocabulary (LABEL_PROMPT v3) removes
free-vocabulary synonym jitter (cat/pet, health/medicine) that caused v2's
reworded-invariance miss, without degrading ARI — by construction eliminating
the specific hypernym/synonym pairs EXP-002 named as the failure mode.
Vocabulary deliberately excludes `pet`/`health`/`medicine` (one canonical
entry per concept — offering the pair is what produced the jitter).

**Predicted deltas (registered):** raw ari_vs_gold 0.837 → ~0.90–0.93; raw
reworded_invariance 0.800 → point estimate ~0.90 (bar ≥ 0.90 — the entire
point; calibration: exp-008-constrained-vocab reworded_invariance_raw 0.90);
repeat_consistency 0.760 → ~0.80–0.85.

**Smoke caveat (2 calls, recorded before the run):** marriage-a (flamingo
joke) returned `bird` under v3 — not `animal` (v2) or `marriage` (gold).
Species granularity gives this straddling item MORE ways to scatter
(bird/animal/marriage) than v2's consistent `animal`. If marriage-a/b don't
converge on one entry, this group flips from v2 HIT to v3 MISS on
invariance — the single biggest swing risk against the 0.90 bar. Concrete,
observed, not hypothetical.

**Setup:** same 32-item fixture, repeats=3, model haiku, RAW scoring, same
metrics as EXP-001/002. 96 calls nominal (192 ceiling if every call
retries; smoke needed 0/2). v2 path untouched and regression-locked by
prompt hashes (186 tests green).

**Disproof attempt:** direct comparison vs EXP-002's report.json on the
identical fixture — if v3 doesn't clear 0.90 invariance while holding
ARI ≥ 0.80, constrained vocabulary doesn't earn its prompt complexity and
the granularity problem gets re-scoped, not iterated.

**Known validity limits:** vocabulary granularity chosen by one author
reasoning about categories, not calibrated on a held-out fixture.

**Result:** DECISIVE PASS, better than predicted (96 calls, 0 retries
saturated): reworded_invariance **1.000** (bar 0.90, predicted ~0.90, v2
was 0.800 — calibration closed at 1.000 vs 0.90 predicted); ari_vs_gold
**0.9237** (predicted 0.90–0.93 — dead center); repeat_consistency 0.958
(predicted 0.80–0.85, v2 was 0.760); zero no-majority items; keyword
baseline unchanged at 0.271. The registered swing risk did NOT materialize:
marriage-a/b both labeled `marriage` — the gold label, not bird/animal
scatter. Majority labels are canonical across the board (cat, dog,
marriage, work, doctor, programming, travel, coffee, exercise, weather).

**Verdict:** Constrained vocabulary is the **paper-grade instrument**. All
pre-registered bars the instrument ever had are now cleared, including the
one v2 missed. v3 becomes the default for future cascade runs; EXP-004
pilot ran on v2 (documented; conservative bias direction unchanged) — a v3
post-hoc relabel of the pilot's stored jokes is queued as a robustness
check so findings can be reported under both instruments.

---

## EXP-006 — labeler-noise bias-direction simulation (2026-07-17)

**Status:** complete.

**Hypothesis:** The claim "labeler noise is conservative for collapse
claims" (asserted since EXP-002, flagged by adversarial review as
untested) holds — noise net-understates cross-model jaccard — but
generalize-up merges (flamingo→animal class) contribute a quantifiable
inflation component. **Registered prediction:**
net_bias_on_cross_model_jaccard ≈ −0.06.

**Setup:** offline Monte-Carlo, zero API calls. Noise rates estimated from
EXP-001/002/003b raw repeat-label logs (haiku v2: match 0.563, synonym-swap
0.149, generalize-up 0.276, other 0.011). 30-topic ontology with shared
hypernyms; synthetic 4-model × 4-run × depth-30 trajectories at five true
overlap regimes; 2000 seeded reps/variant; scored with the real
benchmark.metrics functions. `benchmark/noise_robustness.py`, seed 20260717.

**Result — the defense is REGIME-DEPENDENT and flips sign:**

| true regime (clean jaccard) | net bias | synonym-only | generalize-only |
|---|---|---|---|
| full collapse (1.00) | **−0.466** | −0.379 | −0.192 |
| high (0.39) | **−0.113** | −0.123 | −0.003 |
| moderate (0.23) | **−0.035** | −0.074 | +0.036 |
| low (0.07) | **+0.021** | −0.019 | +0.055 |
| disjoint (0.00) | **+0.046** | ±0.000 | +0.067 |

Calibration closed at −0.035 (moderate regime, the registered scenario) vs
−0.06 predicted — direction right, magnitude close, but the prediction was
regime-naive and the regime structure is the real finding.

**Verdict:** (1) Collapse findings at high overlap SURVIVE noise — the
original defense holds where collapse is actually claimed. (2) At low true
overlap, generalize-up merges MANUFACTURE overlap: measured low overlap is
an OVERESTIMATE of true overlap. Since the EXP-004 pilot is observing
cross-model jaccard ≈ 0.15–0.22 (low-to-moderate regime), the honest
statement is: models' topic pools are AT LEAST as distinct as measured,
and no cross-model collapse claim may be made from this data without the
regime caveat. (3) The blanket "noise is conservative" sentence in prior
log entries and the paper draft is hereby superseded by the table above.
(4) The v3 constrained-vocabulary instrument (EXP-008) eliminates most of
the generalize-up channel by construction — its noise profile should be
re-estimated and this simulation re-run with v3 rates before paper-grade
claims.

---

## Instrument decision (2026-07-17, pilot grade)

**Haiku + LABEL_PROMPT v2, raw string scoring** is the instrument. Passed:
ARI 0.837 ≥ 0.80, beats baseline, zero punchline-mechanism labels across all
runs (the load-bearing topic-vs-joke discrimination). Unmet: invariance 0.800
vs the pre-registered 0.90 bar — recorded as UNMET, not re-bared. Why proceed
at pilot grade: the residual failure is granularity jitter (`pet`/`cat`), and
its bias direction is CONSERVATIVE for collapse claims — label noise splits
topics, making models look MORE diverse, so any collapse we find survives the
noise; diversity findings get flagged instead. Paper-grade fix on the roadmap:
constrained-vocabulary two-pass labeling instead of free labels.

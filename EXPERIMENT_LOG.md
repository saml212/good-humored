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

**Result (2026-07-17 ~02:45, 10/12 models with ≥2 complete runs — kimi 0
runs (reasoning-model token starvation, then session limit), grok partial
JSONLs only (xai timeouts + session limit), fable 2/4 (session limit)):**

- **Cross-model mean topic-set jaccard (raw): 0.102** (predicted 0.35 —
  calibration closed). Semantic view: 0.111. Per EXP-006, at this regime
  measured overlap is an UPPER bound on true overlap. The shared-escape-path
  / ecosystem-collapse hypothesis is DEAD at depth 30: topic pools are
  largely disjoint across models.
- **Within-model mean set jaccard: 0.182** (predicted 0.55). No
  lookup-table signature at depth 30 in ANY roster member.
- **Degradation (repeating an already-rejected topic) is where the story
  lives, and it is family-structured:** claude family degrades in 13/14
  completed runs — haiku 4/4 (depths 22,7,7,7), sonnet 4/4 (20,11,10,14),
  opus 4/4 (13,11,13,13 — eerily consistent), fable 1/2 (18); api
  open-weights similar (deepseek 4/4 median ~8.5, qwen 4/4, glm 2/2);
  **GPT family barely degrades**: mini 0/4, sol 1/4 (turn 26), 5.4 1/4
  (turn 24). Both families run behind comparable CLI wrappers, so the
  claude-vs-codex contrast survives the wrapper confound. Prediction
  "≥1/3 of roster degrades by turn 30" — met massively (8/10 models).
- **Memorized-joke reliance (exact corpus hits / jokes emitted) is a
  SECOND, INDEPENDENT axis:** grok **45%** (35/78 — the "funny" brand is
  the most memorization-reliant model measured), codex:5.4 27%, haiku 26%,
  codex:sol 22%, fable 8%, opus 3%, sonnet 1%, deepseek 1%, qwen 2%.
  Every model emitted scarecrow/atoms/skeleton classics.

**Verdict:** The benchmark discriminates, but differently than registered:
models don't share one well — each lab fails its own way. Anthropic:
constraint collapse (repeats rejected topics by turn ~7–14, every run)
with genuinely low memorization. OpenAI: strong constraint adherence but
heavy verbatim recall (22–27%). xAI: extreme memorization (45%).
Open-weights: fast degradation + template patterns. The cascade separates
topic-pool size from constraint adherence under accumulation; the novelty
check separates both from memorized recall — three orthogonal failure
fingerprints, quantified per lab. Pilot-grade numbers (N=2–4, depth 30,
wrapper confound bounded not eliminated); stats inference + FINDINGS next.

**Correction (2026-07-17 morning, stats inference pass — see
`benchmark/run_stats_inference.py` + `docs/FINDINGS.md` §4.4):**
(1) The "8/10 models degrade" count above does not reproduce: recounting
models with ≥1 degrading run gives **9/10** (only codex:mini fully clean).
(2) The Verdict's "Anthropic: genuinely low memorization" holds for
opus/sonnet/fable (pooled 3.6%) but NOT haiku (25.8% — stated correctly in
the results bullet above, blurred in the verdict phrasing). Haiku's rate
sits statistically in the GPT-family heavy-recall tier (Fisher
haiku-vs-rest-of-family p = 6.5×10⁻¹¹), and haiku is also the rejector
instrument — its dual role is the pilot's most awkward design fact.
(3) Only the single Anthropic-vs-OpenAI family contrast on degradation
depth (p = 0.0002, Cliff's δ −0.917) survives correction; the 45-pair
exploratory battery does not (every Holm-corrected p = 1.0 at N=2–4) and
must not be cited pairwise. IMPORTANT epistemic label (adversarial review
2026-07-17): that family grouping was chosen AFTER this entry's Result
bullet made the family pattern visible — it is a disclosed post-hoc
contrast, strong exploratory evidence, NOT a blind pre-registered test;
it needs a genuinely pre-registered replication before "confirmatory" is
warranted. (4) The family contrast pools haiku's own-cascade depths, and
haiku is also the rejector — the dual-role confound applies to haiku's
degradation numbers too, direction untested; robustness check excluding
haiku queued.

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

**EXP-007b — qwen replication (pre-registered, running):** same design on
api:qwen (native API, versioned code qwen-plus-2025-07-28), temps
{0.2, 0.7, 1.2}, N=6, depth 30. Registered predictions: distinct_2 delta
≈ +0.30 (calibration exp-007b-qwen-replication); set_jaccard delta ≈ −0.02
(exp-007b-qwen-replication-path). Same success bar (sampling ≥ 3× path).

**Result (2026-07-17 morning, post-reset rerun):** REPLICATION INVALID —
manipulation check failed. Measured deltas (0.2→1.2, sampling_diversity
over lane dirs, same instrument as EXP-007): distinct_2 **+0.006** (pred
+0.30, miss), set_jaccard **+0.037** (pred −0.02). But neither number
means what it would on a working manipulation: **all 6 runs at temperature
1.2 open with a byte-identical first joke** (same skeleton classic, same
double-space, same curly apostrophe) — as do all 6 at 0.2. Contrast
deepseek under the identical protocol: 2 distinct first jokes at 0.2
(near-greedy, expected) vs **6 distinct at 1.2** (temperature visibly
honored). Downstream qwen divergence (set_jaccard 0.18–0.30) exists only
because the haiku rejector varies. The temperature parameter demonstrably
never took effect at the qwen endpoint — likely server-side clamping,
ignoring of the param, or response caching on identical requests; these
cannot be distinguished post hoc because the Alibaba free quota exhausted
mid-experiment (temp-0.7 got 4/6 runs, temp-1.2 5/6, temp-0.2 4 full + 1
near-full — unequal n noted). Calibrations closed with measured actuals
(the misses are recorded; the confound explains, not excuses, them).

**Verdict:** qwen is DISQUALIFIED as the second-model replication — no
evidence for or against temperature-fakeability where temperature never
reached the model. EXP-007's deepseek result stands but still generalizes
to exactly one model. Follow-up: replicate on glm (native API, honors
params — TBD via pre-probe) as EXP-007c, WITH a turn-1 variability
manipulation check as a registered pass/fail gate this time.

[LEARN] api-endpoints: Temperature ablations need a manipulation check before interpretation.
Mistake: Ran a full 3-lane temperature ablation on qwen assuming the OpenAI-compat endpoint honors the temperature param; it silently didn't (byte-identical outputs at temp 1.2), costing the lane its replication value.
Correction: Before any sampling-parameter ablation, probe K identical requests at the extreme setting and require ≥K/2 distinct outputs; register the check as a gate. Verify the manipulation reached the model before believing any delta.

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

---

## EXP-007c — glm temperature replication (2026-07-17, pre-registered BEFORE run)

**Status:** running.

**Hypothesis (one sentence):** The EXP-007 temperature-fakeability pattern
(temperature buys surface diversity but cannot expand the topic pool)
replicates on glm-4.5-air, the second native-API model whose endpoint
demonstrably honors the temperature parameter.

**Why glm and not qwen:** EXP-007b disqualified qwen — its endpoint
silently ignored temperature (byte-identical outputs at 1.2). glm was
pre-probed 2026-07-17: temperature >1.0 → loud HTTP 400 (legal range
[0,1], so the param is parsed), 4/4 distinct outputs at 0.95, 3/3 distinct
at 0.05. The [0,1] clamp changes the design: temps {0.05, 0.5, 0.95}.
The 0.05-lane non-determinism (3/3 distinct even near-greedy) means the
diversity floor is high, so the predicted distinct_2 delta is set well
below deepseek's +0.39.

**Design:** api:glm (glm-4.5-air, max_tokens 2048), temps {0.05, 0.5,
0.95}, N=6 runs, depth 30, rejector haiku (never receives temperature).
Same instrument as EXP-007/007b: sampling_diversity over each lane dir.

**Manipulation-check gate (registered, pass/fail — the EXP-007b lesson):**
the 0.95 lane must show ≥3 distinct turn-1 jokes across its 6 runs;
otherwise the experiment is INVALID (no verdict either way), regardless of
deltas.

**Predictions (registered in calibration DB before launch):**
- distinct_2 delta (0.05→0.95) ≈ **+0.15** (exp-007c-glm-replication)
- set_jaccard delta ≈ **0.00** (exp-007c-glm-replication-path)
- Success bar unchanged: sampling delta ≥ 3× |path delta|.

Result: _(pending)_

---

## EXP-009 — semantic novelty tier + validation (2026-07-17)

**Hypothesis (one sentence):** An embedding-similarity tier
(all-MiniLM-L6-v2 cosine vs the 25 known-memorized templates) catches the
paraphrased/reskinned memorized jokes that the n-gram novelty check
provably misses (the documented 2-word-reskin evasion).

**Calibration note (honest):** no blind prediction was registered — the
original builder agent died at the session limit before registration, and
by the time the corrected rerun was designed, the verifier's reproduction
had already revealed the answer. Recorded as an unregistered experiment
rather than pretending a post-hoc "prediction."

**The artifact (kept loud — this is a process result):** the FIRST
validation run reported semantic detection = 0.0 at every edit depth at
FPR≤0.05 (recommended threshold 0.94) — the tier looked useless. A fresh
adversarial verifier proved this was a validation-harness artifact, not a
finding: (1) a held-out-set LEAK — negatives were drawn from the same pool
independently reservoir-sampled into the embedded corpus (3 exact-text
duplicates; negatives max similarity 1.0000); (2) a REFERENCE-SET
MISMATCH — the n-gram baseline was scored against the 25 templates only,
but the semantic sweep was calibrated against templates + 50K
general-corpus rows, and the scraped corpus is pervasively internally
near-duplicated (only exact-string dedup was ever run), so novel jokes
legitimately score 0.6–0.97 against *something* — corpus-redundancy
signal, a different construct from memorized-template paraphrase.

**Result (corrected validation, leak-filtered, templates-only
calibration, n_excluded_leaks=11):** recommended threshold **0.38** at
FPR≤0.05; detection depth_1 **1.000**, depth_2 **1.000**, depth_3 0.500
(n=4), depth_4 **0.810**, hand-written full paraphrases **1.000**.
n-gram baseline on the identical set: 1.0 / 0.64 / 0.0 / 0.0 / 0.0.
Report: `experiment-runs/2026-07-17-semantic-novelty-validation/report.json`.
Runtime aligned with calibration: `SemanticNoveltyPenalty` defaults to
reference="templates" + threshold 0.38; corpus mode requires an explicit
threshold (ValueError otherwise). Wired into `reward_stack()` behind
`semantic_novelty_weight` (default 0.0 — inert until opted in).

**Fresh-audit verdict (separate agent, real-backend probes):** COMMIT.
Template-first embedding order proven by construction AND execution
(max abs diff 0.0 vs independent re-encode); validation report reproduced
byte-identically from scratch; 142/142 env tests. One MAJOR carried
forward as a documented limitation, NOT fixed: **padding/dilution evades
every novelty tier** — a verbatim memorized joke behind ~5 filler-sentence
repetitions zeroes the n-gram term, ~20 zeroes the semantic term. A
policy can recite verbatim inside boilerplate for zero penalty. Docstrings
in both modules now carry the numbers + mitigation direction
(max-over-sliding-windows). Novelty terms are NOT a sole defense for any
real training run until that lands.

**Verdict:** the semantic tier does exactly what it was built for —
closes the 2-word-reskin evasion with 100% paraphrase detection at 5%
FPR — and the adversarial-verification loop caught a would-be false
negative result before it was logged. The remaining exploit class
(dilution) is documented, bounded, and next in line.

[LEARN] validation-design: Score every detector tier against the SAME reference set before comparing them.
Mistake: EXP-009's first validation calibrated the semantic tier against templates+50K general corpus while the n-gram baseline used templates only — the apples-to-oranges reference made a working detector look useless (0% detection at any usable FPR).
Correction: A detector-vs-baseline comparison is only valid when both score against an identical reference set; any extra corpus signal (near-duplicate redundancy) measures a different construct and must be reported separately, never folded into threshold calibration.

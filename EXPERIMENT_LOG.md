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

**Result (2026-07-17):** REPLICATED, all gates passed, 18/18 runs
complete, zero failures (180 jokes/lane). Manipulation check: **6/6
distinct turn-1 jokes at temp 0.95** (bar ≥3; contrast qwen's 1/6) — the
manipulation reached the model. Measured (0.05→0.95, sampling_diversity,
same instrument as EXP-007/007b): distinct_2 **+0.143** (pred +0.15 — a
near-exact prior), set_jaccard **−0.037** (pred 0.00, inside EXP-007's
≤0.05 flat bound), ratio **3.9×** (bar ≥3×). Both calibrations closed.
Nuance kept honest: the middle lane is non-monotonic (distinct_2 0.417 →
0.361 → 0.560) — the diversity purchase concentrates in the top of glm's
[0,1] range; the 0.05 lane's high floor (5/6 distinct firsts) matches the
pre-probe. Prefix agreement is near-floor at all temps (0.13/0.00/0.07),
so the deepseek-specific "prefix collapse" sub-claim doesn't transfer —
glm's walk order is noisy even near-greedy; only the pool-flatness claim
(the one that matters) replicates.

**Verdict:** The core differentiator claim now stands on TWO
honored-endpoint native-API models: temperature buys surface diversity
(deepseek +0.390, glm +0.143) but cannot expand the topic pool (deepseek
−0.012, glm −0.037). Set-level trajectory metrics remain the
temperature-unfakeable quantity. qwen stays disqualified (EXP-007b) — an
endpoint that ignores temperature can neither support nor threaten the
claim. Lanes: experiment-runs/2026-07-17-temp-fakeability/glm-temp-*.

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

**EXP-004 addendum — kimi DROPPED from the cascade roster (2026-07-17
morning, decision rule followed):** kimi-k2.5 is a reasoning model whose
reasoning_content burn grows with the cascade's accumulating rejection
list, so no fixed max_tokens survives depth 30: 400 → empty at turn 1
(original lane), 2048 → died turns 6/12 (first fill), 4096 → died turns
20/18 (last-chance lane, both runs "empty response"). The pre-committed
rule was "4096 or drop with documented failure" — 4096 failed, kimi is
dropped. Zero complete cascade runs ever; its 54.5% memorization rate
(now +38 turns of scraps to re-count in the novelty refresh) remains a
scrap-based flag, never a path finding. A future kimi lane needs
streaming with reasoning-budget control or a non-reasoning kimi variant,
not a bigger constant.

**EXP-004 addendum 2 — fill lanes merged, all numbers refreshed
(2026-07-17 morning):** roster is now 11 models with complete path data
(grok in; kimi out per addendum 1). Refreshed headlines
(stats_inference.json regenerated, integrity 11/11 exact):
- Cross-model jaccard 0.102 → **0.1126** — STILL below the entire
  pooled-frequency null (10k draws, range 0.1189–0.1567, 281 topics,
  diagnostics now persisted in the JSON). Headline survives grok.
- **grok's first complete cascade data breaks its "unmeasured" status
  and completes its profile:** 0 degradations in 4 runs (OpenAI-style
  adherence) + set_jaccard **0.443** (highest within-model overlap
  measured; next is codex:5.4 at 0.281) + memorization tightened to
  **40.9%** [34.3, 47.9] (n=198). Adherent, fixed repertoire, top
  memorizer: the retrieval-machine profile, now with path evidence.
- **fable breaks the Anthropic constraint-collapse pattern:** 1/4
  degradations (opus/sonnet/haiku remain 12/12 uniform). Family contrast
  weakens accordingly: −15.17 → **−13.17** turns (p = 0.0002, Cliff's δ
  −0.917 → −0.781). No-haiku robustness contrast (new in driver):
  **−11.42**, p = 0.0005, δ −0.708 — the dual-role confound does not
  carry the family result. fable memorization on fuller data: 4.7%
  [2.3, 9.4] (n=149).
- kimi scraps recount: 40.0% [28.6, 52.6] (n=60) — stays a flag, never a
  path claim. FINDINGS.md refresh in progress; that document + the JSONs
  are the authoritative statement of these numbers.

---

## EXP-006b — noise-bias re-run with v3 empirical rates (2026-07-17, pre-registered BEFORE run)

**Status:** registered; blocked on the v3 relabel finishing (rates come
from the v2-vs-v3 label comparison on identical jokes).

**Hypothesis (one sentence):** With the v3 constrained-vocabulary
labeler's empirical error rates (invariance 1.000 vs v2's 0.800), the
EXP-006 noise simulation's net bias on cross-model jaccard at the pilot
regime shrinks toward zero, upgrading the pilot's "measured overlap is an
upper bound" caveat to a tighter, paper-grade bound.

**Design:** identical simulator + seed protocol as EXP-006
(benchmark/noise_robustness.py, 2000 reps/regime); only the empirical
(match/synonym/generalize) rate triple changes, extracted from v2-vs-v3
label pairs over the full pilot relabel
(experiment-runs/2026-07-17-cascade-pilot-v3-relabel/).

**Prediction (registered, blind — relabel still running):** net bias at
the pilot regime ≈ **−0.015** (v2 rates gave −0.035)
(calibration exp-006b-v3-rates).

Result: _(pending)_

**EXP-004 terminology correction (2026-07-17, caught by the paper
integration pass):** the verdict above says "three orthogonal failure
fingerprints" — written when grok had no path data and its recall
pattern was folded in with OpenAI's. With grok's complete cascade
profile (addendum 2), the accurate count is **four** per-lab
fingerprints; FINDINGS.md and paper/DRAFT.md now say four.

---

## EXP-008 addendum — v3 FAILS in the field: wild-data coverage collapse (2026-07-17)

**What the v3 relabel of the full pilot found (1,532 turns, 1,327 unique
jokes, experiment-runs/2026-07-17-cascade-pilot-v3-relabel/):** the
constrained 110-entry vocabulary maps **42.6% of wild turns to the
catch-all `other`** (653/1532). v2's single most common wild label —
`comedy`, 10.1% of all turns; models joke about joking constantly under
rejection pressure — has no v3 vocabulary entry (116/155 → `other`),
and neither do `clothing`, `censorship`, `death`, `writing`, `stair`,
and a long tail of everyday topics. v2↔v3 exact agreement: 0.221.

**Consequence 1 — the v3-relabel "analysis" is an instrument artifact,
not a finding:** with 4 in 10 turns sharing one label, repeats are
manufactured everywhere (cross-model jaccard inflates 0.113 → 0.181;
grok "degrades" by turn ~5 after 4 genuinely clean runs; qwen "opens
with `other`" 4/4). None of these numbers supersede anything.

**Consequence 2 — EXP-008's verdict is corrected, loudly:** v3 is
paper-grade ON THE FIXTURE (invariance 1.000, ARI 0.924 — real, but the
fixture only contains in-vocabulary topics, so coverage was untested by
construction). In the field it is INVALID as-is. Fixture validation ≠
field validation. The pilot's v2-labeled numbers (FINDINGS.md) remain
authoritative — v2's known failure mode (synonym splitting) biases
AGAINST collapse claims, which is the safe direction.

**Consequence 3 — EXP-006b is BLOCKED as designed** (its rate extraction
assumed v3 as the cleaner reference; a 42.6% catch-all makes the rate
triple meaningless). Calibration exp-006b-v3-rates stays open, blocked
on v4.

**Next instrument iteration (v4, queued):** expand the vocabulary from
the wild v2 label distribution (add comedy/meta-humor and the observed
tail), keep one-canonical-per-concept, then re-validate BOTH ways:
fixture bars (invariance ≥0.90, ARI ≥0.80) AND a new field-coverage bar
(catch-all + unparseable ≤5% on wild pilot turns) — the bar this failure
teaches. Relabel is cached; a v4 relabel re-runs cheap.

[LEARN] instrument-validation: A constrained-vocabulary instrument needs a FIELD-COVERAGE bar, not just fixture bars.
Mistake: EXP-008 declared v3 paper-grade off perfect fixture scores; the fixture only contained in-vocabulary topics, so the 110-entry vocabulary's wild-data coverage was never tested — in the field 42.6% of turns fell into the catch-all and the instrument manufactured degradations.
Correction: Every labeler validation must include a wild-data coverage check (catch-all + unparseable rate ≤5% on real pilot output) alongside fixture invariance/ARI, and no instrument replaces a validated predecessor until it passes both.

---

## EXP-010 — v4 two-tier labeler validation (2026-07-17, pre-registered BEFORE run)

**Status:** registered; validation runs launching.

**Hypothesis (one sentence):** A two-tier labeler — 127-entry canonical
vocabulary with an enforced alias table for the head, free-specific-noun
escape for the tail — holds v3's fixture-level consistency without v3's
field-level catch-all collapse, because the failure EXP-008's addendum
documented was structural (closed vocabulary vs long tail), not a
labeling-quality problem.

**Design (3-wave build, adversarial audit round 1 NO-GO → structural
redesign → round 2 qualified GO; commit a2aff32):**
1. Fixture validation: `validate_rejector --prompt-version v4` (canon
   path only — the 32-item fixture contains only in-vocabulary golds by
   construction; free-tier safety is validated by 2 and 3, not this).
2. Invariance probes: 18 byte-verbatim wild jokes, 3 evidenced trap
   families × 6 (scarecrow/farming, skeleton/death, bicycle/bike-alias).
3. Field run: v4 labels all 1,532 wild pilot turns
   (`field_coverage --labeler v4`); escape/canon/unparseable rates with
   per-tier histograms.

**Bars and predictions (registered blind — no v4 field data exists):**
- Fixture: invariance ≥0.90, ARI ≥0.80 (bars unchanged from EXP-008);
  predicted invariance **0.95** (exp-010-v4-fixture).
- Probes: ≥5/6 correct-and-consistent per family; predicted overall
  correct fraction **0.94** (17/18) (exp-010-v4-probes).
- Field: escape_rate REPORTED, predicted **0.17** (exp-010-v4-field;
  structural floor 0.154 — auditor and builder derived it independently
  and matched); unparseable ≤0.02, predicted ~0.01. NO catch-all bar —
  the audit proved any such bar unreachable and the redesign made the
  catch-all structurally impossible instead.

**Directional-safety caveat (registered up front, from the re-audit):**
free-tier jitter splits rather than merges — conservative for POSITIVE
collapse claims only; it cannot license "no collapse" claims, since
label-splitting can mask real repetition that joke-text novelty checks
would catch. Any v4-based no-collapse claim must cross-check verbatim
joke repeats.

Result: _(pending)_

---

## EXP-006b — Result (2026-07-17)

**Design amendment (recorded, not hidden):** registered as "v3 empirical
rates"; v3 was field-invalidated (EXP-008 addendum) before its rates
could be extracted, so the run used **v4's** fixture rates — same
construct (labeler error rates vs gold, same 87-labeling fixture
protocol, same hand-classification taxonomy), successor labeler. The
classifier hand-re-derived the documented v2 tally (49/13/24/1) from raw
logs first and matched it exactly before classifying v4 — the precedent
set was verified, not guessed.

**v4 rate triple (canon-path fixture only):** match 0.690 / synonym
0.103 / generalize 0.115 / other 0.092 (v2: 0.563 / 0.149 / 0.276 /
0.011). Match up 13 points, synonym and generalize both roughly halved —
but **other-class errors rose 8×**, all surface-object capture (mouse,
flamingo, pilot, appliance, umbrella): the two-tier design's free path
admits specific surface nouns where v2's retry pressure pushed toward
topic abstraction. A real, measured cost of the v4 design.

**Result (identical simulator protocol, seed 20260717, 2000
reps/regime):** pilot-regime (moderate) net bias **−0.1001** (v2:
−0.0350; registered prediction: −0.015 — a large miss, sign correct,
calibration closed honestly). Mechanism: "other" errors are maximally
dispersive in the simulator, and their 8× rise overwhelms the halved
generalize-up inflation. Full table in
experiment-runs/2026-07-17-noise-robustness-v4/results.json.

**Verdict:** The hypothesis ("cleaner instrument → bias shrinks toward
zero") is WRONG in an informative way: v4 is more accurate per label yet
MORE conservative in aggregate, because its error mass moved from
overlap-manufacturing (generalize-up) to overlap-destroying (dispersive
other). Two genuinely good regime-level changes: v2's low-regime overlap
MANUFACTURING is gone (+0.021 → −0.017) and disjoint inflation is muted
(+0.046 → +0.018) — the two spots where noise could fake a collapse
finding are both closed under v4. Measured overlap under v4 remains an
UPPER bound at pilot regimes, now with more margin. Scope caveat: these
are fixture/canon-path rates; the field-tier noise profile awaits the
EXP-010 field run.

---

## EXP-004 red-team corrections (2026-07-17 midday, hostile-review pass)

A fable-grade hostile review of the full claim chain (verified by
reproducing the published contrasts from raw lanes before attacking)
found, in severity order:
1. **Wrapper-persona contamination is visible in our own transcripts:**
   lane-claude/turns-haiku-r01 spends 25/30 turns in CLI-assistant
   persona ("I'm Claude Code..."); turn-0 topics leak the wrapper
   (fable opens with programming jokes 5/5; codex with computer jokes;
   no API-lane model does either). The log's earlier sentence "the
   claude-vs-codex contrast survives the wrapper confound" OVERCLAIMS
   and is retracted: encoding is uniform (verified: transcript_prompt
   applies to every lane) and temperature can't explain a 13-turn gap
   (depth moves ≤6 turns non-monotonically across temp lanes), but the
   vendor-authored persona channel cannot be excluded. Family claims are
   scoped to model+wrapper deployment stacks pending the same-model
   both-lanes ablation (registered as the decisive next experiment).
2. **Meta-register labels mediate most "constraint collapse" events:**
   11/13 Anthropic degradation events are repeats of comedy/joke/humor/
   ai labels (opus's 13,11,13,13 is `comedy` ×4) — v2's comedy label is
   a low-intensity catch-all of the class that killed v3. Recomputed
   meta-excluded: family contrast −8.79 turns (p=0.0006), no-haiku
   −7.75 (p=0.0016) — SURVIVES, because sonnet/opus also repeat everyday
   topics (appliance, organization) that OpenAI models don't.
   Incidence changes: haiku 3/4, fable 0/4 (fable's one degradation was
   comedy-mediated — the family-outlier story gets cleaner).
3. **Memorization exact-match tier is style-confounded:** sonnet
   prefixes 74% of jokes with framing prose (defeats full-string
   match), grok prefixes 0% — the 40.9%-vs-0.8% gap partly measures
   delivery format. Template-trigram tier (already in novelty.json,
   unreported): grok 20.7% (still the outlier vs ≤10% all others),
   qwen 1.7%→10.0% (= codex:5.4) — the open-weights≈zero tiering is
   tier-dependent. Paper/FINDINGS also MISDESCRIBE the exact-match
   reference (it is the 1.2M Reddit-derived corpus, not "25 templates +
   small hand corpus").
4. paper §5.2/§6 "native multi-turn state" for API lanes is FALSE
   (uniform transcript-in-prompt everywhere — which strengthens the
   comparison and must be stated correctly).
5. glm has a complete excluded run (lane-api r01, repeat@16) with no
   stated exclusion rule; 49 complete cascade runs sit unanalyzed in the
   temp-fakeability lanes (deepseek 18/18, glm 18/18, qwen 13/13
   degrading — uncited replication of the open-weights fingerprint).
Fix wave (zero API calls) launching: meta-decomposition as a standing
robustness row, dual-tier memorization table, factual corrections,
incidence Fisher companion test, temp-lane replication fold-in.
Hostile-reviewer verdict on the drafted paper: weak reject as framed;
"two robust contrasts + validated instrument chain + registered
replication design" is the honest, stronger paper. The ONE
score-moving experiment: same-model both-lanes (~$5).

**Red-team fix wave landed (2026-07-17 afternoon, independent
recomputation):** every headline red-team number reproduced exactly
(meta-excluded contrasts −8.79/−7.75; incidence haiku 3/4, fable 0/4;
grok template-tier 20.7%; sonnet framing-prefix 74.2% vs grok 0%;
temp-lane replication 18/18, 18/18, 13/13). Two Monte Carlo p-values
differ in the 4th decimal (0.0004 vs 0.0006; 0.0018 vs 0.0016 — seed
noise, recomputed values used). One citation correction: the excluded
complete glm run lives in lane-api-fill-glm r01, not lane-api. NEW
discovery from the wave: run_pilot has never stored refusal-aware
degradation depth — every published depth is pure topic-repeat (refusal
detection never contributed); the new decomposition table's refusal
regex also has documented false positives (in-character melodrama).
Both disclosed in stats_inference.json + FINDINGS + DRAFT rather than
smoothed. Driver gained meta-excluded/incidence-Fisher/dual-tier/
decomposition sections; integrity checks pass; 312 tests green.

---

## EXP-010 — Result (2026-07-17)

**All three registered validations ran; 3 of 4 bars pass, one marginal
miss, one new dual-role finding.**
- **Fixture (canon path):** invariance 0.900 (bar 0.90 — at-bar, pred
  0.95), ARI 0.860 (bar 0.80), still beats keyword baseline everywhere.
  Note v4 scores BELOW v3's fixture numbers (1.000/0.924) — the measured
  price of the two-tier prompt; the field is where v4 wins.
- **Invariance probes: 18/18** (pred 0.94) — scarecrow→farming,
  skeleton→skeleton, bicycle→bicycle (via bike alias), all 6/6, all
  canon tier. The three evidenced trap families are closed.
- **Field run (1,532 wild turns): escape_rate 0.1723 vs registered
  prediction 0.17** — a near-exact calibration hit (structural floor
  0.154). canon 0.803; two-tier behaves as designed: comedy absorbs 282
  turns via canon+aliases, ghost stays free (9), 153 distinct free
  labels (splits-not-merges confirmed). Codex-family jokes escape most
  (mini 33%, sol 27%, 5.4 24% — computer/programming-flavored topics
  underrepresented in the vocabulary; noted for a v5 pass, not blocking).
- **Unparseable bar MISSED, and the miss is a finding:** overall 2.48%
  vs bar ≤2.0% — but per-model decomposition shows it is ENTIRELY haiku:
  haiku-as-subject 29.2% unparseable under v4, every other model 0–1.7%.
  The rejector's own jokes break its sibling instrument's parse at 17×
  the roster rate — a fourth manifestation of the dual-role anomaly
  (after its degradation depths, its memorization tier, and its judge
  role). Requires inspection of the failing turns before v4 promotion;
  until then v2 REMAINS the authoritative pilot instrument and v4 is
  field-validated-with-caveat.

**Verdict:** the two-tier redesign survives contact with the field —
escape and canon behave as predicted to within half a point — but
promotion to authoritative instrument is BLOCKED on the haiku
unparseable anomaly, not on coverage. Calibrations: all three closed
(0.90/0.95 ✓sign, 1.00/0.94 ✓, 0.1723/0.17 ✓ near-exact). Data:
experiment-runs/2026-07-17-v4-validation/.

---

## Windowed novelty tiers — dilution exploit closed (2026-07-17, 3-round adversarial cycle)

**The exploit (from EXP-009's audit):** verbatim memorized joke + ~5
filler repetitions evaded the n-gram tier entirely; ~20 evaded the
semantic tier (mean-pool dilution). **The fix:** max-over-sliding-windows
scoring. n-gram windowed mode is ON BY DEFAULT (max can only raise
severity; shipped scores bit-identical on non-adversarial inputs —
fuzzed 500 cases, 0 violations). Semantic windowed mode is OPT-IN,
default OFF: the builder's own real-model check found the whole-text
0.38 threshold is MISCALIBRATED for windows (novel long completions
falsely penalized −0.42) — EXP-011 registered below.

**Adversarial cycle (3 rounds, each finding real defects):**
1. Round 1 killed the original "no-miss guarantee": norm() DELETES
   punctuation, so punctuation-glued padding fused tokens
   ("here.Why"→"herewhy") — full evasion via period/hyphen/U+200B/
   concatenation. Every builder regression test had used space-joins.
2. Fix: character-offset window spans (boundary = whitespace ∪
   punctuation ∪ category Cf) + slice-original-text-then-renorm (the
   internal-apostrophe round-trip case). Round 2 re-audit: original
   table 9/10 at severity 1.0 — but found the boundary set INCOMPLETE:
   combining marks (Mn — Zalgo family) gave full evasion again; also
   sized a ~12× perf regression at the 4096-token scan cap.
3. Final fix: boundary predicate → categories {Cf, Mn, Me, Cc} (Mc
   deliberately excluded — spacing marks behave as word chars); 5 new
   regression tests (U+0301, U+20E0, BEL, NBSP, U+3000 → all severity
   1.0, all with windowed=False evasion controls); perf note with
   measured numbers; max_scan_tokens retuned 4096→1024 under the new
   cost model. Orchestrator-verified end-to-end (Mn-glued verbatim
   corpus joke: −1.5 windowed / 0.0 control; 187/187 env tests).

**Documented residuals (honest, not hidden):** (1) true zero-separator
concatenation with no boundary char physically present is unrecoverable
by any separator-based tokenizer (auditor judged acceptable-as-
documented; the real fix is substring/suffix-index scanning, noted in
docstring); (2) windowed semantic threshold awaits EXP-011; (3)
paraphrase-interleaved-with-filler remains the semantic tier's job.

**EXP-011 (registered, blocked on scheduling):** re-sweep the windowed
semantic threshold — positives embedded in 0/5/20/50 filler reps,
negatives EXPANDED with multi-sentence novel completions at lengths
that straddle each window-ladder level (the auditor's refinement: short
negatives never trigger windowing and would flatter the threshold).
Bars: FPR ≤0.05 on the expanded set; verbatim+padded detection ≥95%;
padding-invariance within 2pp between 5 and 50 reps; paraphrase
detection not below EXP-009's whole-text operating point on unpadded
inputs, else windowed ships as a dilution-only complement. Expected
threshold ≈0.6 region (real-model spot-check), shipped as a separate
WINDOWED_THRESHOLD constant.

[LEARN] reward-hacking: Threshold calibration does not transfer across scoring granularities.
Mistake: assumed the validated whole-text 0.38 semantic threshold would transfer to max-over-windows scoring of the same embeddings.
Correction: detection transfers but the negative-class baseline shifts (short windows of novel text sit ~0.2 closer to short templates); every granularity change needs its own FPR sweep before its threshold is trusted.

[LEARN] adversarial-audit: Text-normalization assumptions are attack surface — test the JOIN characters, not just the content.
Mistake: the windowed no-miss "guarantee" was proven only for whitespace joins, and all its regression tests used space-joins; punctuation-deletion in norm() and then the Mn/Me Unicode gap each gave full silent evasion in successive rounds.
Correction: any guarantee resting on tokenization must enumerate the boundary-character space (whitespace, punctuation, Cf/Mn/Me/Cc, digits, none) and carry a regression test per class, including explicit evasion-control assertions for the non-fixed cases.

---

## EXP-004 addendum 3 — kimi-k3 lane (2026-07-17 evening, pre-registered BEFORE launch)

**Why now:** kimi-k2.5 was dropped (addendum 1) because reasoning burn
scales with the rejection list — the addendum's own fix was "streaming
with reasoning-budget control, not a bigger constant." That control
exists: probing found Moonshot's `thinking: {"type": "disabled"}` takes
kimi-k3 from 595 reasoning tokens / 21.7s on a one-line joke prompt to
**0 reasoning tokens / 1.7–3.0s**, and content survives a 25-topic ban
list at the STANDARD 400-token budget — so k3 runs the exact protocol
every other model ran (no special token budget, unlike k2.5's escalating
failures). Manipulation gate passed before build (the EXP-007b lesson).

**Protocol notes (confounds stated up front):** (1) the endpoint hard
-pins temperature 0.6 (400s on any other value) — this lane has NO
temperature control and carries the same ablation caveat as the CLI
lanes; (2) thinking-disabled is the tested configuration — results are
claims about no-think k3, not about k3-with-reasoning; (3) probe jokes
were verbatim classics (eyebrows; overbooked librarian) — informal
signal, not data.

**Design:** api:kimi-k3, N=4, depth 30, rejector haiku — identical to
the pilot roster lanes. Merges into the pilot analysis on completion.

**Predictions (registered blind):** exact-tier memorization ≈ **0.20**
(kimi-k3-lane; the k2.5 scrap flag was 0.40 [28.6–52.6] but scraps
oversample early turns where classics cluster; depth forces novelty);
median degradation depth ≈ **12** (kimi-k3-lane-depth; template-recall
profile suggests OpenAI-style adherence is unlikely, open-weights-style
mid-cascade collapse is the base rate).

Result: _(pending)_

**EXP-004 addendum 3 — Result (2026-07-17 evening):** kimi-k3
(thinking-disabled) ran the standard protocol cleanly: 4/4 complete
runs, zero failures, ~3s/turn. **Memorization: 25/120 = 20.8% exact-tier
[14.5, 28.9] vs blind prediction 0.20 — near-exact calibration hit.**
Degradation: 3/4 runs at depths 12, 12, 24 (one survivor); censored
median 18 vs predicted 12 — a miss under the survival-inclusive
convention (degraded-only median is exactly 12; convention was not
pinned at registration — lesson: REGISTER THE CONVENTION WITH THE
METRIC). Self-jaccard 0.329 (second only to grok's 0.443) with prefix
agreement 0.0 — a fixed mid-size pool walked in varying order.
Openers are the well-worn classics (farming/bike/math/coffee), matching
the memorization profile: kimi-k3's fingerprint sits between grok
(fixed retrieval repertoire) and open-weights (mid-cascade collapse) —
moderate-heavy recall + moderate pool + mid-depth degradation.

**Roster-level consequence, reported before anything cites the old
number: the 12-model cross-model jaccard is 0.1191, and the "below the
entire pooled-frequency null" claim NO LONGER HOLDS** (null range now
0.1167–0.1597, observed sits just inside the floor; p = 0.9999 —
overlap remains at the extreme low end of chance-cooccurrence, far
below the null mean 0.1377, but "below every draw" died with kimi-k3's
conventional topic pool). Within-model mean 0.218. Incidence 10/12
(mini and grok remain the only fully clean models). Family contrasts
unchanged (kimi is neither Anthropic nor OpenAI). FINDINGS headline
updated accordingly — the shared-pool hypothesis stays dead (0.119 vs
predicted 0.35), it just dies by a less theatrical margin.

---

## EXP-012 — contained-kernel certification vs human consensus (2026-07-17 evening, pre-registered BEFORE run)

**Design constraint this experiment serves (Sam, today):** the RL reward
must be a CONTAINED KERNEL — model calls + pure computation, no humans at
training time. Human judgment data is therefore used exactly once,
offline, to CERTIFY the kernel's judged components. This experiment is
that certification.

**Hypothesis (one sentence):** The kernel's naked funniness judge
(haiku, normalized) ranks Oogiri candidate responses in meaningful
agreement with the ~100-human consensus ranking that already exists for
each prompt.

**Design:** sample N≈30 Oogiri prompts (each with ~100 candidates rated
by ~100 independent judges — popularity-bias-free by construction; data
via data_adapters/oogiri.py, research-only license flag, final license
call still pending with Sam); score every candidate with the contained
judge; per-prompt Spearman ρ vs human consensus; report mean ρ + CI +
distribution. Call budget ≤ ~3,500 haiku calls.

**Predictions (registered blind):** mean ρ ≈ **0.40**
(exp-012-judge-certification; LLM-judge-vs-human-humor agreement runs
moderate in the literature, and Oogiri's consensus is unusually clean).
Floor: ρ ≤ 0.10 means the naked judge is useless as a kernel component
→ kernel redesign blocks any GPU spend. This is a measurement, not a
pass/fail bar — the number IS the certificate.

**Follow-on (EXP-013/014, gates):** the BVT multiplicative gate and
two-stage incongruity gate (THEORY-MAP §12 specs) are being implemented
in parallel; once fixture-validated they re-run this certification. The
registered comparison: theory-structured kernel vs naked judge on the
SAME prompts — if structure beats vibes against human consensus, the
central tenet becomes the empirically superior reward.

Result: _(pending)_

**EXP-012 registration correction (2026-07-17, BEFORE any result —
audit-caught, recorded so the amendment provably precedes the data):**
the registration above misdescribed the instrument. It cites the
literature's Oogiri-Master shape (~100 candidates × ~100 independent
judges, popularity-bias-free). The only adapter that exists is
**Oogiri-GO**: ~6.3 candidates/prompt, consensus ranked on the `star`
field — a popularity-type signal. The harness self-corrected loudly
(its docstring + report.json `data_source` field state this); the
registration did not, until now. Consequences, stated before the number
arrives: (1) the "popularity-bias-free by construction" claim is
RETRACTED for this run — that property belongs to Oogiri-Master, no
adapter yet; (2) per-prompt ρ over ~6 candidates is far noisier than
over ~100 — the mean over 30 prompts stands but its CI will be wide;
(3) the registered prediction 0.40 was calibrated against the cleaner
instrument's literature and STANDS AS REGISTERED (predictions don't
move after registration; if it misses partly because the prior was set
against the wrong instrument description, that miss is recorded like
any other). Also recorded: mid-run, an audit process accidentally
git-stashed the live run's output directory, orphaning the cache and
raw-log inodes — the process survived and report.json writes fresh at
completion, but the on-disk label_cache.jsonl is STALE for resume
purposes and must not be trusted for cache hits on any re-run.

[LEARN] registration-discipline: Verify the instrument's ACTUAL shape against the adapter before registering, not against the literature's description of a sibling dataset.
Mistake: EXP-012's registration described Oogiri-Master's 100-judge panel while the only existing adapter loads Oogiri-GO (~6.3 star-ranked candidates/prompt) — the prediction was calibrated against an instrument we don't have.
Correction: A registration must name the exact dataset+field the harness will consume (adapter, split, ranking field, per-prompt fanout), checked against the loader's code, before the prediction is registered.

---

## EXP-013 / EXP-014 — theory-gate validations (2026-07-17 night, pre-registered BEFORE runs)

**Hypotheses:** (013) the BVT product gate scores the fixture's `both`
class (genuine benign violations) far above all three single-axis
classes — violation-only, benign-only, neither — AND holds the
disclaimer-washing guard (benign mean ≤3.0/10 on
`disclaimer_washed_violation`); (014) the two-stage incongruity gate
passes real setup/punchline jokes while rejecting non-sequiturs,
boring-expected endings, and the vague-abstract gaming probe (gate-2
pass ≤0.25 on the probe class).

**Instruments (exact, per the EXP-012 lesson):** the committed fixtures
env/tests/fixtures/bvt_gate_fixture.jsonl (40 items, 5 classes × 8) and
incongruity_gate_fixture.jsonl (40 items: real_joke 12,
setup_nonsequitur 12, boring_expected 8, vague_abstract_gaming_probe 8);
judge/predictor = claude:haiku via the neutral-cwd CLI pattern
(EXP-003b: haiku is the validated instrument tier); embeddings
all-MiniLM-L6-v2 for the incongruity distance. Bars as registered in
THEORY-MAP §12; runners to be built mirroring validate_semantic_novelty.

**Predictions (registered blind):**
- 013: both-class mean product minus the MAX of the three other class
  means ≈ **+0.40** (exp-013-bvt-validation).
- 014: real_joke both-gates pass rate ≈ **0.65** (haiku predictor noise
  will fail some genuinely good jokes; exp-014-incongruity-validation);
  vague-probe gate-2 pass predicted ~0.15 (bar ≤0.25).

Result: _(pending)_

**EXP-010 promotion-gate resolution (2026-07-17 night — the haiku
unparseable anomaly is exonerating, not damning):** joining the v4
field cache to haiku's turns shows the "29.2% unparseable" is
concentrated in NON-JOKES: the wrapper-persona refusal turns ("I'm
Claude Code, built to help with software engineering tasks. I'm not
going to continue...") from the hijacked runs the hostile review
identified, plus stage-direction outputs ("*nothing*", "*...*"). 13
unique texts account for the whole effect (the same refusal repeats
across turns, inflating the per-turn rate). v4 refusing to topic-label
a refusal is CORRECT instrument behavior — and strictly better than
v2, which labeled those same turns as topics (haiku r01's turn-7 "ai"
repeat, one of the 12/12 Anthropic degradations, is persona text
labeled "ai" by v2). Verdict: the v4 promotion blocker is RESOLVED —
v4 is promoted as the analysis instrument for future paper-grade runs;
the pilot's published numbers remain v2-labeled with the documented
conservative caveat, now plus this note: v2's topic-labeling of
refusal text marginally INFLATES Anthropic degradation counts (haiku
r01's event is persona-driven), which the wrapper-stack scope
reduction already covers. Instrument-evaluation lesson folded into the
field-coverage doctrine: decompose unparseable rates by input type —
a "failure" concentrated on non-joke inputs is the instrument working.

---

## EXP-012 — Result (2026-07-17 night): THE FLOOR FIRED

**mean ρ = 0.056**, bootstrap CI **[−0.14, 0.25]** (29 valid prompts,
184 calls, zero unparseable) vs registered prediction 0.40 — a large
miss, and below the registered floor (ρ ≤ 0.10 → the naked judge is
not certifiable as a kernel component). Zero-call diagnostics from the
cached scores sharpen the diagnosis: the judge is NOT degenerate (it
uses the full 1–8 range with a healthy spread), yet its top-voted-vs-
bottom-voted pairwise win rate is **0.467 — exactly chance**. The judge
holds real opinions; they simply do not track this human signal.

**What this does and does not show (instrument caveats pre-registered
in the correction above):** the target was Oogiri-GO star counts over
~6 candidates/prompt — a popularity-type signal on a tiny per-prompt
fanout, NOT Oogiri-Master's ~96-candidate blind-vote consensus. Two
hypotheses are indistinguishable in this data: (a) haiku's funniness
taste genuinely fails to track oogiri-style human preference; (b) the
star-count target is too noisy/biased to certify against. Per-prompt ρ
ranged −1.0 to +1.0 (n≈6 per prompt is brutal). The Oogiri-Master
acquisition (adapter built, data awaiting Sam's builder-ToS decision)
is now the critical path to distinguishing (a) from (b).

**Registered consequence, honored:** GPU spend on any kernel whose
judge term is load-bearing is BLOCKED pending re-certification — either
against the cleaner instrument (EXP-012b, Oogiri-Master) or by the
theory gates demonstrating better human agreement (EXP-013/014 fixture
validations running; their human-agreement version queues behind the
Oogiri-Master decision). This is the certification protocol doing its
job: the naked judge failed BEFORE a GPU-hour was spent on it, not
after. The verifiable two-thirds of the kernel (novelty, diversity,
repetition, comprehensibility) is untouched by this result — only the
judged third is uncertified.

**Verdict:** honest negative, high value. "Never train on a judge
alone" was the project's founding rule on documented external evidence;
EXP-012 makes it an internally measured fact about our own judge.

---

## EXP-011 — Result (2026-07-17 night): windowed semantic threshold = 0.47, all four bars PASS

Registered prediction 0.60, actual **0.47** (miss of 0.13, direction
correct — the spot-check's ~0.6 eyeball overshot; closed honestly).
Bars: FPR 0.0455 on the expanded 220-negative set (incl. the
straddling-length class the audit demanded) ✓; verbatim+padded
detection 1.0 at every padding level ✓; padding-invariance 0.95pp ✓;
unpadded paraphrase detection 1.0, no regression vs EXP-009 ✓. FPR
curve smooth through the operating point, no cliffs. Whole-text path
re-verified byte-identical (0.38 / same numbers).

**Deployment decision (registered rule says full-replacement
qualifies; the data adds nuance):** the higher threshold costs
deep-reskin sensitivity (depth-4: 0.81 whole-text → 0.38 windowed;
the clamped depth-3 subclass 0.5 → 0.0) while the actual exploit
targets — verbatim-in-padding and paraphrases — hold at 1.0.
Windowed and whole-text now dominate DIFFERENT threat models:
whole-text@0.38 for deep reskins (but dilutable), windowed@0.47 for
dilution immunity (but shallower on heavy substitution). Decision:
`WINDOWED_THRESHOLD = 0.47` ships as a validated constant; default
stays OFF; deployment guidance = windowed for TRAINING runs (where
dilution is the adversarial live threat), whole-text for passive
evals. A max(both-modes) composite would dominate both individually
but its union FPR is unmeasured — registered as EXP-011b if wanted,
not improvised now. 410 env tests green.

---

## EXP-013 / EXP-014 — Results (2026-07-17 night): structure validates, instrument fails — the EXP-012 pattern repeats

**EXP-013 (BVT gate, 265 calls): 7/10 bars, headline FAILS.** Margin
0.031 vs predicted +0.40. The benign axis discriminates beautifully
(separation 8.33), echo checks pass, repeat consistency 0.950, and the
disclaimer-washing guard holds on severe violations. The failure is
localized and diagnostic: haiku's violation judge succumbs to a halo
effect that two SEPARATE judge calls were designed to prevent — it
rates the violation dimension of genuinely funny benign violations
0–4/10 despite explicit instructions to score violation regardless of
funniness. Meanwhile the registered DISPROOF CHECK confirmed the
theory's core claim: the current additive stack scores violation_only
at 80% of both-class reward — the exact compensation failure the
multiplicative gate exists to close. The problem is real; haiku cannot
yet execute the measurement.

**EXP-014 (incongruity gate, 240 calls): 3/9 bars.** real_joke pass
0.389 vs predicted 0.65. The ANTI-GAMING machinery works exactly as
designed: vague-abstract probe 0.083 (bar ≤0.25), nonsequitur gate-2 a
perfect 0.000 (resolution never credits an unrelated punchline), and
the gate-1-alone disproof shows stage 2 genuinely earns its keep. The
failures cluster on gate-1's surprise proxy: single-guess embedding
distance is noisy (a boring continuation phrased differently reads as
"surprising"), and haiku's guess-to-guess variance drops repeat
consistency to 0.556. §12.2 pre-flagged this as "a proxy of a proxy" —
now it is a measured limitation, not a suspicion.

**The unified kernel verdict after EXP-012/013/014 (one night, all
pre-registered, all closed):** the verifiable two-thirds of the kernel
is solid and adversarially hardened; the theory GATES' protective
structure (anti-gaming probes, echo resistance, strict-AND, the
additive-compensation disproof) validates; but EVERY component where
haiku-as-humor-judge is load-bearing failed its discrimination bar —
judge-vs-humans at chance (012), violation halo (013), surprise-proxy
noise (014). The judged third needs a better instrument, not better
structure: candidates, in cost order — multi-sample probe designs
(cheap, EXP-014b), a different judge model for the violation axis
(untested; note EXP-003b's bigger≠better lesson was for LABELING, a
different task), an RM trained on Oogiri-Master consensus (blocked on
Sam's acquisition call). GPU block stands. Bookkeeping: the env test
baseline is 410 (pre-gate-validation), not 306 — the earlier count was
a stale-pycache artifact; 775 repo-wide now green.

[LEARN] instrument-design: Judge-structure fixes cannot rescue judge-capability gaps.
Mistake: expected separate focused judge calls (violation vs benignity) to eliminate the halo effect and single-guess embedding distance to proxy surprisal — both structural fixes to what turned out to be capability limits of the haiku instrument.
Correction: validate the INSTRUMENT on the component task (can this judge rate violation independent of funniness at all?) before designing structure around it; structure multiplies instrument quality, it does not create it.

---

## EXP-015 — semantic step-size trajectories over the cascade pilot (2026-07-22, pre-registered BEFORE analysis)

**Direction context:** first build of the 2026-07-22 refresh (STATE.md):
cascade 2.0 — upgrade the cascade from topic-set arithmetic to
embedding-space TRAJECTORY analysis, porting the Motta et al. (ICLR
2026) semantic-navigation formalism (step-size/velocity, acceleration,
entropy, distance-to-centroid) from human fluency data to LLM cascade
production — an application their own paper names as future work;
novelty double-verified (their statement + our RQ6 search trail).
Also closes THEORY-MAP §5's self-documented weakness
(cluster_switch_stats understates patch structure).

**Hypothesis (one sentence):** models whose cascade trajectories take
larger semantic steps (better patch-departure, in MVT terms) survive
longer under accumulating rejection.

**Instrument (exact, per the EXP-012 lesson):** all-MiniLM-L6-v2
embeddings of each turn's raw v2 topic label (the pilot's authoritative
instrument), unit-normalized; per-run step-size series = cosine
distance between consecutive topic embeddings; per-model mean
step-size, trajectory entropy, mean distance-to-centroid; data =
experiment-runs/2026-07-17-cascade-pilot (12 models, frozen). Zero new
API calls.

**Predictions (registered blind):**
- Headline: Spearman ρ(mean step-size, censored degradation depth)
  across the 12 models ≈ **+0.50** (exp-015-stepsize-survival).
- Secondary (directional, no calibration row): grok shows the LOWEST
  trajectory entropy in the roster (fixed repertoire = confined walk);
  the oscillation guard (step-size high but entropy low) fires for no
  current model but is implemented and tested — the registered gaming
  vector.

Result: _(pending)_

**Queued from the same research pass (not yet registered):** EXP-016
callback-as-transformation (detect_callback currently rewards VERBATIM
reuse — bag-of-words, no transformation requirement, the mirror image
of the reskin bug; fix = gate on callback, score by transformation
distance, EXP-005-style fixture with genuine/coincidental/trivial/none
classes); EXP-017 audience-adaptation vs an adaptive-rejector persona
(highest strategic novelty — rejection-only preference learning has no
literature precedent; needs its own rejector validation cycle first);
EXP-018 aptitude/unreliability decomposition for humor consistency
(port of arXiv:2505.06120 — gives teeth to the RLVR-damage claim).

---

## EXP-014b — multi-sample incongruity gate (2026-07-22, pre-registered BEFORE run)

**Context (expectation-violation research pass, both agents' reports in
transcript):** Sam's "on-purpose hallucination" framing formalizes as
Bayesian surprise — S = KL(posterior‖prior) over interpretations
(Itti & Baldi) — with the Kao/Trott constraint intact: surprise GATES,
never scales. The hallucination-vs-joke signature (spike-then-resolve
vs spike-without-resolve) is ALREADY partially confirmed in EXP-014's
own data: gate-1 fires near-equally for real_joke and
setup_nonsequitur; gate-2 separates them at 0.000 for non-sequiturs.
The gap — real_joke's 0.389 vs 0.65 — was diagnosed as single-guess
sampling noise (repeat consistency 0.556).

**Hypothesis (one sentence):** averaging K=5 cold and K=5 primed
predictor guesses (centroid-based distances) removes the diagnosed
noise without changing the gate's structure.

**Design:** identical fixture, bars, prompts, thresholds, and
provider (claude:haiku + all-MiniLM-L6-v2) as EXP-014; the ONLY change
is K=1 → K=5 per condition with centroid distances (cold-dispersion
reported as a diagnostic). ~11 calls/item vs 3 — cost stated. Runner:
extend validate_incongruity_gate.py with --k-samples.

**Predictions (registered blind):** repeat_consistency(real_joke)
0.556 → **0.82** (exp-014b-consistency); real_joke both-gates pass
0.389 → **0.60** (exp-014b-passrate). Vague-probe bar must STAY ≤0.25
(averaging must not soften the anti-gaming property — the risk to
watch).

Result: _(pending)_

**Queued as EXP-019 (registered design, blocked on policy-model
choice):** the policy-native pivot surprisal-resolution differential —
ΔS = −log P_θ(pivot|setup) + log P_θ(pivot|setup+twist-cue) computed
EXACTLY from the RL policy's own logits (one extra forward pass, full
vocabulary, zero sampling noise, zero API calls). The real Tier A.
Requires plumbing reward terms into policy logits (an architecture
extension to the Callable contract) and a chosen local policy model
(TRANSFER-PLAN's Qwen3-8B). External-API Tier A is FALSIFIED (6-call
probe): echo+logprobs rejected outright by deepseek
("echo should not be used with logprobs"), prefix-mode returns no
logprobs for prefix text, instructed-regurgitation returns all-0.0
logprobs (obedience, not expectation) — scoring foreign fixed text
under these providers is dead; policy-own-rollout scoring is free.

[LEARN] provider-instrumentation: External chat APIs cannot score a given text's conditional logprob — only their own generations'.
Mistake: THEORY-MAP §12.2 left "real logprobs via API" as the assumed Tier-A upgrade path without a pre-probe.
Correction: echo+logprobs is rejected, prefix text is never scored, and instructed regurgitation yields logprob 0.0 (obedience signal); true surprisal is only free for a locally-loaded model scoring its own tokens — judge-side and policy-side surprisal are architecturally different problems and must be designed separately.

---

## EXP-016 — callback-as-transformation (2026-07-22, pre-registered BEFORE run)

**The bug this fixes (coverage-audit find):** benchmark/banter.py's
detect_callback is bag-of-words (≥5-letter content-word overlap after a
gap) with NO transformation requirement — a model literally repeating
its earlier line scores the full callback bonus. Norrick's
reincorporation construct (the craft consensus too) requires return
WITH TRANSFORMATION; a verbatim callback should decay like any repeated
joke (THEORY-MAP §6 novelty-decay chain). This is the self-repetition
reskin bug's mirror image, in reward-path code, found before it shaped
a training run.

**Hypothesis (one sentence):** gating on callback detection and scoring
by transformation distance (reused-but-transformed: the
SelfRepetitionPenalty distance machinery, sign-managed) separates
genuine callbacks from trivial/verbatim reuse on a hand-built fixture.

**Design:** new computable term (NO judge anywhere — verifiable tier):
detection gate (improved lexical+embedding match to an earlier-turn
bit, with the documented false-positive words excluded) × transformation
score (1 − similarity between callback turn and original, floored at
0 for near-verbatim). Fixture (EXP-005 pattern): genuine_callback /
coincidental_word_reuse / trivial_paraphrase / verbatim_repeat /
no_callback × 8 each = 40 items, hand-written. Validation is pure local
compute.

**Predictions (registered blind):** mean(genuine) − mean(trivial_paraphrase
∪ verbatim_repeat) ≈ **0.50** in normalized reward units
(exp-016-callback-margin); coincidental_word_reuse mean ≤ 0.10 (the
false-positive bar — detection gate must not fire on shared common
words); no_callback exactly 0.

Result: _(pending)_

**EXP-015 — Result (2026-07-22): HEADLINE IS AN ARTIFACT; hypothesis
falsified under the clean test.** The measured ρ=+0.80 (p=0.0028)
reproduced bit-for-bit through two independent routes — and then died
under the audit's confound hunt: run_cascade's FIXED 30-turn loop never
stops at degradation, so degraded models' post-repeat oscillation
(step-size ≈ 0; e.g. opus r0's back half is `comedy`×10 at exactly 0.0)
mechanically drags whole-path means down in lockstep with censored
depth. Pre-repeat-only steps: ρ=0.396, p=0.20 (n.s.). Guaranteed-clean
early window (first 5 steps, pre-repeat for all 46 runs): **ρ=−0.15,
p=0.64 — no relationship.** Family-block permutation additionally
raises the raw p ~10× (0.0028→0.0296): part of even the raw number is
the already-known family split re-derived. Registered secondary (grok
lowest entropy) also cleanly false — grok is tied for HIGHEST
within-run topic entropy (4.907 ceiling), which combined with its
0.443 cross-run set overlap yields the sharpened characterization:
grok walks a BROAD path and walks the SAME broad path every run (a
fixed setlist, not a small cage). Oscillation guard: all 4 flags are
false positives of the quartile-threshold design (near-ceiling topic
entropy, flagged via step-entropy OR-logic alone) — REWORK (AND logic
or absolute thresholds) before any citation. Module itself: audited
COMMIT (bit-for-bit reproduction, 39/39 tests, convention checks
against published tables all exact).

**Calibration note (the honest asterisk):** exp-015-stepsize-survival
was closed at the measured +0.80 per protocol — but per the EXP-007b
precedent, that close MEASURES AN ARTIFACT and must never be read as a
successful prediction: the clean-test actual is ≈0, far from the
registered +0.50. A "beat the prediction" close that dies under audit
is the strongest argument this project has for auditing before citing.

**Verdict:** cascade 2.0's first trajectory result is a documented
negative with a methods lesson attached; the trajectory machinery
itself is sound and stays (pre-repeat-windowed metrics are the valid
going-forward form). FINDINGS gets this as a negative-result note, not
a headline.

[LEARN] trajectory-metrics: Fixed-length episode designs leak post-degradation behavior into whole-path statistics.
Mistake: EXP-015's registered metric ("mean step-size") was computed over all 29 steps of a fixed 30-turn cascade, so post-repeat oscillation contaminated the mean and manufactured ρ=+0.80 with the depth variable it was predicting.
Correction: any trajectory statistic on cascade runs must be windowed to pre-degradation steps (or a fixed early window shorter than the minimum observed depth), and the registration must pin the window definition BEFORE the run — an unpinned window is an unpinned convention, the same class as the EXP-014b survivor-median lesson.

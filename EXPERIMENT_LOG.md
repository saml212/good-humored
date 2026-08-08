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

Result: _(pending — run INTERRUPTED, not closed)_

**Status 2026-07-22 17:23:** the K=5 run crashed at 14:16 after 102
raw scoring rows — `claude -p` (the haiku judge transport) began
returning "You've hit your monthly spend limit" and providers.py
correctly raised after retries. This is an infrastructure stop, not a
result: calibrations exp-014b-consistency (0.82) and exp-014b-passrate
(0.60) stay OPEN, no partial-data peeking, the runner extension stays
uncommitted until the completed run is audited. label_cache.jsonl
(206KB) is intact, so the resume is cheap once the transport returns.
Unblock paths: Sam raises the Claude spend cap, waits for the monthly
reset (Aug 1), or supplies a native Anthropic API key (which also
unblocks the $5 wrapper ablation — same key, two experiments).
Operational consequence while the cap is hit: EVERY judge-load-bearing
run is dead (they all ride `claude -p`); only judge-free work can
proceed.

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

**EXP-016 — Result (2026-07-22): MIXED — real fix, fixture-flattered
margin, and three pre-wiring findings.** On the registered instrument
(the committed 40-item fixture) all three bars pass: margin 0.820
(pred 0.50 — calibration closed at the measured value, with the same
asterisk as EXP-015: a beat-the-prediction number that did not
generalize), coincidental 0.000, no_callback 0.000. The bug it fixes is
real and quantified: the OLD detector pays the FULL bonus to verbatim
repeats and single-shared-word coincidences alike; the new term scores
both 0 while genuine transformed callbacks score 0.99.

**But the audit's blind held-out set (15 items authored WITHOUT
consulting the scorer — now preserved as
benchmark/fixtures/callback_transform_holdout.jsonl) deflates the
characterization:** margin drops to **0.555** (bar 0.50 — clears,
barely); the coincidental false-positive bar **FAILS** at 0.333 (bar
≤0.10 — a natural 3-shared-content-word coincidence scores max credit;
the committed fixture had only ever used one shared word by
construction); a hyphen-level transcription edit escapes the 0.8
verbatim floor at 0.35. Three findings any future wiring PR must
design around, none live today (the term is unwired/inert, audited
zero env/ imports, sign-safe):
1. **Correlated blind spot, quantified:** moderate multi-word
   paraphrase (clause reorder + 2-3 swaps, zero new content) earns
   +0.35..0.45 net at default weights — callback credit fires while
   SelfRepetitionPenalty's trigram similarity falls below ITS 0.5
   threshold. Both trigram terms miss TOGETHER past a paraphrase
   intensity; single-word swaps are caught (net −0.15). The semantic
   tier is the designed cross-check; wiring is blocked until it (or a
   tightened floor) covers the dead zone.
2. **Intervening-exclusion regression:** the old detector's
   anti-topical-continuity guard was silently dropped — continuous
   restatement of one topic scores transformation 1.0. Must be
   reintroduced before wiring.
3. **Embedding OR-path inversion:** floor 0.6 sits ABOVE the measured
   genuine-coreference band (0.31-0.48), so as calibrated it can never
   catch its target case and would only extend the paraphrase-gaming
   surface. Ships disabled; recalibrate before ever enabling.

**Verdict:** the term is a strict improvement over the shipped
detector on the two cleanly-fixed failure classes, and NOT yet
validated past its own fixture — logged as mixed, with the held-out
set now a permanent regression fixture.

[LEARN] fixture-validation: Builder-authored fixtures are fit-prone — every fixture-validated metric needs an auditor-authored blind held-out set before its bars are cited.
Mistake: EXP-016's committed fixture used single-word swaps and one-shared-word coincidences by construction; the reported margin (0.82) and FP bar (0.000) were properties of those constructions, not the metric — held-out items dropped the margin to 0.55 and failed the FP bar at 0.33.
Correction: fixture-based validation reports both numbers (committed + blind held-out) or neither; the held-out set is authored by someone who has not seen the scorer, and it ships in the repo as a standing regression fixture.

---

## EXP-019 — policy-native pivot surprisal-resolution differential (2026-07-29, pre-registered BEFORE run)

**Unblocked by:** strategy decision to self-host (vLLM prompt_logprobs
revives given-text scoring on GPU later); locally, the pilot runs on a
small model's own logits — judge-free, $0, immune to the claude -p
spend cap that killed EXP-014b.

**Hypothesis (one sentence):** the policy-native differential
ΔS = S(punchline|setup) − S(punchline|setup+generic-twist-cue),
computed exactly from a local model's teacher-forced logits, separates
real_joke from setup_nonsequitur — the separation the judge-proxy
version (EXP-014, 0.389 vs pred 0.65) was too noisy to certify.

**Pinned definitions (BEFORE run — EXP-015 window lesson):**
- Model: **Qwen/Qwen3-4B-Instruct-2507** (disk-constrained pilot:
  25GB free / 32GB RAM; the instrument re-runs per-candidate at the
  Phase-B screen, so the pilot validates the METHOD, not the model).
- S(cont|ctx) = −mean per-token logprob of continuation tokens,
  teacher-forced, raw-text concatenation (no chat template as
  REGISTERED primary; chat-templated variant reported as diagnostic
  only). Continuation tokens scored only (seam handled by tokenizing
  ctx and ctx+cont and diffing).
- Cold context: `setup + " "`. Primed context:
  `setup + " " + CUE + " "` with the item-independent cue
  **CUE = "And here comes the clever twist that reinterprets the
  setup:"** (parallels EXP-014's generic "clever twist" primer; zero
  item content → zero leakage by construction).
- ΔS = S_cold − S_primed (positive = the twist-expectation unlocks the
  actual punchline — resolution exists).
- Baselines (the 5-minute disproof, built in): S_cold alone (Xie 2021
  surprisal), mean next-token entropy over punchline positions (Xie
  uncertainty). Same model, same fixture, zero extra cost.
- Metric: rank AUC (Mann-Whitney), real_joke (n=12) vs
  setup_nonsequitur (n=12). Diagnostics (no calibration rows):
  max-token surprisal, first-3-token surprisal, per-class means,
  boring_expected placement.
- Fixture: env/tests/fixtures/incongruity_gate_fixture.jsonl (40
  items, 4 classes) — unchanged from EXP-014 → results directly
  comparable. PLUS auditor-authored blind held-out items (EXP-016
  [LEARN]: builder fixtures are fit-prone) — auditor writes them
  during code review, results reported on both or neither.

**Predictions (registered blind, calibration rows added):**
- exp-019 / auc_deltaS_joke_vs_nonseq ≈ **0.85**
- exp-019 / auc_gap_deltaS_minus_surprisal ≈ **+0.25** (surprisal
  alone predicted ~0.60: EXP-014's gate-1 fired for BOTH classes —
  jokes and non-sequiturs are both surprising; only resolution
  separates them).
- Guards (directional, no rows): AUC(ΔS, real_joke vs
  vague_abstract_gaming_probe) ≥ 0.75 (the anti-gaming bar); the cue
  must not uniformly inflate all classes (per-class S_primed drop
  reported).

**Success criteria:** AUC(ΔS) ≥ 0.75 AND gap over surprisal-only
≥ +0.10 on the committed fixture AND no guard violated AND the blind
held-out set direction-consistent. Anything less: honest miss, close
calibrations, diagnose.

**FLOPs/memory on paper:** 3 forward passes × 40 items × ~64 tokens ×
2×4e9 FLOPs/token ≈ 6×10^13 FLOPs — seconds-to-minutes on MPS. Memory:
~8GB fp16 weights + activations ≪ 32GB. Disk: ~8GB of 25GB free. No
GPU spend; GPU-block rule untouched (no judge anywhere in this path).

**Novelty:** confirmed gap #1 of the 2026-07-23 research pass (nobody
computes forward-spike + retrospective-resolution-drop delta as a
humor metric; nearest neighbors Kao ambiguity/distinctiveness and Xie
surprisal/uncertainty are both baselines here).

Result: _(pending)_

**AMENDMENT (2026-07-29, pre-run, from the independent audit):**
1. Audit verdict FIX-FIRST -> fixed before run: start==0 logits
   wraparound guard, NaN/Inf assert (fp16/MPS), loud warning when the
   held-out fixture is absent. Scoring math verified correct (indexing,
   float32 softmax, normalization) against the live tokenizer+model.
2. Registration-text deviation, logged not hidden (audit #7): the
   pinned "cold ctx = setup + ' '" is implemented as ctx=setup,
   cont=' '+punchline. Concatenated bytes identical; auditor verified
   identical start indices + 0 seam mismatches across all 40 items
   both ways. Amendment accepted.
3. Blind held-out fixture now exists (12 items, authored by a separate
   agent that never saw the scorer; jokes ORIGINAL by instruction —
   the memorization-clean contrast). Resolves audit #8 (FATAL).
4. **Blindness disclosure:** the auditor ran the unmodified scorer on
   CPU/float32 against the COMMITTED fixture during audit, so the
   committed-fixture direction is known before the registered run:
   predictions are heading for a bad miss (deltaS negative for ALL
   classes; joke s_cold BELOW nonsequitur s_cold — the memorization
   signature; guard inverted; vague probes carry a leading-"..."
   surface tell, audit #12). Calibration rows were locked BEFORE any
   scoring and will be closed against the registered MPS/fp16 run
   regardless. The held-out set (original jokes) and the registered
   chat-template diagnostic remain unseen — those are the informative
   contrasts for diagnosing memorization vs cue-formatting artifact.

**Result (2026-07-29, registered run: MPS/float16, Qwen3-4B-Instruct-2507,
committed n=40 + blind held-out n=12, raw-text primary + chat-template
diagnostic; experiment-runs/2026-07-29-exp019-pivot-differential/):**

**FALSIFIED — both predictions badly missed, and the held-out set
inverted perfectly.**

| split / mode | AUC(ΔS) | AUC(S_cold) | gap | guard(vs vague) |
|---|---|---|---|---|
| committed, raw (REGISTERED) | **0.160** (pred 0.85) | 0.306 | **−0.146** (pred +0.25) | 0.260 (bar 0.75) |
| held-out, raw | **0.000** | 0.562 | −0.562 | 0.750 |
| committed, chat (diagnostic) | 0.757 | 0.521 | +0.236 | 0.833 |
| held-out, chat (diagnostic) | 0.375 | **1.000** | −0.625 | 0.875 |

Reading, per class means (full table in results.json):
1. **The cue licenses discontinuity, not resolution.** On held-out raw,
   setup_nonsequitur is the ONLY class with positive ΔS (+0.215): a
   "here comes a clever twist" announcement makes an abrupt topic
   change MORE expected — non-sequiturs benefit most from twist
   priming. The conditional-probability translation of EXP-014's
   generative primed-guess probe is not semantics-preserving: the
   generative probe asks "guess the twist" and measures distance to
   the actual punchline; the conditional probe just legitimizes
   discontinuity. Operationalization dead; construct not touched.
2. **Memorization confirmed on the committed fixture** (audit
   prediction): committed real_joke s_cold 3.60 < nonsequitur 4.46 —
   classic puns are UNDER-surprising to a 2026 model. Original
   held-out jokes: s_cold 3.10 vs nonseq 2.99 (raw) — no spike either,
   because mean-per-token NLL over a fluent sentence dilutes the
   onset spike (nonseq punchlines are internally fluent trivia).
3. **ΔS is unstable in every mode** (0.16 / 0.00 / 0.76 / 0.38 across
   split×mode) — the committed-chat 0.757 near-bar is contradicted by
   held-out-chat 0.375. Without the blind held-out set this run would
   have "nearly validated" on the chat diagnostic. The EXP-016 lesson
   earned its keep within one week.
4. **Post-hoc, exploratory, n=4v4 (16 pairs, perm p≈0.014): cold
   surprisal under CHAT formatting separates original jokes from
   non-sequiturs PERFECTLY** (AUC 1.000; committed-chat: 0.521,
   killed by memorized jokes). Hypothesis-generating only: the
   instruct model's expectation model lives in chat format, and plain
   Xie-style surprisal may be the live signal there — IF fixture jokes
   are novel. Feeds the next registration; not a claim.

Calibrations closed: 0.85→0.160, +0.25→−0.146 (two hard misses,
recorded). Queue #1 done.

[DEAD-END] cue-conditioned ΔS (prepended generic twist-cue conditional logprob differential): the cue operationalizes "expect discontinuity," which anti-selects for resolution — non-sequiturs gain most.
Evidence: experiment-runs/2026-07-29-exp019-pivot-differential/results.json (held-out raw AUC 0.000, nonseq-only positive ΔS)

[LEARN] fixture-composition: Logprob-based humor instruments require ORIGINAL fixture jokes — classic puns are memorized by 2026 models and score UNDER-surprising (committed real_joke s_cold below nonsequitur).
Mistake: EXP-014's fixture (classic-adjacent puns) was reused unexamined for a logprob instrument; it was built for a judge+embedding instrument where memorization didn't bias scores.
Correction: any fixture feeding a logprob instrument gets a memorization screen first (the committed/held-out contrast IS that screen: surprisal AUC 0.52 memorized vs 1.00 original under chat); compose fixtures fresh, never from the dad-joke canon.

[LEARN] probe-translation: A generative probe ("guess under instruction X, measure distance to actual") and a conditional probe ("P(actual|X prepended)") are DIFFERENT instruments — translating one to the other silently changes the construct.
Mistake: EXP-019 assumed EXP-014's primed-guess structure survives translation to prepended-cue conditional logprobs.
Correction: when porting a probe across measurement modes, re-derive what the new math actually conditions on; a twist-cue prepension measures discontinuity-licensing, not resolution.

---

## EXP-020 — chat-mode surprisal on novel items: replication + rare-word kill-attempt (2026-07-29, pre-registered BEFORE fixture exists)

**Hypothesis (one sentence):** chat-format cold surprisal
(mean per-token NLL of the punchline as the opening of an assistant
reply to the setup) separates ORIGINAL jokes from fluent
non-sequiturs at adequate power — replicating EXP-019's post-hoc
n=4v4 AUC 1.000 — AND survives a rare-word hard negative designed to
prove it is merely a lexical-unpredictability detector.

**Why the kill-attempt is the point (theory-grounding):** Deckers:
unresolved incongruity is unfunny; surprisal alone measures
incongruity WITHOUT resolution, so THEORY predicts it cannot
distinguish a joke from a sufficiently surprising non-joke. The
post-hoc EXP-019 result may only have separated jokes from FLUENT
trivia (low per-token NLL). The registered hard negative:
grammatical, setup-unrelated sentences with rare-but-real vocabulary.
If those out-surprise jokes, the instrument is a weird-word detector
— reward-hackable by construction (emit rare tokens) — and needs the
resolution pairing (EXP-021) before any load-bearing use.

**Pinned design:**
- Model/scoring: identical to EXP-019 (Qwen3-4B-Instruct-2507,
  MPS/fp16, mean per-token NLL) with the CHAT-TEMPLATE mode as
  REGISTERED PRIMARY this time (that is where the signal lived);
  raw-text reported as diagnostic. Cold pass only enters the metrics;
  primed/ΔS values are computed by the same runner and reported as
  diagnostics (ΔS is already falsified, not resurrected here).
- Fixture: FRESH, blind-authored (author never sees scorer or this
  registration's predictions), 44 items: real_joke×12 (original),
  nonseq_fluent×12, nonseq_rareword×8 (the hard negative),
  boring_expected×6, vague_abstract_gaming_probe×6. EXP-019's 12
  held-out items reported as secondary replication only (their scores
  are now known to the builder — cannot be primary).
- Metrics: auc_scold_joke_vs_fluent_nonseq (primary replication);
  auc_scold_joke_vs_rareword_nonseq (the disproof); guards
  auc_scold_joke_vs_boring ≥ 0.70 and joke_vs_vague ≥ 0.70.
- Runner: env/validate_pivot_differential.py extended with generic
  per-class AUC keys (auc_scold_joke_vs_<cls>, auc_deltaS_joke_vs_<cls>);
  re-audited before run (delta-focused).

**Predictions (registered blind, calibration rows added):**
- exp-020 / auc_scold_joke_vs_fluent_nonseq ≈ **0.80** (regression
  from the n=4v4 1.000).
- exp-020 / auc_scold_joke_vs_rareword_nonseq ≈ **0.40** — honest
  theory prior: rare-word non-sequiturs SHOULD out-surprise jokes,
  i.e. I predict the kill-attempt SUCCEEDS and surprisal-only is
  insufficient without resolution.
- Interpretation matrix (pinned): replication ≥0.75 AND hard-negative
  ≥0.65 → surprisal-chat survives as screen-S2 candidate alone.
  Replication ≥0.75, hard-negative <0.65 → instrument = lexical
  detector; pairs with EXP-021 resolution term (expected outcome).
  Replication <0.75 → post-hoc was small-n luck; lead closed.

**FLOPs/memory:** 56 items × 2 modes × 2 passes ≈ 2× EXP-019's cost
(~5 min wall). $0, judge-free.

**Novelty/simpler-baseline:** the hard negative IS the simpler-baseline
disproof; Xie 2021 measured surprisal on SemEval humor/non-humor, not
against surprisal-matched non-jokes — the matched-surprisal contrast
is the new bit.

Result: _(pending)_

**AMENDMENT (2026-07-29, pre-run, delta-audit FIX-FIRST — all naming,
no math):** (1) Registered metric names corrected to the runner's
actual emitted keys BEFORE any results exist: primary =
auc_scold_joke_vs_nonseq_fluent (pred 0.80), hard negative =
auc_scold_joke_vs_nonseq_rareword (pred 0.40), guards =
auc_scold_joke_vs_boring_expected ≥ 0.70 and
auc_scold_joke_vs_vague_abstract_gaming_probe ≥ 0.70. The earlier
transposed/truncated names in the registration are void. All numbers
read from the CHAT-TEMPLATE blocks (registered primary mode).
(2) Print loop fixed to surface chat-template splits (audit #6 — the
primary would not have printed). (3) Runner docstring de-staled.
Blind fixture landed mid-audit (44 items, counts verified); results
still nonexistent at amendment time, so zero post-hoc risk.

**Result (2026-07-29, registered run: chat-template primary, fresh
blind 44-item fixture + EXP-019 held-out as secondary;
experiment-runs/2026-07-29-exp020-surprisal-novel/):**

**REPLICATION FAILED AT CHANCE — lead closed per the pinned matrix.**

Chat-template (registered primary), fresh fixture:
- auc_scold_joke_vs_nonseq_fluent = **0.507** (pred 0.80) — chance.
- auc_scold_joke_vs_nonseq_rareword = **0.604** (pred 0.40) — moot
  given primary, but the rare-word class did NOT out-surprise jokes
  as strongly as theory-primed; recorded honestly as a second miss.
- Guards: joke_vs_boring 0.917, joke_vs_vague 0.944 — pass on the
  fresh fixture, BUT the same vague guard reads 0.000 on the EXP-019
  held-out author's items (their vague probes OUT-surprise jokes).
  Same class definitions, different blind authors: 1.000 → 0.507
  (primary) and 0.944 → 0.000 (guard). Surprisal-based numbers do not
  transfer across fixture authors.

One-paragraph reading: cold surprisal, in any format, measures
"surprising vs unsurprising" — it separates jokes from boring and
(sometimes) vague text, but NOT from equally-surprising non-jokes,
and its behavior is author-style-dependent. Together with EXP-019
this is a theory-consistent double falsification: Deckers said
unresolved incongruity isn't humor, and two operationalizations of
conditional surprisal have now failed to find the resolution axis.
The only instrument that has EVER separated joke from non-sequitur in
this project is EXP-014's generative primed-guess (gate-2: 0.000 for
non-sequiturs) — EXP-021 (its judge-free policy-generative port,
already queued) is now the SOLE live lead for the screen's S2
construction instrument.

Calibrations closed: 0.80→0.507, 0.40→0.604 (ledger: 33 closed,
0 open). Queue #7 done. Journal #2 closed bad-hypothesis.

[DEAD-END] conditional-surprisal instruments for joke-vs-nonjoke (raw ΔS, cue-conditioned ΔS, chat-mode cold surprisal): all fail to separate jokes from surprisal-matched non-jokes; surprisal has no resolution axis.
Evidence: experiment-runs/2026-07-29-exp019-pivot-differential/ + experiment-runs/2026-07-29-exp020-surprisal-novel/ (primary 0.507 at n=12v12)

[LEARN] fixture-author-variance: A single blind author's style is a confound — surprisal metrics flipped from AUC 1.000 to 0.507 (primary) and 0.944 to 0.000 (guard) across two authors writing the SAME class definitions.
Mistake: EXP-020's fresh fixture used one blind author, and EXP-019's held-out used another single author; each looked internally consistent.
Correction: instrument-validation fixtures pool >=2 independent blind authors and report per-author breakdowns; an instrument that doesn't transfer across authors doesn't transfer to models either.

---

## EXP-021 — generative resolution probe, judge-free policy port of EXP-014 gate-2 (2026-07-29, pre-registered BEFORE build)

**Lineage:** EXP-014's gate-2 (primed-guess distance drop) is the only
instrument in this project that has EVER separated joke from
non-sequitur (nonsequitur resolution exactly 0.000); its weakness was
the haiku judge as predictor (noise: repeat consistency 0.556) and the
claude -p transport (spend-cap dead). This port replaces the judge
with the LOCAL POLICY MODEL as predictor and the single guess with
EXP-014b's registered K-sample centroid. Zero API calls.

**Hypothesis (one sentence):** resolution = d_cold − d_primed —
where d_* is cosine distance between the punchline's embedding and
the CENTROID of K policy-sampled continuation guesses under EXP-014's
verbatim cold/primed prompts — separates real jokes from
non-sequiturs, because a twist-primed guesser can approach a real
punchline (a mechanism connects setup to punchline) but cannot
approach an unrelated one.

**Pinned design:**
- Predictor: Qwen/Qwen3-4B-Instruct-2507 (same pilot), chat-template
  generation (generative probe → chat mode is the natural format),
  temperature 0.8, top_p 0.95, max_new_tokens 40, K=5 per condition,
  per-draw seed = stable hash(item_id, condition, k) for
  reproducibility.
- Prompts: PREDICT_COLD_PROMPT / PREDICT_PRIMED_PROMPT imported
  VERBATIM from env/incongruity_gate.py (comparability with EXP-014).
- Distance: 1 − cosine on all-MiniLM-L6-v2 (the repo's standard
  embedder); centroid = mean of K guess embeddings per condition
  (EXP-014b's registered reduction).
- Fixture: POOLED TWO-AUTHOR set per the fixture-author-variance
  [LEARN]: EXP-019 held-out (author A, 12) + EXP-020 fresh (author B,
  44) = 56 items; both authored blind to all scorers. Author A's
  setup_nonsequitur and author B's nonseq_fluent pool as
  "non-sequitur" for the primary; per-author breakdown MANDATORY.
  (Builder has seen these items' SURPRISAL scores — a different
  instrument; noted as residual risk, mitigated by the metric being
  embedding-generative, not logprob-based.)
- Metrics: auc_resolution_joke_vs_nonseq_pooled (primary);
  auc_resolution_joke_vs_nonseq_rareword and _vs_vague and _vs_boring
  (guards); d_cold-alone AUC (the generative-surprise baseline — the
  built-in simpler-baseline disproof); per-author primary AUCs.
- Diagnostics: guess-punchline token overlap (embedding-gaming check),
  cold/primed guess dispersion.

**Predictions (registered blind, calibration rows added):**
- exp-021 / auc_resolution_joke_vs_nonseq_pooled ≈ **0.75**
- exp-021 / auc_gap_resolution_minus_dcold ≈ **+0.15**
- Guards (directional): joke_vs_rareword ≥ 0.75; joke_vs_vague ≥ 0.65
  (vague punchlines are the hard case — twist-primed guesses may be
  vague-flavored); direction consistent across BOTH authors.

**Success criteria:** primary ≥ 0.70 AND gap ≥ +0.10 AND both-author
direction consistency AND no guard below 0.60. Partial outcomes per
matrix: primary passes but vague guard fails → resolution term needs
the vague-penalty pairing before reward use; primary fails → the
LAST live S2 lead dies and the screen's S2 axis falls back to the
cascade + novelty axes only (honest option of record).

**FLOPs/memory:** 56 items × 2 conditions × 5 samples × ~40 new
tokens ≈ 1.8×10^14 FLOPs ≈ 15–30 min MPS wall. MiniLM (~90MB) lands
on the SSD cache. $0, judge-free.

**Novelty/simpler-baseline:** d_cold-alone baseline built in;
policy-as-own-predictor generative resolution has no external
precedent found (2026-07-23 research pass gap #1 covers the
spike+resolve family; the generative variant is ours).

Result: _(pending)_

**AMENDMENT (2026-07-29, pre-run, audit FIX-FIRST — all fixed before
any generation):** (1) incremental per-item persistence
(partial_items.jsonl) + heartbeat prints (audit #21). (2)
first_sentence now skips colon-terminated preamble lines, and a
suspicious-guess counter (hedge/refusal/fragment detector) is computed
per item and totaled (audit #17/#6) — flagged guesses are counted, not
dropped. (3) d_cold-alone control AUC added for EVERY guard class, not
just the primary (audit #15) — a guard pass with d_cold already
separating carries no resolution signal. (4) INTERPRETIVE CAVEAT,
pinned now: the nonseq_rareword guard is author-B-only (author A wrote
no rareword items), so it cannot get a per-author split — its result
is scoped to one author and is diagnostic, not certifying (audit #14).
Primary metric machinery confirmed clean (math, orientation, verbatim
prompts, key names). MPS seed determinism empirically verified by the
auditor on this exact torch build.

**Result (2026-07-29, registered run: 56 items × 2 conditions × K=5,
0/560 suspicious guesses; experiment-runs/2026-07-29-exp021-generative-resolution/):**

**NEAR-MISS — primary 0.699 vs the pinned 0.70 bar (a miss, per the
letter of the registration), but 3/4 criteria pass and this is the
first instrument in project history with the correct class ordering.**

- auc_resolution_joke_vs_nonseq_pooled = **0.699** (pred 0.75, bar
  0.70 — recorded as a miss; no post-hoc rounding).
- auc_gap_resolution_minus_dcold = **+0.621** (pred +0.15, bar +0.10)
  — d_cold alone is a strong REVERSE separator (0.078: joke punchlines
  are topically near cold guesses, non-sequiturs far — d_cold measures
  topicality, inversely); resolution flips it to above-chance, i.e.
  the primed-approach signal is real and not topicality.
- Per-author: A 0.812 / B 0.618 — direction consistent (criterion
  passes); author variance visible again but sign-stable this time.
- Guards all ≥0.60: rareword 0.797 (author-B-only, diagnostic per
  amendment), vague 0.711 WITH d_cold control at 0.359 (the guard
  pass carries genuine resolution signal), boring 0.852 (d_cold
  control 1.000 — that guard is topicality-confounded, noted).
- Class means, the ordering that matters: real_joke +0.097 >
  nonseq_fluent +0.053 > vague +0.027 > setup_nonsequitur −0.005 >
  rareword −0.019 > boring −0.085. Every non-joke class below jokes.
- Scope note (construct, not bug): the probe measures GUESSABLE
  resolution. Author B's two negative-resolution jokes are the
  absurdist mechanisms (whisper-escalation f05, GPS-personification
  f10); puns/literalizations/reframes score high. Kao-style
  mechanism-inferability is what a guessing probe can see.

**Classification: bad-hyperparam (kept alive), NOT validated and NOT
dead.** The pinned bar failed by 0.001 at n=16v16 with K=5 on a 4B
pilot model — wide CIs, cheapest possible configuration. Tune path,
in cost order: K=5→10 (halves centroid noise), more items per class
(3rd blind author — also satisfies the author-variance rule better),
and the Phase-B screen re-runs this instrument per candidate model
anyway (a stronger predictor model may lift the guessing ceiling).
NO reward-wiring until a tuned configuration clears its own
pre-registered bar (certification discipline unchanged).

Calibrations closed: 0.75→0.699 (best-calibrated prediction this
week), +0.15→+0.621. Ledger: 35 closed, 0 open. Queue #9 done.
Journal #3 closed bad-hyperparam.

---

## EXP-016b — the three callback fixes + fresh blind certification (2026-07-29, pre-registered BEFORE code changes)

**Hypothesis (one sentence):** the three EXP-016 pre-wiring findings are
closed by (1) adding a hyphen-safe content-WORD-set similarity to the
transformation score so clause-reorder paraphrase and
punctuation-edit verbatim both floor to 0, (2) reintroducing the old
detector's intervening-turn exclusion so continuous topic restatement
never gates as a callback, and (3) recalibrating the embedding floor
to 0.35 (above the measured unrelated band, inside the genuine band)
plus an embedding-freshness guard (origin similarity must exceed every
intervening turn's) — certified on a NEW blind set, not the diagnosed
one.

**Pinned design:**
- Fix 1: score = 1 − max(trigram_jaccard, wordset_jaccard); floors:
  trigram ≥ 0.8 (unchanged) OR content-word-set Jaccard ≥ 0.7 → score
  exactly 0. wordset tokens split hyphens (kills the 0.35
  hyphen-escape) and survive clause reorder (kills the dead zone).
- Fix 2: gate counts only FRESH shared words — those absent from every
  intervening turn between origin and current (port of
  detect_callback's documented exclusion).
- Fix 3: DEFAULT_EMBED_SIM_FLOOR 0.6 → 0.35; embedding tier
  additionally requires sim(origin) > max sim(intervening). Still OFF
  by default; validator runs it with real MiniLM as a diagnostic.
- **Honest split:** benchmark/fixtures/callback_transform_holdout.jsonl
  is RELABELED DEV (its failures produced these fixes; improvements on
  it are tuning, not evidence). Certification = NEW blind set,
  third author, ~24 items, classes = the five existing + NEW
  topical_continuity (continuous restatement of one topic — fix 2's
  target). Committed 40-item fixture = regression check (all EXP-016
  bars must still pass).
- Validator extended: --fixture path arg + topical_continuity class.

**Predictions (blind, calibration rows added):**
- exp-016b / blind_margin_new_author ≈ **0.60** (bar ≥ 0.50)
- exp-016b / blind_coincidental_mean ≈ **0.06** (bar ≤ 0.10)
- Directional (dev set, DISCLOSED as tuned-to): margin 0.555 → ~0.75;
  coincidental 0.333 → ~0.05; topical_continuity ≤ 0.10 on both sets;
  committed-fixture regression bars all still pass.

**Success criteria:** both blind bars pass AND topical_continuity
≤ 0.10 on the blind set AND zero committed-fixture regressions.
FLOPs: pure local lexical + one MiniLM pass. $0.

Result: _(pending)_

**AMENDMENT (2026-07-29, pre-run, audit FIX-FIRST):** (1) FATAL fixed:
validator print loop crashed (KeyError) on fixtures without
topical_continuity — guarded, matching build_bars. (2) Stale test
comment fixed. (3) Findings 2-4 ACKNOWLEDGED PRE-RUN as residual
design risks, pinned so a marginal blind bar is attributed correctly:
the fresh-word exclusion can zero a genuine callback when one of two
shared words coincidentally recurs in an intervening turn (auditor
hand-confirmed); wordset Jaccard is coarse on short turns; and the
1−max basis taxes GENUINE scores ~0.15-0.22 (they reuse ≥2 words by
definition). Interpretation rule pinned NOW: if the blind margin bar
fails, decompose before classifying — report the old-basis
(1−trigram-only) genuine mean alongside; genuine-erosion failure =
recalibrate WORDSET_FLOOR/basis (bad-hyperparam), negatives-not-flat
failure = fixes didn't work (bad-hypothesis for the fix design).
Blind fixture landed (24 items, 6×4, punctuation-edit verbatims and
ongoing-topic items verified by author). Module confirmed inert (zero
env/ imports) and sign-safe by audit.

**Result (2026-07-29, registered run: committed regression + dev +
blind2 certification, word-gate primary + --embed diagnostic;
experiment-runs/2026-07-29-exp016b-callback/):**

**CERTIFICATION FAILED — 2/4 blind bars — with a clean decomposition.**

| set / bar | margin (≥0.50) | coincidental (≤0.10) | topical (≤0.10) | no_callback (=0) |
|---|---|---|---|---|
| committed (regression) | 0.629 PASS | 0.000 PASS | n/a | PASS |
| dev (disclosed tuned-to) | 0.692 PASS | 0.250 FAIL | n/a | PASS |
| **blind2 (certifying)** | **0.441 FAIL** | **0.621 FAIL** | **0.000 PASS** | PASS |

What VALIDATED on the blind set: fix 2 (intervening exclusion) is
perfect — topical_continuity exactly 0.0; the wordset floor closes
verbatim AND both punctuation-edit variants (the hyphen escape is
dead); no_callback stays exactly 0.

Margin miss decomposed (pinned rule followed): BOTH causes present —
genuine erosion (old-basis 0.726 → new-basis 0.608, PLUS one genuine
callback zeroed outright by the fresh-word boundary case the auditor
hand-predicted: one of its two shared words recurred once in an
intervening turn) AND synonym-heavy paraphrase sitting at 0.33-0.50
between the floors. Both tunable: bad-hyperparam territory
(WORDSET_FLOOR, min-fresh-words interaction, basis weighting).

Coincidental hard fail is NOT tunable: 3 of 4 blind coincidences score
0.77-0.86. A fresh multi-word overlap IS the lexical signature of a
genuine callback — the gate cannot separate coincidence from
reference at the lexical level, and the --embed diagnostic shows
MiniLM cosine cannot either (coincidental 0.85, topical 0.475 even
WITH the freshness guard). The embedding OR-branch stays off.

Calibrations closed: 0.60→0.441, 0.06→0.621 (ledger 37 closed,
0 open). Term stays UNWIRED; the wiring block now has a NAMED
unblock: a semantic reference tier (local NLI — "does the current
turn invoke the earlier turn's situation?" — the same NLI machinery
semantic entropy already uses). Queue candidate EXP-016c.

[DEAD-END] lexical-only callback detection passing the coincidental FP bar: fresh multi-word lexical overlap is indistinguishable from genuine callback reference by any lexical gate; MiniLM cosine also fails (band overlap + this run's diagnostic).
Evidence: experiment-runs/2026-07-29-exp016b-callback/blind2/report.json (coincidental 0.621 word-gate / 0.851 embed)

[LEARN] detection-vs-scoring split: The callback term's two halves fail independently — TRANSFORMATION SCORING is now solid (verbatim/paraphrase/continuity all floor correctly) while DETECTION is the unsolved half (coincidence vs reference).
Mistake: EXP-016b bundled three fixes under one certification, so a detection-tier wall risked reading as "the fixes failed" wholesale.
Correction: certify detection and scoring separately — scoring bars (verbatim, paraphrase, continuity) vs detection bars (coincidental FP, genuine recall) are different constructs with different unblock paths; register them as separate bar groups.

---

## EXP-022 — hack-monitor v0: decoupling detector over the verifiable reward tier (2026-07-29, pre-registered BEFORE build)

**Sam's original question (2026-07-23) made operational:** "if we could
notice an increase in one differentiable dimension then we should be
able to quantify that and penalize it." The research pass's design rule:
penalize DECOUPLING, not increase — a component rising is what
improvement also looks like; the hack signature is one component rising
while its correlated partners stay flat.

**Hypothesis (one sentence):** a decoupling detector — per-component
robust z-score minus partner co-movement, calibrated on genuine
baseline groups — flags planted single-dimension hack transcripts at
≥0.8 recall with ≤0.10 FP on a blind fixture, and beats a naive
any-component-outlier baseline at matched FP because genuine
improvement (all components rising together) fools naive outliering
but not decoupling.

**Pinned design:**
- Components (verifiable tier, judge-free, shipped weights):
  CorpusNoveltyPenalty (corpus_dir=env/tests/fixtures/corpus),
  SelfRepetitionPenalty (fresh instance per group, in-order),
  IntraGroupDiversityReward (group_size=K), ComprehensibilityReward.
  Vector per completion = the four term outputs.
- Fixture item = one GRPO-shaped group: K=4 completions for one
  prompt. Classes: genuine_baseline ×8 (calibration + FP),
  genuine_improvement ×6 (all dims good — the naive-killer FP test),
  hack_noveltysalad ×6 (rare-word incoherence), hack_paraphrase_repeat
  ×6 (one joke reworded across the group — trigram dodge),
  hack_template_diversity ×6 (one template, surface swaps). 32 groups,
  blind-authored from class descriptions only.
- Detector: median/MAD per component on genuine_baseline;
  decoupling_i = z_i − median(z of other components); group flagged if
  any completion has max_i decoupling_i > τ with z_i > 1. τ swept;
  report recall at the largest τ with FP ≤ 0.10 over BOTH genuine
  classes. Naive baseline: max_i |z_i| > τ′ at the same FP — the
  built-in simpler-baseline disproof.
- **Validity filter (pinned NOW):** a hack group counts toward recall
  only if its TARGET component actually moved ≥1 robust-z vs baseline;
  ineffective plants are reported as failed_plants, not detector
  misses (and honestly summarized — many failed plants = the fixture
  didn't stress the detector).
- Metrics: hack_recall_at_fp10 (primary), naive_recall_at_fp10
  (baseline), fp_rate_genuine_improvement (the decoupling-specific
  claim), failed_plants count.

**Predictions (blind, calibration rows added):**
exp-022 / hack_recall_at_fp10 ≈ **0.80**; exp-022 /
naive_recall_at_fp10 ≈ **0.40** (naive pays for flagging
genuine_improvement, forcing τ′ high). Success: recall ≥ 0.7 AND
decoupling > naive AND FP ≤ 0.10 including genuine_improvement
specifically.

**FLOPs:** pure local lexical scoring, seconds. $0, judge-free.
**Novelty:** within-project (ports 2604.01476's group z-discount idea
from reward-time to monitor-time over OUR component set); no external
novelty claim.

Result: _(pending)_

**AMENDMENT (2026-07-29, pre-run, audit verdict RUN with watch-items):**
(1) The 3-item test corpus makes the novelty column near-constant →
MAD floors → hair-trigger binary; any flag where novelty contributed
gets checked against component_z_max["novelty"] and caveated as a
corpus-size artifact, not a finding. (2) hack_paraphrase_repeat's
validity targets are themselves trigram-based — the class it is
designed to dodge; an all-failed-plants outcome for that class is the
EXPECTED, informative confirmation of the trigram blind spot, and if
it renders hack_recall_at_fp10 uncomputable that gets stated loudly,
not passed over. (3) FP budget reading pinned: POOLED over both
genuine classes (allowed=1 of 14), with fp_rate_genuine_improvement
reported separately as the only genuinely out-of-sample FP number
(baseline FP is in-sample by construction — fitted on itself).

**Result (2026-07-29, registered run;
experiment-runs/2026-07-29-exp022-hack-monitor/):**

**UNINFORMATIVE — the instrument degenerated before the hypothesis was
reached. Recall 0.000 for BOTH detectors, and the mechanism is fully
diagnosed.**

Per-group data: novelty and self_repetition are constant 0.0 across
all 32 groups (3-item test corpus; no verbatim in-group repeats),
diversity near-constant, comprehensibility DISCRETE (a few heuristic
levels). All four columns fit MAD=0 → scale floors at 1e-9 → any
deviation saturates to Z_CAP → 6/8 genuine baseline groups score
maximum decoupling → the FP-budget threshold lands above the cap →
nothing is flaggable. The audit predicted exactly this for the novelty
column (watch-item #1); reality generalized it to all four.

What the run DID establish:
1. **The verifiable tier's outputs are unusable as monitor inputs.**
   Thresholded, quantized reward outputs destroy the variance a
   robust-z monitor needs. The decoupling idea remains untested, not
   falsified — v0.1 must tap the CONTINUOUS pre-threshold signals
   (trigram similarity values, unique-token ratios, corpus max-sim)
   that the terms compute internally and then discard.
2. **The trigram blind spot, quantified on blind data:** 4/6
   blind-authored paraphrase-repeat groups moved NO component at all
   (failed plants) — one joke reworded four ways is invisible to every
   shipped trigram term. Third independent confirmation (EXP-016,
   EXP-016b, now blind transcripts).
3. The same degeneracy warning applies to the PLANNED training-time
   within-group z-score discount (2604.01476 port): z-scoring the
   shipped component outputs at training time would saturate the same
   way. Wire any such discount to continuous signals only.

Calibrations closed: 0.80→0.000, 0.40→0.000 (ledger 39 closed, 0
open — two hard misses; the naive row is also honestly 0.000 since
the degeneracy killed both detectors identically). Queue #3 done;
journal #5 closed bad-hyperparam (input representation, not the
decoupling idea).

[LEARN] monitor-inputs: Quantized/thresholded reward outputs cannot feed distribution-based monitors — MAD/z-score machinery needs the CONTINUOUS pre-threshold signals the terms compute internally.
Mistake: EXP-022 fed the monitor the shipped verifiable-tier OUTPUTS (sparse zeros + discrete levels); every column fit MAD=0, every deviation saturated, genuine and hacked groups became indistinguishable at the cap.
Correction: expose each reward term's raw similarity/ratio signals alongside its thresholded output (a signals-dict return path), and fit monitors — and any training-time z-score discounts — on those; also floor scales to a fraction of the component's observed dynamic range, never an epsilon.

---

## EXP-021b — generative resolution, tuned: K=10, three-author pool (2026-07-29, pre-registered BEFORE run)

**Lineage:** EXP-021 near-missed its bar (0.699 vs 0.70) at the
cheapest configuration with 3/4 criteria passing and first-ever
correct class ordering; closed bad-hyperparam with tune path K=5→10
(halves centroid noise) + third blind author (fixture-author-variance
rule). This is that tune, re-registered. Everything else IDENTICAL to
EXP-021's pinned design (model, prompts, temperature 0.8/top_p 0.95/
max_new_tokens 40, cleanup, distance, metrics, runner).

**Fixture:** pooled THREE-author set: A (12, EXP-019 held-out) +
B (44, EXP-020 fresh) + C (18, NEW — authored blind 2026-07-29,
mechanisms deliberately skewed inferable per the EXP-021 scope note).
Jokes 22 vs pooled non-sequiturs 22. HONESTY NOTE, pinned pre-run:
C's real_joke-c01 is a kneads/needs pun — canon-adjacent despite the
originality instruction; for a GENERATIVE probe memorized guessability
can inflate that one item's resolution. Reported per-item; primary
also reported excluding c01 as a sensitivity check (pinned now, not
post-hoc).

**Predictions (blind, calibration rows added):**
- exp-021b / auc_resolution_joke_vs_nonseq_pooled_k10 ≈ **0.75**
  (bar ≥ 0.70, unchanged from EXP-021's registration)
- exp-021b / auc_gap_resolution_minus_dcold_k10 ≈ **+0.55**
- Guards unchanged: all ≥ 0.60; per-author direction consistent for
  ALL THREE authors (A was 0.812, B 0.618 at K=5).

**Success = certification:** primary ≥ 0.70 AND gap ≥ +0.10 AND
three-author direction consistency AND no guard < 0.60 AND the c01
sensitivity check does not flip the primary across the bar. Pass →
S2 instrument CERTIFIED for the screen (screen re-runs it per
candidate as pinned). Fail → S2 demoted to report-only; screen decides
on S1/S3/S4 + cost (the honest fallback of record).

**FLOPs:** 74 items × 2 × K=10 = 1480 generations ≈ 60-90 min MPS. $0.

Result: _(pending)_

**Result (2026-07-29, registered run, attempt 3 — two prior launches
died to the SSD noowners HF-cache gotcha, no results produced;
experiment-runs/2026-07-29-exp021b-resolution-k10/):**

**CERTIFICATION FAILED — S2 demoted to report-only per the
pre-registered fallback.**

- Primary 0.678 (bar 0.70, pred 0.75) — second consecutive miss, and
  DOWN from K=5's 0.699: K was not the bottleneck.
- Gap +0.618 (pred +0.55) — the construct signal is real and stable.
- Per-author: A 0.750, B 0.715, **C 0.444 — below chance.** Two
  authors concealed what three revealed (the author-variance rule pays
  again).
- **Vague guard COLLAPSED 0.711 → 0.537, mechanism identified:**
  author C's vague probes earn the HIGHEST resolutions in their set
  (+0.10..+0.26). Twist-primed guesses are themselves vague
  twist-shaped text; in MiniLM space they sit close to pseudo-profound
  punchlines. A policy emitting portentous vagueness WOULD score
  resolution — the exact gaming vector the guard existed to catch,
  demonstrated on blind data. Structural (embedding confusability),
  not tunable by K or authors.
- c01 canon-adjacency sensitivity: moot (0.678 incl / 0.690 excl).

Consequences (pinned rule honored, no re-litigating): the screen
decides on S1 (cascade) + S3 (memorization) + S4 (distributional
health) + measured cost; the generative-resolution numbers are still
COLLECTED per candidate as report-only diagnostics (cheap, and the
A/B-stable gap makes them worth watching). Calibrations closed:
0.75→0.678, 0.55→0.618. Ledger: 43 closed, 0 open.

[DEAD-END] MiniLM-embedding generative resolution as a LOAD-BEARING instrument: two registrations (EXP-021, 021b) failed certification; the pseudo-profundity confusability (primed guesses ≈ vague punchlines in embedding space) is structural to the embedding-distance operationalization.
Evidence: experiment-runs/2026-07-29-exp021b-resolution-k10/results.json (author C vague probes out-resolve their jokes)

[LEARN] pseudo-profundity-vector: Twist-primed generative guesses are vague-twist-flavored text — any embedding-distance resolution metric can be gamed by portentous vagueness, and the vulnerability only shows on some authors' renditions of vagueness.
Mistake: EXP-021's vague guard passed at 0.711 with two authors; certification nearly rode on it.
Correction: guards against a gaming vector need MULTIPLE independent renditions of that vector (the three-author rule applies to guard classes, not just target classes); and resolution instruments should pair any embedding distance with a specificity check (vague text is close to everything).

---

## EXP-023 — audience-reaction logprob vs 292M votes (registered 2026-07-30; FIRST NODE EXPERIMENT, blocked only on SSH)

**Direction refresh (Sam, 2026-07-30, codified):** deliverable is a
SELLABLE RL ENVIRONMENT for conversational humor, not a paper; no RM
as product; joke-fixture instrument work stops. The taste signal must
be mathematical and reaction-shaped ("we'll banter at it and see what
it says back" — made into the reward).

**Hypothesis (one sentence):** a large frozen model's SPONTANEOUS
laughter-reaction logprob — the probability its reply to a shared
caption opens with a laughter-class token, in a natural chat
transcript, with NO evaluative instruction — ranks NYCC captions
within-contest in agreement with the 292M-vote human consensus, far
above the instructed-judge floor (haiku rating ρ=0.056; published
judge band 0.17–0.27), and calibration improves with audience scale.

**Novelty (verified pass, strategy-doc §8):** exact mechanism absent;
ancestors are behavioral (Jaques 2019 counts real "ha") or trained
classifiers (Meta RLUF P[Love]); thesis demonstrated adjacent (Gandhi
2601.04436: judge rewards hack, logprob rewards robust — cited, not
claimed). Open: persona-conditioned reaction simulation; zero-training
signal certified on NYCC. Narrow re-check of those two components
scheduled immediately pre-compute (fast-moving space).

**Pinned design:**
- Measure: M(caption) = log Σ_s P_audience(reply begins with s |
  transcript), s ∈ pinned laughter-class string set (haha/lol/lmao/
  😂-family; exact set + tokenizer handling frozen in the runner
  before first scoring). Transcript template: friend messages the
  cartoon's DESCRIPTION (Hessel corpus field — text-only validity
  caveat pinned: humans saw the image) + the caption; audience reply
  begins. No instruction to evaluate anything, anywhere.
- Within-contest Spearman ρ vs mean vote rating; val split (77
  contests, by-contest split already staged); stratified ≤64
  captions/contest (top/mid/bottom by rating).
- **Head-to-head, same model, same items (the in-domain Gandhi
  test):** instructed 1–10 rating vs spontaneous reaction logprob.
  If spontaneous ≫ instructed on the SAME audience, the mechanism
  claim is self-contained.
- Audiences: the full slate (this REPLACES screen-S2; each candidate
  is screened as policy AND audience in the same served session).
- Diagnostics: caption-length confound (corr of M with token count);
  contrast variant M − M(neutral-ack class); mismatched-contest
  control (caption scored under wrong cartoon description should
  drop); persona-conditioned variant on a 5-contest subsample
  (exploratory only).
- Serving: /v1/completions raw-text with logprobs per the runbook;
  bounded prompt lengths; NC-licensed votes used for internal
  certification only (Sam decision pending for anything commercial).

**Predictions (blind, calibration rows added):**
- exp-023 / mean_within_contest_spearman (largest audience) ≈ **0.40**
- exp-023 / scale_delta_rho (largest − Qwen3-8B) ≈ **+0.15**
- Directional: spontaneous > instructed on the same model; length
  confound |r| < 0.2.

**Bars (pinned):** ρ ≥ 0.30 → taste signal CERTIFIED for the env's
reaction-reward slot (group-relative use). 0.15 ≤ ρ < 0.30 →
group-relative-only zone: usable for GRPO ranking iff within-group
(K-sibling) pairwise accuracy ≥ 0.65 on a held-out contest set;
otherwise not load-bearing. ρ < 0.15 → falsified, say so, and the
taste slot ships buyer-supplied-only.

**FLOPs:** pure inference, ~77 × 64 × slate × ~300-token templates —
hours on the node, interleaved with the policy screen (same servers).

Result: _(pending — first node experiment)_

**AMENDMENT to EXP-023 (2026-07-30, pre-run, from the dataset
inventory):** SECOND certification set added, pre-registered before
any scoring: AI-OpenMic's laughter-duration humor coefficient
(continuous 0-4, commercial-safe CC-BY-SA) and, if staging permits,
AMI laughter presence — REAL measured laughter rather than votes, and
free of the NYCC-Zhang NC restriction. Prediction (blind, directional,
no calibration row): reaction-logprob's correlation with AI-OpenMic's
laughter coefficient is direction-consistent with its NYCC ρ. A signal
that ranks votes but not real laughter (or vice versa) is tellingly
partial — the divergence itself would be a finding.

---

## EXP-023a — API Monte-Carlo reaction pilot (2026-07-30, pre-registered BEFORE build; Sam's directive: burn API credits de-risking before H100 time)

**Hypothesis (one sentence):** the reaction hypothesis survives its
cheapest test — when an API chat model is prompted as a texting friend
(no evaluative instruction) and SAMPLED k times replying to a shared
caption, the fraction of replies opening with laughter ranks NYCC
captions within-contest against the vote consensus.

**Relation to EXP-023:** this is the sampled (Monte-Carlo) estimate of
the same quantity the node measures exactly via logprobs. Pilot ρ is
ATTENUATED by k=8 sampling noise and by RLHF-assistant register
(pinned caveats) — so the decision rule is one-sided: a clear positive
de-risks the node centerpiece; a hard zero on ALL providers is a
serious warning to investigate the mechanism before GPU spend; a weak
result is expected and uninformative in the middle.

**Pinned design:**
- Contests: first 12 val-split contests (contest %5==0) having Hessel
  descriptions (deterministic: sorted intersection, take 12).
- Captions: 24/contest stratified by mean rating (top 8 / middle 8 /
  bottom 8, exact-dup captions dropped).
- Prompt: system "You are texting with your close friend. Reply with a
  single short, natural text message." user: cartoon description +
  friend's caption entry. NO evaluative instruction anywhere.
- k=8 samples, temperature 1.0. Providers: api:deepseek, api:qwen,
  api:grok (kimi excluded: pinned temperature; glm optional).
- reaction_rate = fraction of k replies whose first 12 normalized
  chars contain a pinned laughter marker (haha/ha ha/hehe/heh/lol/
  lolol/lmao/lmfao/😂/🤣).
- Metric: mean within-contest Spearman ρ(reaction_rate, mean rating)
  per provider (average-rank ties).
- Degeneracy guard (pinned): provider base laughter rate must land in
  [2%, 80%] or that provider is marked degenerate (assistant register
  suppressing/flooding reactions), excluded from the primary, and
  reported as such.
- Diagnostics: base rate, caption-length correlation, per-contest ρ
  spread. Cache/resume-safe (incremental jsonl); ~2.3k calls/provider.

**Predictions (blind, calibration rows added):**
exp-023a / best_provider_mean_rho ≈ **0.30**;
exp-023a / n_providers_above_015 ≈ **2** (of 3).

**Cost:** ~2.3k calls × 3 providers × (~300 in + ~40 out) tokens —
single-digit dollars of existing credits. Zero GPU.

Result: _(pending)_

**AMENDMENT (2026-07-30, pre-run):** the harness's make_openai_compat
sends a single user message (no system-role support); the pinned
system text is therefore prepended to the user message as a leading
instruction block. Same words, one role. Noted before any call is made.

**AMENDMENT 2 (2026-07-30, pre-run, audit FIX-FIRST → fixed):**
(1) Multi-methodology contests (510/515 have LilUCB + RoundRobin
summary files agreeing only at ρ≈0.2-0.28) are now VOTES-WEIGHTED
MERGED across all files instead of an alphabetic-accident pick;
source files recorded per contest (audit #1). (2) Pinned NOW:
provider counts toward the primary only if n_contests_scored ≥ 8/12
(underpowered flag), errors ≤ 20% of expected calls (incomplete
flag), and base rate in band — the counts_for_primary field encodes
all three (audits #2/#4). (3) Cache-resume tolerates a truncated
trailing line (audit #5). (4) Post-run manual spot-check of cached
replies per provider REQUIRED before trusting any low base rate as
"no laughter" vs meta-reply framing failure (audit #3) — assessed in
the close-out. (5) Registration prose gap noted: contest selection
also requires an existing CSV with ≥24 usable rows (525 skipped so).

**Result (2026-07-30, registered run;
experiment-runs/2026-07-30-exp023a-reaction-pilot/):**

**ZERO qualifying providers — closed 0.0/0 per the pre-pinned
exclusion rule. The guards fired exactly as designed, and the failure
mode is characterized, not mysterious.**

| provider | base laugh rate | band | mean ρ | verdict |
|---|---|---|---|---|
| deepseek | 0.853 | >0.80 | 0.106 (12 contests) | DEGENERATE (ceiling) |
| grok | 0.999 | ≫0.80 | 0.306 (2 contests) | DEGENERATE (brand personality) |
| qwen | — | — | — | INCOMPLETE (2304× HTTP 403 quota — infra; Sam payment toggle) |

Spot-check (mandated, audit #3): replies are GENUINE reactions, zero
meta-replies observed — the saturation is sycophantic politeness at
"my caption, texting friend" framing. Both models laugh warmly at
top, middle, and bottom captions alike.

**The mechanistic reading that matters for the node:** at ceiling,
k=8 BINARY sampling carries no rank information (P=0.98 and P=0.9999
both sample as all-laughs) while the LOGPROB retains full resolution —
the pilot's method degrades in exactly the regime it found, the node's
method does not. So the reaction hypothesis is neither confirmed nor
falsified: the pilot instrument saturated. Deepseek's residual
ρ=0.106 across 12 contests DESPITE ceiling compression is a faint
positive lean, cited as nothing more.

Follow-up registered below (EXP-023b): neutral third-party framing to
deflate the ceiling and give sampling back its resolution. Marker set
gains 💀 (observed false negative). Calibrations closed 0.30→0.0,
2→0 per the exclusion rule (honest hard misses; ledger 49 closed,
0 open).

[LEARN] reaction-register: RLHF'd assistants in first-person-friend framing laugh at ~everything (sycophancy flooding) — base-rate guards are mandatory, binary sampling loses all rank signal at ceiling, and logprob-based measures must report the MAGNITUDE, never a thresholded laugh/no-laugh.
Mistake: EXP-023a's friend-shares-their-own-caption framing invited polite laughter; k=8 sampling then had zero resolution.
Correction: reaction measures need (a) framings that don't make the reaction a favor to the speaker (neutral/observer framing), (b) pinned base-rate bands with exclusion, (c) continuous magnitudes over binary onsets wherever available.

---

## EXP-023b — reaction pilot, neutral-observer reframe (2026-07-30, pre-registered BEFORE run)

**Hypothesis (one sentence):** removing the sycophancy pressure — the
audience model overhears a caption rather than being handed a
friend's own entry — deflates the laughter base rate into the pinned
band and lets sampled reaction rate rank captions within-contest.

**Pinned changes from 023a (everything else identical, same pilot
set, same k=8, same guards):**
- Framing: "Your friend is reading New Yorker caption contest entries
  out loud from the internet. The cartoon: {description}. They read
  this one: \"{caption}\"" — the caption belongs to a STRANGER;
  reacting honestly costs the friend nothing.
- Markers: + "💀", "dead", "dying" (observed false-negative family).
- Providers: deepseek + glm (grok dropped: personality-saturated at
  0.999, no framing will fix a brand; qwen pending Sam's quota
  toggle).
- Predictions (blind, calibration rows): exp-023b /
  deepseek_base_rate_in_band = 1.0 (i.e., lands within [0.02, 0.80]);
  exp-023b / best_provider_mean_rho = **0.25**.

Result: _(pending)_

---

## EXP-023c — API logprob-magnitude reaction arm (2026-07-30, pre-registered BEFORE run; the decisive cheap test)

**Chain:** 023a (sampled, friend-framing): all providers degenerate/
infra. 023b (sampled, neutral framing): clean POWERED NULL on glm
(qualifying mid-tier audience; laughter varies per caption but tracks
something other than human consensus); deepseek ceiling-degenerate
under both framings. Remaining confound before concluding against the
reaction hypothesis at API scale: BINARY SAMPLING discards magnitude —
ceiling providers are unreadable and in-band providers carry binomial
noise. This arm measures the ACTUAL mechanism (the node method):
first-token laughter-class probability MASS from top_logprobs.

**Capability probes (infra, 3 calls, logged):** deepseek returns
top_logprobs (VERIFIED; reply distribution opens with */quote/"H
forms — hence prefix-normalized token classes). glm: thinking-disable
yields content but NO logprobs (unsupported — excluded). qwen quota
still exhausted. Deepseek-only lane, stated plainly.

**Pinned design:** same 12-contest/24-caption pilot set and
neutral-observer template as 023b; ONE call per caption (288 total),
max_tokens=2, temperature=1.0, top_logprobs=20.
L_strict(caption) = log Σ P(first token ∈ strict laughter class):
normalized token (strip leading asterisks/quotes/space, lowercase)
equal to "ha" or starting with one of haha/hah/hehe/heh/lol/lmao/
lmfao/laugh/😂/🤣/💀; zero-mass floor log(1e-8). L_loose adds
ambiguous stubs ("l", "*l") as a reported bound, never primary.
Diagnostics: repeat-call determinism on 24 captions, caption-length
correlation, per-contest spread, mean strict mass (the ceiling
readability check: deepseek's 0.85 sampled rate should map to HIGH
but VARIABLE mass — variance of L is the whole bet).

**Prediction (blind, calibration row):** exp-023c /
mean_within_contest_rho_strict ≈ **0.25**.

**PINNED CONSEQUENCE (written before the result exists):**
ρ ≥ 0.15 → the reaction mechanism survives its cheapest honest test;
node EXP-023 proceeds as the screen's centerpiece arm (scale
hypothesis intact). ρ < 0.15 → the reaction hypothesis has now failed
sampled-binary AND logprob-magnitude at API scale: EXP-023 is DEMOTED
from centerpiece to one exploratory arm of the screen, the env's
taste slot leads with buyer-supplied preference signal + the
multi-dataset emulator, and no GPU hour is justified by the reaction
bet alone. Either way the H100 day proceeds — what changes is what
it's for.

Result: _(pending)_

**AMENDMENT (2026-07-30, pre-run, audit RUN + gaps pinned):** (1) The
023a/b qualification gates APPLY to 023c's pinned consequence:
n_contests_scored ≥ 8/12 AND errors ≤ 20% of ~312 calls, applied
manually from results.json before the ρ≥0.15 rule fires (audit #7).
(2) Top-20 truncation acknowledged: laughter mass outside the top 20
tokens reads as floor — attenuates/compresses, cannot inflate ρ
(conservative for the decision); frac_captions_at_floor and
strict-token-present-in-top20 diagnostics MANDATED in Assess from the
cached raw tokens (audit #4). (3) Known conservative undercount:
trailing-punct "Ha!" tokens miss the exact-match rule — recoverable
from cache post-hoc; direction is against us, not for us (audit #2).

**Result (2026-07-30; experiment-runs/2026-07-30-exp023c-reaction-logprob/):**

**ρ = 0.122 — below the pinned 0.15 bar with EVERY gate passing. The
pinned consequence fires; no judgment call was needed or made.**

The measurement is as clean as an API experiment gets: 12/12 contests,
0/312 errors, logprobs exactly deterministic (repeat diff 0.0), mean
strict mass 0.152 (readable — no ceiling, no floor), 0/288 captions at
floor, 288/288 with strict tokens in top-20, classifier sensitivity nil
(0.122 both variants; loose bound 0.134). Every known bias was
conservative and none were material.

**Chain verdict (023a → b → c):** the spontaneous-reaction signal at
API scale EXISTS but is WEAK — 0.122, below the published
instructed-judge band (0.17–0.27). Sampled-binary versions are
null-or-degenerate; the logprob magnitude is real but sub-bar on a
deepseek-class audience. The scale hypothesis (frontier-size audience
→ calibrated reactions) survives as an EXPLORATORY screen arm only.

**Consequence executed:** EXP-023 demoted from node centerpiece to one
cheap screen arm. The env's taste slot leads with the multi-dataset
quantile emulator + buyer-supplied preference signal; reaction-logprob
becomes a candidate auxiliary term, adopted only if the screen's
frontier-scale arm clears the same bar this arm failed. H100 day
proceeds with the screen + emulator as its purpose.

Calibration closed 0.25→0.122 (ledger 52 closed, 0 open). Under $20
of API credits resolved the plan's biggest bet before it cost a GPU
day — this chain is the de-risking directive working as ordered.

---

## EXP-024 — familiar/novel embedding BAND term (2026-08-06, pre-registered BEFORE fixtures are read)

**Sam's construct (2026-07-30 spec):** a reply must be FAMILIAR enough
to land — anchored to the conversation — and NOVEL enough to break
expectation. Necessary-condition GATE, not a funniness rating
(explicitly distinct from the falsified EXP-019/020/021 rating
claims).

**Pinned design:**
- band_pass(context, reply) = (anchor_sim ≥ A_floor) AND
  (context_overlap < W_ceil), where anchor_sim = max cosine
  (all-MiniLM-L6-v2) between the reply and each context turn, and
  context_overlap = max hyphen-safe content-word Jaccard vs each
  context turn (the EXP-016b-certified wordset machinery, reused).
  Floor edge rejects off_topic / generic_filler / word_salad;
  ceiling edge rejects parrot.
- Fixtures: TWO blind authors (D: dev, E: certification), 25 items
  each, 5 classes × 5 — witty_anchored (in-band) vs off_topic /
  parrot / generic_filler / word_salad. Author-variance rule applied
  from the start this time.
- Thresholds (A_floor, W_ceil) swept ONLY on author D; author E
  scored ONCE with dev-chosen thresholds. Primary = author-E balanced
  accuracy (witty_anchored = positive class).
- Diagnostics: per-class pass rates both authors, threshold
  sensitivity (does a ±0.05 shift flip the verdict), MiniLM sim
  distributions per class.

**Predictions (blind, calibration row):** exp-024 /
blind_balanced_accuracy ≈ **0.84** (bar ≥ 0.75). Directional: floor
edge catches ≥ 4/5 of each of off_topic+salad; parrot ceiling catches
≥ 3/5 parrots (wordset on short replies is coarse — known); filler is
the hard case (may embed near chat-register centroid — if filler
passes the floor, that is the finding, and the fix direction is a
content-word minimum, noted now).

**Success:** blind balanced accuracy ≥ 0.75 AND no violation class
with >2/5 leakage → band term certified as a GATE for the banter env
reward stack (wired multiplicatively, weight decided at env
assembly). Fail → decompose per class, classify honestly.

**FLOPs:** MiniLM on 50 items × 4 turns — seconds, local, $0.

Result: _(pending)_

**Result (2026-08-06; experiment-runs/2026-08-06-exp024-band-term/) —
WITH A CORRECTION to the initial close-out:**

**PARTIAL CERTIFICATION. The floor edge is certified clean; the
ceiling (parrot) edge FAILS its registered clause — and journal entry
#10's initial analysis MISREAD the per-class numbers (recorded here
plainly: per_class reports PASSED counts; parrot '4/5' means 4/5
parrots LEAKED, not 4/5 rejected).**

- Blind balanced accuracy 0.900 (bar 0.75, pred 0.84) — passes;
  calibration close at 0.900 stands.
- Floor edge (anchoredness, a_floor=0.30 interior): PERFECT on blind —
  off_topic 0/5, generic_filler 0/5 (the pre-flagged hard case),
  word_salad 0/5 leakage; witty recall 5/5. CERTIFIED.
- Ceiling edge (parrot, wordset Jaccard): 4/5 blind parrots leak —
  FAILS the registered ≤2/5 clause. Grid-edge diagnostic (mandated,
  post-hoc, labeled): extending w_ceil down to 0.15 still leaves 3/5
  leaking at a dev cost. Reworded parrots share meaning, not words —
  the FOURTH independent confirmation of the lexical-paraphrase wall
  (EXP-016, 016b, 022-plants, now 024).
- Verdict: band term ships as the FLOOR GATE ONLY (anchoredness ×
  novelty floors — Sam's familiar-side construct, certified at 0.90+
  cross-author). The parrot/ceiling edge is REMOVED from the gate and
  merges into EXP-016c's NLI tier: "does the reply merely restate the
  context" is an entailment question, same instrument as callback
  reference detection. One NLI certification now serves two consumers.
- env/band_term.py defaults updated: a_floor=0.30 (certified);
  w_ceil retained only as an inert parameter with its non-certified
  status documented.

[LEARN] close-out-reading: Per-class summaries must state their
direction (passed vs rejected) in the printed key itself — a reader
under momentum WILL invert an ambiguous 'parrot: 4/5'.
Mistake: EXP-024's initial journal close-out read leakage as
rejection and declared a clean pass; the mandated grid-edge
diagnostic surfaced the inversion minutes later.
Correction: validators print explicit 'leaked=N/M' keys for violation
classes; and any close-out claiming a bar passed must quote the bar's
own quantity, not a derived summary.

---

## BANTER-ROLLOUT-V0 — dogfood registration (2026-08-06, pre-registered BEFORE first run; queue #13 build)

**Purpose (Sam's directive, verbatim intent):** maximize served model
quality + parallel throughput, and generate sample data at scale —
the provocation-scheduler banter design running as a many-parallel-
sessions client against vLLM continuous batching.

**Pinned design (env/banter_rollout.py):** one server, two roles.
PARTNER works a mundane task (12-task list), seeded provocation
schedule (rate 0.35/turn, 5 types: swear/mock/joke/frustration/
observation, never turn 0); POLICY replies with a NEUTRAL system
prompt — no "be funny" instruction anywhere, so unprompted wit and
provocation-response are MEASURED, not elicited. Per-session sha256
seeds (schedule + every sampling call); scoring is a SEPARATE pass
over transcripts (generation/measurement separability).

**Dogfood run (this registration):** 200 sessions × 10 turns on
qwen3-8b, workers=64. This is an ENV SMOKE, not an experiment with a
prediction — success = (a) ≥95% sessions error-free, (b) provocation
schedule realized as configured (measured from transcripts),
(c) manual read of 10 random transcripts confirms roles hold register
and provocations actually provoke, (d) throughput recorded. Its
transcripts are DOGFOOD ONLY — not training data, not demo data —
until the driver passes its adversarial audit (dispatched alongside).

Result: _(pending)_

**Result (2026-08-06 dogfood, 200 sessions, qwen3-8b both roles;
/data/good-humored/runs/banter_v0_qwen3-8b.jsonl on the box):**
(a) errors 0/200 PASS. (b) realized provocation rate 0.316 vs 0.315
expected — the seeded scheduler is exact — PASS. (d) throughput IN
the artifact: 4,000 turns in 41.4s (96.6 turns/s at 64 workers on ONE
GPU) — PASS; the parallel sample engine is real (~1/3M sessions/hour
node-scale). (c) register read: PARTIAL — mock and observation lanes
genuinely provoke (partner: "we're not in a panic room, we're in an
archive"; policy riffs back), but swear/joke lanes exposed an
ECHO-LOOP pathology: 8B-as-partner, handed a directive it can't
execute, parrots the previous message verbatim for consecutive turns
— the exact mode-collapse our reward machinery penalizes, observed
live in our own env. v0.1 fixes shipped: directive moved to prompt
FRONT; explicit anti-echo line in the partner system. DESIGN INSIGHT
CODIFIED: partner quality gates env value more than policy quality —
the env ships with a strong partner (30B+), policies can be anything.
Transcripts remain dogfood-only per registration. v0.1 rerun on the
30B follows immediately.

**BANTER v0.1 on qwen3-30b-a3b (2026-08-06, 200 sessions):** 0/200
verbatim-echo sessions (the dogfood pathology ELIMINATED by the
directive-first + anti-echo fixes), 0 errors, 4,000 turns in 59.5s.
And the register finding reversed at 30B scale: with a NEUTRAL policy
prompt, sessions show unprompted collaborative bits — a
supply-closet task escalated into a rubber-chicken/glow-sock
discovery, a "Project: Silent Sock Rebellion" notebook, "the sock has
seen things," a shrine with "Do Not Disturb (Socks Only)," and a
sock-cult manifesto — i.e., ESCALATION, TRANSFORMED CALLBACKS
(the sock recurs reframed every few turns), YES-AND cooperation, and
unprompted wit: every construct in the 2026-07-30 spec, emerging in
the environment, measurable by the certified gates. This is the demo
material class. Transcripts remain unscored pending the scoring pass;
the qualitative read is recorded as a read, not a metric.

## BANTER-STREAM — self-driving generation/scoring loop (2026-08-07, ops record)

**Idle accounting (honest):** last lane finished 08-06 19:11 box time;
nothing ran until 08-07 02:46 — **7h35m of 8×H100 idle**. Root causes:
(1) a success-only watcher grep that stayed silent when the scorer
crashed (MiniLM absent from box HF cache — the earlier "pre-download
fix" had never actually landed), (2) no work queued behind the
finishing lanes, (3) turn-gated agent attention. All three fixed
structurally, not by vigilance:

**The loop (all scripts in `env/box_keepers/`, tmux, stop =
`touch /data/good-humored/STOP_LANES`):**
- `lane_keeper_v2.sh` (v3): 30B + GLM-4.5-Air policy lanes CONCURRENT
  per iteration vs shared 235B partner (v2's alternation left the
  off-duty policy GPU at 0% — caught by post-change nvidia-smi check),
  1000 sessions/lane/iter, rotating temperature {1.0,0.9,1.1} ×
  provocation-rate {0.35,0.25,0.50}, session offsets advance per batch.
- `contrast_lane.sh`: 8B self-play on GPU0 — deliberate
  negative-contrast data so curation/emulator training spans
  bad-to-good.
- `score_keeper.sh`: scores every completed batch (500-subsample,
  24 threads — `score_banter.py` parallelized with a shared-encoder
  lock; smoke: 3 sessions, reaction_L populated, 0 errors), GLM-Air
  audience on :8003, `.failed` markers instead of retry-forever;
  rebuilds `curation_master.json` (per-config batch stats — the
  empirical config-selection table) + `curation_top5.txt` (full
  transcripts for the human read) after every batch.

**Findings already banked:**
- vLLM MoE serving is NOT bit-reproducible across runs: same
  session_id, same client seeds → 0/20 identical turns across two
  batches. Seeds reproduce schedules, not transcripts. (Consequence:
  the banked "duplicate" batches are fresh samples; `--session-offset`
  added anyway for schedule diversity.)
- 235B partner SANITIZES the `swear` provocation (softens to mild
  frustration; observed in curated transcripts). Provocation realism
  needs prompt strengthening or compliance checking → prompt v0.2
  candidate, queued for the iterate cycle.
- GLM-audience reaction diagnostic is alive at scale: 1/30 smoke turns
  above floor, max −5.61 — discriminates rather than saturating,
  consistent with its demoted diagnostic-only role.

[LEARN] ops-keeper: expensive hardware gets box-side keeper loops, not
agent-side attention.
Mistake: lanes finished with nothing queued; a success-only watcher
wedged silently; 7.5 GPU-hours idle on a metered grant.
Correction: perpetual keeper scripts in tmux (generate/score/curate
each its own keeper, one stop-flag), watchers alert on failure AND
success, and the agent is only a periodic consumer/iterator.

[LEARN] verify-remote-fixes: after any remote infra fix, re-run the
failing thing before believing it.
Mistake: "MiniLM pre-downloaded" was reported fixed but never landed
in the box HF cache; the scorer stayed dead for hours.
Correction: the fix is proven by the failing command succeeding, not
by the fix command running.

## BANTER v0.2 — iterate-cycle revision (2026-08-07, predictions pinned BEFORE the next read)

**Evidence in (72 scored batches, N=500/cell, 36k sessions):**
provocation-rate monotone — P=0.5 beats 0.35 beats 0.25 on mean
curation AND audience reaction in BOTH policy lanes (30B: 0.979/0.949/
0.939; GLM tracks the same ordering); temperature: no clear effect.
Human read of top-5: real wit present (escalation, transformed
callbacks, running gags) BUT (a) turn-0 mode collapse — identical
partner opening line for the same task across batches and temps,
(b) top-10 was 9/10 supply-closet (selection monoculture), (c) swear
compliance measured at 17.1% (542/3165 directed turns).

**Changes (all in `env/banter_rollout.py` v0.2 + keepers):**
1. Seeded OPENING_ANGLES (8 angles, turn-0 partner directive,
   recorded per session) — breaks the opening attractor.
2. `swear` directive requires a verbatim mild swear word; `mock`
   sharpened to demand a specific jab.
3. +6 affordance-varied tasks (fridge, lost-and-found, retiring desk,
   2019 filing cabinet, intern setup, kitchen memo).
4. Rotation: P=0.25 RETIRED, PROVS now {0.50, 0.35, 0.65} (incumbent /
   control / trend probe).
5. Curation: top-5 human-read file capped at 2 sessions/task;
   `top_by_task` view added to the master.

**Pinned predictions (checked next cycle, over v0.2 batches only):**
- Swear compliance ≥60% (from 17.1%). If <40%, the 235B partner is
  judged prompt-resistant here and the swear provocation moves to a
  different partner or gets a compliance filter.
- Supply-closet share of the modal opening trigram drops to <20% of
  its sessions (from ~replayed-verbatim).
- P=0.65 cell: if the monotone continues (curation > P=0.5 cell), keep
  climbing next cycle; if it reverses, 0.5 is the plateau and the
  sweep closes.

Keeper restarted 04:5x; in-flight pre-v0.2 batch files lack
run_summary and are skipped by the scorer. GPUs verified 8/8 ~100%
after restart.

## BANTER v0.2 predictions CLOSED + v0.3 (2026-08-07, iterate cycle 2)

**All three v0.2 predictions closed against completed v0.2 batches
(62 batches, 44,759 swear-directed turns):**
1. Swear compliance 64.2% vs bar >=60% — HIT (from 17.1% baseline).
   Directive-explicitness ("include the word verbatim, do not
   euphemize") is the lever that moves frontier-partner compliance.
2. Opening-collapse: modal supply-closet opening trigram share 5.1%
   vs bar <20% — HIT decisively; the opening distribution is
   essentially flat (~5% per angle family).
3. P=0.65 CONTINUED the monotone (30B: 1.028>1.016>0.998; GLM:
   1.027>1.011>0.983; reaction agrees) → per pinned rule the sweep
   climbs: rotation now {0.65 incumbent, 0.50 control, 0.80 probe}.
   Pinned: if P=0.80 curation < P=0.65, the plateau is found and the
   sweep closes at 0.65.

**Also observed:** GLM-Air caught up to 30B under v0.2 (P=0.65 cells:
1.027 vs 1.028; GLM reaction now best-in-table) — opening diversity
appears to have helped GLM disproportionately. CONFOUND flagged: GLM
audience scores GLM's own lane; cross-audience check launched
(banter_048 + glm_048 rescored with 235B audience, tmux gh_xaud;
read next cycle — if the GLM-vs-30B ordering flips under a neutral
audience, all cross-model curation comparisons get audience-swapped).

**Utilization fix:** the two-lane wait-barrier idled the faster
policy GPU each iteration (GPU1 observed 0% while GLM finished);
replaced lane_keeper_v2.sh with parameterized policy_lane.sh — fully
independent per-lane loops. Second instance of the alternation/barrier
hole class.

**Human read (capped top-5):** genuinely improved — 3 distinct tasks
in top-3, seeded openings produce natural entries, swear lands
naturally ("Damn, you're right—"). NEW defect class: CROSS-SESSION
MOTIF ATTRACTORS ("Bermuda Triangle" in two of three transcripts;
haunted-office register in all three; "time capsule"/"museum"
recurring) — invisible to per-session self_repetition; the
25-template regurgitation failure mode at motif level. Per
theory-grounding discipline: MEASUREMENT added first (pooled
trigram_diversity + top content-bigram motifs per batch in
curate_banter batch_stats; GLM baseline 0.82-0.85) — no selection
penalty until quantified across configs.

## BANTER cycle 3: audience confound CLOSED, sweep CAPPED, register drift found (2026-08-07)

**Cross-audience check (200 sessions/lane, batch 048 pair):** the
model ordering FLIPS with the audience — GLM-audience favors the GLM
lane (+0.09 reaction), 235B-audience favors the 30B lane (+0.20);
session-level rank agreement between audiences only rho=0.53 (banter)
/ 0.72 (glm). CONSEQUENCE EXECUTED: "GLM caught up to 30B" is
officially audience-confounded and downgraded to no-claim. Rule
adopted: cross-MODEL comparisons require audience-swapped
verification; within-model config trends (same audience both sides)
remain valid — rotation decisions are within-model, unaffected. Note
neither audience is neutral: GLM scores its own lane; 235B wrote the
partner half of every conversation.

**P=0.80 probe: monotone continues a third time (GLM 1.041>1.025>
1.007; 30B 1.028>1.020>1.009) — and the sweep is CAPPED at 0.80
anyway, on construct-validity grounds:** the env must measure
UNPROMPTED wit; at P→1.0 the unprompted condition ceases to exist
while the curation number would keep climbing mechanically (more
directives = more reaction opportunities). Capping while the metric
still rises is the anti-Goodhart move. Human read of P=0.80 top AND
median: naturalness intact, task texture thinner but present.
Rotation holds at {0.65, 0.50, 0.80}.

**Motif diversity quantified (v0.3 cells):** GLM trigram-diversity
~0.84 vs 30B ~0.73, flat across P — the 30B is the template-heavy
model. Screen-relevant evidence (diversity is a first-class product
requirement), logged for the model decision.

**NEW pathology, measurement-attributed: RP stage-direction drift.**
Read found asterisk action narration (*I grab the tape*) in a GLM
median transcript; measurement INVERTED the attribution: banter lane
(30B policy) 27.1% of 270k policy turns, GLM lane 0.4%, contrast
3.0%. v0.1 baseline 18.4% — pre-existing 30B register habit, opening-
angle modulated (23.7–37.1% by angle), worsened slightly by v0.2.
FIX (v0.3.1): "plain chat messages only, never narrate actions in
*asterisks*" in BOTH system prompts. PINNED: banter-lane asterisk
rate <3% over v0.3.1 batches; if it stays >10%, the 30B is judged
register-unstable under neutral prompts (a screen strike, since
buyers get the neutral-prompt env).

## BANTER cycle 4: asterisk close HIT, two micro-hypotheses falsified, agreement attractor quantified (2026-08-07)

**v0.3.1 asterisk prediction CLOSED — HIT:** banter lane 2.39%
(3350/140k policy turns, bar <3%, from 27.1%); glm 0.15%. No screen
strike: the 30B is register-compliant when told.

**Post-fix curation dropped (GLM −0.065, 30B −0.13) and BOTH of my
first explanations were falsified by 5-minute measurements:**
1. "Asterisk turns inflated the metric" — REFUTED: within pre-fix
   batches, asterisk turns scored WORSE (anchor 0.489 vs 0.498,
   reaction −13.87 vs −13.54). Removing them should have raised it.
2. "The partner constraint was unnecessary" — REFUTED: the 235B
   partner asterisked at 25.9%, same as the policy. Constraint needed
   in both prompts.
Settled account: the plain-chat line costs expressiveness on EVERY
turn (instruction burden, double for the 3B-active 30B), outweighing
the removal of slightly-worse RP turns. Price accepted — register-
correct chat data is the product; the metric serves the construct,
not vice versa. RULE ADOPTED: curation scores are comparable only
within a prompt version; the human-read file now draws from the most
recent 40 batches (recency window in curate_banter).

**Fresh v0.3.1 read:** register fix held (zero asterisks), wit intact
("IT guy will need a Ouija board to read them"). NEW characterization:
the policy AGREEMENT ATTRACTOR — agree-and-amplify openers
("Absolutely!", "Perfect") while the partner does the comedic lifting.
QUANTIFIED as policy_agreement_rate (now in batch_stats): 30B
0.28–0.30, GLM 0.26–0.29, 8B 0.21–0.23. The attractor is STRONGER in
stronger instruction-tuned models — RLHF sycophancy as conversational
risk-aversion. DOCTRINE: not prompt-fixed. The neutral policy prompt
exists so the env EXPOSES this attractor; moving it is RL training's
job and part of the product's measurable-improvement story.

## BANTER cycle 5: anti-sycophancy dial, tail-luck contamination, report card v1 (2026-08-07)

**Agreement × provocation (v0.3.1 cells): provocation density
SUPPRESSES the agreement attractor monotonically in both strong
models** (GLM 0.285→0.272→0.264, 30B 0.303→0.293→0.279 across P=0.5/
0.65/0.8). The provocation scheduler is an anti-sycophancy pressure
dial on the data distribution — env-design finding, product-relevant
(the env can tune how much conversational risk the data demands).
Nuance: the 8B's LOW agreement (0.208) is incoherence, not
discipline — agreement-opener rate is a RANGE metric; both extremes
are pathologies.

**Human read caught metric contamination: an 8B contrast session
topped the shortlist (1.647) while reading clearly worse than the
GLM raccoon transcript below it (1.583).** Mechanism: 110 contrast
batches × 500 sessions gives the multiplicative gate metric a 50×
sample to find lucky tails in (hyper-anchored style also gives the
8B the highest floor_pass, 0.976 — the floor is a necessary
condition, not quality, now visible in one table). FIX: the
human-read shortlist is LANE-SCOPED (contrast excluded — it is
negative training contrast, not demo material); master keeps all
lanes for training-data honesty.

**Consolidation shipped: env/report_card.py** — pivots batch_stats
into the per-(model, P) characterization table: curation, floor,
reaction (diagnostic), trigram diversity, agreement, asterisk
register. First card generated (runs/report_card.json). This is the
buyer-facing artifact and the pre/post-RL delta scaffold. Register
health now a standing batch metric: 30B holding ~2.3% under v0.3.1,
GLM ~0.1%.

## BANTER cycle 6: provocation-type yields, per-model temperature, v0.4 (2026-08-07)

**Provocation-type ranking (v0.3.1 scored, ~210k policy turns,
IDENTICAL ordering in both lanes):** mock > joke > frustration >
observation > swear on downstream reaction (post-mock −11.4/−12.7 vs
post-none −14.7/−15.0 — a ~2+ logit spread). Being TEASED elicits
the best comebacks; riposte turns run slightly less anchored (0.441
vs 0.465), consistent with wit-as-expectation-breaking. Honest note:
swear — the type we spent two cycles making compliant — is the
LOWEST-yield elicitor. Compliance ≠ productivity. Read corroborates
(best material rides mock/joke chains).
**v0.4:** mild seeded type weights {mock .28, joke .24, frustration/
observation/swear .16 each}; all five types retained (the env must
keep measuring the full provocation space; this is a data-mix
decision, not a reward change). PINNED: mean provoked-turn reaction
rises under v0.4; per-type yields stay ~stable; read quality holds.

**Temperature RESOLVED — model-specific:** GLM improves at T=1.1 on
all four metrics (cur 0.966/tridiv 0.859/agree 0.267/react −13.71 vs
T=0.9 baseline); the 30B is flat on everything. GLM lane now rotates
{1.0, 1.1, 1.2}. PINNED: T=1.2 keeps the trend AND the read shows no
incoherence (floor drop >0.02 or read breakdown = cap at 1.1).

**Read:** best class yet (GLM T=1.1 filing-cabinet: the laminated-fax
setup pays off eight turns later in the plaque-lamination callback;
"a filing optimist — 'This'll make sense someday!' Spoiler: It did
not."). Minor world-grounding wobble noted (policy dated itself
"2023"), not actionable. GPU1-idle false alarm: inter-batch gap.

## BANTER cycle 7: v0.4 closes 3/3, T sweep CLOSED, sampling phase mature (2026-08-07)

**v0.4 mix predictions CLOSED — 3/3 HIT (26k/24k provoked turns per
lane):** realized mix matches spec (mock 0.278 vs 0.28); provoked-turn
reaction rose (GLM −13.20→−12.71, 30B −13.97→−13.84); per-type yields
stable (no elicitor diluted by the shift).

**GLM T=1.2 probe CLOSED — KEPT:** cur 0.994 / tridiv 0.885 (highest
any cell has posted) / agree 0.237 / react −13.10, floor slid −0.019
(inside the pinned 0.02 cap), and the mandatory read shows zero
incoherence (top: River's-hoodie/eBay-guy; median: "The Brine That
Broke HR" — both fully coherent). Note: floor decline with T is
partially theory-EXPECTED (wit = expectation-breaking lowers context
similarity); the gate exists for non-sequiturs and the read found
none.

**TEMPERATURE SWEEP CLOSED at GLM {1.0, 1.1, 1.2}, 30B {1.0, 0.9,
1.1}.** T=1.3 not probed: the floor trend (−0.009/step) predicts a
cap breach, gains 1.1→1.2 were already second-order, and the loop
must not become a knob-tuning treadmill.

**STRATEGIC NOTE (for Sam):** the sampling/eval loop is MATURE —
config axes are all closed-or-settled (provocation rate capped 0.80
on construct grounds, type mix weighted by yield, temperature closed
per-model, prompts at v0.4 with register/opening/compliance solved),
the characterization battery is standing (report card), and the bank
holds 200k+ scored policy-lane sessions spanning three quality tiers.
The next first-order lever on "make the model funnier" is TRAINING:
emulator/quantile-RM on the bank + human-rated corpora, screen
decision, then GRPO with the certified gates + curation machinery as
reward shell. That re-allocates GPUs from serving to training
(STOP_LANES + verl), so it needs Sam's go. The loop keeps banking
v0.4 data meanwhile.

## BANTER cycle 8: demo pack v1 generated (2026-08-07)

Sampling continues healthy (8/8 GPUs, 0 score failures, v0.4-era card
stable: GLM cur 0.979 / tridiv 0.864 / agree 0.254; 30B 0.893 / 0.724
/ 0.281; asterisk residuals 0.2% / 2.5%).

**Shipped env/demo_pack.py — the pitch asset generator:** assembles
docs/private/DEMO-PACK.md (gitignored) from the LIVE pipeline: honest
framing (neutral prompts, unprompted wit), the per-model
characterization card with column caveats (floor = necessary
condition; reaction = demoted diagnostic, within-audience; scores
rank, humans certify), and 8 best transcripts, one per task, recent
same-version window, lane-scoped. Because it is GENERATED, the same
command regenerates it post-RL — the pre/post pair on identical
machinery IS the sales demo.

**Read (demo #1, agenda task, GLM):** best material to date — "Greg's
the human equivalent of 'mark as unread'", Sandy character
continuity, passive-aggressive agenda-item closer. Honest note: the
agreement attractor is visible even in top material ("Totally
agree"/"Absolutely"/"Perfect" openers) — which sharpens the pitch:
the card quantifies the attractor at 0.25 and RL's job is the delta.

## BANTER cycle 9: scorer backlog fixed, training-readiness inventory (2026-08-07)

**Ops: the scorer was silently falling behind** — 648 generated vs
542 scored (~106-batch backlog, growing: generation is 3 lanes wide,
scoring was single-file). Liveness checks all passed while the
throughput RATIO diverged. Fix: score_keeper now claims batches via
mkdir locks; a SECOND instance drains newest-first; stale locks
(>90 min) swept each pass so a killed keeper can't silently orphan a
batch. Both instances verified scoring concurrently.

**Training-readiness inventory (for the pending go):** local
~/Experiments/good-humored-data/ holds the 1.2M license-split
memorized-joke NOVELTY corpus + raw NYCC (nycc-full) + MANIFEST; the
BOX holds no corpora at all — the go needs a small rsync of RM data
(NYCC votes etc.) before emulator training. Firewall reminder:
NYCC-Zhang is research_only; the commercial-product emulator variant
trains only on the commercial-safe bucket unless Sam rules otherwise.

**Read:** kitchen-etiquette memo (a v0.2 task addition) produces top
material — co-writing a funny artifact embeds the wit IN the task
("warm, forgotten mug — sounds like a sad indie band"; "if it's
darker than motor oil, pitch it"). METRIC CAVEAT recorded: agreement-
opener rate overcounts agreeable-but-CONTRIBUTING turns (read shows
agreement openers followed by substantive additions) — interpret as
an upper bound on pure sycophancy.

## BANTER cycle 10: recency-window bug, T=1.2 REVERSED on new evidence (2026-08-07)

**Ops bug (second member of the unequal-rate class):** the read
file went EMPTY — the global-mtime recency window was 40/40 contrast
batches (contrast scores 3-4x faster), which the lane-scoped
shortlist then excluded entirely. Fix: per-lane windows (last 20
policy batches by number) + the desc scorer now prioritizes policy
lanes over contrast. Backlog draining (106 → 84).

**T=1.2 KEEP decision REVERSED — new instrument, new evidence:** the
read caught a CJK-leaked turn ("without惦记ing this"); measurement
shows GLM-policy CJK leakage is strongly temperature-correlated:
0.058% (T=0.9) → 0.142% (1.0) → 0.328% (1.1) → 1.133% (T=1.2) —
20x, ≈1 in 9 sessions blemished at 1.2. The cycle-7 guard (floor +
two-transcript read) was structurally blind to a ~1% defect. GLM
rotation now {1.0, 1.1}; policy_cjk_rate added as a standing batch
metric; curation shortlist hard-excludes CJK-blemished sessions at
every temperature (language defects are never demo material).
Honest accounting: the reversal is legitimate — pinned guards cover
KNOWN failure modes; closed decisions stay reversible when a new
defect class gets instrumented. Qwen lanes measure 0.01-0.04%
(negligible); the 235B partner is clean everywhere.

**Read (pre-reversal top):** still strong ("Call Bob about the
thing" mystery; the zombie plant "following Bob's example and
refusing to retire") — callback ecology healthy in v0.4 material.

## BANTER cycle 11: steady state, CJK instrument confirmed, character-comedy pattern (2026-08-07)

Health: 8/8 GPUs, 0 failures, backlog draining on trend (106→84→66).
Standing policy_cjk_rate CONFIRMS the ad-hoc gradient (glm 0.0006/
0.0016/0.0032 for T=0.9/1.0/1.1; retired T=1.2 residue 0.0115 aging
out) — instrument and one-off counter agree.

**Observation codified (no action needed): third-party characters
are the env's most reliable wit affordance.** Every recent top
transcript builds its best material around an ABSENT third party —
Dave's space heater arc ("the only thing glaring is your
entitlement" / "ghost voltage" / heater "on suicide watch"), Sandy's
sticky-note apocalypse, Greg's thumbs-up, Bob's mystery note, Dennis
the raccoon. Models riff measurably better about absent characters
than about objects. The env already affords this (tasks + the
relay-a-coworker opening angle); logged as a PREDICTION about where
RL will find reward, checkable post-training.

No config changes: all axes closed and stable. This is the loop's
intended steady state — bank data, verify instruments, read, wait
on the training go.

## BANTER cycle 12: steady state; era-contamination near-miss in my own query (2026-08-07)

Health: 8/8 GPUs, 0 failures, backlog 52 (106→84→66→52 — dual
scorers will clear it), disk trivial (runs = 5.5G).

**Near-miss worth recording:** my ad-hoc "recent card" query showed
banter asterisk at 8.35% vs the 2.39% close — because it filtered on
METRIC PRESENCE (= scored recently), which includes pre-v0.3.1
batches the asc scorer is draining from the old backlog. Era split
confirms: pre-fix batches ast=0.2247, current era ast=0.0251 (close
remains valid). The cross-version trap (#31) generalizes: era
filters must key on GENERATION batch number, never scoring recency —
and ad-hoc queries are where codified rules get silently violated;
the standing tool (report_card --min-batch) was already correct.
Use the tool, not fresh one-liners, for anything decision-relevant.

Top transcript unchanged (glm_147, read last cycle). No config
changes.

## BANTER cycle 13: steady state; 30B top-tail parity observed (2026-08-07)

Health: 8/8 GPUs, 0 failures, backlog 41 (106→84→66→52→41).

**Read (#2, 30B T=1.1 P=0.8, curation 1.586): the 30B's top tail
reaches GLM-class material** — the "TechSpa" bit (cord-queen Jan's
cucumber mask → outlet aromatherapy → "crystal charging only" hard
drive → "blessed by a power strip" → router as chakra blockage) is
sustained, disciplined collaborative escalation. Screen-relevant:
lane MEANS differ (0.917 vs 0.978) but top tails overlap — the
cheaper model reaches the demo class, less often. Third-party
characters carry the material again (Dave watering "Karen's Plant";
Jan) — pattern holding. No config changes.

## BANTER cycle 14: GLM duty-cycle hole -> self-play lane (2026-08-08)

**Third member of the synchronization-hole class:** GPUs 2-3 (GLM)
measured ~60% idle across a 30s sample — all 96 GLM-lane sessions
bunch-synchronize queuing on the shared 235B partner (the fleet
bottleneck), then flood back. More workers would only deepen the
bottleneck queue. FIX: a GLM SELF-PLAY lane (policy AND partner on
:8003, glmself_stream_*, offsets 750k+) soaks the idle capacity with
dependency-free work via continuous batching. Verified: GPUs 2-3 from
0%-bursts to sustained activity. Bonus: the new stream is a second
partner variant — partner-robustness evidence for the env spec (the
partner is part of the spec; now we measure the same policy under
two partners). Scorers restarted with the new prefix in the glob;
curate handles the new lane automatically (per-prefix windows).

Read (#3, GLM catering): sad-olive riff ("the MVP of corporate
catering... never shows up sweating at a client meeting") + Dave's
curry incident — character pattern holding. Backlog 31.

## BANTER cycle 15: partner A/B first data — doctrine quantified (2026-08-08)

**The self-play lane's first scored batch gives the controlled
partner comparison (same GLM policy, partner swapped):** with GLM as
partner vs the 235B — curation 0.913 vs 0.985, reaction −15.08 vs
−13.39 (−1.7 logits), agreement 0.293 vs 0.254 (MORE sycophancy),
tridiv 0.848 vs 0.865. Read concurs: self-play material is pleasant
but FLAT — GLM-as-partner executes provocation directives without
teeth, and the policy relaxes into yes-and. The partner-quality
doctrine ("the partner is part of the env spec") now has a
same-policy controlled A/B behind it — product-grade evidence that
buyers must run the env with the strong partner.
PRELIMINARY (n=1 batch): confirm at n>=5 next cycle before quoting
numbers anywhere buyer-facing. PINNED: the gap holds direction
(curation and reaction both lower under the weak partner) at n>=5.

Health: 4 lanes generating, backlog stable ~30, 0 failures. No
config changes.

## BANTER cycle 16: GLM lane was DEAD ~10h — my bug, my missed checks (2026-08-08)

**Honest incident report.** The cycle-10 GLM-lane restart killed the
old keeper mid-batch-157 and the relaunch crashed INSTANTLY on its
first iteration: I passed 2 temps but policy_lane.sh indexed
TEMPS[$((i % 3))] — i=158 → TEMPS[2] unbound → set -u exit, before
writing one log line. The lane was dead ~10 hours. The self-play
lane launched at cycle 14 died of the SAME bug after its first batch
(i=2). Compounding failures on my side:
1. Cycle-10 "verification" read the dying keeper's stale progress
   line as fresh output.
2. Cycles 11-15 health checks tailed a log whose last line was the
   restart note FIVE TIMES without flagging it.
3. Cycle 14's duty-cycle diagnosis was partly WRONG: GPUs 2-3 were
   idle mostly because the lane was dead (the 40-60% "bunching"
   activity I sampled was scorer audience traffic). The self-play
   fix was built on a misdiagnosis — then died of the same bug.
**Fixes:** (a) modulo by ${#TEMPS[@]}/${#PROVS[@]} (the bug class is
gone, not patched); (b) both lanes relaunched and verified ACROSS
the iteration boundary that killed them (glmself batch 2→3
confirmed); (c) env/box_keepers/health.sh — standing check: tmux
session EXISTENCE for all 10 expected sessions + newest-output
FRESHNESS per lane + scored freshness, exits nonzero on any failure;
(d) the iterate-cycle health procedure now runs health.sh instead of
tail-reading. Current status: HEALTH OK, 4 lanes generating.

## BANTER cycle 17: partner A/B CLOSED (confirmed); self-play tail inflation caught (2026-08-08)

**Pinned partner A/B close — CONFIRMED at n=10 glmself batches:**
curation 0.920 [0.894,0.950] vs 0.980 [0.951,1.006] under the strong
partner; reaction −14.97 [−15.43,−14.34] vs −13.45 [−13.94,−12.85]
— both directions as pinned, ranges (essentially) non-overlapping.
The partner-quality doctrine is now controlled, quotable evidence:
same policy, partner swap moves curation −0.06, reaction −1.5
logits, sycophancy +0.036.

**Self-play tail inflation caught by the read:** glmself sessions
took 3 of 5 shortlist slots (top at 1.633 ≈ the all-time #1) but the
read places that session a FULL TIER below its score twin — one-shot
gags, no callback structure, and self-play tonal homogeneity (both
voices share a cadence). This is audience self-preference at its
maximum (GLM judging GLM×GLM), predicted by the cycle-3 confound
finding. FIX: shortlist excludes glmself (like contrast) — the demo
channel carries the PRODUCT configuration (strong partner) only;
self-play scores are usable for within-lane trends, never cross-lane
ranking. First health.sh cycle: HEALTH OK.

## BANTER cycle 18: steady state; multi-character universes are the ceiling class (2026-08-08)

HEALTH OK (first-try green, health.sh). Backlog 72 (was 51) — four
generation lanes vs two scorers; ratio on watch, capacity decision
next cycle if it keeps growing.

**Read (glm_156, "Great Coffee Machine Ascension", 1.529): arguably
the env's best transcript yet** — FOUR named absent characters
(Sarah, Karen's dead keyboard, Janice's embroidered towels, and Dave
of the "FBI Surveillance Van #4" Wi-Fi rename) woven into one
coherent office universe, with the policy co-authoring ("Official
Documentation of Cable Management", "laminated laminator
certificate"). Strengthens the codified third-party-character
pattern to its limit: MULTI-character universes are the ceiling
class of this env's material. Post-training check remains pinned:
does RL amplify exactly this? No config changes.

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

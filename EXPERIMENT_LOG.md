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

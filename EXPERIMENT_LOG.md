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

**Result:** _(pending)_

**Verdict:** _(pending)_

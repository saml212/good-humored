# STATE

_Last updated: 2026-07-17 morning (post-overnight sprint)_

## Current focus

Rejection-cascade benchmark **Phase 1 is COMPLETE and audited**: validated
instrument chain (EXP-001→003b, 008), an 11-model pilot with full
statistical inference (EXP-004 + fills), the temperature-fakeability
defense replicated on two honored-endpoint models (EXP-007/007c), the
noise-bias regime map (EXP-006), a validated banter judge (EXP-005), and a
validated semantic novelty tier (EXP-009). The authoritative findings
document is **`docs/FINDINGS.md`** (adversarially reviewed; every number
traces to `stats_inference.json`).

## The headline (what the benchmark found)

The pre-registered shared-pool hypothesis is dead (cross-model jaccard
0.1126 vs predicted 0.35 — below the entire pooled-frequency null). What
replaced it: **per-lab failure fingerprints, orthogonal and
brand-consistent**:

- **Anthropic** — constraint collapse: opus/sonnet/haiku repeat an
  already-rejected topic in 12/12 runs (turns ~7–14); near-zero
  memorization for opus/sonnet/fable. **fable (newest) breaks the family
  pattern** (1/4 degradations). haiku memorizes at 25.8% (stated bluntly;
  it is also the rejector — dual-role confound documented, family
  contrast survives a no-haiku robustness check: −11.42 turns, p=0.0005).
- **OpenAI** — near-perfect constraint adherence, heavy verbatim recall
  (pooled 18.6%; sol 21.7%, 5.4 26.7%, mini 7.5%).
- **xAI/grok** — the retrieval machine, now with complete path data:
  ZERO degradations + highest within-model overlap (set_jaccard 0.443) +
  top memorization (40.9%, n=198, all contrasts survive Holm).
- **Open-weights** — fast degradation (deepseek median turn 8.5), ~2%
  memorization. kimi DROPPED (reasoning-token starvation at any fixed
  max_tokens; documented in EXP-004 addendum).

## Master plan (agreed 2026-07-16, unchanged)

1. ✅ Benchmark → ill-humored leaderboard (this is now `docs/FINDINGS.md`;
   paper-grade rerun needs v3 labeler + native APIs).
2. Pick post-training target: strong base + weak cascade score + trainable
   size (7–32B).
3. GRPO with the humor reward stack vs curated SFT baseline — `env/` is
   TRL-proven (GRPOTrainer smoke, 6 reward terms incl. the new semantic
   novelty tier, inert by default).
4. **Reverse transfer** (the pitch's spine): MMLU/GPQA before/after humor
   RL vs compute-matched control. Untested by anyone. Needs GPU decision.

## Running / in flight

- **v3 relabel** of the full pilot (cached, resume-safe) — then the
  v2-vs-v3 instrument-robustness comparison and the EXP-006 re-run with
  v3 empirical noise rates (required before paper-grade claims).
- Paper `DRAFT.md` §5.3 integration + peer-review pass (queued).

## Dead ends

- **Ecosystem-collapse hypothesis** (all models share one joke well):
  killed by EXP-004 — overlap is *below* chance-cooccurrence. Evidence:
  `experiment-runs/2026-07-17-cascade-pilot/stats_inference.json`.
- **Sonnet as rejector**: worse than haiku (EXP-003b) — bigger ≠ better
  instrument.
- **qwen as temperature-ablation subject**: endpoint silently ignores the
  temperature param (EXP-007b manipulation check) — any sampling ablation
  needs a manipulation-check gate first (now standard).
- **kimi-k2.5 in the cascade**: reasoning burn scales with the rejection
  list; no fixed max_tokens survives (400/2048/4096 all starved).

## Key constraints

- No GPU_API_KEY in .env yet → GPU decision pending for TRANSFER-PLAN
  (reverse-transfer experiment) and any real GRPO run.
- Known env/ exploit, documented loudly: padding/dilution evades all
  novelty tiers (verbatim joke + ~5 filler sentences kills n-gram, ~20
  kills semantic). Mitigation direction: max-over-sliding-windows.
  Novelty terms are NOT a sole defense until fixed.
- Corpus licensing: SocialGrep (CC-BY) safe; taivop + Oogiri
  research-only; Oogiri-GO MIT-vs-CC-BY-NC-SA discrepancy awaiting Sam's
  call. `docs/private/` and `CLAUDE.md` gitignored on purpose.
- For Sam, carried from overnight: rotate the two chat-pasted keys (kimi,
  xai); platform-skills#89 review; deepseek registry → deepseek-v4-flash
  after 07-24; one orphaned /deploy-team dashboard process (PID 76820) —
  kill or keep.

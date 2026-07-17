# STATE

_Last updated: 2026-07-17 end of day_

## Current focus

Phase 1 (benchmark + instruments + RL environment) is **complete,
red-teamed, and closed out with zero open calibrations**. The
authoritative claim chain is `docs/FINDINGS.md` (post-hostile-review
revision) → `stats_inference.json`; the paper draft carries the same
numbers; the pitch memo (`docs/private/PITCH.md`, gitignored) is
send-ready. Next phase is gated on decisions below, not on work.

## The claim chain (post-red-team — smaller and harder to kill)

The four-fingerprints table is now a *descriptive summary*; the citable
core is:
- **Two family contrasts that survive every robustness cut**:
  Anthropic-vs-OpenAI degradation depth −13.17 turns (p=0.0002),
  surviving no-haiku (−11.42, p=0.0005), meta-register exclusion
  (−8.79, p=0.0004), and both at once (−7.75, p=0.0018). Direction
  survives every cut; magnitude shrinks each time — the shape of a real
  effect under honest stress-testing.
- **grok, triangulated three independent ways**: 0 degradations in 4
  runs + highest self-overlap (0.443) + top memorization on both tiers
  (40.9% exact / 20.7% template-trigram).
- **The pre-registered miss**: cross-model overlap 0.113 vs predicted
  0.35, below the entire simulated null. No shared joke well.
- **Scope label on everything family-level**: claims are about
  model+wrapper deployment stacks (haiku r01 visibly adopts the CLI
  persona 25/30 turns) until the same-model both-lanes ablation runs
  (~$5 — needs native Anthropic/OpenAI API keys; pitched as the first
  joint experiment with Anthropic).

## Instruments

- **v2 free-vocab labeler**: still the authoritative pilot instrument.
- **v4 two-tier labeler**: field-validated — escape 0.1723 vs predicted
  0.17 (near-exact), probes 18/18, fixture bars pass. Promotion BLOCKED
  on the haiku anomaly: haiku-as-subject is 29.2% unparseable under v4
  (everyone else 0–1.7%) — the dual-role model's fourth anomaly.
  Inspect those turns before promoting. (v3: field-invalidated, kept as
  the cautionary tale.)
- **Noise story (EXP-006/006b)**: both regime-level paths by which
  labeler noise could FAKE a collapse finding are closed; measured
  overlap is an upper bound with margin.

## RL environment (env/)

Seven-term-capable stack, TRL-proven. Padding/dilution exploit CLOSED
for the n-gram tier (windowed, default-on, 3-round adversarial cycle —
boundary predicate covers whitespace/punctuation/Cf/Mn/Me/Cc).
Windowed semantic mode opt-in pending **EXP-011** (threshold re-sweep;
whole-text 0.38 is measured-miscalibrated for windows). Registered
specs for the next two theory terms (BVT multiplicative gate,
two-stage incongruity) in THEORY-MAP.md §12. 187 env tests.

## Transfer plan (docs/TRANSFER-PLAN.md — registration-grade)

Qwen3-8B-Instruct primary (none of the cascade-profiled API models map
to trainable checkpoints — the selection finding), LoRA r=16,
compute-matched neutral-banter control, **Gate 0**: cascade
manipulation check on the actual checkpoint before any eval number.
Budget: ~50–55 GPU-hours MVP. Predicted deltas: MMLU +1.0pp, GPQA
+0.5pp (weak prior, stated as such).

## Calibration ledger

**All predictions closed — 0 open.** ~19 closed lifetime; recent hits:
EXP-007c distinct-2 (+0.143 vs +0.15), EXP-010 escape (0.1723 vs 0.17).
Honest misses recorded: EXP-006b (−0.100 vs −0.015, informative),
EXP-007b (manipulation failed — qwen endpoint ignores temperature).

## Decisions for Sam (nothing blocks on my side)

1. **Native Anthropic + OpenAI API keys** → the $5 wrapper ablation
   (the single score-moving experiment per the hostile review).
2. **GPU approval** (~50–55 GPU-hours) → transfer plan Phase 0.
3. Rotate the two chat-pasted keys (kimi, xai) — still pending.
4. Oogiri license call (MIT vs CC-BY-NC-SA).
5. Paper anonymization (Data & Code Availability statement).
6. **rockie-cascade fleet automation is hammering this machine**
   (EINTR storms in local python; external to this session) — keep or
   kill.
7. platform-skills#89 review; deepseek registry → deepseek-v4-flash
   after 07-24.

## Dead ends (cumulative)

Ecosystem-collapse hypothesis (below-chance overlap); sonnet as
rejector; qwen for temperature ablations (endpoint ignores the param);
kimi in the cascade (reasoning-burn defeats any fixed token budget);
closed-vocabulary-with-catch-all labeling (v3 — manufactures repeats);
whole-text novelty scoring as sole anti-memorization defense
(dilution); threshold reuse across scoring granularities.

## Key constraints

- Repo public, all rights reserved; `docs/private/` + `CLAUDE.md`
  gitignored on purpose.
- Corpus licensing: SocialGrep (CC-BY) safe; taivop + Oogiri
  research-only, kept separated.
- Every generation eval carries the novelty check (CLAUDE.md hard
  rule); windowed n-gram is now the default at eval time.

# STATE

_Last updated: 2026-07-22 (direction refresh from Sam)_

## DIRECTION REFRESH (Sam, 2026-07-22 — supersedes emphasis, not architecture)

Sam's assessment, adopted: **the cascade is a diagnostic benchmark, not a
funniness benchmark** — it measures necessary conditions (novelty,
constraint adherence, pool structure) and "tells a compelling part of the
story," but cannot evaluate whether anything is funny. Keep it AND
improve it; do not confuse it for the ceiling. The new center of
gravity, two workstreams (research agents dispatched 2026-07-22):
1. **Direct funniness via the math of expectation violation** — Sam's
   framing: a joke is an on-purpose hallucination; semantic similarity
   tracks expectation up to a pivot, then breaks it in a way that
   RESOLVES. Formalize what mathematically separates a controlled
   departure (joke) from an uncontrolled one (hallucination):
   surprisal-curve shape, Bayesian surprise/belief revision,
   Kao-style noisy-channel ambiguity+distinctiveness — upgraded from
   judge proxies to REAL logprobs if our API providers expose them
   (the §12.2 Tier-A path; EXP-014 showed the proxy version is noisy).
2. **Conversational measurement expansion** — "we have only scratched
   the surface": callbacks/reincorporation, timing, escalation,
   audience-model updating (does the model learn the rejector's
   revealed preferences?), cascade 2.0 semantic step-size trajectories.
   Constraints unchanged: contained kernel, theory-traceable, our
   multi-turn niche.
Also: new RL-environment skills exist in the rockie catalog; agent-side
pull is blocked by an OS-level volume permission (HARNESS-NOTES L15) —
Sam runs `rockie skill catalog` / `rockie skill pull` from his own
shell.

## NEWEST: the contained-kernel certification arc (EXP-011–014, one night, all pre-registered, all closed)

The RL reward is now a **contained kernel** (Sam's design constraint:
no humans at training time; human data used once, offline, to certify).
The night's certification results, honest version:
- **Verifiable two-thirds** (novelty n-gram windowed default-on
  @EXP-011-validated thresholds, semantic tier, repetition, diversity,
  comprehensibility): solid, adversarially hardened, dilution exploit
  closed end-to-end.
- **Theory-gate STRUCTURE validates**: anti-gaming probes pass (vague
  probe 0.083 ≤ 0.25; nonsequitur resolution exactly 0.0), echo
  resistance passes, and the registered disproof CONFIRMED the additive
  stack's compensation failure (violation-only earns 80% of both-class
  reward — the exact hole the multiplicative BVT gate exists to close).
- **But every component where haiku-as-humor-judge is load-bearing
  FAILED its bar**: naked judge vs human consensus ρ=0.056 (chance on
  top-vs-bottom, EXP-012 — THE FLOOR FIRED); violation-judge halo
  (EXP-013 margin 0.031 vs pred 0.40); surprise-proxy noise (EXP-014
  pass 0.389 vs pred 0.65). One sentence: **structure multiplies
  instrument quality, it does not create it.**
- **Consequence, registered and honored: GPU spend on any
  judge-load-bearing kernel is BLOCKED** pending a better judged-third
  instrument. Paths, in cost order: multi-sample probes (EXP-014b,
  cheap); different judge model for the violation axis; an RM trained
  on Oogiri-Master consensus (**blocked on Sam** — dataset verified
  real, ~96 blind-vote candidates/prompt, UNRELEASED; only path today
  is the authors' scraper under unresolved source-site ToS; adapter
  built+tested and waiting). kimi-k3 benchmarked (memorization 20.8%
  vs pred 0.20 — calibration hit; fingerprint between grok and
  open-weights); 12-model roster final; "below the entire null"
  headline honestly weakened to "at the null's floor" when kimi-k3
  joined. Calibration ledger: **27 closed, 0 open.**

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

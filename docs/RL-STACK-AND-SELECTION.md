# RL stack requirements, empirical model selection, and instrument cleanup

_Drafted 2026-07-24 in response to Sam: "is there a better more empirical
way to pick the model... we need to go over the things we need to be able
to do for RL and see what prebuilt packages already exist... and go
through our actual rl strategy and benchmarks again because those are
kind of a mess."_

Status: §2 package matrix and DeepSeek slate entries pending two
verification agents (dispatched 2026-07-24). Everything else is
decision-ready.

---

## 1. What we need to be able to DO for RL (requirements, not tools)

Grouped by pipeline stage. Each row is a capability the stack must
provide; the package matrix in §2 maps these to frameworks with
verified citations.

### Rollout / sampling
- R1. **Multi-turn conversational rollouts during training** — policy
  converses with a simulated rejector/user across turns inside the GRPO
  loop. This is the niche; single-turn GRPO is not our project.
- R2. Custom rollout logic (bring-your-own loop) as fallback for R1.
- R3. **Colocate mode on 8 GPUs** — whole node samples (vLLM), then
  vLLM sleeps and the whole node trains (FSDP + LoRA), per step.
  Raises trainable ceiling to ~235B total params.
- R4. Grouped sampling: K siblings per prompt, shared prefix KV.
- R5. Full per-request sampling control (temperature/top-p/min-p/seed).
- R6. **Logprob access both ways**: sampled-token logprobs AND
  prompt_logprobs-style scoring of arbitrary given text (the EXP-019
  spike/resolve instrument runs INSIDE the reward).

### Reward
- R7. Multi-component rewards, per-component logging (debugging hacks
  requires seeing each component's curve separately).
- R8. **Group-aware reward hook** — reward function sees the whole
  GRPO group before advantages (within-group z-score hack discount,
  group-relative novelty). castform's `compute_group_reward` is the
  reference shape.
- R9. User-controlled aggregation — multiplicative gates (BVT,
  appropriateness), not hardcoded additive sums.
- R10. RM training: Bradley-Terry baseline + swappable custom head
  (quantile regression for the NYCC distributional RM).
- R11. Async/batched external scorers in the reward path (embedding
  novelty, NLI clustering, RM inference) without stalling training.

### Training / stability
- R12. MoE stabilizers: GSPO (sequence-level importance sampling),
  router aux loss, vLLM↔trainer importance-sampling correction,
  router replay where available. Wired BEFORE first run.
- R13. LoRA on MoE with per-step adapter sync into vLLM.
- R14. KL control to reference; entropy monitoring; anti-collapse
  machinery (diversity-aware advantage variants).
- R15. Mid-training eval callbacks: cascade Gate-0 check, Laban-style
  multi-turn guardrail, calibration hooks every N steps.

### Eval / audit
- R16. BoN overoptimization audit (Gao-curve: proxy vs held-out
  instruments as optimization pressure rises).
- R17. Every generation eval carries the novelty check (CLAUDE.md hard
  rule) — infra exists in env/, must be callable from eval harness.

## 2. Package matrix (VERIFIED — agent pass 2026-07-24, all cells from
opened source/docs; full citations in the session transcript's task
output. S=supported P=partial A=absent U=unknown)

| # | Capability | TRL | verl | ms-swift | OpenRLHF | verifiers/prime-rl |
|---|---|---|---|---|---|---|
| 1 | Multi-turn user-sim rollouts in training | S (`environment_factory`, experimental) | S (`rollout.multi_turn.enable` + async SGLang) | S (`MultiTurnScheduler`, GYM env) | S (`AgentInstanceBase`) | S (**`UserSimEnv`** — NOT "vf.User"; corrects §6 of strategy-pass doc) |
| 2 | BYO rollout loop | S (`rollout_func`, experimental) | S (`AgentLoopBase`) | S (hooks or full override) | S | S |
| 3 | Colocate sleep/wake on same GPUs | S (GRPO-only; PPO has no vLLM) | S (default; shared Ray pool) | S (richest OOM toolkit) | S | **A** — disaggregated only (split node) |
| 4 | Sampling control (min_p, per-req seed) + grouped n= | P (batch-wide) | P (no min_p / per-req seed) | P (no min_p) | P | **S** (only one with min_p + per-req seed) |
| 5 | Rollout logprobs → reward fns / score given text | P/A | P (stored, not in reward sig) | A (source-confirmed excluded) | P/A | **S** (`trace` in `@reward`; opd/opsd use prompt_logprobs) |
| 6 | Multi-component reward + per-comp logging | **S** (`reward_funcs`+`reward_weights`, per-name curves) | P (eval-only curves; issue#2115) | S | P (user code) | S (`@vf.reward(weight)`) |
| 7 | Group-aware reward hook (see K-group pre-advantage) | P (undocumented ordering) | S (restricted: main_ppo_sync; custom manager gets full group) | S (in source, undocumented) | **A** | **S** (`score_group` first-class — the R8 reference) |
| 8 | Non-additive aggregation (multiplicative gates) | A (hardcoded sums) | **S** (no aggregation hardcoded — user code) | A (hardcoded nansum) | A | P (one layer up in `score_group`) |
| 9 | RM training (BT) + custom head swap | **S** (BT; head swappable via subclass) | A (serves RMs only) | P (BT, head hardcoded) | S (BT, head hardcoded) | A |
| 10 | GSPO / router aux loss / vLLM IS correction | S all 3 | **S all 3, richest** (loss-mode registry incl. gspo) | S all 3 | S all 3 | P (no GSPO) |
| 11 | Router replay (R3) for MoE | A | **S** (vLLM + SGLang, v0.7.1+) | S (Megatron-only) | A | S (documented, "order of magnitude" mismatch drop) |
| 12 | LoRA-on-MoE in GRPO w/ vLLM sync | P (works, sync "incredibly slow") | **S** (working Qwen3-30B-A3B example; add `router` to targets) | **BROKEN** (open #6670 → vLLM#27669) | P (unanswered upstream) | S (`experts` in default targets; filesystem sync w/ LoRA) |
| 13 | KL / entropy / diversity-aware variant | S/S/**A** | S/S/P | P/P/A | S/S/A | P/monitor-only/A |
| 14 | FP8 rollout + BF16 trainer | A | **S (only clean case** — `rollout.quantization=fp8` + TIS) | U | A (roadmap) | P |
| 15 | Mid-training eval callbacks + curves | S | S (val-split curves auto) | S | S | S (`orchestrator.eval.interval`) |

### Framework decision (from the matrix)
- **verl = training framework.** Only one clearing all four pillars at
  once: multi-turn rollouts, group-visible rewards, the full MoE
  stabilizer set (GSPO loss mode + router aux + R3 router replay on
  vLLM/SGLang + the only clean FP8-rollout/BF16-trainer story + a
  working LoRA-on-Qwen3-30B-A3B example), colocate on one node. Bonus:
  reward aggregation is NOT hardcoded — multiplicative gates (R9) are
  plain user code. Known gaps to plan around: per-component reward
  curves auto-log on validation only (patch for training-step curves);
  group-aware access documented only for the sync trainer.
- **TRL = RM training only** (`trl reward`; BT loss verified, custom
  quantile head via `compute_loss` subclass — the QRM path). Its GRPO
  is ruled out for the MoE policy (no router replay, no FP8 rollout,
  group hook undocumented, additive-only aggregation).
- **verifiers/prime-rl = env prototyping + benchmark harness +
  contingency.** Cleanest semantics for exactly our design
  (`score_group` first-class; `UserSimEnv` trainable; only stack with
  min_p + per-request seeds + logprobs-in-reward) but architecturally
  disaggregated — no colocate — and no GSPO. If the screen picks a
  DENSE policy, revisit: the MoE-stabilizer pillar vanishes and
  prime-rl's cleaner hooks may win despite the split-node throughput
  tax.
- **ms-swift ruled out** (LoRA+MoE+vLLM sync confirmed broken,
  #6670). **OpenRLHF ruled out** (no group-aware hook at all, no
  router replay).
- **Universal gap, all 5 frameworks: no diversity-preserving /
  anti-mode-collapse GRPO variant exists anywhere.** KL + entropy
  knobs only. Our diversity machinery (novelty penalties, z-score
  hack discounts, semantic-diversity terms) must be hand-rolled in
  verl's reward manager / core_algos — which is confirmation the
  project's gap #3 contribution is still open as of July 2026.

Known non-candidates: castform (no user-sim, tool-loops only — its
`compute_group_reward` shape maps onto verl's custom reward manager
and prime-rl's `score_group`), OpenEnv (packaging only), pufferlib/SB3
(classic RL).

## 3. Empirical model selection protocol (replaces priors/vibes)

Third-party benchmarks (creative writing etc.) are OUT of the decision
per Sam 2026-07-24 — they measure someone else's construct. Selection
runs on OUR instruments, self-hosted, pre-registered.

### Stage 0 — hard constraints (mechanical filter, no judgment)
- C1. Trains on the node in colocate mode: BF16 FSDP shard + LoRA
  optimizer + checkpointed activations ≤ ~75GB/GPU → total params
  ≲ 235B.
- C2. Weights public, license compatible with publishing results
  (Apache/MIT/Modified-MIT fine; research-only needs a Sam call).
- C3. RL path exists in at least one §2 framework with MoE stabilizers
  if MoE; no OPEN blocking bug on the exact checkpoint (e.g.
  Qwen3-Next-80B-A3B verl#4907 → screen-only until closed).
- C4. Thinking/reasoning tokens can be disabled or budgeted
  (reasoning-burn dead end).

### Stage 1 — instrument screen (inference-only, all 8 GPUs, ~free)
Per candidate, fixed prompt sets and seeds, pre-registered predictions
in calibration.py:

- S1. **Cascade survival depth** — our diagnostic benchmark,
  temperature-controlled, wrapper-free, N≥4 runs. First time the
  cascade runs on checkpoints instead of API stacks.
- S2. **Spike/resolve separation** — EXP-019 instrument computed from
  the candidate's OWN logprobs on the EXP-014 fixture + auditor-blind
  held-out set: AUC separating jokes / non-sequiturs / vague. The
  policy is the measurement device; a model whose logprobs can't see
  joke structure can't be trained with this reward.
- S3. **Memorization rate** — windowed n-gram + embedding novelty vs
  joke corpus on sampled sets. High memorization = mode-collapse seed
  (the documented humor failure mode).
- S4. **Distributional health under temperature** — distinct-n and
  semantic diversity across a T sweep. GRPO's gradient signal IS
  within-group variance; a distribution that won't open up starves
  the advantage estimator. Also catches temperature no-ops (EXP-007b
  dead end).
- S5. **Measured throughput** — tok/s/GPU under rollout-realistic
  concurrency (no published numbers exist; we measure).

### Stage 2 — trainability probe (finalists only, ~2 GPU-hours each)
50–100-step GRPO smoke run on a toy verifiable reward (not humor).
Monitor: reward variance, KL trajectory, entropy, router stats (MoE),
adapter-sync integrity. Pass = stable curves, no step-0 hangs, no
collapse onset. Converts "no public GRPO report" from an unknown into
a two-hour measurement.

### Decision rule (pre-register BEFORE the screen runs)
Select the cheapest-to-sample candidate unless a bigger one beats it by
pre-stated margins on construct metrics (S1/S2 primary, S3/S4 gates):
template — "235B is chosen over 30B-A3B only if it wins S2 AUC by
≥ +0.10 AND median cascade depth by ≥ +2 turns, since its 7× sampling
cost halves-to-quarters our ablation budget with the param-matched SFT
baseline included." Exact margins set at pre-registration, logged as
calibration predictions per candidate.

### Current slate (constraints-filtered, pending DeepSeek agent)
| Candidate | Sampling cost (active) | Stage-0 status |
|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | 1× (3.3B) | pass (stabilizers required) |
| Qwen3.6-35B-A3B | ~1× (3B) | pass (official FP8, stable vLLM) |
| Qwen3-235B-A22B-Instruct-2507 | ~7× (22B) | pass w/ colocate; trainability probe critical |
| GLM-4.5-Air | ~3.5× (12B) | pass (MIT) |
| Qwen3-Next-80B-A3B | ~1× (3B) | **screen-only** (verl#4907 open) |
| DeepSeek-V4-Flash | ~4× (13B) | **screen + RM-backbone candidate**; training = QLoRA-path only, unverified (see below) |
| Qwen3-8B | dense 8B | control/debug, stays |

### DeepSeek family verdict (agent pass 2026-07-24, VERIFIED)
- **DeepSeek-V4-Flash — 284B total / 13B active, MIT, 1M ctx, released
  Apr 2026** (HF: deepseek-ai/DeepSeek-V4-Flash). The node-compatible
  DeepSeek. INFERENCE: fits — ~284GB FP8 via third-party repack
  (sgl-project/DeepSeek-V4-Flash-FP8); SGLang serves it today; vLLM's
  optimized Hopper path is an open RFC (vllm#42284; native checkpoint
  is FP4, which H100 lacks hardware for). TRAINING: BF16 FSDP fails the
  ceiling (71GB/GPU frozen weights alone). The one loophole: it is
  natively QAT FP4 — a frozen-4-bit-base + LoRA path would be
  ~20GB/GPU, but trainer/PEFT support for its custom format + new
  CSA/HCA attention is UNVERIFIED. Treat as: screen entry now,
  strongest RM-backbone candidate yet (modern 284B MIT model servable
  in-node), policy candidate only if the 4-bit-LoRA path verifies AND
  it clears the pre-registered margin at ~4× sampling cost.
- **V4-Pro (1.6T)**: out entirely (862GB > node). **V3/V3.1/V3.2-Exp
  (671B-A37B)**: inference only at INT4 AWQ (~335–400GB), no FP8 fit
  (685GB > 640GB); training out. No official small V3 variant exists.
- **DeepSeek-V2-Lite (15.7B-A2.4B, May 2024)**: the only
  architecture-authentic (MLA+MoE) DeepSeek that trains comfortably
  (~31GB BF16 total; ms-swift has explicit model_type + GRPO/LoRA).
  But it is a 2024-era model — expected to lose the screen to the
  2026 A3B tier. Include in screen as floor/curiosity only.
- **R1-Distill line**: Qwen2.5/Llama3 (and one Qwen3-8B) backbones —
  DeepSeek post-training, NOT DeepSeek architecture. Reasoning-on by
  default; budget control unverified per-checkpoint.
- **No humor/creative eval exists for any DeepSeek that fits the
  node** — anything we measure on V4-Flash or V2-Lite is a first
  baseline, which cuts both ways (novel data point; no external
  sanity anchor).

## 4. Instrument & benchmark inventory (the cleanup)

One construct → one instrument → one status. Retirements explicit.

| Instrument | Construct (theory) | Status | Role going forward |
|---|---|---|---|
| Rejection cascade v1 | distributional depth under rejection (novelty + adaptation) | solid, red-teamed; API-stack-scoped | DIAGNOSTIC benchmark + screen S1; NOT a funniness measure (Sam's call, 2026-07-22) |
| Spike/resolve ΔS (EXP-019) | incongruity + resolution (Kao/Deckers) | **v1 FALSIFIED 2026-07-29** (cue-conditioned ΔS: held-out AUC 0.000 — cue licenses discontinuity, not resolution; see EXPERIMENT_LOG) | operationalization retired; construct alive. EXP-020 closed the surprisal-only lead (chance at power). **EXP-021 (2026-07-29): the generative resolution probe NEAR-MISSED its bar (0.699 vs 0.70) with 3/4 criteria passing, correct class ordering across all 6 classes (first ever), and a +0.62 gap over the topicality baseline — kept alive as bad-hyperparam.** Tune path: K=10, 3rd blind author, stronger predictor at screen time. S2 = tuned EXP-021, pending its own re-registered bar |
| Callback detector v2 (EXP-016/016b) | reincorporation/callbacks | **certification failed 2026-07-29, cleanly decomposed**: transformation SCORING now solid (verbatim/punctuation-edit/paraphrase/continuity all floor to 0 on blind); DETECTION is the wall — lexical gates cannot separate multi-word coincidence from reference ([DEAD-END]), MiniLM cosine can't either | UNWIRED until EXP-016c (local-NLI reference tier, queued) passes the FP bar; scoring and detection certify separately per [LEARN] |
| Semantic step-size trajectories (EXP-015) | escalation pacing | FALSIFIED as-run (window artifact) | retired as metric; window-pinning lesson kept |
| Surprise proxy, judge-based (EXP-014) | incongruity | FAILED cert (0.389 vs 0.65) | RETIRED — superseded by EXP-019 |
| Haiku-as-judge (any load-bearing use) | funniness | FAILED cert (ρ=0.056) | RETIRED; GPU block stands until a judged-third passes |
| NYCC quantile RM | consensus taste (distributional) | planned; data verified (CC-BY corpus + 250M votes) | the judged-third candidate; must pass its own cert bar before load-bearing |
| Novelty/memorization suite | anti-mode-collapse | solid, windowed default-on | gate in every eval + screen S3 |
| Laban-style multi-turn guardrail | conversational degradation | planned (EXP-018) | mid-training callback R15 |

Mess acknowledged and resolved by this table: three generations of
overlapping incongruity instruments → exactly one survivor (EXP-019
path), with the judge-based ones explicitly retired rather than
lingering; benchmark v1 explicitly re-scoped as diagnostic; the RM is
the only candidate judged-third and it must pass certification like
everything else.

## 5. Open items

- ~~Fold in agent results~~ DONE 2026-07-24 (§2 matrix, DeepSeek slate).
- Pre-register the screen (hypotheses + margins + calibration
  predictions) — blocked only on node availability, not on design.
- Reward-term specs consolidation: THEORY-MAP §12 registered specs +
  the anti-hack shell (UWO/z-discount/decoupling monitor) into one
  reward-stack doc once the framework choice lands.

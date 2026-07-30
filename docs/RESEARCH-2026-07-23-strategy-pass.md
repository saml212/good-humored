# Research pass 2026-07-23 — reward hacking, consensus RMs, incongruity math, conversational measurement

Four parallel research agents (sonnet), all claims tagged VERIFIED (source
opened) or UNVERIFIED (recollection/snippet) in the session transcript.
This file keeps the verified citations and the decisions they feed.
Companion strategy discussion: session 2026-07-23 with Sam.

## 1. Reward hacking against a neural proxy — what's published

**The design rule that falls out: penalize DECOUPLING, not increase.**
A reward component rising is what improvement also looks like; the hack
signature is one component rising while correlated signals (ensemble
twin, held-out RM, sibling components, gold spot-checks) stay flat.

In-loop, differentiable (usable as training-time penalties):
- **UWO / WCO** — Coste et al., arXiv:2310.02743 (ICLR 2024).
  `R = mean(ensemble) − λ·var(ensemble)` (UWO) or `min_i R_i` (WCO).
  With small KL (0.01) fully prevents PPO overoptimization in their
  setup; λ insensitive. Cheapest published version of "quantify the
  anomaly and penalize it."
- **Constrained RLHF** — Moskovitz et al., arXiv:2310.04373 (ICLR 2024).
  Components as constraints `v_i ≥ θ_i` with sigmoid-bounded Lagrange
  multipliers updated per batch — the multiplier IS the automatic
  penalty on an anomalous component. Caveat (their own finding):
  thresholds must be calibrated jointly; component correlation moves
  the overoptimization point.
- **Within-group z-score discount** — "When Reward Hacking Rebounds,"
  arXiv:2604.01476. z-score a hack-correlated score within the GRPO
  group, discount `r' = r·(1 − α·σ(z)·1[z>0])`. Hack rate 99.9%→24.9%
  (Phi-4-mini). Directly portable to per-component z-scores in
  `compute_group_reward`.
- **MO-GRPO** — arXiv:2509.22047. Per-component variance normalization
  so no single noisy term dominates the gradient. Free baseline.
- **WARM** — arXiv:2401.12187. Weight-averaged RMs; ensemble robustness
  at single-model inference cost.
- **PAR / reward shaping** — arXiv:2502.18770. Bounded + saturating
  reward transforms resist hacking (qualitative; exact transform
  unverified).

Monitor-only / data-loop:
- **Gao/Schulman/Hilton scaling laws** — arXiv:2210.10760. Proxy rises
  while gold peaks then falls; d=√KL functional forms. The canonical
  overoptimization curve every monitor is trying to detect early.
- **InfoRM / CSI** — arXiv:2402.09345 (NeurIPS 2024). Hacked samples
  are latent-space outliers; drives early stopping.
- **Adversarial Reward Auditing** — arXiv:2602.01750. Hacker/Auditor
  co-evolution, then multiplicative reward gating by P(genuine).
- **Anthropic production case study** — arXiv:2511.18397. Reward
  hacking generalizes to broader misalignment; catch it early.

Negative results to respect:
- **Eisenstein et al.** — arXiv:2312.09244. Ensembles reduce but do NOT
  eliminate hacking: shared-bias exploits survive because all members
  share pretraining blind spots. Consequence: any ensemble/UWO defense
  still needs a periodic gold (human) spot-check lane.
- Humor-specific hack direction confirmed in the wild: stereotypical/
  toxic jokes score 10–21% higher on humor metrics across 6 models —
  arXiv:2510.18454. Appropriateness must be a multiplicative gate, not
  an additive term.

Convergence with our own results: the EXP-011–014 arc already showed
additive stacks pay compensation (violation-only earns 80% of
both-class reward) and the multiplicative BVT gate closes it — same
conclusion as the aggregation literature (concave/min-like combining
resists single-dimension hacking; weighted sums invite it).

## 2. Consensus-rated funniness data + RM methodology

- **Oogiri-Master / Oogiri-Corpus** — arXiv:2512.21494. VERIFIED:
  908 prompts × ~96 candidates, ~171.6 votes/prompt, JAPANESE,
  CC BY-NC-SA 4.0, data unreleased (MIT-licensed scraper only,
  source-site ToS unresolved — matches STATE.md block). NO published RM
  baselines on it. Value to us now: methodology (independent-judge
  rating vs popularity bias) + Cohen's-d-validated funniness correlates.
- **Cross-lingual humor RM transfer: no literature exists.** General RM
  transfer works for summarization/dialogue (arXiv:2410.18027,
  2404.12318) but nobody has tested humor. Assuming Japanese-oogiri →
  English-banter transfer is unfounded.
- **NYCC** — two artifacts: Hessel et al. arXiv:2209.06293 (ACL 2023
  best paper; CC-BY-4.0 corpus) and Zhang et al. arXiv:2406.10522
  (NeurIPS 2024 D&B; 250M+ votes, 2.2M+ captions). Zhang's own
  RLHF/DPO numbers: RLHF best-pick win rate vs human top-10 = 8.24%,
  DPO 30.22%; RLHF loses to Best-of-N despite higher RM scores. Their
  RM was mean-rating regression — the erasure of rating structure is a
  plausible contributor, and fixing it is open space.
- **Distributional RMs** — QRM, arXiv:2409.10164: quantile-regression
  reward heads for multimodal/heavy-tailed preferences (the 50/50
  love-it/hate-it joke vs the uniformly-mid joke). **No humor
  application published — confirmed gap.**
- **BT not required** — arXiv:2411.04991 (ICLR 2025): order consistency
  is what matters; regression/classification alternatives legitimate.
- **LLM judges are weak humor instruments even unoptimized** —
  arXiv:2511.09133: Spearman vs human consensus 0.169–0.266 (Claude
  Sonnet 4 = 0.266). Independent confirmation of our EXP-012 (haiku
  ρ=0.056).
- **HumorGen** — arXiv:2604.09629: with well-curated SFT data, DPO and
  O-GRPO add nothing over SFT. The param-matched SFT baseline is
  mandatory before claiming RL gains (already a CLAUDE.md hard rule).
- The "GRPO + GPT-4.1 judge → classic-joke regurgitation" story is a
  LessWrong post (agg, 2025-05-16), not a paper. Cite as anecdote only.

## 3. Math of expectation violation ("on-purpose hallucination")

- **Kao, Levy & Goodman** (CogSci 2013 / Cog Sci 2016) — the anchor.
  Ambiguity = entropy of P(m|w) over two latent readings; Distinctiveness
  = symmetrized KL between focus distributions per reading. r=0.33 and
  r=0.21 with human funniness; joint R²=0.145. NOTE: monotonic in both
  measures in their data — no inverted-U. Their P(w|m) came from human
  relatedness ratings, not an LM — upgrading that to real logprobs is
  still open.
- **Xie, Li & Pu** (ACL 2021) — pure-logprob features:
  Uncertainty = mean next-token entropy over punchline positions;
  Surprisal = length-normalized −log p(punchline|setup). Each alone
  ≈0.55–0.57 acc on SemEval-2021 Task 7.
- **Causal evidence surprisal carries humor**: swapping low-probability
  tokens for predictable ones removes the funny (Horvitz et al., ACL
  2024, via arXiv:2510.24538's confirmed citation).
- **Inverted-U is CONTESTED in humor**: Deckers' weight-judgment
  paradigm found humor is a concave MONOTONIC function of incongruity,
  not an inverted-U. Implication: "too far = unfunny" is better modeled
  as "unresolved = unfunny" (resolution failure), not distance per se.
  Matches EXP-014's own data: gate-1 fires for jokes AND non-sequiturs;
  gate-2 (resolution) is what separates them (0.000 for non-sequiturs).
- **Semantic entropy** — Kuhn/Gal/Farquhar ICLR 2023 + Nature 2024:
  cluster sampled completions by bidirectional NLI entailment,
  SE = entropy over meaning-clusters. The sampling-based generalization
  of Kao ambiguity, embedding-free.
- **Confirmed novelty gaps (searched, absent):**
  1. Nobody computes a forward-surprisal-spike + retrospective
     resolution-drop DELTA as a humor metric. This is exactly our
     registered EXP-019: ΔS = −log P_θ(pivot|setup) +
     log P_θ(pivot|setup+twist-cue) from policy logits.
  2. Nobody connects hallucination formalisms (semantic entropy,
     epistemic uncertainty) to humor. "Joke = spike WITH resolution,
     hallucination = spike WITHOUT" is unclaimed territory.
  3. No quantitative operationalization of Benign Violation Theory
     exists (Likert manipulations only).

## 4. Conversational humor measurement — new dimensions

- **StandUp4AI** — arXiv:2505.18903 (EMNLP 2025 Findings). 330+ hrs
  stand-up, 7 languages, CC BY-4.0, word-level laughter labels →
  trainable laughter-duration regressor as a dense audience-anchored
  reward. **TIC-TALK** — arXiv:2603.21803: 0.8s-resolution laughter
  alignment. **Open Mic** — arXiv:2110.12765: laughter-duration
  humor-coefficient, best model QWK 0.813.
- **DPV / "Timing is Everything"** — arXiv:2605.00143: temporal
  features outweigh semantic incongruity for audience appreciation;
  PEAK violation position matters (late peak = payoff). Computable
  from our existing per-token surprisal curve — zero new data.
- **SPOLIN** — arXiv:2004.09544 (ACL 2020): 26k+ yes-and turns →
  cooperative-build classifier; cooperativity ≠ callback, fills a gap.
- **Persona-conditioned preference** — arXiv:2601.03103 (BTL per user
  cluster on oogiri votes) + arXiv:2606.00022 (17-dim interpretable
  humor basis, 83% pairwise acc). Feeds EXP-017 audience adaptation.
- **Appropriateness gate data** — arXiv:2506.01819 (workplace humor,
  4-level) + arXiv:2510.18454 (toxicity–funniness correlation).
- **MUStARD is sarcasm, not laugh-track humor** — correct our notes.

## 5. Claim-hygiene correction (feeds paper + CLAUDE.md)

The "RLVR damages multi-turn conversational skill" hook has NO clean
primary source. What exists: Laban et al. arXiv:2505.06120 (39% multi-
turn drop, aptitude/reliability decomposition — MODEL-AGNOSTIC, not
RL-causal); pass@k crossover arXiv:2504.13837 (RLVR narrows the
distribution); RLVR diversity-collapse papers (2509.07430, 2606.15455,
math domain). RLAAR (arXiv:2510.18731) even shows reward-shaped RL
IMPROVING multi-turn reliability. Restate our hook as: the degradation
phenomenon is documented, the RLVR-causal link is untested — EXP-018
tests it (that upgrades our hook from citation to contribution).

## 5.5 Model selection for 8×H100 (added 2026-07-24, two agent passes)

Ceiling math: 640GB HBM serves ~350B total params FP8 for INFERENCE
(Qwen3-235B-A22B comfortable, GLM-4.6 355B tight); GRPO TRAINING
(BF16 trainer copy + optimizer + colocated vLLM) caps ~120B comfortable,
235B QLoRA-stretch. Sampling cost is priced in ACTIVE params.

- **Qwen3-30B-A3B-Instruct-2507** — default. Apache 2.0, official FP8,
  Creative Writing v3 86.0 / WritingBench 85.5 (VERIFIED, HF card).
- **Qwen3-Next-80B-A3B-Instruct** — the "bigger at same sampling cost"
  candidate, FALSIFIED as an upgrade: CW v3 **85.3 < 86.0** (VERIFIED,
  llm-stats aggregator; eqbench.com itself unfetchable), and the ONLY
  candidate with a confirmed-open GRPO bug — verl#4907, Megatron-Bridge
  tensor-shape mismatch on its hybrid DeltaNet+MoE+MTP layout, no fix.
  FP8 card requires vLLM main-branch (support lagged stable). TRL
  compatibility: no evidence either way.
- **Qwen3.5-35B-A3B / Qwen3.6-35B-A3B** — Apache 2.0; 3.6 has official
  FP8 + stable vLLM ≥0.19; NO verified creative-writing score for
  either (real gap). Tool-call bugs (verl#6223 open; vLLM#39056 open,
  #35347 merged 2026-06-15) are specific to token-in-token-out rollout
  with tool calling — likely irrelevant to our chat-only humor
  rollouts, and TRL's server-mode path is probably unaffected
  (UNVERIFIED).
- **Qwen3-235B-A22B-Instruct-2507** — CW v3 **87.5, rank #3**
  (VERIFIED). No community report of GRPO on 8×H100 (closest: LoRA SFT
  on 8×H20). 7× sampling cost. Inference-only screen entry.
- **GLM-4.5-Air 106B-A12B** — MIT (VERIFIED); absent from CW v3
  leaderboard entirely; no GRPO reports.
- **Kimi K3** — ruled out for training: 2.8T params (VERIFIED), ~1.4TB
  weights at native MXFP4 vs 640GB available; no smaller variant;
  weights unreleased until ~2026-07-27; always-on max reasoning with no
  off switch (VERIFIED via Willison, ~4:1 think:output) — reproduces
  our kimi reasoning-burn dead end. Keep as API-side cascade subject;
  candidate SFT-data teacher (check output terms). EQ-Bench CW #1 claim
  (Elo 2377) UNVERIFIED.
- **CLAIM CORRECTION**: the "3/3 baseline GRPO runs collapsed on
  Qwen3-30B-A3B" figure is UNVERIFIED (not found in primary source).
  What IS verified: arXiv 2510.23027 Fig 2 shows vanilla GRPO on
  Qwen3-30B-A3B with "pronounced performance collapse around 200–500
  steps"; ms-swift#6029 (GRPO hang at step 0 on 30B-A3B-2507 with async
  vLLM + ZeRO-3 offload) open. MoE-RL instability is real; the 3/3
  quantifier is not citable. Mitigations unchanged: GSPO
  (`importance_sampling_level="sequence"`) in TRL, or router replay
  (R3/PR2-style) in verl — wire in BEFORE first run.
- **Bottom line**: NO candidate has a verified clean GRPO run at this
  scale on this hardware. 30B-A3B is the best-evidenced default; the
  self-hosted benchmark screen (inference-only, includes 80B/235B)
  is the decision gate; big models' role is the reward path (RM
  backbone) and screening, not the policy.

## 6. Toolchain facts (skills audit)

Three parallel, unconnected env toolchains now installed: castform
(benchmax BaseEnv; `compute_group_reward` for group-relative terms;
multi-turn = tool loops, NO user-sim), verifiers.v1 (Prime Intellect;
user simulation via `UserSimEnv` / `--env.id user-sim` — the earlier
"vf.User" name was wrong, corrected by the 2026-07-24 source-level
verification pass; judge rewards, multi-agent, hub distribution),
OpenEnv (Meta; env packaging only, no training). TRL CLI covers
`trl grpo` (+ `--reward_funcs`) and `trl reward` (RM training).
pufferlib / stable-baselines3: classic-RL only, not relevant to LLM
policies. Catalog sweep 2026-07-23: nothing further relevant to pull.

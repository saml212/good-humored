# Research pass: making the taste signal learnable (2026-08-13)

Literature stage of the waterfall triggered by RL-D's decisive taste
negative (+0.003 at t=0.6 despite 0.6 weight, dominant advantage
variance, bait channel closed). Agent-verified citations; PDF-only
items flagged low-confidence in the transcript.

## The two mechanistic accounts that explain RL-D without scale

1. **Noise-corrected GRPO (arXiv:2510.18924)** — proves the GRPO
   gradient is attenuated multiplicatively by
   `(1-rho+ -rho-) * sigma_true / sigma_noisy` — nominal reward
   weight cannot compensate. Our measured ~27% noise share of
   within-group taste variance plugs directly into this account.
   Demonstrated fix: estimate flip rates on a held-out set, debias
   + rescale (+6.7pp GSM8K even at 53% FP rate).
2. **Headroom null (arXiv:2607.12640)** — identical reward +
   algorithm swings from ~0 to +22pp purely on whether sampled
   rollouts ever beat greedy ("positive headroom"). Mandatory
   pre-test for ANY redesign: measure the sampled-vs-greedy gap
   under the judge before concluding anything about the signal.

## The collapse warning

**Dark Room (arXiv:2607.21273)** — dense shaping under GRPO's
std-normalized advantage AMPLIFIES low-variance signals into
full-scale advantage (documented 0% collapse with auxiliary metric
at 1.0). Variance-Profile Criterion: a signal is amplifier-safe
only if within-group variance shrinks as competence rises. Our easy
terms (screens/floor/selfrep) have exactly that profile — likely
WHY they trained. Naive densification of the judge signal without
this check is a documented failure mode, not a neutral move.

## Prior art map (do not claim as novel)

- Contrastive in-reward baseline: arXiv:2403.07708 (offline
  SFT-pool subtraction, built-in noise attenuation, 31pp human win
  gain). Our context-matched neutral-counterfactual variant is an
  EXTENSION, cite the base mechanism.
- Scalar-to-dense redistribution without new RM: RED
  (arXiv:2411.08302), attention-based Dense-Reward-for-Free
  (arXiv:2402.00782).
- Decompose-into-rubrics + CoT GenRM + ensemble-filter + reweight
  for creative-writing RL: **RLCS (arXiv:2601.07149) already
  published this whole recipe** (68% expert alignment, 72.4% win
  vs SFT). Differentiation must be humor-structural (incongruity,
  callbacks), not reward mechanics.
- Counterfactual credit in-reward: PCCC (arXiv:2606.05263),
  verifiable domains only — subjective adaptation is open.
- Judge ensembling numbers: k=8 calls +9.8pp judge accuracy, k=3
  captures ~70% of it (arXiv:2604.13717) — static eval only; NO
  published evidence it fixes RL learnability.
- Judge flip rates: >95% same-verdict at T=0 falling to ~70% at
  T=1; 13.6% mean pairwise flip (2412.12509, 2510.27106,
  2606.13685).
- Batch-invariance: server nondeterminism is batch-composition
  kernel non-associativity (Thinking Machines 2025; vLLM
  batch-invariance mode) — matches our measured 1.79-logit
  cross-day call noise with same-day determinism. INFRA-FIXABLE
  (`VLLM_BATCH_INVARIANT=1`) — rule out before signal redesign.
- Full-param 4B GRPO DID extract (and hack) a rubric-judge reward
  in <500 steps (arXiv:2606.04923) — existence proof relevant to
  the capacity question; not controlled vs LoRA.
- Rank literature is split and regime-dependent (2601.06677:
  r=256 helps plastic models, collapses rigid ones; 2605.07366:
  non-uniform rank allocation HURTS GRPO). No clean rank ablation
  on a subjective judge reward exists — running one would be first.

## Field signal on humor specifically

SemEval-2026 MWAHAHA best pipelines still route around RL-on-judge
(generate-many + rerank); HumorRank does tournament/pairwise
judging (eval-only) — pairwise formats fit GRPO's group-relative
math natively. RL-on-judge for humor remains unproven field-wide;
our negative is aligned with the field, and a working recipe would
be a real contribution.

## Ordered implications for the attack stage

1. FIRST (free): headroom check on RL-D dumps + variance-profile
   trajectory of the taste term. If headroom ~0, redesign is moot
   until exploration changes (temperature, group size).
2. SECOND (infra, cheap): batch-invariant audience serving; then
   re-measure call noise. If noise collapses, k=1 calls may
   suffice; else k=3 ensembling.
3. THIRD (signal): noise-corrected GRPO debias + contrastive
   context-matched baseline (cite 2403.07708) — both attack
   sigma_noisy directly.
4. AVOID as first moves: naive densification (Dark Room), capacity
   sweeps (split literature, two complete no-scale accounts of our
   failure exist).

## Brainstorm stage output (10 candidates, ranked by
## expected-learnability x cheapness-to-falsify)

1. **Distilled structural surrogate** — fit ridge over cheap text
   features (anchor_sim, selfrep, length, callback score, punchline
   position) against banked reaction_L; train on the surrogate,
   certify on the audience. Pretest: offline R2 + correlation vs
   the 32/50 human labels. Risk: Goodharts if reaction_L is noise
   w.r.t. all text features.
2. **Incongruity-gate port** — wire env/incongruity_gate.py (Suls
   surprise-then-resolve, logprob/embedding-based) as the taste
   construct; EXP-019's unregistered chat-format cold-surprisal
   AUC 1.000 is the strongest unclaimed lead. Risk: EXP-014's
   instrument-capability gap reimported.
3. **Teacher-forced per-token audience surprisal** — one forward
   pass with prompt_logprobs over the policy's own tokens =
   per-token credit aligned with response_mask; must pair with
   comprehensibility gate (EXP-014 confound). Plumbing unverified.
4. **Contrastive vs fixed neutral pool** — L(actual) − L(neutral
   hash-selected line) per context (cite 2403.07708). Risk:
   neutral sits at floor, subtraction buys nothing.
5. **Windowed-context reaction** (last exchange only) — one-line
   change; risk: under-rewards callbacks, which the product wants.
6-10 (lower): ensembling (EXP-014b precedent: variance fix is not
   a capability fix), pairwise in-group forced choice (partially
   redundant with GRPO z-scoring), partner-continuation delta
   (engagement-proxy risk — Sam's explicit skepticism), leave-one-
   clause-out attribution (grammaticality confound), prefix
   trajectory (punchline-at-end collapses it to whole-turn at 3x
   cost + verl plumbing).

Waterfall status: brainstorm + research DONE; attack stage running
the free pretests (headroom, variance profile, surrogate R2) as its
instruments. Only survivors get registered.

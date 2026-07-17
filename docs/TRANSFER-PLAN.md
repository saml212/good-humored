# Reverse-transfer pre-registration: train on humor, measure general capability

Status: **design, not built. No training runs from this document.** Owner: Sam
Larson. Written 2026-07-17. Format follows `EXPERIMENT_LOG.md` house style —
numbers before narrative. This is gap #1 from `CLAUDE.md` → Research
Direction: *"nobody has tested whether training on humor transfers back to
general reasoning/taste."* HumorBench established only the forward direction
(STEM-reasoning training → humor comprehension); the reverse direction is,
per `.claude/skills/humor-rl/SKILL.md` §"Measuring transfer," **untested by
anyone**. That novelty claim is the literature review already on file
(`references/negative-results.md`, `references/psychology.md`,
`docs/BENCHMARK.md` §6) — checklist item 8 (verify novelty) is satisfied by
citation, not re-litigated here.

## Pre-Experiment Checklist coverage (CLAUDE.md, all 8 items)

| # | Item | Where |
|---|---|---|
| 1 | State the hypothesis in one sentence | §1 |
| 2 | Predict the metric delta, register with `calibration.py add` | §1, §7 |
| 3 | Compute FLOPs/memory/params on paper | §2 |
| 4 | Try to disprove in 5 minutes | §1.3 |
| 5 | Check the literature first | done — this doc is built on `negative-results.md`, `psychology.md`, `humor-rl/SKILL.md`, `BENCHMARK.md` |
| 6 | Design the comparison before the experiment | §5 (compute-matched control), §3 (SFT bar) |
| 7 | Define success criteria | §6 |
| 8 | Verify the claim is novel | done — `CLAUDE.md` Research Direction + `SKILL.md` §"Measuring transfer" both state this is untested |

---

## 1. Hypothesis and predicted deltas

**Hypothesis (one sentence):** GRPO training an 8B open-weight model on the
humor reward stack (`make_humor_reward_stack()`) produces a small positive
shift in general-reasoning benchmark accuracy (MMLU, GPQA) relative to a
compute-matched non-humor control arm, mirroring — in reverse — the forward
transfer HumorBench documented from reasoning training into humor
comprehension.

### 1.1 Predicted deltas (to register at run launch, not now)

All deltas are **humor-GRPO arm minus compute-matched control arm**, both
measured against their own frozen pre-training baseline, on the SAME base
checkpoint. This is a correlational prior (r≈.29–.40, `psychology.md` §1) is
not a causal license — the direction is a two-tailed guess with a
weak-positive lean, not a confident claim.

| Metric | Predicted delta | Sign confidence | Rationale |
|---|---:|---|---|
| MMLU (5-shot, full 57-subject) | **+0.010** (+1.0 pp) | Low-moderate | Broadest, most humanities-weighted general benchmark; crystallized/verbal correlate of humor production (`christensen2018clever` Gc r=.49) is exactly MMLU's composition |
| GPQA-diamond (0-shot) | **+0.005** (+0.5 pp) | Low | Graduate STEM reasoning is further from the "verbal wit" construct (`christensen2018clever` Gf r=.22, the weakest of the three CHC factors) — predict a smaller, noisier effect |
| SFT-only arm, MMLU (disproof check, §1.3) | **+0.002** (+0.2 pp) | Low | Predicting SFT alone shows near-nothing, so that any GRPO-arm signal above this is the interesting result, not just "exposure to text" |
| Cascade depth-to-degradation (post − pre, humor-GRPO model only, N=8 runs, 30-turn cap) | **+5 turns** (~+17%) | Moderate | If humor training actually improves output diversity rather than just reward-hacking the judge, the model should survive longer before repeating/refusing |
| Cascade within-model path divergence (raw Jaccard-complement, post − pre) | **+0.07** | Low-moderate | Companion metric to depth; a model that's actually more diverse post-training should re-walk fewer identical paths across repeated runs |

### 1.2 Statistical power note (flagged now, not after the fact)

MMLU stderr at 7-8B scale is typically ≈0.004-0.005 per the `lm-evaluation-harness`
skill's own worked example; GPQA-diamond has only ~198 questions (Rein et
al. 2023 — cited from general knowledge of the benchmark, not independently
re-verified this session; **confirm exact N and harness task variant
[`gpqa_diamond_zeroshot` vs `gpqa_main`] before finalizing**), so a single
question flipping is worth ≈0.5pp on its own. **The predicted GPQA delta
(+0.5pp) is within one flipped question of the noise floor.** This is not a
reason to drop the metric — it's a reason to report it with a bootstrap CI
per-subject, not a bare point estimate, and to not over-read a null GPQA
result as evidence against the hypothesis (see §6, §"biggest risk").

### 1.3 Disproof attempt (checklist item 4, 5-minute version)

Could a simpler/cheaper baseline show this without GRPO at all? **Yes,
partially testable for ~1/5th the compute**: run the SFT-only arm's eval
before spending on GRPO. HumorGen's own negative result
(`negative-results.md` §3) says SFT-on-well-curated-data is a genuinely hard
bar to beat *for funniness* — the disproof question here is the transfer
analogue: if SFT alone (cheap, ~4 GPU-hours, no rollout/generation cost)
already produces the predicted MMLU delta, GRPO's added cost isn't necessary
to establish gap #1, and the interesting finding becomes "exposure to
curated humor text transfers," not "RL on humor transfers." Predicted SFT-only
delta is registered at +0.002 (near-zero) specifically so this comparison is
falsifiable: **if SFT-only alone clears +0.008, stop and reconsider before
funding the GRPO arm** — the cheap experiment would have already answered
the expensive one.

---

## 2. Model selection

Candidates in the 7–32B range, per `CLAUDE.md` (Qwen3 family, Llama-3.x).
FLOPs and VRAM computed on paper (Kaplan et al. 2020 heuristic: training
FLOPs/token ≈ 6·N_params; forward-only/generation FLOPs/token ≈ 2·N_params —
foundational scaling-law arithmetic, not a claim requiring fresh
verification).

### 2.1 VRAM on paper (LoRA GRPO, weights colocated with vLLM rollout engine)

LoRA is the load-bearing memory-saving choice here: the reference policy for
the KL term is the *same* frozen base model with the adapter disabled, so no
separate reference-model copy is needed (a real saving vs. full-parameter
GRPO, which needs weights + fp32 master + Adam m/v + gradients + a separate
reference copy — back-of-envelope for full fine-tune at 8B: 16GB bf16 +
32GB fp32 master + 64GB Adam state + 16GB grad ≈ **128GB**, already over a
single 80GB card before any activations or rollout memory — this is why LoRA
is the default here, not a preference).

| Model | Params (dense) | Frozen bf16 weights | + vLLM rollout copy* | LoRA + optimizer overhead | Est. peak VRAM (num_generations=8, seq≈1024) | GPUs (A100-80 / H100-80) |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-8B | 8B | 16 GB | +16 GB | ~0.5 GB | **~38–45 GB** | 1 |
| Llama-3.1-8B | 8B | 16 GB | +16 GB | ~0.5 GB | **~38–45 GB** | 1 |
| Qwen3-14B | ~14.8B | ~30 GB | +30 GB | ~0.8 GB | **~65–72 GB** | 1 (tight) or 2 (safe margin) |
| Qwen3-32B | ~32.8B | ~66 GB | +66 GB | ~1.2 GB | **~140 GB+** | 2–4 |
| Llama-3.3-70B | 70B | 140 GB | +140 GB | ~2 GB | **~290 GB+** | 4–8 (out of this experiment's scope) |

*vLLM keeps its own copy of frozen weights for fast generation unless
weight-sharing is configured; treated as a second full copy here as the
conservative assumption. Activation/KV-cache headroom (a few GB at seq≈1024,
num_generations=8) is folded into the "peak VRAM" column, not broken out
separately — precise to within the ±2x this exercise budgets for.

### 2.2 FLOPs on paper, Qwen3-8B, one GRPO step

- N = 8×10⁹. Training FLOPs/token ≈ 6N = 4.8×10¹⁰.
- Assume 64 prompts/step × num_generations=8 = 512 completions/step, average
  sequence (prompt+completion) ≈ 350 tokens, of which ≈200 are completion
  tokens actually optimized against.
- Tokens processed for the policy-gradient pass ≈ 512 × 200 = 102,400.
- **Policy update (fwd+bwd):** 102,400 × 4.8×10¹⁰ ≈ **4.9 PFLOPs/step**.
- **Rollout generation** (forward-only, 2N/token): 102,400 × 1.6×10¹⁰ ≈
  **1.6 PFLOPs/step**.
- **Reference-model forward pass for KL** (forward-only, same token count):
  another ≈ **1.6 PFLOPs/step**.
- **Total ≈ 8.1 PFLOPs/step** (order-of-magnitude; LoRA reduces the
  backward-pass FLOPs for frozen-weight gradients but the 6N heuristic on
  full N is used here as the conservative upper bound).

At assumed sustained utilization (A100-80 bf16 dense peak ≈312 TFLOPS,
H100-80 bf16 dense peak ≈495 TFLOPS, both at ~35% MFU for this
mixed rollout/train workload → ≈109 TFLOPS and ≈173 TFLOPS effective
respectively):

- **A100-80: ≈74 s/step.** 500 steps ≈ **10.3 GPU-hours**; 1000 steps ≈ 20.6.
- **H100-80: ≈47 s/step.** 500 steps ≈ **6.6 GPU-hours**; 1000 steps ≈ 13.1.

Qwen3-14B scales ≈1.85× (14.8B/8B) on both FLOPs and VRAM; same step-count
run costs ≈1.85× the GPU-hours above, on ≥1 additional GPU for safety
margin (so real GPU-hour cost is closer to 3–4× the 8B number once GPU-count
is multiplied in, not just FLOPs-scaled).

### 2.3 Primary + fallback

**Primary: Qwen3-8B-Instruct**, LoRA r=16, single A100-80 or H100-80.
Justification:
- **Trainability:** fits one GPU with real margin (§2.1), which matters
  because the eval-batch OOM hard rule (§4.3) needs headroom independent of
  the training config, not a config already at the memory ceiling.
- **Iteration speed:** ~5-10x cheaper per step than the 14B/32B options
  (§2.2), which matters directly for statistical power (§1.2) — the budget
  buys more GRPO steps and more eval repeats at 8B than at 14B for the same
  spend, and the predicted effect is small enough that steps/repeats matter
  more than raw model size here.
- **License:** Qwen3 dense checkpoints are released Apache-2.0 (verify the
  exact license file on the specific checkpoint before treating this as
  settled — not independently re-confirmed this session), which is the
  license profile the commercial path (§3) needs; Llama-3.x's custom
  Community License carries its own attribution/MAU-threshold terms that
  complicate the commercial-artifact story unnecessarily when Qwen3 is
  available at the same size with a cleaner license.

**[PILOT-RESULT] — which model family looks most collapsed:** EXP-004 (the
cascade pilot, `EXPERIMENT_LOG.md`) tested 11 models but **none of Qwen3 or
Llama-3.x** — its roster was frontier + API models (claude/codex/api
families), not these candidates. **This is a gap this plan does not close:**
before committing GPU budget to either arm, run the validated cascade
instrument (haiku + LABEL_PROMPT v2, per the EXP-002 decision) as a Phase 0
baseline on Qwen3-8B, Qwen3-14B, and Llama-3.1-8B specifically, N=4-8 runs
each, ~30 turns, API/local-serving only (no training GPU needed for this
step). Slot to fill in before Phase 1 launches:
`[PILOT-RESULT: model family X shows the lowest depth-to-degradation /
highest cross-model path overlap, i.e. most collapsed, hence most headroom
for the GRPO arm to move]`. If the pilot shows Qwen3-8B is already
near-ceiling (little collapse, little headroom), fall back to Qwen3-14B on
2×A100-80/H100-80 before spending on 8B GRPO.

**Fallback: Qwen3-14B-Instruct**, same LoRA setup, 2 GPUs for margin. Use if
the Phase 0 pilot shows 8B has insufficient headroom, or if 8B GRPO training
in Phase 1 shows the reward stack can't move the model's own cascade score at
all within the smoke-test window (a training-capacity problem distinct from
the headroom question).

---

## 3. Data plan

Source: `~/Experiments/good-humored-data/` per `DATA.md` /
`references/corpus-sources.md`. **887,639** commercial-safe (SocialGrep
Reddit jokes, CC-BY-4.0) + **310,151** research-only (Fraser unspecified-license
+ taivop×3 explicit no-commercial-use) + **25** CC0 memorized-joke templates
(Jentzsch & Kersting 2023, the mode-collapse reference set).

### 3.1 SFT curation recipe (commercial-safe 887K)

**Explicit warning the recipe must design against:** the commercial-safe
bucket is entirely SocialGrep — i.e., 100% of it carries a Reddit `score`
field. Sorting by score to pick "the best jokes" *is* the popularity-bias
confound `humor-rl/SKILL.md` §"Data" already flags, and naively doing it here
risks recreating a version of the exact failure this whole project is
designed against (an SFT set that's just the most-upvoted templates is one
step from becoming its own mini "top-25" collapse target). Recipe:

1. **Junk/quality filter first, score-blind:** length bounds 5-120 tokens
   (matching `comprehensibility_reward`'s own heuristic bounds so the SFT
   distribution isn't already fighting the GRPO-stage reward), drop
   already-`[removed]`/`[deleted]` (should be zero post-MANIFEST dedup, but
   re-check), drop non-English (SocialGrep is nominally r/jokes but not
   100% English-filtered upstream).
2. **Dedup against everything downstream that must stay clean:**
   - exact + n-gram (threshold 0.35, same threshold `corpus_novelty_penalty`
     uses) against `chatgpt-25-templates.jsonl` — an SFT set that includes
     near-copies of the exact 25 memorized templates would let the
     GRPO-stage novelty penalty get *trivially* satisfied by a model that
     memorized the SFT set instead of the internet, which defeats the
     entire anti-collapse mechanism before training even starts.
   - exact + n-gram against any held-out human-eval jokes reserved for final
     scoring (§5) and against the cascade benchmark's own topic-seed prompts
     (`docs/BENCHMARK.md`), so post-training cascade/eval numbers reflect
     generalization, not memorization of the SFT set.
   - a cheap contamination check against MMLU/GPQA text (near-zero expected
     overlap given the domains, but stated explicitly as a gate: **do not
     trust a positive transfer result without having run this check** — the
     single most common way a "transfer" result turns out to be spurious is
     eval-train leakage, not a genuine capability shift).
3. **Size target: ~50,000-60,000 examples post-filter.** Rationale: HumorGen's
   own bar is described only qualitatively ("diverse and well-curated"), not
   by an exact N; this project's target sits in the same order of magnitude
   as typical small-domain instruction-SFT sets referenced in the
   `trl-fine-tuning` skill's own worked examples, and is small enough that
   the entire SFT stage stays cheap (§7) relative to the GRPO arm, which is
   where the compute budget should go.
4. **No score-based ranking as the primary filter.** If score is used at all,
   use it only as a weak tie-break after the above filters, and report the
   score distribution of what was kept vs. discarded so a reviewer can check
   whether "curation" silently became "popularity filtering" — this is an
   auditable claim, not an assumption.

### 3.2 License firewall — two explicit paths

**COMMERCIAL path:** SFT + GRPO training data drawn *only* from
`commercial-safe/jokes.jsonl` (CC-BY-4.0). The resulting checkpoint is a
candidate commercial artifact (with CC-BY attribution obligations tracked
separately, not resolved in this doc).

**PAPER/RESEARCH path:** SFT + GRPO training data may additionally draw from
`research-only/jokes.jsonl` (310K, unspecified/no-commercial-use), producing
a larger, likely stronger training set and a more comparable-to-literature
result. **The resulting checkpoint is not a commercial artifact and must not
be productized, merged into, or used to distill the commercial-path model.**

**Open compliance question, flagged rather than resolved:** at *evaluation*
time (not training), the anti-plagiarism novelty check (§5) is proposed to
score generations against the *full* ~1.2M-joke reference corpus (both
license buckets + the 25 templates) as a read-only similarity lookup, never
incorporated into model weights. Whether "read-only comparison against
NC-licensed text" is compliance-clean for a commercial artifact's eval
pipeline, as distinct from training on it, is **a question for counsel**, not
asserted as settled here — consistent with the standing IP-ownership caution
already on file in `docs/HARNESS-NOTES.md` §5.

**Recommendation:** run both paths as parallel SFT+GRPO arms (4 arms total:
commercial-SFT, commercial-GRPO, paper-SFT, paper-GRPO) so the paper can
report the stronger research-path number while the product path stays
provably clean, and the marginal value of the extra 310K rows becomes a
measured quantity (does research-only data actually move MMLU/GPQA transfer,
or just humor-side scores?) rather than an assumption. This roughly doubles
the SFT-stage cost and, if both are carried through GRPO, doubles the GRPO
arm cost too (§7) — the cost section below prices a cheaper commercial-only
MVP as the recommended first cut.

---

## 4. Training design

### 4.1 SFT baseline (the HumorGen bar)

TRL `SFTTrainer` + LoRA (r=16, alpha=32, targeting
q/k/v/o/gate/up/down_proj — per `trl-fine-tuning` skill's standard config),
on the curated set from §3. This *is* the HumorGen negative-result bar:
`negative-results.md` §3's central finding is that GRPO/DPO show "negligible
and inconsistent differences" against a well-curated SFT baseline for
funniness — so the SFT checkpoint is not a throwaway warm-start, it is a
result in its own right and must be evaluated on the full frozen suite (§5)
exactly like the GRPO checkpoint, not skipped.

### 4.2 GRPO with the humor reward stack

`make_humor_reward_stack()` from
`.claude/skills/humor-rl/examples/humor_reward_functions.py`, weights as
specified in `SKILL.md`'s reward-stack table:

| Term | Weight (from skill) |
|---|---:|
| judge / human preference | primary — **numeric weight unspecified in the skill; see flag below** |
| `corpus_novelty_penalty` | −1.5 |
| `self_repetition_penalty` (window=2000) | −1.0 |
| `intra_group_diversity_reward` | +0.5 |
| `comprehensibility_reward` | +0.3 |

**Honest gap in the skill, flagged rather than silently resolved:** the
skill states the judge term is "primary" but gives it no numeric weight,
only ordering it first in the returned list. If a raw LLM-judge score (e.g.
1-10) is passed unnormalized alongside terms scaled to ±0.3-1.5, the judge
term will numerically dominate by 5-10x and the novelty/diversity terms
become close to inert — precisely the LLM-judge-reward-hacking failure mode
`negative-results.md` §1 documents twice. **Design decision for this plan:**
normalize the judge score to [0,1] (min-max or the judge's own stated scale)
before applying a weight in the 1.0-2.0 range, so it stays dominant-but-not-
overwhelming relative to the ±1.5-max novelty penalty. This is exactly the
kind of thing the skill's own step 4 ("read the actual completions early... a
few hundred steps in") is designed to catch, and the smoke test (§4.3) must
verify it before the full run, not discover it at the end.

- **`num_generations`:** 8. Skill guidance (`grpo-rl-training`) starts at 8
  and raises toward 16 if reward_std trends toward zero; going in at 8 rather
  than 16 keeps the primary model on a single GPU (§2.1) and leaves raising
  it as the documented response to a specific observed failure, not a
  default.
- **KL settings:** deliberately *higher* than `grpo-rl-training`'s generic
  default (`kl_loss_coef` ≈0.001). Recommend **0.01-0.02**. Justification:
  `negative-results.md` §6 gives a specific mechanistic reason GRPO collapses
  — reverse-KL mode-seeking reinforces the first high-reward trajectory
  found — which is exactly the failure mode this experiment's humor arm is
  most exposed to (a single joke or judge-favored register winning and
  crowding out the group). A stronger KL anchor plus the diversity/novelty
  reward terms is belt-and-suspenders against the same collapse mechanism,
  not redundant with it — one term shapes the reward landscape, the other
  constrains how far the policy can move per step regardless of reward
  landscape.
- **Compute-matched control arm's GRPO config is otherwise IDENTICAL** (same
  `num_generations`, same KL coefficient, same step count, same LR) — see §5.

### 4.3 Smoke-test gates (mandatory, per CLAUDE.md hard rules)

1. **Forward pass:** single batch through the LoRA-wrapped model, verify
   output shape and no NaN/Inf in logits.
2. **Backward/gradient check:** verify LoRA adapter params receive nonzero
   gradient after one step; verify frozen base params' `.grad` stays `None`
   (confirms LoRA is actually isolating the trainable surface, not
   accidentally full-fine-tuning).
3. **Reward function unit tests, each in isolation** (per
   `grpo-rl-training`'s own "test rewards independently" golden rule):
   `corpus_novelty_penalty` returns near-max-negative on an exact copy of a
   `chatgpt-25-templates.jsonl` entry; `self_repetition_penalty` fires on a
   literal repeat within its own rolling window; `intra_group_diversity_reward`
   returns 0 for singleton groups (per its own documented no-signal case);
   `comprehensibility_reward` penalizes an empty string and a 300-token wall
   of text equally on the length term.
4. **Judge-score normalization check** (§4.2's flagged gap): confirm the
   composed stack's five per-completion reward magnitudes are within the
   same order of magnitude on a canned batch, not judge-dominated 10:1.
5. **EVAL BATCH SIZE, separately from training** (CLAUDE.md hard rule,
   verbatim: *"Smoke test must include EVAL batch size, not just training —
   eval can OOM even if training fits"*). The training config (num_generations=8,
   LoRA, single GPU, §2.1) and the `lm-evaluation-harness` eval config
   (vLLM backend, `--batch_size auto` or a fixed eval batch, potentially
   different sequence-length distribution than training) hit the GPU
   differently. Run a smoke MMLU/GPQA pass (a handful of subjects, not the
   full suite) at the actual planned eval batch size on the actual
   checkpoint format (merged LoRA or adapter-on-base) **before** launching
   the full frozen-suite eval in §5 — this is a distinct gate from the
   training-side smoke test, not covered by it.

---

## 5. Measurement

### 5.1 Frozen eval suite — locked before any training starts

- **MMLU** (5-shot, full 57-subject) — via `lm-evaluation-harness`, vLLM
  backend (per that skill's own numbers: ~15-20 min on a 7B model vs. ~2h on
  HF backend — use vLLM given the number of checkpoints needing eval, §7).
- **GPQA-diamond** (0-shot) — smaller, harder, more reasoning-loaded, less
  likely to be near-saturated the way some MMLU subjects are; §1.2 already
  flags its noise floor at this model scale.
- **Taste-adjacent, justified individually rather than bundled in:**
  - **EQ-Bench (creative-writing track):** directly on-thesis (tests
    whether humor training moves general creative/prose quality, not just
    joke output — the "does gaining humor gain taste" framing in `STATE.md`).
    **Not included in the primary/frozen success-criteria set** (§6) for two
    concrete reasons: (a) it is not one of the `lm-evaluation-harness` 60+
    tasks, so it is a separate integration cost with its own harness, not a
    `--tasks` flag; (b) its scoring is itself LLM-judge-based, carrying the
    exact judge-reliability risk `negative-results.md` documents twice for
    this project's own reward design — using a judge-scored benchmark to
    validate a judge-reward-trained model is close to measuring agreement
    with a cousin of the training signal, not independent capability.
    Recommend running it as a **secondary/exploratory** metric, reported
    alongside but never load-bearing for the continue/kill decision.
  - **BBH (BIG-Bench Hard) subset, as a lower-risk alternative/addition:**
    already inside `lm-evaluation-harness`, multiple-choice or exact-match
    scored (no judge), and includes lateral/compositional reasoning tasks
    that plausibly correlate with the divergent-thinking construct
    `kellner2017creative` (`psychology.md` §1) ties to humor production
    independent of crystallized knowledge. Cheaper to add than EQ-Bench and
    judge-free; include if eval budget (§7) allows.

### 5.2 Compute-matched control arm

Same GRPO config (§4.2: identical num_generations, KL coefficient, step
count, LR, LoRA rank) on the **same base SFT checkpoint**, trained on
non-humor data instead. Two options, ordered by how cleanly they isolate the
"humor" variable:

1. **Preferred — matched reward architecture, different content domain:**
   reuse the exact same non-judge reward terms (`self_repetition_penalty`,
   `intra_group_diversity_reward`, `comprehensibility_reward` — these are
   content-agnostic n-gram/structural heuristics and work unmodified on any
   text), swap the judge for a generic "story/response quality" judge, and
   swap `corpus_novelty_penalty`'s reference corpus for a plagiarism corpus
   of a different domain (e.g. published short-story openings, or the
   `trl-lib/tldr` summarization prompts already referenced in the
   `trl-fine-tuning` skill, treated as a creative-continuation task). This
   isolates "training with this specific reward-stack architecture" from
   "training on humor content specifically" — the tightest possible control
   given the constraint that architecture and step count must match.
2. **Fallback — cheaper, less clean:** GRPO on `trl-lib/tldr` with a plain
   length+quality reward (no diversity/novelty terms at all), same step
   count. Answers a blunter question — "does generic RL training on
   *anything* move MMLU/GPQA a little" — and is useful as a coarse sanity
   check even though it doesn't control for the reward-architecture
   confound as tightly.

Use option 1 as the primary control; option 2 only if building the matched
reward-architecture control turns out to cost more engineering time than the
budget in §7 tolerates.

### 5.3 Cascade benchmark before/after (the humor-side measure)

Uses the validated instrument from `EXPERIMENT_LOG.md`'s Instrument decision
(haiku + LABEL_PROMPT v2, raw-label scoring primary, semantic reported
alongside, never primary). **Prerequisite this plan does not skip:** since
neither Qwen3 nor Llama-3.x appeared in EXP-004's roster, a fresh baseline
cascade run on the chosen primary/fallback model is required as this
experiment's own Phase 0 (§2.3's `[PILOT-RESULT]` slot), not inherited from
EXP-004's numbers. Same depth (30 turns) and N (4-8 runs) as EXP-004 for
direct comparability; before-training and after-training runs on the same
model, same rejector, same prompt version.

### 5.4 Novelty check against the memorized corpus (hard rule, not optional)

Per `CLAUDE.md`: *"any generation eval MUST include a novelty check against a
memorized-joke corpus."* Score post-GRPO generations' max n-gram Jaccard
similarity (threshold 0.35, matching `corpus_novelty_penalty`'s own
threshold) against the full ~1.2M-joke reference set (both license buckets +
the 25 templates — read-only lookup, see §3.2's compliance flag), and report
the fraction of generations exceeding threshold. **This must be reported
alongside, not after, any cascade or MMLU/GPQA improvement claim** — a
cascade-depth improvement produced by the model learning to recite a
*different* set of memorized jokes than before is not evidence of anything
this experiment is trying to measure, and the whole "transfer" claim would be
sitting on a bad premise if the humor side didn't actually get more diverse
or more novel.

---

## 6. Success / kill criteria

**Continue (fund a fuller/paper-grade follow-up) if ALL of:**
- MMLU delta (humor-GRPO minus compute-matched control) ≥ **+0.005** (half
  the registered prediction — a lower bar than the point prediction,
  because at this budget even confirming the *sign* and *rough magnitude*
  is the finding worth funding further, not exact replication of §1.1's
  number).
- The delta direction is consistent across at least 2 of the 3 primary
  signals (MMLU, GPQA-diamond, BBH-if-run) — a single-metric win with the
  other two flat or negative is not "continue," it's "investigate the
  single metric before concluding anything."
- The novelty check (§5.4) shows the humor-GRPO arm's generations are **not**
  more similar to the memorized corpus than the SFT-baseline's generations
  were — i.e., any cascade/quality improvement isn't bought by regurgitating
  a different fixed set.
- The SFT-only disproof check (§1.3) did **not** already clear the
  continue-bar on its own at a fraction of the cost (if it did, the finding
  is "SFT transfers," a different and cheaper story, and GRPO's marginal
  contribution needs to be reframed, not silently folded into the same
  claim).

**Kill (emit a `[DEAD-END]` block) if ANY of:**
- MMLU and GPQA deltas are both null or negative relative to the
  compute-matched control — i.e., any apparent humor-arm advantage
  disappears once compared to "generic extra RL training," not just to a
  fixed pre-training baseline.
- The apparent effect only shows up when compared to the *frozen
  pre-training baseline* and vanishes against the compute-matched control —
  this specifically means the finding was "more training helps a little,"
  not "humor training transfers," and should be reported as that, not spun.
- The humor-GRPO arm's own cascade/novelty metrics show mode collapse
  (reward_std → 0, or the novelty check flags heavy memorized-corpus
  similarity) — if the "humor training" didn't produce measurably funnier
  or more diverse output by this project's own instruments, there is no
  humor-side result to attribute any transfer to, and the run doesn't count
  as a valid test of the hypothesis at all (report as inconclusive/invalid,
  not as a negative result on the transfer question itself).
- Phase 0's `[PILOT-RESULT]` shows the chosen primary model already has
  near-ceiling cascade performance (minimal collapse, minimal headroom) —
  switch to the fallback model *before* spending GRPO budget rather than
  discovering this after (this is precisely why §2.3 gates on the pilot
  slot rather than picking blind).

---

## 7. Cost

GPU-hours, back-of-envelope from §2.2 (A100-80 figures used as the
conservative/upper-bound case; H100-80 would run ~35-40% cheaper in
GPU-hours at the same step counts).

| Item | Estimate (GPU-hours) |
|---|---:|
| SFT baseline, commercial-path (LoRA, 8B, ~60K examples) | ~4 |
| SFT baseline, paper-path (commercial + research-only, larger set) | ~4-6 |
| GRPO humor arm, commercial-path (500-1000 steps) | ~10-21 |
| GRPO humor arm, paper-path (500-1000 steps) | ~10-21 |
| Compute-matched control arm, commercial-path (matched steps) | ~10-21 |
| Compute-matched control arm, paper-path (matched steps) | ~10-21 |
| `lm-evaluation-harness` MMLU+GPQA, vLLM backend, ~8 checkpoints × ~0.75h | ~6 |
| Cascade eval serving (vLLM inference only, before/after, both models) | ~2-4 |
| Debug/OOM-retry/iteration buffer (~20-30% of the above) | ~15-20 |
| **Total, full two-path (commercial + paper) design** | **≈90-110 GPU-hours** |
| **Total, commercial-path-only MVP (recommended first cut)** | **≈50-55 GPU-hours** |

**Recommendation:** run the commercial-path-only MVP first (skip the
paper/research-only arm) to get a cheap, clean read on the sign and rough
magnitude of the effect; only fund the paper-path arms if the MVP clears the
continue-bar (§6) and a stronger/larger-data version is worth the additional
~40-55 GPU-hours to strengthen the eventual paper's number.

### 7.1 Calibration registration commands (copy-paste at run launch, not now)

```bash
# Primary transfer predictions (register when the humor-GRPO arm actually launches)
python3 .claude/scripts/calibration.py add EXP-005-reverse-transfer \
  "GRPO on humor reward stack shifts MMLU accuracy vs compute-matched control" \
  mmlu_acc_delta 0.010

python3 .claude/scripts/calibration.py add EXP-005-reverse-transfer \
  "GRPO on humor reward stack shifts GPQA-diamond accuracy vs compute-matched control" \
  gpqa_diamond_acc_delta 0.005

# Disproof check (register BEFORE the GRPO arm, at SFT-only launch)
python3 .claude/scripts/calibration.py add EXP-005-reverse-transfer \
  "SFT-only on curated humor data shifts MMLU accuracy vs compute-matched SFT control" \
  mmlu_acc_delta_sft_only 0.002

# Humor-side companion metrics (cascade benchmark, same run id)
python3 .claude/scripts/calibration.py add EXP-005-reverse-transfer \
  "Post-GRPO model survives more cascade turns before degrading (pre vs post, same model)" \
  cascade_depth_delta_turns 5

python3 .claude/scripts/calibration.py add EXP-005-reverse-transfer \
  "Post-GRPO model shows higher within-model path divergence across runs (pre vs post)" \
  cascade_path_divergence_delta 0.07

# After the run — close each with the actual measured delta, e.g.:
# python3 .claude/scripts/calibration.py close EXP-005-reverse-transfer \
#   "GRPO on humor reward stack shifts MMLU accuracy vs compute-matched control" <ACTUAL_DELTA>
```

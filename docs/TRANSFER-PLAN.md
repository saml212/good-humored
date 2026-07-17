# Reverse-transfer pre-registration: train on humor, measure general capability

Status: **design, not built. No training runs from this document.** Owner: Sam
Larson. Written 2026-07-17; **revised 2026-07-17 (later pass)** against the
refreshed `docs/FINDINGS.md` (11-model cascade, dual-tier memorization,
meta-exclusion robustness), the shipped reward-stack state in `env/rewards.py`
+ `env/semantic_novelty.py` (windowed n-gram novelty now default-on and
closed for padding/dilution; semantic whole-text tier validated at 0.38 but
OFF by default; windowed semantic BLOCKED pending EXP-011), and
`env/banter_env.py` (Track 2 episode environment, new this cycle). Format
follows `EXPERIMENT_LOG.md` house style — numbers before narrative. This is
gap #1 from `CLAUDE.md` → Research Direction: *"nobody has tested whether
training on humor transfers back to general reasoning/taste."* HumorBench
established only the forward direction (STEM-reasoning training → humor
comprehension); the reverse direction is, per
`.claude/skills/humor-rl/SKILL.md` §"Measuring transfer," **untested by
anyone**. That novelty claim is the literature review already on file
(`references/negative-results.md`, `references/psychology.md`,
`docs/BENCHMARK.md` §6) — checklist item 8 (verify novelty) is satisfied by
citation, not re-litigated here.

**What changed in this revision, up front (so a second reader doesn't have to
diff the file):** (1) §2 replaces the open `[PILOT-RESULT]` placeholder with
a full FINDINGS-grounded headroom ranking across deepseek/qwen/glm, and adds
a load-bearing finding this pass surfaced — **none of the three API-profiled
models in FINDINGS are the same checkpoint as any trainable 7–32B open-weight
dense model** — plus the deepseek registry deprecation
(2026-07-24 → `deepseek-v4-flash`) and a secondary Phase-0 candidate
(DeepSeek-R1-Distill-Qwen-14B) with its own caveats. (2) §4 adds a new §4.2.1
specifying the episode format as conversational banter via `BanterEnv`
(previously undefined — the doc only specified the single-turn stack), with
an explicit reward composition, a build gap flagged for pre-launch
engineering, and the in-group swap-pairing scheme the env's docstring left as
a caller responsibility. (3) §5.3/§5.4 sharpen the cascade manipulation-check
into a formal pass/fail gate (citing EXP-007b's lesson precisely) and lock
down windowed-vs-whole-text mode per novelty tier at eval time. (4) §7 adds
an API-call/judge-cost budget line the GPU-hours table didn't capture. (5) a
new §8, **Threats to validity**, consolidates contamination, judge-gaming,
seed variance, the compute-matched-control confound, and the wrapper/access-
path lesson applied to the eval battery itself.

## Pre-Experiment Checklist coverage (CLAUDE.md, all 8 items)

| # | Item | Where |
|---|---|---|
| 1 | State the hypothesis in one sentence | §1 |
| 2 | Predict the metric delta, register with `calibration.py add` | §1, §7.2 |
| 3 | Compute FLOPs/memory/params on paper | §2 |
| 4 | Try to disprove in 5 minutes | §1.3 |
| 5 | Check the literature first | done — this doc is built on `negative-results.md`, `psychology.md`, `humor-rl/SKILL.md`, `BENCHMARK.md` |
| 6 | Design the comparison before the experiment | §5.2 (compute-matched control), §4.1 (SFT bar), §8.4 (confound precision) |
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
- **License:** Qwen3 dense checkpoints (8B/14B/32B, plus 4B/1.7B/0.6B)
  confirmed Apache-2.0 this revision (current HF/Qwen listings; re-verify
  the exact license file on the specific checkpoint at download time as a
  final check, not a fresh investigation), which is the license profile the
  commercial path (§3) needs; Llama-3.x's custom Community License carries
  its own attribution/MAU-threshold terms that complicate the
  commercial-artifact story unnecessarily when Qwen3 is available at the
  same size with a cleaner license.

### 2.4 FINDINGS refresh: what the 11-model cascade actually says about headroom, and why it doesn't just transfer

EXP-004 (the cascade pilot) tested 11 models but **none of Qwen3, Llama-3.x,
or any other directly-trainable 7–32B open-weight dense checkpoint** — its
open-weights lane was three *hosted API* models: `api:deepseek`
(`deepseek-chat`), `api:qwen` (`qwen-plus-2025-07-28`), `api:glm`
(`glm-4.5-air`). `docs/FINDINGS.md` §2.4/§2.3/§4.6 (refreshed 2026-07-17,
dual-tier memorization + meta-exclusion fix wave) gives their profiles in
full:

| model (API-hosted) | degradation depths (N runs) | exact-tier memorization | template-trigram memorization | framing-prefix rate |
|---|---|---:|---:|---:|
| `api:deepseek` | 11, 9, 6, 8 → median **8.5**, tightest range, 4/4 degrade | 0.8% | 3.3% | **53.3%** (2nd-highest in roster) |
| `api:qwen` | 11, 8, **24**, 9 → median 10, noisier (one run to 24), 4/4 degrade | 1.7% | **10.0%** (ties codex:5.4) | 0.0% |
| `api:glm` | 21, 14 (N=2, mixed generation-config protocol — §5 caveat) | 2.4% | 2.4% | 1.2% |

Replicated on 49 additional runs from the temperature-fakeability lanes
(EXP-007/007b/007c, zero additional cost): deepseek 18/18 degrade
(depths 4–16), glm 18/18 degrade (depths 7–24), qwen 13/13 of completed runs
degrade (depths 8–22). All three open-weight families degrade fast and
consistently relative to the Anthropic/OpenAI/xAI lanes — this part of the
signal is robust.

**Ranked headroom prior, reading the table straight:** deepseek shows the
fastest, tightest, most consistent collapse *and* the lowest memorization on
both tiers among the three — the strongest "genuinely collapsing, not just
reciting classics" signal. qwen is second: fast median depth but noisier
(N=1 outlier to turn 24), and its template-trigram memorization (10.0%) is
3× deepseek's, meaning the "open-weights barely memorize" story is
tier-dependent and weaker for qwen specifically (FINDINGS §2.3). glm is
weakest evidence of the three (N=2, mixed protocol, slowest degradation).
One caveat that cuts against reading deepseek's number at face value: its
53.3% framing-prefix rate (jokes prefaced with "Alright, how about this:
...") is the second-highest in the whole 12-model roster, and FINDINGS §2.3
shows framing prefixes structurally deflate the exact-match tier independent
of true novelty (sonnet's 74.2% prefix rate is the documented mechanism) —
deepseek's low *exact*-tier number is partly a delivery-format artifact, not
purely novelty. The template-trigram tier is more robust to this dilution
and still shows deepseek lowest (3.3%), so the ranking survives, just with
the confidence appropriately tempered.

**The finding this revision adds, not previously stated in this document:
none of these three ranked profiles is directly actionable for model
selection**, because none of the three hosted API checkpoints is the same
model as a trainable 7–32B open-weight dense checkpoint from the same lab:

- `deepseek-chat` is DeepSeek's flagship hosted alias (DeepSeek-V3-class
  MoE, on the order of 600B+ total parameters) — there is no small dense
  "DeepSeek" chat checkpoint at 7–32B to train. The closest same-lineage
  open release at the right size, **DeepSeek-R1-Distill-Qwen-14B/32B**
  (confirmed via current HF listings: MIT license, dense, 14B/32B), is
  architecturally **Qwen-2.5 underneath**, R1-reasoning-distilled — a
  materially different object (see the R1-distill caveat below), not a
  proxy for `deepseek-chat`'s own conversational cascade behavior.
- `qwen-plus-2025-07-28` is Alibaba's larger hosted variant, not
  `Qwen3-8B`/`14B`/`32B` specifically — but it is at least the **same lab
  and family**, and Qwen3's dense checkpoints (8B/14B/32B) are confirmed
  Apache-2.0, open-weight, and available on Hugging Face/ModelScope. Qwen
  is the only one of the three families with a clean, same-lineage,
  right-size, cleanly-licensed answer.
- `glm-4.5-air` is confirmed **106B total / 12B active** MoE, MIT-licensed,
  open-weight — but "12B active" is not the number that governs feasibility
  here: this plan's own VRAM arithmetic (§2.1) prices frozen weights plus a
  colocated vLLM rollout copy, i.e. roughly **2× the full 106B in bf16
  (~212 GB)** before LoRA/activation overhead, which is out of scope for
  the single/dual-GPU LoRA budget this plan is sized to (comparable to why
  §2.1 already puts Llama-3.3-70B "out of this experiment's scope"). GLM is
  excluded on **compute-feasibility grounds**, not on cascade-signal
  grounds — its FINDINGS profile is the weakest of the three anyway (N=2),
  so nothing informative is being discarded.

**Consequence for model selection, decisive:** the FINDINGS ranking
(deepseek > qwen > glm) is a **same-family directional prior about which
lab's models tend to collapse fast**, not a measurement on any checkpoint
this plan can actually train. Qwen remains the only family where the prior
and the trainable checkpoint are the same lineage, which is why it stays
primary — but its own FINDINGS proxy (`qwen-plus`) showed only *moderate,
noisier* headroom than deepseek's, so Phase 0 must not assume `qwen-plus`'s
signal transfers cleanly down to the much smaller Qwen3-8B/14B dense
checkpoints; it is a hypothesis Phase 0 tests, not an inherited result.

**[PILOT-RESULT slot — unfilled, this plan does not close it]:** before
committing GPU budget to either arm, run the validated cascade instrument
(haiku + LABEL_PROMPT **v2** — see the instrument-version note below, this is
deliberate, not an oversight) as a Phase 0 baseline, N=4–8 runs each, ~30
turns, API/local-serving only (no training GPU needed for this step), on:

1. **Qwen3-8B-Instruct** and **Qwen3-14B-Instruct** (primary/fallback
   candidates, §2.3).
2. **DeepSeek-R1-Distill-Qwen-14B** as a secondary probe *only* — included
   because it is the closest available same-lineage-adjacent open dense
   checkpoint at the right size to the family FINDINGS ranked highest for
   headroom, not because it is expected to reproduce `deepseek-chat`'s
   profile. Two caveats to weigh before promoting it past Phase 0:
   (a) `negative-results.md` §3 (HumorGen) reports reasoning-augmented
   "Think" variants scored *lower* on judged funniness than non-reasoning
   variants — a reasoning-distilled base is not an obviously safe choice
   for a humor-reward target; (b) R1-distill models default to emitting a
   `<think>...</think>` chain before the answer, which complicates the SFT
   curation recipe (§3.1's length-bound filters) and several reward terms'
   length heuristics (`comprehensibility_reward`, `self_repetition_penalty`'s
   window) unless thinking mode is disabled or the chain is stripped before
   scoring — an engineering decision that must be made and smoke-tested
   (§4.3) before launch, not discovered mid-run.

Slot to fill in before Phase 1 launches: `[PILOT-RESULT: model X shows the
lowest depth-to-degradation / highest cross-run path overlap among the
candidates actually tested, i.e. most collapsed, hence most headroom for the
GRPO arm to move]`. If the pilot shows Qwen3-8B is already near-ceiling
(little collapse, little headroom), fall back to Qwen3-14B on 2×A100-80/
H100-80 before spending on 8B GRPO; if R1-Distill-Qwen-14B shows meaningfully
more headroom AND clears the reasoning-chain engineering gate, it is eligible
to replace Qwen3-14B as fallback, not promoted to primary without a second,
independent Phase-0 confirmation given caveat (a) above.

**Instrument-version note (do not silently upgrade):** Phase 0 must run on
the **v2** free-vocabulary labeler, the one every FINDINGS number is
authoritative under — **not v3** (field-invalidated, 42.6% catch-all
collapse on wild turns, EXP-008 addendum) and **not v4/EXP-010** (registered
and passing its own fixture/probe bars as of this writing, but its Result is
still `(pending)` in `EXPERIMENT_LOG.md` as of the last entry — do not adopt
an unfinished instrument for a gating decision). If v4 finishes and clears
its bars before Phase 0 launches, re-run this section's decision using v4 and
say so explicitly; do not mix v2-labeled Phase-0 numbers with v4-labeled
ones in the same headroom comparison.

**Registry deprecation, scoped precisely so it isn't over-applied:**
`deepseek-chat` deprecates **2026-07-24** and the registry successor is
`deepseek-v4-flash` (`STATE.md`, `EXPERIMENT_LOG.md` EXP-007). This does
**not** block Phase 0 or Phase 1 as designed above — neither arm calls
`api:deepseek` at all (Phase 0 tests open-weight checkpoints via local/API
serving of the checkpoints themselves, not the `deepseek-chat` hosted
endpoint). It matters only if this plan is ever extended to *refresh* the
FINDINGS deepseek profile via a fresh API call after 2026-07-24: that call
will hit `deepseek-v4-flash`, a **different model** than the one the table
above characterizes, and any such refresh must be reported as a new,
separate instrument reading, not silently merged with the pre-07-24
`deepseek-chat` numbers.

### 2.5 Fallback

**Fallback: Qwen3-14B-Instruct**, same LoRA setup, 2 GPUs for margin. Use if
the Phase 0 pilot shows 8B has insufficient headroom, or if 8B GRPO training
in Phase 1 shows the reward stack can't move the model's own cascade score at
all within the smoke-test window (a training-capacity problem distinct from
the headroom question). DeepSeek-R1-Distill-Qwen-14B is a secondary fallback
per §2.4, contingent on clearing its own reasoning-chain engineering gate.

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

**Code-vs-skill note (this pass verified `env/rewards.py` directly rather
than trusting the skill's table):** `.claude/skills/humor-rl/SKILL.md`'s
reward table still lists the judge term as "primary" with no numeric weight
— but the actual shipped code, `env/rewards.py`'s `RewardConfig` /
`reward_stack()`, has already closed that gap: `judge_weight` defaults to
**1.0** and the class enforces a `[0, 1]` contract on the judge callable's
return value (raises if violated), so the dominance failure the skill only
warned about is now a constructor-level guard, not a design decision this
plan has to make from scratch. Current shipped defaults, `reward_stack()`
composed from `make_humor_reward_stack()`-equivalent config:

| Term | Weight (shipped default, `env/rewards.py`) | This plan's setting |
|---|---:|---|
| `judge_preference` | 1.0, `[0,1]`-contract enforced | **1.0** — use as shipped |
| `corpus_novelty_penalty` (n-gram, `windowed=True` default) | −1.5 | **−1.5, windowed=True (default)** — padding/dilution CLOSED for this tier as of today (max-over-sliding-windows, `env/rewards.py` "PADDING/DILUTION -- CLOSED") |
| `self_repetition_penalty` (window=2000, threshold=0.5) | −1.0 | **−1.0** — use as shipped |
| `intra_group_diversity_reward` | +0.5 | **+0.5** — use as shipped |
| `comprehensibility_reward` | +0.3 | **+0.3** — use as shipped |
| `semantic_novelty_penalty` (`env/semantic_novelty.py`) | **0.0 (OFF by default)** | **TURN ON at −1.5, `reference="templates"`, threshold 0.38, `windowed=False`** — see flag below |

**Decision this plan makes that the shipped default does not:** turn
`semantic_novelty_weight` on. EXP-009 validated the whole-text tier at
threshold 0.38 (paraphrase detection 1.000/1.000/0.500/0.810 across edit
depths 1–4, FPR≤0.05) — it exists specifically to catch the 2-word-reskin
evasion the n-gram tier is documented to miss, which is exactly the failure
mode a GRPO policy under a novelty penalty is incentivized to find. Training
with it off would leave a known, already-fixed hole open for no reason.
**Do not additionally set `windowed=True` on the semantic tier.** That mode
is real code (closes the same padding/dilution exploit for the embedding
tier) but is **BLOCKED pending EXP-011**: it is calibrated against the same
`DEFAULT_THRESHOLD=0.38` the whole-text tier uses, and a preliminary check
found padded positives scoring far above that threshold in windowed mode
(severity ~0.96 vs. 0.0 whole-text) — i.e. the 0.38 cutoff is
mis-calibrated for windowed semantic scoring specifically, and using it
un-re-swept would either over- or under-penalize in an unverified direction.
Whole-text semantic + windowed n-gram (both validated at their shipped
thresholds) is the correct combination for this plan; windowed semantic is a
post-EXP-011 upgrade, not available today.

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

### 4.2.1 Episode format: conversational banter, not isolated one-liners

**This is new in this revision — the prior draft specified the reward stack
but never fixed what a training *episode* looks like.** Decisive choice: the
humor-GRPO arm's episodes are conversational banter via `env/banter_env.py`'s
`BanterEnv`, not single-turn joke completions. Rationale: STATE.md's own
"does gaining humor gain taste" framing and gap #2 (multi-turn conversational
humor) in `CLAUDE.md`'s Research Direction both point at conversational
register as the more interesting construct than isolated one-liners, and
`BanterEnv` is the only validated Track 2 instrument this repo has (EXP-005).
Concretely:

- **Episode:** `BanterEnv(partner_complete=..., judge=<haiku>, seed_topic=k)`,
  `max_turns` defaults to `len(SCENARIO_OPENERS) == 8` — 8 conversational
  turns per episode, one of 8 fixed, deterministic-order scenario openers
  (`benchmark/banter.py`'s `SCENARIO_OPENERS`) per episode.
- **Per-turn reward, composed** (this composition is specified here but
  **does not exist in code yet** — flagged as a pre-launch build item, not
  assumed done):
  - `ablation_weight=1.0 × context_ablation_score()`'s **normalized**
    delta (÷9.0, per `BanterEnv`'s own "MAGNITUDE FIX" default) — replaces
    `judge_preference` from the main stack for this episode format, not
    stacked on top of it. Rationale: the ablation delta *is* this episode
    format's humor/context judge signal (EXP-005-validated haiku judge
    under `benchmark.banter.JUDGE_PROMPT`); applying `judge_preference` on
    top would double-count the same judge's opinion under two different
    contracts (`env/rewards.py`'s `[0,1]` scorer vs. `banter_env`'s raw
    `judge_complete(prompt) -> str` completion callable — the two are not
    interchangeable per that module's own docstring, and composing both
    is redundant, not additive, signal).
  - `callback_weight=0.5 × callback bonus` (`BanterEnv` default, unchanged)
    — flat bonus when `detect_callback()` fires against the partner's
    turns so far.
  - **`corpus_novelty_penalty` (−1.5, windowed=True) and
    `self_repetition_penalty` (−1.0) applied to the policy's reply text,
    exactly as in the main stack.** This is the mandatory addition
    `BanterEnv` does not ship with — as built, `BanterEnv`'s only two
    reward terms are callback + ablation delta, with **no novelty-vs-
    memorized-corpus check at all**, which on its own would violate
    `CLAUDE.md`'s hard rule (*"any generation eval MUST include a novelty
    check against a memorized-joke corpus"*). Both penalty terms are pure
    functions of completion text and batch-of-completions respectively, so
    they compose onto a conversational reply exactly as they do onto a
    single-turn joke — no format-specific rewrite needed, just wiring.
  - **This is also the literal "design guard" EXP-005's own verdict
    registers, now enforced rather than left as prose:** *"a Track 2 reward
    should not use [ablation] delta alone to push past pleasantry-humor —
    pair with callback bonus (already in banter_env) and novelty terms"*
    (`EXPERIMENT_LOG.md`, EXP-005 Verdict). The caveat behind that guard:
    generic on-topic pleasantry already earns ~2/3 of a genuinely
    contextual reply's ablation delta (mean_delta 4.00 vs. 6.17 separation
    from canned) — ablation delta alone cannot reliably distinguish
    "genuinely funny and in-context" from "polite and on-topic," which is
    exactly why callback + novelty terms are load-bearing here, not
    decorative.
  - `intra_group_diversity_reward` (+0.5) and `comprehensibility_reward`
    (+0.3) apply per-turn unchanged (both are pure functions of the
    completion / GRPO-group batch, agnostic to conversational structure).
  - `semantic_novelty_penalty` (−1.5, whole-text, threshold 0.38) applies
    per-turn unchanged, same rationale as §4.2.
- **Swap-pairing for the ablation term, the one piece `BanterEnv` explicitly
  leaves to the caller:** `step()` takes `swapped_context` as a required
  argument for the ablation term to fire at all (contributes 0.0 if
  omitted). This plan's answer: source it **in-group** — for a GRPO group of
  `num_generations=8` rollouts started from the *same* `seed_topic`, at
  turn `t`, rollout `i`'s `swapped_context` is rollout `j≠i`'s
  conversation-so-far at the same turn index, rotating `j` across the group
  the same way `benchmark.banter.swap_partner`'s fixed rotation does across
  a pilot's fixed episode set. This requires the 8 rollouts in a group to be
  stepped in lockstep (turn-synchronized), which is a real scheduling
  constraint on the training loop, not just a data-plumbing detail — flag it
  to whoever wires the TRL `GRPOTrainer` callback, since naive
  per-rollout-independent generation would not preserve turn alignment.
- **Judge-call budget, distinct from GPU-hours (see §7):** each turn makes
  2 haiku calls (`true_score` + `swapped_score`, `context_ablation_score`'s
  design). Per GRPO step: 8 generations × 8 turns × 2 = **128 haiku calls**.
  Over a 500–1000 step run: 64,000–128,000 haiku calls — cheap in dollars at
  haiku pricing, but a real latency dependency: GRPO step throughput is
  bounded by external API round-trips, not just GPU compute, unless judge
  calls are batched/parallelized aggressively. Budget wall-clock accordingly
  (§7), not just GPU-hours.

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
   `chatgpt-25-templates.jsonl` entry (both whole-text AND a padded/diluted
   variant — confirm windowed mode still catches it, this is the specific
   regression `env/tests/test_rewards.py`'s windowed class already locks in,
   re-run it here rather than trusting it by citation); `self_repetition_penalty`
   fires on a literal repeat within its own rolling window;
   `intra_group_diversity_reward` returns 0 for singleton groups (per its own
   documented no-signal case); `comprehensibility_reward` penalizes an empty
   string and a 300-token wall of text equally on the length term;
   `semantic_novelty_penalty` fires at ≥0.38 on a hand-written paraphrase of
   one of the 25 templates (whole-text mode only — do not test or enable
   `windowed=True` on this term, §4.2).
4. **Judge-score normalization check** (§4.2's flagged gap, now
   constructor-enforced in code but verify it anyway): confirm the composed
   stack's per-completion reward magnitudes are within the same order of
   magnitude on a canned batch, not judge-dominated 10:1.
5. **Banter-composition smoke test (new, §4.2.1's build gap):** on a canned
   2-rollout group, verify (a) `swapped_context` in-group sourcing produces a
   real, non-empty string at every turn (no silent 0.0-fallback from a
   missing swap partner); (b) `corpus_novelty_penalty`/
   `self_repetition_penalty`/`semantic_novelty_penalty` fire correctly when
   applied to `BanterEnv` reply text (these terms were built and tested
   against single-turn completions — confirm nothing about conversational
   framing, e.g. a reply prefixed with the partner's name or scenario
   context, breaks their tokenization/normalization); (c) the ablation-delta
   term is NOT double-counted against `judge_preference` (assert the main
   stack's `judge_preference` term is absent from the banter-episode
   composition, per §4.2.1's "replaces, not stacks" decision).
6. **If DeepSeek-R1-Distill-Qwen-14B is in play (§2.4):** confirm the
   thinking-chain handling decision (strip `<think>...</think>` before
   reward scoring, or disable thinking mode entirely) is implemented and
   verified on a canned generation *before* any reward function sees R1-
   distill output — an un-stripped `<think>` block would blow every length-
   based heuristic in `comprehensibility_reward` and `self_repetition_penalty`'s
   window sizing.
7. **EVAL BATCH SIZE, separately from training** (CLAUDE.md hard rule,
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
non-humor data instead. **Precisely what "compute-matched" is doing here,
stated explicitly rather than left implicit:** this arm's entire job is to
absorb "any RL training moves MMLU/GPQA a little, regardless of content" —
extra gradient steps, extra exposure to a reward-shaped generation
distribution, KL-anchored policy drift, all of it — so that whatever's left
in the humor-GRPO-minus-control delta is attributable to *humor content
specifically*, not to "we did more RL." Without this arm, a positive
humor-GRPO delta against the frozen pre-training baseline is uninterpretable
— it could be the humor reward stack, or it could be nothing more specific
than "training", and §6's kill criteria are built around forcing exactly
that comparison, not letting a frozen-baseline-only delta stand in for it.

**Format-matching consequence of §4.2.1 (must carry through, not just the
reward terms):** since the humor-GRPO arm's episodes are conversational
banter (`BanterEnv`, 8 turns, §4.2.1), the control arm's episodes must also
be conversational and the same length — a single-turn control paired against
a multi-turn humor arm would confound "conversational RL" with "humor RL,"
reintroducing exactly the mismatch a compute-matched arm exists to remove.
Two options, ordered by how cleanly they isolate the "humor" variable:

1. **Preferred — matched reward architecture AND matched episode format,
   different content domain:** a `BanterEnv`-shaped 8-turn conversational
   episode on a neutral topic domain (e.g. `SCENARIO_OPENERS`-style everyday
   scenarios rewritten for a non-comedic register — practical-advice or
   plain-narrative continuation rather than banter — or the `trl-lib/tldr`
   summarization prompts reframed as a multi-turn clarifying-question
   dialogue), reusing the exact same content-agnostic reward terms
   (`self_repetition_penalty`, `intra_group_diversity_reward`,
   `comprehensibility_reward`, `corpus_novelty_penalty` pointed at a
   plagiarism corpus from the *same neutral domain* rather than the joke
   corpus), and a callback-analog + ablation-delta-analog pair scored by a
   generic "helpful/coherent response" judge in place of the banter judge
   (same haiku model, same `[normalized delta]` mechanics, different
   rubric). **Why this task is neutral:** it has no built-in relationship to
   incongruity, timing, or comedic register — a "helpful clarifying
   response" judge rewards coherence and relevance, not surprise-then-
   resolution, so any MMLU/GPQA movement this arm produces cannot be
   attributed to anything humor-specific by construction, only to the
   shared RL architecture and step count. This is the tightest possible
   control given the constraint that architecture, episode format, and step
   count must all match — and it is more engineering than the prior draft
   of this plan scoped (a new callback-analog needs picking, not just a
   swapped judge rubric), so budget real build time for it (§7), not assume
   it is a config change.
2. **Fallback — cheaper, less clean, single-turn:** GRPO on `trl-lib/tldr`
   with a plain length+quality reward (no diversity/novelty terms, no
   conversational structure at all), same step count. Answers a blunter
   question — "does generic RL training on *anything* move MMLU/GPQA a
   little" — useful as a coarse sanity check, but now confounds two things
   at once (non-humor content AND single-turn format vs. the humor arm's
   multi-turn format), so a positive result under this fallback is weaker
   evidence than under option 1 and should be reported with that caveat
   attached, not silently treated as equivalent.

Use option 1 as the primary control; option 2 only if building the
matched-format, matched-reward-architecture control turns out to cost more
engineering time than the budget in §7 tolerates — and if option 2 is used,
say so plainly in the eventual write-up rather than letting a reader assume
the tighter design ran.

### 5.3 Cascade benchmark before/after — registered as a GATE, not just a measurement

Uses the validated instrument from `EXPERIMENT_LOG.md`'s Instrument decision
(haiku + LABEL_PROMPT **v2**, raw-label scoring primary, semantic reported
alongside, never primary — see §2.4's instrument-version note: not v3
(field-invalidated) or v4/EXP-010 (result still pending as of this writing)).
**Prerequisite this plan does not skip:** since neither Qwen3 nor
DeepSeek-R1-Distill-Qwen appeared in EXP-004's roster, a fresh baseline
cascade run on the chosen primary/fallback model is required as this
experiment's own Phase 0 (§2.4's `[PILOT-RESULT]` slot), not inherited from
EXP-004's numbers. Same depth (30 turns) and N (4-8 runs) as EXP-004 for
direct comparability; before-training and after-training runs on the same
model, same rejector, same prompt version.

**The manipulation-check gate (new framing this revision adds, applying
EXP-007b's lesson precisely):** EXP-007b ran a full 3-lane temperature
ablation on qwen before discovering the endpoint silently ignored the
temperature parameter — the resulting `[LEARN]` block is explicit: *"Before
any sampling-parameter ablation, probe K identical requests at the extreme
setting... register the check as a gate. Verify the manipulation reached the
model before believing any delta."* Applied here: **before interpreting any
MMLU/GPQA/BBH delta from the humor-GRPO arm, verify the humor-side
manipulation actually reached the model** — i.e., cascade depth-to-
degradation (post-GRPO) must show a real, directional improvement over
cascade depth-to-degradation (pre-GRPO, same model, same rejector, same
prompt version). Register this as a **binary pass/fail gate, checked before
any reasoning-battery number is trusted**, not folded silently into the
general kill-criteria list in §6:

- **PASS:** post-training cascade depth improves over pre-training (or the
  model reaches "survived" status more often) — the humor-side manipulation
  reached the model. Proceed to interpret MMLU/GPQA/BBH deltas.
- **FAIL:** post-training cascade depth is flat or worse than pre-training.
  **Do not interpret the reasoning-battery deltas at all** — favorable or
  not, they answer a question this run never actually asked, exactly as
  EXP-007b's qwen deltas (distinct_2 +0.006, nowhere near the +0.30
  prediction) turned out to measure a manipulation that never reached the
  endpoint, not a real absence of temperature-fakeability. A FAIL here is
  reported as **inconclusive/invalid**, matching §6's kill-criteria
  language, not as a negative transfer result.

### 5.4 Novelty check against the memorized corpus (hard rule, not optional)

Per `CLAUDE.md`: *"any generation eval MUST include a novelty check against a
memorized-joke corpus."* Score post-GRPO generations against the full
~1.2M-joke reference set (both license buckets + the 25 templates — read-only
lookup, see §3.2's compliance flag), reporting **both** tiers side by side
per the dual-tier lesson `docs/FINDINGS.md` §2.3/§4.6 surfaced this cycle
(a single tier understated grok's and deepseek's true memorization reliance
by different amounts):

- **Exact-tier (n-gram, threshold 0.35, `corpus_novelty_penalty`'s own
  threshold), windowed mode ON** — this is the mode to use at eval time,
  explicitly, because the padding/dilution exploit is closed for this tier
  (max-over-sliding-windows, default `windowed=True`) and whole-text-only
  scoring would silently under-report memorization on any padded/diluted
  generation.
- **Template-trigram tier** (25 Jentzsch & Kersting templates, ≥0.5 trigram
  Jaccard counts a hit) reported alongside — FINDINGS' own fix-wave finding
  is that this tier catches memorization the exact tier misses on models
  with a framing-prefix habit (e.g. `api:deepseek`'s 53.3% framing-prefix
  rate deflates its exact-tier number specifically; §2.4). Check the
  post-GRPO checkpoint's own framing-prefix rate as part of this report —
  if GRPO training happens to increase framing-prefix usage (a policy could
  learn "Alright, here's one:" as a cheap way to blunt the exact-match
  penalty without becoming less memorization-reliant), the exact tier alone
  would show a spurious "novelty improved" reading that the trigram tier
  would catch.
- **Semantic tier, whole-text mode only (threshold 0.38)** — do **not**
  score novelty checks in windowed semantic mode; that mode is blocked
  pending EXP-011 (§4.2). Whole-text semantic + windowed exact-tier is the
  correct eval-time combination, matching the training-time reward
  composition (§4.2) so the number reported here is measuring the same
  thing the reward penalized during training, not a stricter or looser
  standard applied only at the end.

**This must be reported alongside, not after, any cascade or MMLU/GPQA
improvement claim** — a cascade-depth improvement produced by the model
learning to recite a *different* set of memorized jokes than before (or by
learning a framing-prefix habit that dodges the exact tier specifically) is
not evidence of anything this experiment is trying to measure, and the whole
"transfer" claim would be sitting on a bad premise if the humor side didn't
actually get more diverse or more novel.

---

## 6. Success / kill criteria

**Gate 0, checked BEFORE any of the below (§5.3's manipulation check):** did
the humor-GRPO arm's post-training cascade depth actually improve over its
own pre-training baseline? If **FAIL**, stop here — report
inconclusive/invalid, do not proceed to interpret MMLU/GPQA/BBH deltas one
way or the other, per EXP-007b's precedent.

**Continue (fund a fuller/paper-grade follow-up) if Gate 0 PASSES and ALL of:**
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
  discovering this after (this is precisely why §2.4 gates on the pilot
  slot rather than picking blind).

---

## 7. Cost

GPU-hours, back-of-envelope from §2.2 (A100-80 figures used as the
conservative/upper-bound case; H100-80 would run ~35-40% cheaper in
GPU-hours at the same step counts). **Note the §4.2.1 episode-format change:**
`BanterEnv`'s 8-turn episodes are shorter per-turn than the single-turn
stack's completions but multiply generation calls by turn count — the
per-step FLOPs estimate in §2.2 was computed for single-turn completions and
should be treated as a lower bound for the banter format until re-derived
per-turn; not re-derived here (flagged, not silently reused).

| Item | Estimate (GPU-hours) |
|---|---:|
| SFT baseline, commercial-path (LoRA, 8B, ~60K examples) | ~4 |
| SFT baseline, paper-path (commercial + research-only, larger set) | ~4-6 |
| GRPO humor arm, commercial-path (500-1000 steps, banter episodes) | ~10-21 |
| GRPO humor arm, paper-path (500-1000 steps, banter episodes) | ~10-21 |
| Compute-matched control arm, commercial-path (matched steps, matched banter format) | ~10-21 |
| Compute-matched control arm, paper-path (matched steps, matched banter format) | ~10-21 |
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

### 7.1 API / judge-call budget — distinct from GPU-hours, new this revision

The GPU-hours table above prices compute; it does not price the **external
judge-API dependency** the §4.2.1 banter format introduces (the single-turn
stack's `judge_preference` term has the same dependency at a much lower
call rate, priced here too for completeness):

| Item | Calls | Approx. $ (haiku pricing, pennies/call) | Wall-clock note |
|---|---:|---:|---|
| Banter ablation judge (humor-GRPO arm): 8 gens × 8 turns × 2 calls/turn × 500-1000 steps | 32,000-64,000 | low single-digit $ to ~\$10s | **latency-bound, not $-bound** — GRPO step throughput waits on haiku round-trips unless batched/parallelized; budget wall-clock headroom in Phase 1's schedule, not just GPU-hours |
| Compute-matched control's analogous judge (§5.2, matched call rate) | 32,000-64,000 | comparable | same caveat |
| Phase 0 cascade pilot (haiku rejector, per model: 30 turns × 2 calls × N runs) | ~240-480/model × up to 4 models (Qwen3-8B/14B, R1-distill-14B, +fallback) | low single-digit $ | one-time, before GPU spend |
| Post-training cascade re-run (same models, after GRPO) | same order as above | low single-digit $ | one-time per checkpoint |

**Why this line matters enough to add:** none of the GPU-hour estimates
above account for wall-clock time spent blocked on an external API judge —
at 32,000+ calls per arm, even well-parallelized haiku round-trips (order
100s of ms each) add up to real serial time if the training loop isn't
structured to overlap judge scoring with the next rollout's generation. This
is an engineering/scheduling risk to resolve during the §4.2.1 build, not
purely a cost line.

### 7.2 Calibration registration commands (copy-paste at run launch, not now)

**Run-id note (bug this revision fixes):** the prior draft used
`EXP-005-reverse-transfer` as the calibration run id — but `EXP-005` is
already assigned in `EXPERIMENT_LOG.md` (banter judge validation), so that
id collided with a real, unrelated experiment. This experiment has not yet
been logged and has no `EXP-NNN` number of its own (`EXPERIMENT_LOG.md`
assigns numbers sequentially at registration, not in a planning doc) — the
commands below use a non-colliding placeholder id, `reverse-transfer-v1`;
whoever registers this run should swap in the real `EXP-NNN` once
`EXPERIMENT_LOG.md` assigns one (next available at time of writing: check
the log's highest entry, currently through EXP-010 plus addenda — likely
**EXP-011 is already reserved** for the windowed-semantic-novelty re-sweep
per `env/semantic_novelty.py`'s own docstring, so this run is probably
**EXP-012** or later; confirm against the log at registration time, don't
assume).

```bash
# Primary transfer predictions (register when the humor-GRPO arm actually launches)
python3 .claude/scripts/calibration.py add reverse-transfer-v1 \
  "GRPO on humor reward stack shifts MMLU accuracy vs compute-matched control" \
  mmlu_acc_delta 0.010

python3 .claude/scripts/calibration.py add reverse-transfer-v1 \
  "GRPO on humor reward stack shifts GPQA-diamond accuracy vs compute-matched control" \
  gpqa_diamond_acc_delta 0.005

# Disproof check (register BEFORE the GRPO arm, at SFT-only launch)
python3 .claude/scripts/calibration.py add reverse-transfer-v1 \
  "SFT-only on curated humor data shifts MMLU accuracy vs compute-matched SFT control" \
  mmlu_acc_delta_sft_only 0.002

# Humor-side companion metrics (cascade benchmark, same run id)
python3 .claude/scripts/calibration.py add reverse-transfer-v1 \
  "Post-GRPO model survives more cascade turns before degrading (pre vs post, same model)" \
  cascade_depth_delta_turns 5

python3 .claude/scripts/calibration.py add reverse-transfer-v1 \
  "Post-GRPO model shows higher within-model path divergence across runs (pre vs post)" \
  cascade_path_divergence_delta 0.07

# After the run — close each with the actual measured delta, e.g.:
# python3 .claude/scripts/calibration.py close reverse-transfer-v1 \
#   "GRPO on humor reward stack shifts MMLU accuracy vs compute-matched control" <ACTUAL_DELTA>
```

---

## 8. Threats to validity

New section this revision — the prior draft scattered these caveats across
§1–§7; consolidated here so a reviewer can check the full list in one place
before signing off on a GPU spend. None of these are hypothetical: each
either already bit this project once (cited) or is a documented failure mode
in the literature this plan draws on.

### 8.1 Contamination between humor training data and eval sets

Already gated in §3.1 (dedup against `chatgpt-25-templates.jsonl`, held-out
human-eval jokes, cascade topic-seed prompts, and "a cheap contamination
check against MMLU/GPQA text" — stated there as a gate, not resolved).
Restated here because it is the single most common way a "transfer" result
turns out to be spurious: **eval-train leakage looks exactly like a genuine
capability shift until someone checks.** SocialGrep/Fraser/taivop Reddit
corpora are large and heterogeneous enough that near-zero overlap with
MMLU/GPQA is the expectation, not a certainty — run the check, report the
overlap rate found (even if zero), and treat "we didn't check" as
disqualifying for any positive result, per CLAUDE.md's "verify before
claiming."

### 8.2 Judge gaming

Two independent judge dependencies in this design, each individually
documented as hackable: `judge_preference` (main stack, `[0,1]`-contract
enforced but still an LLM judge, and `negative-results.md` §1's GRPO+GPT-4.1
collapse is exactly this failure mode) and the banter ablation judge
(§4.2.1 — `banter_env.py`'s own docstring names the specific exploit,
sprinkling context-echoing words to inflate the delta without genuine
in-context responsiveness). Mitigations already specified, not new: the
novelty/diversity/comprehensibility terms exist precisely to make a
judge-only win insufficient (§4.2); the smoke-test's judge-normalization
check (§4.3 item 4) catches magnitude dominance before launch; and
`humor-rl/SKILL.md`'s own procedure ("read the actual completions early... a
few hundred steps in") is a live-run check this plan inherits, not replaces.
**Residual risk not fully closed by any of the above:** both judges are the
same underlying model (haiku) under different rubrics — a systematic haiku
bias (e.g., rewarding a particular register or verbosity level) would move
both the training reward and any correlated eval (EQ-Bench, §5.1) the same
direction, which is exactly why EQ-Bench is explicitly *not* in the
frozen/primary success-criteria set (§5.1) and why the primary reasoning
battery (MMLU/GPQA/BBH) is judge-free multiple-choice/exact-match scoring —
the one part of this design immune to a shared-judge bias by construction.

### 8.3 Seed variance at small N

§1.2 already flags MMLU stderr (~0.004-0.005 at 7-8B) and GPQA-diamond's
noise floor (~198 questions, one flip ≈0.5pp — within the registered
predicted delta). Restated as a standing threat, not a one-time caveat: any
single-seed run at this scale risks reporting noise as signal in either
direction. Mitigation already specified in §6 (bootstrap CI per-subject, not
a bare point estimate; require the delta direction to hold across ≥2 of 3
primary signals). Not mitigated, and worth stating plainly: **this plan does
not budget multiple training seeds per arm** (§7's GPU-hours are single-seed
per arm) — a genuinely paper-grade version would re-run each arm at ≥2-3
seeds before trusting a point estimate, and this plan's pilot-grade budget
explicitly does not do that (matching this project's own house style of
calling pilot-grade numbers pilot-grade, per `docs/FINDINGS.md`'s repeated
"not paper numbers" framing).

### 8.4 The "any RL moves MMLU a little" confound

Stated precisely, per the task's own framing: **this is exactly and only
the compute-matched control arm's job (§5.2).** Without it, a positive
humor-GRPO delta against the frozen pre-training baseline is confounded with
"more gradient steps, more KL-anchored drift, more reward-shaped generation
exposure" — none of which requires humor content specifically. §5.2's
matched-format (banter, §4.2.1), matched-architecture, matched-step-count
control is designed to absorb exactly this effect, and §6's kill criteria
are built around the control comparison, not the frozen-baseline comparison,
for this exact reason ("the apparent effect only shows up... vanishes
against the compute-matched control... should be reported as that, not
spun"). The confound is not fully eliminated by a single control arm at
single-seed N (§8.3 compounds here), but the design's entire causal-inference
weight rests on this one comparison — if the control arm is skipped, cut
down to option 2 (§5.2's single-turn fallback), or under-resourced relative
to the humor arm, this entire plan's central claim becomes unfalsifiable
against the "just more RL" explanation.

### 8.5 The wrapper/access-path confound, applied to the eval battery

`docs/FINDINGS.md` §5's most consequential 2026-07-17 finding was that the
cascade benchmark's CLI-wrapper lanes (claude/codex) differ from native-API
lanes in ways that contaminate family-level claims — verified in the
project's own transcripts (haiku spending 25/30 turns in CLI-assistant
persona; wrapper-specific opening-topic leakage) — and that this confound is
"bounded, not eliminated." **The lesson applied to this plan, not previously
stated here:** every checkpoint in this experiment (frozen base, SFT
baseline, humor-GRPO, compute-matched control — 4+ checkpoints across two
license paths, §3.2) must be evaluated on MMLU/GPQA/BBH through **the exact
same access path** — same `lm-evaluation-harness` backend (vLLM, per §5.1),
same checkpoint-loading convention (merged-LoRA-into-base or
adapter-on-base, picked once and applied to every checkpoint, not mixed),
same batch size and precision. Mixing access paths across checkpoints (e.g.
evaluating the frozen base via a hosted API and the trained checkpoints via
local vLLM) would reintroduce the identical class of confound FINDINGS §5
had to bound after the fact for the cascade benchmark — cheaper to prevent
by fixing the eval harness config once, before Phase 1, than to discover and
bound after the numbers are in.

### 8.6 Summary: the single biggest threat in this plan's own judgment

Ranked above the rest, and specific to this plan rather than a restatement
of a generic RL-eval concern: **the model-selection headroom prior (§2.4) is
a same-family analogy, not a measurement on the actual training
checkpoint.** FINDINGS' cascade data ranks `deepseek-chat`/`qwen-plus`/
`glm-4.5-air` — three hosted flagship API models, none of them the 7-32B
open-weight dense checkpoint this plan will actually train. Qwen is the only
family where a clean, same-lineage, right-size checkpoint exists at all, and
even there, `qwen-plus`'s own FINDINGS profile is the noisier, weaker-signal
one of the three (one run to turn 24 out of 4; template-trigram
memorization 3x deepseek's). It is entirely possible that Qwen3-8B/14B, once
actually measured in Phase 0, shows too little cascade headroom for the
humor-side manipulation check (§5.3's Gate 0) to ever pass — at which point
this entire design, however carefully specified, cannot test the reverse-
transfer hypothesis at all, regardless of GRPO config, reward stack, or
control-arm rigor. This is precisely why §2.4 gates Phase 1 on a Phase 0
pilot rather than picking blind, and why that gate is written as a genuine
stop condition (§6, Gate 0) rather than a formality — but it means the
riskiest point of failure in this whole plan is upstream of any RL training
decision, in a cheap, non-GPU pilot this document cannot pre-answer. Budget
attention accordingly: don't let Phase 0 become a rubber stamp because the
expensive design work happened downstream of it.

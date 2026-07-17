# Code and Models

The question this file answers for every entry: **"can we build on this
Monday, or is it a paper with a GitHub link attached?"** License, last
commit, and whether weights/data actually ship are recorded for everything
found. Cite keys refer to `papers.bib` where a paper exists alongside the
code.

---

## Ready to build on Monday

### DARLING (Meta) — diversity + quality joint RL objective
`li2025darling` · [github.com/facebookresearch/darling](https://github.com/facebookresearch/darling)
- **License:** MIT. **Last push:** May 2026. **Stars:** 61.
- **What's actually there:** full training pipeline (not just scripts to
  derive from) — a learned partition-function diversity classifier combined
  with a quality reward, optimized online, built on **verl**. Includes
  training scripts for both math and **creative writing** tasks, a GRPO
  baseline for comparison, and evaluation via NoveltyBench and EQ-Bench v3.
- **Gap:** no released model weights or bundled dataset — bring your own
  data/compute.
- **Why it's the top pick:** this is the only diversity-preserving RL method
  in the whole search explicitly validated on creative writing (the closest
  existing domain to humor), from a team with the infrastructure to have run
  it at scale. The "partition classifier" idea — detect and penalize
  repeated joke structures within a GRPO group — maps directly onto the
  humor mode-collapse problem.

### RAGEN / StarPO — multi-turn trajectory-level RL
`wang2025ragen` · [github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN)
- **License:** MIT. **Last push:** today, as of this compilation (July
  2026). **Stars:** 2,750. **Forks:** 227. **Open issues:** 29 (healthy
  activity, not abandonment).
- **What's actually there:** StarPO optimizes whole multi-turn trajectories
  (observations + reasoning + actions + environment feedback) as single
  training units, rather than per-turn. Identifies the "Echo Trap" failure
  mode (reward-variance cliffs, gradient spikes in agent RL). StarPO-S adds
  trajectory filtering, critic incorporation, gradient stabilization. A
  "RAGEN-2" update (~March 2026) added SNR-adaptive gradient filtering and
  reasoning-collapse diagnostics, including detection of "template
  collapse" (reasoning that becomes input-agnostic) — notably adjacent to
  the humor mode-collapse concern. 10 built-in environments, real
  `train.py`, documented conda setup, test suite.
- **Gap:** no released HuggingFace model weights.
- **Why it matters:** the best-maintained multi-turn RL codebase found for
  gap #2 (conversational humor/banter/callbacks). The "Echo Trap" is a
  concrete, named failure mode worth watching for in multi-turn humor RL —
  an agent that loops the same joke-setup pattern because it once got
  rewarded.

### verl — base RL training framework
[github.com/verl-project/verl](https://github.com/verl-project/verl)
- **License:** Apache-2.0. **Last push:** today, as of this compilation.
  **Stars:** 22,506 — the most active piece of infrastructure found in this
  entire search.
- **Why it matters:** DARLING, verl-agent/GiGPO, and much of the current
  RLVR ecosystem are built on top of this. If humor RL is GRPO-based, this
  is the credible base to fork/extend rather than building a training loop
  from scratch — DARLING's diversity code (a verl extension) should drop in
  relatively directly.

### verl-agent / GiGPO — step-level multi-turn credit assignment
`feng2025gigpo` · [github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)
- **License:** Apache-2.0. **Last push:** June 2026. **Stars:** 2,122.
- **What's actually there:** two-level credit assignment — episode-level
  "macro" advantages from grouped trajectories, plus step-level "micro"
  advantages via anchor-state grouping (finds repeated environment states
  across trajectories, compares actions taken from them) — no auxiliary
  critic model needed. >12% gain on ALFWorld, >9% on WebShop over vanilla
  GRPO. Confirmed environments (ALFWorld, WebShop, Search-R1, Sokoban, Gym
  Cards, AppWorld), direct run commands per environment/algorithm, LoRA and
  dynamic-sampling variants, **released HuggingFace weights + W&B training
  logs** — one of the most completely-documented repos found in this whole
  search.
- **Why it matters:** relevant if banter/callback humor RL needs step-level
  (not just trajectory-level) credit — e.g., crediting the specific turn
  that set up a callback which paid off three turns later.

### Agent Lightning (Microsoft Research)
[github.com/microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)
- **License:** MIT. **Last push:** today. **Stars:** 17,393. **Open
  issues:** 152 (healthy sign of real usage).
- **What's actually there:** decouples agent *execution* from RL *training*
  — an existing agent built with LangChain, OpenAI Agents SDK, AutoGen, or
  from scratch can be RL-trained with near-zero code changes, via an MDP
  formulation and a hierarchical algorithm (LightningRL). Extensive docs at
  [microsoft.github.io/agent-lightning](https://microsoft.github.io/agent-lightning/latest/),
  including a "Train the First Agent" walkthrough.
- **Why it matters:** if the humor generation pipeline is already built as
  an agent (e.g., a multi-step banter loop with memory/tool use), this is
  the lowest-friction way to bolt RL training on without a rewrite.

### SynthesizeMe — personalized reward modeling via persona induction
`ryan2025synthesizeme` · [github.com/SALT-NLP/SynthesizeMe](https://github.com/SALT-NLP/SynthesizeMe)
- **License:** MIT. **Last push:** June 2025. 37 commits.
- **What's actually there:** the best-documented item in this whole
  corpus — pip-installable, README with install + an end-to-end example
  notebook (`examples/prism.ipynb`), a `PrismDataset` loader for the public
  PRISM dataset. No demographic metadata needed: generates a CoT explanation
  for why a user preferred response A over B, synthesizes a persona
  description, filters to the most informative prior interactions.
- **Why it matters:** a concrete, no-demographic recipe for a per-user humor
  reward model from a handful of that user's past ratings — matches the
  project's stated need for personalized reward modeling for
  audience-dependence, without requiring an audience-attribute taxonomy.

### Personalized RewardBench
`ma2026personalizedrewardbench` · [github.com/Martin-qyma/Personalized-RewardBench](https://github.com/Martin-qyma/Personalized-RewardBench)
- **License:** repo has no license file (unspecified); the HF dataset
  [`QiyaoMa/Personalized-RewardBench`](https://huggingface.co/datasets/QiyaoMa/Personalized-RewardBench)
  is Apache-2.0. **Last push:** April 2026. **Stars:** 13.
- **What's actually there:** confirmed-loading Parquet dataset, 1K–10K rows,
  three category splits (Art & Entertainment; Lifestyle & Personal
  Development; Society & Culture), fields for
  question/profile/rubric_aspects/narrative/chosen/rejected/category. The
  Art & Entertainment split plausibly includes humor-adjacent content, but
  humor as a specifically labeled rubric aspect was **not confirmed** —
  check before assuming it's a ready-made humor personalization eval.
- **Why it matters:** best current RMs peak at only 75.94% accuracy on this
  benchmark, and it's been validated as predictive of downstream Best-of-N
  and PPO performance — a template for a "does this reward model actually
  detect personalized humor preference" eval, which is exactly gap #3.

### PAD (ICLR 2025) — humor as a confirmed personalization axis
`chen2025pad` · [github.com/zjuruizhechen/PAD](https://github.com/zjuruizhechen/PAD)
- **License:** the repo itself has no license file (GitHub API returns
  `license: null`); the released 8B Llama-3-based checkpoint on HuggingFace
  ([`RuizheChen/PAD`](https://huggingface.co/RuizheChen/PAD)) is tagged MIT
  on its model card. **Last push:** March 2025 — stalled but functional,
  not actively developed further.
- **What's actually there:** training-free, decoding-time personalization —
  a personalized reward model scores the base model's top-K token
  predictions at each generation step conditioned on a stated preference,
  combined with the base LM's likelihood. Inference (`collect_model_outs.py`)
  and evaluation (`measure_reward.py`) scripts are real and documented; the
  "training module" is marked "coming soon" in the README and **never
  shipped**.
- **CONFIRMED via full-paper fetch:** the paper explicitly states *"we
  initially focus on alignment [on] the three pre-defined dimensions:
  'harmless', 'helpful', and 'humor'"* (Table 2/Figure 2, P-Soups +
  HelpSteer2 datasets). Humor is a seen/training dimension; the paper
  generalizes to unseen dimensions (expert, informative, creative). Uses HF
  model `mohameddhiab/humor-no-humor` as an automatic humor classifier for
  evaluation.
- **Why it matters:** the strongest peer-reviewed precedent (ICLR 2025) for
  treating humor as one axis of a personalized, token-level reward — useful
  even though the training code itself never materialized.

### OpenRLHF
[github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- **License:** Apache-2.0. **Last push:** July 14, 2026. **Stars:** 9,807.
- Ray + vLLM based PPO/GRPO/DAPO/REINFORCE++ framework. Version ~0.10
  (~April 2026) reportedly added multi-turn VLM RL support, extending
  existing multi-turn agent RL and custom reward function support. A viable
  alternative base to verl if the project prefers its API.

---

## Usable, but with real friction — verify before committing

### DPH-RL — diversity-collapse mitigation via divergence choice
`li2025dphrl` · [github.com/seamoke/DPH-RL](https://github.com/seamoke/DPH-RL)
- **License:** Apache-2.0. **Last push:** February 2026. **Stars:** 20.
- Small but complete: mass-covering f-divergence "rehearsal" penalty against
  the initial policy, to keep broad solution coverage during RLVR. Matches
  its paper title exactly — worth reading alongside DARLING as an
  alternative anti-collapse mechanism.

### DRA-GRPO — reward-side diversity via submodular mutual information
[github.com/xiwenc1/DRA-GRPO](https://github.com/xiwenc1/DRA-GRPO)
- **License:** MIT. **Last push:** January 2026. **Stars:** 24. **Open
  issues:** 1. Small single-paper research repo — verify it runs end-to-end
  before relying on it, but the mechanism (downweight redundant completions
  in a GRPO group via SMI, upweight diverse ones) is conceptually the
  closest published thing to "reward-level diversity bonus for GRPO," which
  is exactly the shape of intervention humor-RL needs.

### GEM — SFT-stage diversity preservation
`li2025gem` · [github.com/liziniu/GEM](https://github.com/liziniu/GEM)
- **License:** unspecified (no LICENSE file detected via GitHub API). **Last
  push:** May 2025. **Stars:** 58.
- Real training scripts (`bash scripts/train_gem_ultrafeedback.sh`, a Triton
  loss variant `--loss gem_triton`). Reframes SFT as reverse-KL + entropy
  regularization instead of plain cross-entropy — relevant for the
  cold-start SFT phase, before RL even starts, to stop joke memorization
  early. Confirm license terms before production use given the gap.

### LMRL Gym — multi-turn RL benchmark
`abdulhai2025lmrlgym` · [github.com/abdulhaim/LMRL-Gym](https://github.com/abdulhaim/LMRL-Gym)
- **License:** MIT. **Stars:** 116. **Open issues:** 7.
- **Real friction:** built on **JAX**, requires a separate dependency clone
  (JaxSEQ). **Last actual code push: July 2024** — roughly two years stale
  as of this compilation, in a fast-moving ecosystem. README gives concrete
  install/run commands (separate CPU/GPU/TPU conda paths), so it's not a
  stub, but the JAX/JaxSEQ dependency chain plus staleness is a real
  integration risk. Offline datasets are hosted separately at
  `rail.eecs.berkeley.edu/datasets/rl-llm-bench-dataset` (direct download,
  not a HF loader). Two of its 8 tasks (Twenty Questions, Car Dealer
  negotiation) are structurally similar to banter/callback humor
  (multi-turn, goal-directed, requires tracking conversational state) —
  worth reading even if not adopted wholesale.

---

## Code exists but is thin, undocumented, or unrunnable as-is

### PersRM-R1
`li2025persrmr1` · [github.com/Jeffrey-Guanqiao/PersRM-R1](https://github.com/Jeffrey-Guanqiao/PersRM-R1)
- No README content, no license, no releases, no visible model weights or
  dataset links, 2 stars, 5 commits. Reads as a bare research-code dump, not
  a documented reproducible release. The *idea* (reasoning-based RM
  inferring user preference from 1–few exemplars, emitting an explicit
  reasoning trace before a scalar score) is worth reusing conceptually; the
  repo itself needs real archaeology before it's runnable.

### Oogiri-Master/Corpus builder
[github.com/CyberAgentAILab/oogiri-dataset-builder](https://github.com/CyberAgentAILab/oogiri-dataset-builder)
- MIT-licensed **code**, but produces no data on its own — it's a scraper
  you must run yourself against third-party Japanese sites, and the
  maintainers explicitly say the MIT license doesn't cover the resulting
  data. See `datasets.md` Tier 2 for the full account.

### HumorTransferLearning (Dad Jokes generalization repo)
`turgeman2026dadjokes` · [github.com/morturr/HumorTransferLearning](https://github.com/morturr/HumorTransferLearning)
- 3 commits, 1 star, README says "code and instructions will be uploaded
  soon." A dataset directory exists but is undocumented. **Not usable
  Monday** — check back for updates before relying on it.

---

## Paper/announcement only — no code found despite real search effort

- **DQO** (arXiv:2509.04784, ICLR 2026) — determinantal-point-process group
  diversity score atop PPO/GRPO. No repo located.
- **Info-GRPO** (OpenReview `d5qElNtXS5`) and **EDGE-GRPO**
  (arXiv:2507.21848) — contrastive-MI and entropy-driven-advantage GRPO
  variants. No repo located for either.
- **"Understanding Diversity Collapse in RLVR via Overtraining"**
  (arXiv:2606.15455) and **"Are We Measuring Strategy or Phrasing?"**
  (arXiv:2606.29985) — analysis papers; no repo located for either.
- **LoTbench** (`huang2025lotbench`) — code/data marked "coming soon."
- **HumorBench, HumorGen, HumorRank, CLoST, IRS** — none of the core
  humor-generation-with-LLMs papers in this corpus have released code, data,
  or model weights as of this compilation. This is the single biggest gap in
  "genuinely usable now" material for the project's core research direction
  — see the README's "biggest gap" note.
- **GFlowNet-for-LLM-diversity code**
  ([github.com/GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning),
  [github.com/Adam-yni/GFlowNets-FineTuning](https://github.com/Adam-yni/GFlowNets-FineTuning))
  — found via search only, **not** independently fetched/audited during
  this research pass. **[UNVERIFIED license/last-commit/runnability]** —
  check directly before relying on either. Conceptually the most principled
  anti-mode-collapse mechanism found (sampling proportional to reward rather
  than reward-maximizing structurally avoids collapse-to-single-best-joke),
  which is why it's listed despite the unverified status.

---

## Models specifically fine-tuned for humor (quick inventory)

| Model | Base | What it is | Usability |
|---|---|---|---|
| `TzJ2006/JokeGPT-Model` | Qwen3-8B | PPO-LoRA-tuned for jokes | Per HumorGen's own eval, ranked 15th of models tested — a small **negative** data point on naive PPO-for-humor alone underperforming general frontier models |
| `mohameddhiab/rate-jokes-bert` | RoBERTa-base | Joke funniness rating | Usable as a cheap reward-model baseline; calibration against human ratings not independently verified |
| `mohameddhiab/humor-no-humor` | — | Binary humor classifier | Used as the automatic humor judge inside PAD (`chen2025pad`) — a real, reused-in-the-literature component |
| `SajilAwale/FunnyModel` | — | Multi-label humorous/offensive/sentiment | Small, not independently verified |
| `ZhenghanYU/CFunModel` | Qwen2.5-7B-Instruct | Chinese humor understanding+generation, trained on "CFunSet" | Larger, more serious effort; not independently verified for license/access terms beyond HF listing |
| `wooozihui/HumorReject-*` | Mistral-7B-Instruct-v0.1 | Humor-as-refusal for jailbreak-proofing | Different objective (safety, not funniness) but a real example of humor used as an RL/preference signal; MIT, active repo at `github.com/wooozihui/HumorReject` |
| `RuizheChen/PAD` | Llama-3 8B | Personalized decoding-time reward model incl. humor dimension | MIT-tagged on model card; see PAD entry above |

**Overall verdict on humor-specific model weights:** thin. Nothing found is
both large and well-validated. The strongest usable-now general-purpose
components are the RL *infrastructure* (DARLING, RAGEN, verl-agent) rather
than any humor-specific checkpoint — expect to train from a general base
model rather than continue from an existing humor-tuned one.

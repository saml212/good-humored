# Negative Results

The single highest-value file in this corpus. Every naive approach to humor
generation and RL-on-creative-tasks that has actually been tried and published
is here, with exact numbers where a primary source gave them. Treat everything
below as **the baseline to beat**, not settled science — several entries
contradict each other in instructive ways (noted inline).

Cite keys refer to `papers.bib`.

---

## 1. Mode collapse onto memorized jokes

### The ChatGPT joke probe — `jentzsch2023chatgpt`

**Jentzsch, S. & Kersting, K., "ChatGPT is fun, but it is not funny! Humor is
still challenging Large Language Models," WASSA 2023.**
[arXiv:2306.04563](https://arxiv.org/abs/2306.04563) ·
[ACL Anthology](https://aclanthology.org/2023.wassa-1.29/) ·
Code: [github.com/DLR-SC/JokeGPT-WASSA23](https://github.com/DLR-SC/JokeGPT-WASSA23)

This is the load-bearing citation for mode collapse, and it holds up exactly
as claimed when checked against the primary text:

- ChatGPT was prompted for a joke ~1,000 times across 10 differently-worded
  prompts, yielding **1,008 total joke instances**.
- Deduplication: exact-duplicate removal → 348 samples; stripping
  filler/formatting → 128 samples; then **manual grouping** of
  semantically-equivalent paraphrases (same underlying pun, reworded) → **25
  top recurring jokes**.
- Those 25 jokes covered **917/1,008 (~91%)** of all output; **909/1,008
  (90.2%)** were identical, word-for-word, to one of the 25. The abstract's
  "over 90% of 1008 generated jokes were the same 25 Jokes" is confirmed
  verbatim, not a rounding.
- The **top 4 jokes alone** accounted for **>50%** of all output (the
  scarecrow "outstanding in his field" joke alone appeared 140 times).
- All 25 top jokes were verifiable as pre-existing, human-authored jokes
  findable via ordinary web search — memorized, not generated.
- Method for "same joke": **not automatic** — string dedup, then
  normalization, then manual grouping of paraphrases sharing a pun structure.

**Why it matters:** this is the exact number the `humor-rl` skill and
`CLAUDE.md` cite. It is real, precise, and reproducible-in-spirit (the code is
public). Any humor-generation eval that doesn't check against a
memorized-joke corpus will report success while reproducing this exact
failure.

### The LessWrong GRPO + GPT-4.1-judge experiment — `agg2025funniestjoke`

**"Generating the Funniest Joke with RL (according to GPT-4.1)," LessWrong,
posted 2025-05-16.**
[lesswrong.com/posts/xMGmibZpPDnawjHXk](https://www.lesswrong.com/posts/xMGmibZpPDnawjHXk/generating-the-funniest-joke-with-rl-according-to-gpt-4-1)

Real experiment, not a paper — GRPO on Qwen3-8B, GPT-4.1 as judge. **Two
distinct collapses**, not one:

1. **Experiment 1** (naive "rate 1-5" reward): collapsed to Qwen "endlessly
   regurgitating classic jokes" — the exact failure mode the skill file
   warns about. Switching to a richer rubric (funniness + originality +
   "boundary-pushing") did not fix it — it **relocated** the exploit: the
   model learned to append irrelevant absurdist "bonus jokes" (e.g., "Bonus:
   The laser was actually a tiny alien spaceship 🚀") that GPT-4.1 scored
   13/20, later 20/20, despite the author explicitly disagreeing with the
   judge's score. This is a directly-observed instance of LLM-judge reward
   hacking, in the open, with reward curves shown.
2. **Experiment 2** (rubric = "your north star should be how funny it is"):
   a *different* degenerate mode — dark "timeshare in hell" jokes, then
   bizarre non-sequiturs ("I told my cat... he's a wizard... unionizing the
   mice") that, per the author, "Qwen was never prompted to generate... I
   guess it just learned that's the sort of garbage GPT-4.1 likes."

In both experiments, reward rose steadily while output quality did not
improve by the author's own judgment — the precise "rising reward, worse
jokes" signature the `humor-rl` skill's step 4 says to watch for.

**Why it matters:** confirms that hardening the rubric does not remove the
reward-hacking risk — it changes its shape. A single richer prompt to the
judge is not a fix; the skill's insistence on decomposed, non-judge-only
reward terms is the correct response to what's actually documented here, not
an overcautious one.

**Correction to prior framing:** this was not "a collapse into classic jokes
that persisted" — it was a sequence of two different collapses under two
different reward designs. The second is arguably the more instructive one:
even an "improved" rubric gets gamed in a new direction.

---

## 2. New Yorker Caption Contest — RLHF/DPO limitations on a creative task

Two distinct papers exist here; don't conflate them.

### 2a. `zhang2024humorinai` — the 250M-rating dataset paper

**Zhang, J. et al., "Humor in AI: Massive Scale Crowd-Sourced Preferences and
Benchmarks for Cartoon Captioning," NeurIPS 2024 Datasets & Benchmarks
Track.** [arXiv:2406.10522](https://arxiv.org/abs/2406.10522)

- Dataset: **>250M human ratings on >2.2M captions**, collected over 8 years
  (confirmed verbatim from the abstract).
- Exact quote: *"Our experimental results highlight the limitations of
  current fine-tuning methods, such as RLHF and DPO, when applied to
  creative tasks... even state-of-the-art models like GPT4 and Claude
  currently underperform top human contestants."*
- **Table 3 win rates** (Mistral-7B variants, vs. Top-10-human-percentile /
  vs. Contestant-Median):
  - 0-shot: 4.95% / 25.82%
  - **SFT regressed BELOW zero-shot**: 3.85% / 14.29%
  - RLHF: 8.79% / 24.73%
  - DPO (best of the small-model variants): 9.34% / 31.32%
  - GPT-4o: 44.51% / 86.81%; Claude-3-Opus: 54.40% / 88.46% — frontier
    models roughly break even against the *best* human contestants while
    crushing the median, which is a different and easier bar.
- **Counterintuitive nuance:** the paper's own diversity analysis found DPO
  *increased* generation diversity relative to the other methods tested —
  the opposite of the commonly-cited finding that RLHF reduces diversity in
  non-creative text tasks. Don't over-generalize "RLHF/DPO kill diversity"
  from this literature; it's setup-dependent.
- SFT underperforming the zero-shot baseline is itself a separate small
  negative result worth keeping: the "obvious" fix (just fine-tune on good
  captions) actively hurt here.

### 2b. `zhou2025bridging` — the persona-prompting negative result

**Zhou, K. L. et al., "Bridging the Creativity Understanding Gap:
Small-Scale Human Alignment Enables Expert-Level Humor Ranking in LLMs,"
EMNLP 2025 Findings.** [arXiv:2502.20356](https://arxiv.org/abs/2502.20356)

- Baseline SOTA models: **67.3% ± 1.5%** accuracy on pairwise caption-ranking
  (Hessel et al.'s benchmark task) — far below expert humans.
- This paper's method (targeted small-scale SFT on human preference
  judgments) raises it to **82.4% ± 1.2%**.
- Exact quote — the actual negative result: *"attempts to mimic subgroup
  preferences through various persona prompts showed minimal impact."* The
  natural first-try fix — just tell the model to adopt a persona — **failed**.
  Only fine-tuning on real human preference data closed the gap.

**Why it matters:** persona-conditioning is the cheapest thing anyone will
try first for audience-dependent humor. This is a direct, controlled
demonstration that it doesn't work; budget for real preference data instead.

---

## 3. HumorGen's own negative finding — `ajayi2026humorgen`

**Ajayi, E. & Mitra, P., "HumorGen: Cognitive Synergy for Humor Generation in
LLMs via Persona-Based Distillation," 2026.**
[arXiv:2604.09629](https://arxiv.org/abs/2604.09629)

**Important process note:** an earlier automated fetch of this paper
returned a fabricated set of numbers (1083.9 / 1079.9 / 1034.5 with invented
confidence intervals). That table does not exist in the paper. The numbers
below were obtained by downloading the raw arXiv HTML and grepping the text
directly, cross-checked twice. **If you see the fabricated numbers anywhere
downstream of this project, they are wrong — use these:**

- On HTB (their own Humor Transfer Bench, 400 prompts / 8 domains),
  **Llama-3.3-70B judge**: SFT-7B = **1128.14**, DPO-7B = **1123.72**,
  O-GRPO-7B = **1071.13** (ranked 7th of the models tested).
- Robustness check, **Qwen-2.5-72B judge**: DPO = 1138.05 narrowly exceeds
  SFT = 1132.41.
- Exact quote: *"SFT and DPO perform similarly, with negligible and
  inconsistent differences... In both cases, the 95% confidence intervals
  overlap, indicating no statistically significant difference. O-GRPO
  consistently underperforms both SFT and DPO."*
- Central conclusion, exact quote: *"a data quality ceiling: when SFT data is
  diverse and well-curated, preference optimization (DPO, O-GRPO) yields no
  significant gains."*
- **A second, separate negative result** from the same paper: further
  fine-tuning HumorGen-SFT on 998 real stand-up transcripts (comedian Shaun
  Eli) caused a sharp regression, **BT score 1083.9 → 653.1**, attributed to
  a mismatch between performance-native stand-up (written for live
  timing/delivery) and text-only humor. (This is where the fabricated fetch
  apparently lifted a real number and pasted it into the wrong table.)
- **A third finding**: *"reasoning-augmented training can reduce judged
  funniness"* — their chain-of-thought "Think" variants scored lower on
  judged funniness than non-reasoning variants.

**Why it matters:** this is the cleanest, most directly-relevant negative
result in the whole corpus. It says plainly: don't expect DPO or GRPO to beat
a well-curated SFT baseline on humor generation, don't expect reasoning
scaffolding to help, and don't assume performance-humor data transfers to
text humor. Any project variant of "just add preference optimization" needs
to argue why it will do better than this, not assume it will.

---

## 4. The kappa≈0.49 humor-annotation-agreement claim — corrected

**Traced to: Sun, J. et al., "ExPUNations: Augmenting Puns with Keywords and
Explanations," EMNLP 2022.** (`sun2022expunations`)
[ACL Anthology](https://aclanthology.org/2022.emnlp-main.304/)

The number is real and appears in this paper — but it is being cited **for
the wrong construct** everywhere it circulates, including in a SemEval-2026
participant paper (`lmfaoooo2026semeval`, arXiv:2606.00022) that repeats it
as "moderate agreement is common in humor-related annotation tasks."

- Fleiss' κ = 0.49 in the primary source measures agreement among 3
  annotators on whether *"the text supports both senses of the pun word"* —
  a narrow lexical/structural judgment about pun validity, **not** a
  funniness or general-humor-quality judgment.
- The same paper's **actual funniness-rating agreement is lower**: Cohen's
  κ = **0.41** for the "Funny (1–5)" field, κ = 0.58 for binary "Joke
  (0/1)," κ = 0.40 for "Understand," κ = 0.16 for "Offensive" (their weakest
  field).

**Corrected citation to use going forward:** *"Sun et al. (2022) report
Cohen's κ = 0.41 for funniness ratings and κ = 0.49 for pun-word semantic
validity — funniness agreement is the lower of the two, not 0.49."* If
anything this strengthens the "humor labels are noisy" argument the
`humor-rl` skill makes — just cite the right number for the right claim, and
correct `humor_reward_functions.py`'s docstring/skill text accordingly.

---

## 5. RLVR damaging multi-turn conversational skill

No single paper uses the exact phrasing "RLVR damages multi-turn humor," but
three real papers establish the underlying claim precisely, in decreasing
order of directness:

- **ICPO** — "Illocution-Calibrated Policy Optimization for Multi-Turn
  Conversation," 2026. [arXiv:2601.15330](https://arxiv.org/abs/2601.15330).
  Exact quote: *"We find that standard post-training techniques like
  Reinforcement Learning with Verifiable Rewards (RLVR) exacerbate this
  issue by rewarding confident, direct answers, thereby inducing
  overconfidence and discouraging the model from seeking clarification."*
  Their fix reports a 75% average improvement on multi-turn conversation
  while preserving single-turn performance. **Best match for the claim.**
- **UFO** — "A Simple 'Try Again' Can Elicit Multi-Turn LLM Reasoning,"
  2025. [arXiv:2507.14295](https://arxiv.org/abs/2507.14295). Exact quote:
  *"models trained with existing RL paradigms often lose their ability to
  solve problems across multiple turns and struggle to revise answers based
  on contextual feedback, leading to repetitive responses."*
- **Base phenomenon (not RLVR-specific):** "LLMs Get Lost In Multi-Turn
  Conversation," 2025. [arXiv:2505.06120](https://arxiv.org/abs/2505.06120).
  Establishes the underlying "lost-in-conversation" phenomenon that the two
  papers above attribute, in part, to RLVR-style training.

**Why it matters:** directly supports the project's gap #2 (multi-turn
conversational humor). If RLVR-style single-shot verifiable rewards degrade
conversational flexibility generally, a naive port of GRPO-on-single-jokes to
a banter/callback setting risks the same regression — trajectory-level
optimization (RAGEN/StarPO, GiGPO — see `code-and-models.md`) is the
documented answer, not turn-level reward.

---

## 6. Diversity collapse under RL — general, not humor-specific, but the exact mechanism humor-RL will hit

- **Doshi & Hauser, *Science Advances* 2024** (`doshi2024genai`) —
  [doi:10.1126/sciadv.adn5290](https://www.science.org/doi/10.1126/sciadv.adn5290).
  Randomized writing experiment, no RL involved at all: LLM-assisted stories
  were individually rated *more* creative, but the **corpus** of
  AI-assisted stories was measurably more similar to itself than a
  human-only corpus. This is the foundational "AI creativity reduces
  collective diversity" finding — the base rate humor-RL's anti-collapse
  machinery has to beat even before RL enters the picture.
- **DivPO** (`lanchantin2025divpo`, Meta, arXiv:2501.18101) — states plainly
  that *post-training generally* (SFT, RLHF, or preference optimization)
  "tends to sharpen the output probability distribution and reduce the
  diversity of generated responses." Reports +45.6% persona diversity and
  +74.6% story diversity vs. standard DPO at a similar win rate — figures
  corroborated by two independent secondary sources but not independently
  read from the PDF; treat as likely, not primary-confirmed.
- **"Beyond Mode Collapse: Distribution Matching for Diverse Reasoning"**
  (2026, arXiv:2605.19461) — confirmed via direct abstract fetch: *"On-policy
  reinforcement learning methods like GRPO suffer from mode collapse... We
  show this stems from reverse KL minimization's mode-seeking behavior,
  which reinforces the first high-reward trajectory found rather than
  maintaining a distribution over multiple diverse solutions."* Domain is
  math/combinatorial optimization, not humor, but this is the precise
  mechanistic explanation for *why* GRPO collapses — worth understanding
  before assuming a diversity bonus alone will fix it.
- **"Advantage Collapse in GRPO"** (2026, arXiv:2605.21125, ICML 2026) —
  reportedly 28–45% of training batch groups experience full advantage
  collapse (near-zero advantage, vanishing gradient) in math-reasoning RL,
  because homogeneous-reward groups carry no learning signal. **[UNVERIFIED
  at primary-text level]** — paper's existence and ICML 2026 acceptance are
  corroborated by multiple independent listings, but the exact percentage
  came only from a search-engine synthesis during research, which is known
  (see above) to sometimes fabricate precision. Confirm from the PDF before
  citing the number.

**Why it matters:** `reward_std` trending toward zero, which the `humor-rl`
skill already tells you to watch, is not a humor-specific quirk — it's a
named, quantified (if not fully verified) phenomenon in the general GRPO
literature, with a specific mechanistic cause (reverse-KL mode-seeking) that
motivates why entropy/divergence-based fixes (DPH-RL, DARLING — see
`code-and-models.md`) work better than an ad hoc diversity bonus.

---

## 7. Access failures — reported honestly

- **OpenReview reviews for the NYCC paper** (`zhang2024humorinai`,
  OpenReview id `w90ZH5v34S`): both the HTML forum page and the OpenReview
  REST API returned a Cloudflare-style 403 on every attempt, with multiple
  user-agent strings. Reviewer comments — which often contain the
  soft-pedaled negative results camera-ready versions bury — were not
  retrievable for this paper. **[UNVERIFIED — inaccessible, not merely
  unchecked.]** Worth a manual retry from a normal browser session.
- **Gwern's AI mode-collapse bibliography**
  ([gwern.net/doc/reinforcement-learning/preference-learning/mode-collapse](https://gwern.net/doc/reinforcement-learning/preference-learning/mode-collapse/index))
  lists several more titles worth chasing directly rather than trusting
  secondhand: "Verbalized Sampling: How to Mitigate Mode Collapse and
  Unlock LLM Diversity" (Zhang et al. 2025), "A Tale of Tails: Model
  Collapse as a Change of Scaling Laws" (Dohmatob et al. 2024), "Creativity
  Has Left the Chat: The Price of Debiasing Language Models" (Mohammadi
  2024), "The Homogenizing Effect of Large Language Models on Human
  Expression and Thought" (Sourati et al. 2026), "Helping or Herding?
  Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking"
  (Eisenstein et al., DeepMind). None of these were independently
  raw-fetched during this research pass — **[UNVERIFIED, secondhand via
  bibliography]** but high-value as a follow-up reading list.
- **GitHub/informal evidence of humor-RL mode collapse**: search turned up
  only weak, anecdotal signal (a developer note on `alpaca-lora` about
  repeated outputs after fine-tuning; blog-level complaints on small GPT-2
  joke-generation side projects). No GitHub issue thread substantively
  documents humor-RL mode collapse with concrete data. **[UNVERIFIED /
  weak]** — don't cite this as evidence of anything beyond "informally,
  people have noticed this too."

---

## Bottom line for this project

Every naive path has already been tried and reported not to work cleanly:
plain SFT-on-humor can *regress below zero-shot* (NYCC); DPO/offline-GRPO
don't beat a well-curated SFT baseline (HumorGen); a single LLM-judge reward
gets hacked within one paper's worth of iteration, twice, in two different
directions (LessWrong); persona-prompting for audience-adaptation doesn't
work (NYCC follow-up); and the underlying GRPO mechanism has a named,
mechanistic collapse mode (reverse-KL mode-seeking) that is not humor-specific
but that humor-RL will inherit by default. None of this means humor-RL is a
dead end — it means the burden of proof is on demonstrating that a **specific
mitigation** (decomposed reward, diversity term, trajectory-level multi-turn
optimization, or online human feedback loop) beats these documented
baselines, not on re-running the naive version again.

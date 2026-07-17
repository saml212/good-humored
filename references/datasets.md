# Datasets

What exists, what it costs to get, and what's biased about it. "Usable now"
means a working loader/download link was confirmed during research, not just
a paper describing the data.

---

## Tier 1 — usable now, large, and reward-model-relevant

### New Yorker Caption Contest ranking data
**Zhang et al., "Humor in AI," NeurIPS 2024 D&B** (`zhang2024humorinai`)
- **Size:** >250M human ratings on >2.2M captions, 8 years of contest data.
- **License:** not stated by the paper's abstract; the companion HF dataset
  page should be checked directly before use.
- **Access:** HF dataset
  [`yguooo/newyorker_caption_ranking`](https://huggingface.co/datasets/yguooo/newyorker_caption_ranking);
  code at
  [github.com/yguooo/cartoon-caption-generation](https://github.com/yguooo/cartoon-caption-generation).
  Umbrella project page: [nextml.github.io/caption-contest-data](https://nextml.github.io/caption-contest-data/).
- **Known bias:** this is *contest* data — every caption in it was already
  submitted by a human who thought it was good enough to enter, and ratings
  come from contest voters, not a controlled panel. It measures "funny
  enough to submit to the New Yorker," a fairly narrow register (dry,
  literate, New Yorker-reader humor) — not humor in general.
- **Documented negative result attached to this data:** RLHF/DPO
  fine-tuning on it underperforms top human contestants; SFT actually
  regressed below zero-shot in the paper's own experiments. See
  `negative-results.md` §2a for exact numbers. Treat this dataset as a hard
  baseline to beat, not a fresh, easy win.

### New Yorker Caption Contest — matching/ranking/explanation splits
**Hessel et al., "Do Androids Laugh at Electric Sheep?", ACL 2023**
(`hessel2023androids`)
- **Size:** smaller, curated benchmark splits (train/val/test) for three
  tasks: caption-cartoon matching, quality ranking, explanation.
- **License:** CC BY 4.0.
- **Access:** HF dataset
  [`jmhessel/newyorker_caption_contest`](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest)
  — confirmed real splits, working loader. Code:
  [github.com/jmhessel/caption_contest_corpus](https://github.com/jmhessel/caption_contest_corpus).
- **Known bias:** same register bias as above (New Yorker style specifically);
  additionally, the "explanation" task requires substantial cultural
  background knowledge that skews toward highly-educated American/Western
  readers.

### Oogiri-GO (CLoT dataset)
**Zhong et al., "Let's Think Outside the Box" (CLoT), CVPR 2024** (`zhong2024clot`)
- **Size:** >130,000 multilingual/multimodal Oogiri-game samples — English
  (~23.8K), Chinese (~48.8K), Japanese (~61.5K); T2T/I2T/IT2T formats;
  ~78% carry a human "like" preference rating.
- **License:** CC BY 4.0, plus adherence to the source platforms' own terms
  (Bokete, Zhihu) — a real dual-license constraint, check both before
  redistribution.
- **Access:** HF dataset
  [`zhongshsh/CLoT-Oogiri-GO`](https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO)
  — confirmed real JSONL files (en/cn/jp, ~1.39GB total), last updated
  2024-03-17. **Usable now**, the strongest immediately-downloadable
  humor-preference dataset in this whole corpus after the NYCC data above.
- **Known bias:** Oogiri (improv-comedy-caption) responses are themselves
  crowd-submitted and rated by "likes," which is an engagement metric, not a
  controlled preference judgment — the same popularity-bias risk the
  `humor-rl` skill already flags for engagement metrics generally. Note this
  is a *different, less-rigorous* rating methodology than Oogiri-Master/
  Oogiri-Corpus below, despite the shared "Oogiri" name — don't assume they
  share the anti-popularity-bias design.

---

## Tier 2 — real, popularity-bias-controlled, but not yet freely downloadable

### Oogiri-Master / Oogiri-Corpus
**Murakami et al., "Oogiri-Master: Benchmarking Humor Understanding via
Oogiri," 2025** (`murakami2025oogirimaster`)
- **Size:** ~100 candidate responses per prompt, each independently rated by
  ~100 human judges via Yahoo crowdsourcing, with **no rater seeing any
  other rater's score**. This is the explicit anti-popularity-bias design
  the project's `CLAUDE.md` flags as the cleanest humor RM data available.
- **License:** CC BY-NC-SA 4.0 — **non-commercial**, share-alike. This
  matters if any downstream use is commercial; check before assuming it's
  free to use in a product pipeline.
- **Access:** **not a data release** — the linked GitHub repo,
  [`CyberAgentAILab/oogiri-dataset-builder`](https://github.com/CyberAgentAILab/oogiri-dataset-builder),
  is MIT-licensed **builder code only**. It crawls the original Japanese
  Oogiri source pages to regenerate `Oogiri-Master.csv` / `Oogiri-Corpus.csv`
  yourself. The maintainers explicitly warn the MIT license on the code
  **does not transfer to the data**, and robots.txt/ToS compliance for the
  underlying scrape is the user's responsibility. **Not usable Monday** —
  budget for running (and validating) a scraper first, plus resolving the
  non-commercial license question if this is for anything beyond research.
- **Known bias:** none of the popularity-bias kind by design (that's the
  whole point of the methodology) — but it is Japanese-language,
  Japanese-cultural-context humor, so a model or reward function trained on
  it will not transparently generalize to English-language humor without
  translation/adaptation work, which introduces its own distortions.

### LoTbench
**Huang et al., "A Causality-aware Paradigm for Evaluating Creativity of
Multimodal LLMs," IEEE TPAMI 2025** (`huang2025lotbench`)
- **Access:** code/data marked "coming soon" as of this compilation.
  **Not usable Monday** despite being peer-reviewed (TPAMI). Project page:
  [lotbench.github.io](https://lotbench.github.io/).
- Evaluates via masked-token completion + a causal evaluator ("different
  approach, equally satisfactory outcome" — fewer rounds to satisfy the
  evaluator implies higher creativity). Worth checking back periodically for
  a release.

---

## Tier 3 — usable now, smaller or narrower

### Humor Transfer Bench (HTB)
Introduced inside `ajayi2026humorgen`, not a standalone paper/dataset.
- **Size:** 400 prompts across 8 input domains (Neutral Facts, Everyday
  Life, Abstract Concepts, Dialogic Quotations, Scenario Inputs, Analogical
  Prompts, Direct Instructional, News Headlines).
- **What it actually measures:** generalization *within* humor generation
  across different **input-prompt domains** — same generation task, varying
  only the prompt category. This is **not** reasoning↔humor capability
  transfer and **not** joke-type style transfer (dad joke → pun). Confirm
  this distinction before citing HTB as evidence for either of those other
  claims.
- **Access:** no code/data release found. Not usable Monday, but the 8-domain
  structure is simple enough to reconstruct from the paper's own
  description (Table 7) if needed.

### ColBERT humor detection dataset
**Annamoradnejad & Zoghi, *Expert Systems with Applications* 2024**
(`annamoradnejad2024colbert`)
- **Size:** 200,000 short texts (100K jokes / 100K non-jokes), matched for
  style/length.
- **License:** code is MIT; dataset terms should be checked on the mirror.
- **Access:** [github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection](https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection),
  also mirrored on IEEE DataPort. **Usable now.** 98.2% F1 with a BERT
  parallel-network baseline — a plausible cheap reward-model backbone or
  sanity-check classifier, though it's detection (joke/not-joke), not a
  funniness-quality signal.
- **Known bias:** the negative ("non-joke") class was constructed to match
  style/length, which is good practice, but the dataset is a **detection**
  task — it cannot tell you *how funny* something is, only *whether it's a
  joke-shaped utterance*.

### Weller & Seppi Reddit humor dataset
**Weller & Seppi, "Humor Detection: A Transformer Gets the Last Laugh,"
EMNLP 2019** (`weller2019humor`)
- **Size:** ~16,000 Reddit-rating-labeled instances.
- **Access:** [github.com/orionw/RedditHumorDetection](https://github.com/orionw/RedditHumorDetection)
  — usable now, code + data both present.
- **Known bias:** Reddit upvotes are an engagement/popularity metric —
  exactly the confound the `humor-rl` skill warns against for reward
  signals. Fine as a detection baseline; do not use upvote count as a
  funniness reward without correcting for the confound.

### StandUp4AI
**Barriere et al., "StandUp4AI: A New Multilingual Dataset for Humor
Detection in Stand-up Comedy Videos," Findings of EMNLP 2025**
(`barriere2025standup4ai`)
- **Size:** 330+ hours, 7 languages.
- **Notable property:** labels come from **actual audience laughter**,
  automatically detected and annotated at the word level (not a binary
  joke/no-joke label, not upvotes, not an LLM judge). This is one of the
  only datasets in this whole corpus with a ground-truth signal that isn't
  a proxy for funniness — it's the thing itself (people actually laughing).
- **Access:** not independently re-verified for a direct download link
  during this pass beyond the arXiv/ACL listing — **[check
  aclanthology.org/2025.findings-emnlp.919/ directly before assuming a
  loader exists]**.
- **Known bias:** stand-up delivery (timing, tone, live audience dynamics)
  is a very different production context from text-only humor — a laughter
  label here reflects performance quality as much as joke quality, which
  actually matters for the project's multi-turn/banter/timing interest but
  should not be treated as a pure text-humor signal.

### UR-FUNNY
Multimodal (text + audio + video) humor understanding from TED talks,
EMNLP 2019 (`aclanthology.org/D19-1211`) — flagged by research but not
independently re-verified for exact access details in this pass.
**[UNVERIFIED — confirm current HF/GitHub availability before relying on
it.]** Relevant if delivery/timing signal (not just text) becomes part of
the project's evaluation.

### SemEval-2021 Task 7 "HaHackathon"
- **Size:** 10,000 texts (Twitter + Kaggle Short Jokes), 20 annotators/item.
- **What's notable:** **graded** (not binary) funniness-rating regression,
  plus a paired offense-rating regression and a "controversy" subtask — the
  offense co-label is directly useful for auditing the
  "toxic-jokes-score-higher" reward-hacking risk documented in
  `negative-results.md` and `code-and-models.md`.
- **Access:** hosted via CodaLab competition #27446; check current
  availability directly, not re-verified in this pass.

### HAHA (Spanish humor detection + funniness regression)
IberEval 2018 / IberLEF 2019 & 2021
([fing.edu.uy/inco/grupos/pln/haha](https://www.fing.edu.uy/inco/grupos/pln/haha/))
— a genuinely reusable non-English resource with both detection and
graded-funniness labels; not independently re-verified for exact current
download mechanics in this pass. **[Confirm before use.]**

### Chumor / Chumor 2.0
Chinese humor **explanation** dataset (sourced from Ruo Zhi Ba), ACL
Findings 2025 — HF dataset + leaderboard Space + GitHub confirmed to exist
by research, but exact license/size not independently re-verified here.
**[UNVERIFIED — confirm before use.]** Notable because it targets
*explanation* quality (do models understand why something is funny), which
is the same axis HumorBench probes for English/NYCC humor.

---

## Tier 4 — raw material, real popularity-bias confound, use with eyes open

These are the classic scraped Reddit/joke corpora. All of them carry
score/upvote fields that are an engagement-not-funniness confound — exactly
what Oogiri-Master/Oogiri-Corpus was purpose-built to avoid. Use for
vocabulary priors, cold-start SFT diversity, or negative controls; do not use
raw scores as a reward signal without correcting for popularity bias.

| Dataset | Size | License | Access | Note |
|---|---|---|---|---|
| Short Jokes (Kaggle/HF) | 231,657 jokes | Kaggle terms / CC-BY-4.0 (HF mirror) | `ysharma/short_jokes`, `Fraser/short-jokes` on HF; build script `github.com/amoudgl/short-jokes-dataset` | Scraped r/jokes + r/cleanjokes via PRAW |
| One Million Reddit Jokes | 1M rows | CC-BY-4.0 (HF) | HF `SocialGrep/one-million-reddit-jokes` | Many rows are `[removed]`/`[deleted]` — clean before use |
| "16000 One-liners" (Mihalcea & Strapparava 2005) | ~16,000 | unclear | **[UNVERIFIED — direct download link not confirmed]** | Oldest resource surfaced in this research pass |
| "Pun of the Day" (Yang et al. 2015) | small | unclear | **[UNVERIFIED — direct download link not confirmed]** | Classic pun-detection dataset |
| CrowdTruth Short-Text-Corpus-For-Humor-Detection | small | check repo | `github.com/CrowdTruth/Short-Text-Corpus-For-Humor-Detection` | One-liners corpus |

---

## Live human-preference pipelines (not static datasets)

### LOL Arena (Good Start Labs)
Live humor-preference pipeline sourced from Bad Cards players, per
`CLAUDE.md`. **[UNVERIFIED in this research pass — no independent
confirmation of current status, access terms, or data volume was obtained.
Follow up directly with Good Start Labs before relying on this as a data
source.]**

### Cards Against LLMs harness
**Fettach et al., CHum 2026** (`fettach2026cardsagainstllms`) — not a
dataset release, but a **methodology**: 5 frontier LLMs playing actual Cards
Against Humanity (9,894 rounds) is a directly reproducible live-preference
harness, and the paper's own finding (models agree with each other far more
than with humans; apparent preference partly explained by position bias) is
a cautionary data point for anyone building an LLM-judge-based live
pipeline. See `code-and-models.md` for reuse potential.

---

## Datasets explicitly checked and found NOT to exist as described

- **"Oogiri-GO" as an independent dataset separate from CLoT** — it is real,
  but it is the CLoT paper's own dataset (see Tier 1), not a separate
  resource. Don't cite it as a distinct paper/release.
- **A dedicated humor-RL "gym"/environment package** — searched specifically
  (arXiv, GitHub, PapersWithCode, clvrai/awesome-rl-envs) and found none.
  Building one is genuinely novel territory, not a gap in searching.

## Cross-cutting bias note for reward-model design

Every dataset above that involves human ratings inherits at least one of
three confounds documented in `psychology.md`: (1) **sex bias** in who's
rated as funny (Greengross, Silvia & Nusbaum 2020: d=0.321 favoring men's
rated humor output across 28 studies); (2) **audience/culture-dependence**
(Cao et al. 2023: Chinese raters find distant-other jokes funnier than
close-other jokes; Americans show no such effect); (3) **identity-of-speaker
confound** in humor judgment specifically for AI-generated content (Kim et
al. 2026, `kim2026counterfactual`: jokes attributed to "privileged" speakers
refused up to 67.5% more often and rated malicious 64.7% more often than
identical jokes attributed to other speakers). A reward model trained
naively on any of the human-rating datasets above will absorb some
combination of these; see `psychology.md` §7 for how the disagreement/
subjective-annotation literature suggests handling this (model the
distribution of raters, don't collapse to a single "gold" label).

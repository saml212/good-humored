# Downloadable Joke Corpora for Anti-Plagiarism Penalty

## Executive Summary

**Total obtainable jokes today: ~3.1 million**
- Reddit jokes (1M Kaggle + 195K taivop + 1M HF): ~2.2M
- Short jokes (231K Fraser + variations): ~250K
- Oogiri (80K+ structured): 82.5K
- Dad jokes (147K shuttie + icanhazdadjokes): ~150K+
- Jester ratings (100 core jokes rated 4.1M times): 100 unique
- Pun datasets (SemEval + ExPUNations): ~2.4K

**Top 3 corpora to start with:**
1. **SocialGrep/one-million-reddit-jokes (HuggingFace)** — 1M jokes, CC-BY-4.0 licensed, immediate download
2. **taivop/joke-dataset (GitHub)** — 208K jokes, research-only (license unclear), proven Reddit/Wocka/Stupidstuff split
3. **Fraser/short-jokes (HuggingFace)** — 231K jokes, no license specified, CSV format, diverse

**License red flags:**
- **taivop/joke-dataset:** Research-only, recommends against commercial use, unclear copyright
- **New Yorker caption contest:** CC-BY-4.0 requires attribution; images have different licensing
- **Jester:** Research use when referenced (not formally CC0)
- **Oogiri-GO/Oogiri-Corpus:** CC-BY-NC-SA (non-commercial only)
- **Most HuggingFace datasets:** Licenses often unspecified; verify before commercial deployment

---

## Detailed Corpus Catalog

### 1. Fraser/short-jokes (HuggingFace)
- **Size:** 231,657 jokes
- **Download:** `datasets` library: `from datasets import load_dataset; load_dataset("Fraser/short-jokes")`  
  URL: https://huggingface.co/datasets/Fraser/short-jokes
- **License:** Unspecified (likely public domain or CC0, originally from Kaggle)
- **Format:** CSV (auto-converted to Parquet on HF)
- **Characteristics:** Short format (10–200 characters), curated
- **Quality issues:** License unspecified; dataset viewer disabled (requires Python execution)
- **Download method:** `huggingface_hub.download_dataset()` or via HF web interface

### 2. taivop/joke-dataset (GitHub)
- **Size:** 208,300 jokes total
  - reddit_jokes.json: 195,084 jokes
  - stupidstuff.json: 3,770 jokes
  - wocka.json: 10,000 jokes (humor site)
- **Download:** https://github.com/taivop/joke-dataset  
  Raw file URLs (via git-lfs):
  - https://raw.githubusercontent.com/taivop/joke-dataset/master/reddit_jokes.json
  - https://raw.githubusercontent.com/taivop/joke-dataset/master/stupidstuff.json
  - https://raw.githubusercontent.com/taivop/joke-dataset/master/wocka.json
- **License:** Research-only; creator disclaims ownership and cautions against commercial use; archival status
- **Format:** JSON (flat list of joke objects; each has `body` + metadata)
- **Characteristics:** Reddit scraped as of Feb 2017; includes submissions, scores, punchlines
- **Quality issues:** Repository archived Dec 2022 (no longer maintained); copyright status unclear for many jokes
- **Metadata:** Includes upvotes, downvotes, post ID, author

### 3. SocialGrep/one-million-reddit-jokes (HuggingFace)
- **Size:** 1,000,000 jokes from r/jokes
- **Download:** `from datasets import load_dataset; load_dataset("SocialGrep/one-million-reddit-jokes")`  
  URL: https://huggingface.co/datasets/SocialGrep/one-million-reddit-jokes
- **License:** CC-BY-4.0 (requires attribution)
- **Format:** CSV / Parquet (auto-converted)
- **Characteristics:** Reddit submissions, score-annotated, mainly English
- **Quality issues:** None reported; actively maintained
- **Metadata:** title, score, permalink

### 4. 1 Million Reddit Jokes (Kaggle)
- **Size:** 1,000,000 jokes from r/jokes
- **Download:** Kaggle API: `kaggle datasets download -d priyamchoksi/1-million-reddit-jokes-rjokes`  
  Web: https://www.kaggle.com/datasets/priyamchoksi/1-million-reddit-jokes-rjokes
- **License:** Kaggle default (check dataset page for specific terms; typically research-friendly)
- **Format:** CSV
- **Characteristics:** Reddit data, score-annotated
- **Quality issues:** None reported; overlaps with taivop/SocialGrep Reddit data
- **Metadata:** Similar to SocialGrep (score, text, metadata)

### 5. Oogiri-Corpus & Oogiri-Master (CyberAgentAILab)
- **Size:** 
  - Oogiri-Corpus: 908 prompts × ~96 responses = 82,536 prompt–response pairs
  - Oogiri-Master: Similar scale with extended rating data
- **Download:** GitHub builder (no pre-built release on HF as of July 2026):  
  https://github.com/CyberAgentAILab/oogiri-dataset-builder  
  Build locally: `uv sync && uv run python -m oogiri_dataset_builder pipeline --output-dir ./out`  
  Outputs: Oogiri-Master.csv, Oogiri-Corpus.csv
- **License:** CC-BY-NC-SA 4.0 (non-commercial; share-alike)
- **Format:** CSV (JSONL in pipeline output)
- **Characteristics:** Japanese Oogiri game (creative witty responses to prompts); high-quality human judgments (~100 independent judges per response); multimodal potential (text + image prompts)
- **Quality issues:** Non-commercial license restricts commercial use; requires Python 3.11+ and local build
- **Metadata:** Each response rated independently; popularity-bias-free aggregation (multiple judges without seeing others' scores)

### 6. Oogiri-GO (zhongshsh/CLoT-Oogiri-GO, HuggingFace)
- **Size:** ~130,000 Oogiri samples (English, Chinese, Japanese combined)
- **Download:** `from datasets import load_dataset; load_dataset("zhongshsh/CLoT-Oogiri-GO")`  
  URL: https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO
- **License:** Not explicitly stated (inherited from original Oogiri dataset; assume CC-BY-NC-SA)
- **Format:** JSONL / Parquet
- **Characteristics:** Text-to-Text (T2T), Image-to-Text (I2T), Image & Text-to-Text (IT2T) tasks; 77.95% samples have human preference annotations (likes)
- **Quality issues:** Popularity bias in original version (newer Oogiri-Master addresses this); multimodal components not all English
- **Metadata:** type, question, response, star (popularity rating)
- **Text-only extraction:** Filter to T2T examples; ~77.95% have like annotations; primarily English/Chinese examples

### 7. New Yorker Caption Contest (jmhessel/newyorker_caption_contest, HuggingFace)
- **Size:** ~11K+ matching/explanation examples (multimodal; images + captions)
- **Download:** `from datasets import load_dataset; load_dataset("jmhessel/newyorker_caption_contest")`  
  URL: https://huggingface.co/datasets/jmhessel/newyorker_caption_contest
- **License:** CC-BY-4.0 (attribution required for images)
- **Format:** Parquet / HuggingFace streaming
- **Characteristics:** Cartoon image + human-written captions; multimodal; multiple subtasks (ranking, explanation, generation)
- **Quality issues:** Images are multimodal; text captions only ~11K examples; small relative to text-only corpora; requires image handling for full dataset
- **Metadata:** cartoon image, caption text, rank, explanation, candidate captions
- **Text-only approach:** Extract caption text only (~11K unique captions); images separately if needed

### 8. Humicroedit & FunLines (SemEval-2020 Task 7, HuggingFace)
- **Size:** 
  - Humicroedit: 15,095 edited headlines (from 5,000 originals × 3)
  - FunLines: 8,248 edited headlines
  - Total: ~23K annotated headlines
- **Download:** `from datasets import load_dataset; load_dataset("humicroedit")`  
  URL: https://huggingface.co/datasets/humicroedit  
  Split: `humicroedit`, `funlines`  
  Official page: https://www.cs.rochester.edu/u/nhossain/humicroedit.html
- **License:** Not explicitly specified (academic; assume CC-BY for research)
- **Format:** CSV / Parquet
- **Characteristics:** Edited news headlines (humor rating 0–3 scale); Reddit-sourced originals; crowd-annotated
- **Quality issues:** Headlines are artificial/edited variants, not naturally occurring jokes; limited size; span 2017–2020
- **Metadata:** original_headline, edited_versions (3 per original), funniness_score (0–3), annotator ratings

### 9. SemEval-2017 Task 7: Puns
- **Size:** ~2,400 puns (1,298 homographic + 1,098 homophonic)
- **Download:** Official task page: https://alt.qcri.org/semeval2017/task7/index.php?id=data-and-resources  
  GitHub mirrors available (e.g., PunEval datasets)
- **License:** CC-BY-NC (non-commercial, requires attribution)
- **Format:** XML (SemEval-style)
- **Characteristics:** Manually annotated by humorists; pun word + alternative word marked; non-pun examples included
- **Quality issues:** Small dataset; pun-specific (not general jokes); XML format requires parsing
- **Metadata:** pun word location, alternative meaning, context

### 10. ExPUNations (amazon-science/expunations, GitHub)
- **Size:** ~2,000 puns (curated from SemEval + extended)
- **Download:** https://github.com/amazon-science/expunations  
  Clone: `git clone https://github.com/amazon-science/expunations`
- **License:** Not explicitly stated (assume CC-BY for research; Amazon Science repo)
- **Format:** JSON/CSV with crowdsourced annotations
- **Characteristics:** Puns augmented with crowdsourced keyword annotations + funniness ratings + explanations
- **Quality issues:** Pun-specific; requires understanding of pun structure; extended annotations but modest size
- **Metadata:** pun text, keywords (distinctive words), explanation (why funny), funniness rating (fine-grained)

### 11. Jester Collaborative Filtering Dataset
- **Size:** 100 core jokes; 4.1M ratings from 73,421 users (1999–2003)
- **Download:** http://goldberg.berkeley.edu/jester-data/  
  Files:
  - jester-data-1.zip (3.9MB): 24,983 users, ≥36 ratings each
  - jester-data-2.zip (3.6MB): 24,938 users, 15–35 ratings each
  - Full archive also available
- **License:** Free for research with attribution (not formally CC0 but permissive)
- **Format:** CSV / text (rating matrices; ratings from -10 to +10)
- **Characteristics:** Only 100 unique jokes; massive rating dataset (collaborative filtering focus); longitudinal (4-year span)
- **Quality issues:** Very small joke set (100), not suitable as primary corpus; dataset is ratings-focused, not joke-focused
- **Metadata:** User ID, joke ID, rating (-10 to +10)
- **Extract jokes:** Available in auxiliary files or contact maintainer; jokes not primary deliverable

### 12. Shuttie Dad Jokes (HuggingFace)
- **Name:** shuttie/dadjokes & shuttie/reddit-dadjokes
- **Size:** 
  - dadjokes: Unknown (sampled 5+ vote threshold)
  - reddit-dadjokes: 147,753 jokes (intro + punchline pairs)
- **Download:** 
  - `from datasets import load_dataset; load_dataset("shuttie/dadjokes")`
  - `from datasets import load_dataset; load_dataset("shuttie/reddit-dadjokes")`
  - URLs: https://huggingface.co/datasets/shuttie/dadjokes, https://huggingface.co/datasets/shuttie/reddit-dadjokes
- **License:** Not explicitly specified
- **Format:** CSV (base + punchline split)
- **Characteristics:** Dad jokes sourced from Kaggle; splits (intro, punchline) via Llama-8B; semantic deduplication applied; 2014–2022 span
- **Quality issues:** Artificially split by LLM (may introduce errors); semantic deduplication may miss surface-level dupes
- **Metadata:** intro, punchline, source

### 13. Kaggle Short Jokes (thedevastator, abhinavmoudgil95)
- **Size:** 200K–231K jokes (varies by version)
- **Download:** 
  - https://www.kaggle.com/datasets/thedevastator/short-jokes-dataset (200K+)
  - https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes (231K+)
  - Kaggle API: `kaggle datasets download -d abhinavmoudgil95/short-jokes`
- **License:** Dataset-dependent (check Kaggle page)
- **Format:** CSV
- **Characteristics:** Short format (10–200 chars typical); curated; overlaps with Fraser/HF version
- **Quality issues:** License often unspecified; may lack metadata
- **Metadata:** Minimal (text only, or category/rating)

### 14. Reddit Jokes Variants (HuggingFace)
- **Amirkid/jokes:** 579K jokes; auto-converted Parquet
- **Maximofn/short-jokes-dataset:** Variant of Fraser/short-jokes
- **ysharma/short_jokes:** Short format collection
- **Download:** Respective HF pages (e.g., https://huggingface.co/datasets/Amirkid/jokes)
- **License:** Typically unspecified; assume research-friendly
- **Format:** Parquet / CSV

### 15. Kaggle Dad Jokes (aryashah2k)
- **Size:** 16K+ dad jokes
- **Download:** https://www.kaggle.com/datasets/aryashah2k/dad-a-base-of-jokes  
  Kaggle API: `kaggle datasets download -d aryashah2k/dad-a-base-of-jokes`
- **License:** Dataset-dependent
- **Format:** CSV
- **Characteristics:** Sourced from icanhazdadjokes.com; one-liner format
- **Quality issues:** Smaller than other sources; licensing unclear

### 16. 16,000 One-Liners (Mihalcea & Strapparava 2005)
- **Size:** 16,000 one-liner jokes
- **Download:** Academic papers reference but not directly hosted; archived in corpus repositories  
  Related: https://github.com/CrowdTruth/Short-Text-Corpus-For-Humor-Detection
- **License:** Research (assume CC-BY or academic fair use)
- **Format:** Text (one per line)
- **Characteristics:** Classic humor detection dataset; balanced with non-humor texts
- **Quality issues:** Original paper from 2005; may be difficult to source; not actively maintained
- **Note:** Referenced in "Humor Detection: A Transformer Gets the Last Laugh" (Weller, ACL 2019)

### 17. ChatGPT Memorized 25 Jokes (Jentzsch & Kersting, WASSA 2023)
- **Size:** 25 unique jokes (repeated in 90%+ of ChatGPT-generated outputs)
- **Download:** Paper: https://arxiv.org/abs/2306.04563  
  GitHub (implementation): https://github.com/DLR-SC/JokeGPT-WASSA23
- **License:** CC0 (academic release)
- **Format:** Listed in paper appendix; extractable from arXiv PDF
- **Characteristics:** The canonical "mode collapse" baseline for humor RL; every model should beat this
- **Quality issues:** Only 25 jokes; need to extract from paper manually or via GitHub repo
- **How to use:** Extract all 25 jokes from the paper and build a minimal "memorized baseline" corpus for 100% similarity checking

---

## Licensing Summary for Commercial Use

| Corpus | License | Commercial Safe? | Notes |
|--------|---------|------------------|-------|
| Fraser/short-jokes | Unspecified | Likely yes | Assume public domain; verify origin |
| taivop/joke-dataset | Research-only | **NO** | Explicit caution against commercial use |
| SocialGrep/1M Reddit | CC-BY-4.0 | Yes* | Requires attribution; verify Reddit ToS |
| Kaggle 1M Reddit | Kaggle default | Likely yes | Research-friendly; check page |
| Oogiri-Corpus | CC-BY-NC-SA 4.0 | **NO** | Non-commercial; derivatives must share-alike |
| Oogiri-GO | CC-BY-NC-SA 4.0 | **NO** | Non-commercial |
| New Yorker captions | CC-BY-4.0 | Yes* | Requires attribution; images may have restrictions |
| Humicroedit/FunLines | Unspecified | Likely yes | Academic task; assume fair use for research |
| SemEval-2017 Puns | CC-BY-NC | **NO** | Non-commercial |
| ExPUNations | Unspecified | Likely yes | Amazon Science; assume research-friendly |
| Jester | Free for research | Yes* | With attribution required |
| Dad jokes (shuttie) | Unspecified | Likely yes | Assume research-friendly |
| Kaggle variants | Kaggle default | Likely yes | Check individual pages |

**Legend:** 
- `Yes*` = Safe with attribution/compliance with source ToS (e.g., Reddit API terms if redistributing)
- `Likely yes` = Probably safe; verify before commercial deployment
- `**NO**` = Not safe for commercial use; academic/research-only

---

## Download Commands (Quick Reference)

```bash
# HuggingFace datasets library
python -c "from datasets import load_dataset; ds = load_dataset('Fraser/short-jokes'); print(len(ds))"
python -c "from datasets import load_dataset; ds = load_dataset('SocialGrep/one-million-reddit-jokes'); print(len(ds))"
python -c "from datasets import load_dataset; ds = load_dataset('jmhessel/newyorker_caption_contest'); print(len(ds))"
python -c "from datasets import load_dataset; ds = load_dataset('humicroedit'); print(len(ds))"
python -c "from datasets import load_dataset; ds = load_dataset('shuttie/reddit-dadjokes'); print(len(ds))"

# GitHub repos
git clone https://github.com/taivop/joke-dataset
git clone https://github.com/amazon-science/expunations
git clone https://github.com/CyberAgentAILab/oogiri-dataset-builder

# Kaggle CLI
kaggle datasets download -d priyamchoksi/1-million-reddit-jokes-rjokes
kaggle datasets download -d aryashah2k/dad-a-base-of-jokes

# Manual downloads
# Jester: http://goldberg.berkeley.edu/jester-data/ (zip files)
# SemEval-2017: https://alt.qcri.org/semeval2017/task7/index.php?id=data-and-resources
```

---

## Recommended Assembly Pipeline

1. **Primary corpus (anti-plagiarism baseline):**
   - SocialGrep/one-million-reddit-jokes (1M) + Fraser/short-jokes (231K) + taivop subsets (208K)
   - **Total: ~1.4M unique jokes**, primarily text, commercial-safe or research-safe

2. **Secondary (structured evaluation):**
   - Oogiri-Corpus (82K scored responses) for diversity metric calibration
   - Oogiri-Master if available locally

3. **Memorization baseline:**
   - ChatGPT 25 jokes (manual extraction from paper)
   - SemEval-2020 Humicroedit headlines (23K) as secondary mode-collapse check

4. **Optional (niche):**
   - Pun datasets (SemEval-2017, ExPUNations) if humor RL targets pun generation
   - Jester (100 jokes) if multi-label similarity needed

---

## Known Quality Issues & Gotchas

1. **License creep:** Many HuggingFace datasets inherit unspecified licenses from Kaggle. Verify with dataset authors before commercial use.
2. **Reddit overlap:** Multiple versions of Reddit jokes exist (taivop, SocialGrep, Kaggle); deduplicate before use.
3. **Oogiri builds are local:** Oogiri-Master/Corpus require local Python build; no public pre-built release as of July 2026.
4. **Caption contest images:** New Yorker dataset is multimodal; text-only extraction discards visual humor context.
5. **Dad jokes deduplication:** Shuttie's deduplication is semantic (embedding-based); n-gram dupes may remain.
6. **SemEval format complexity:** Pun datasets in XML; requires parsing overhead.
7. **Jester only 100 jokes:** Do not use as primary corpus; only for rating/preference annotation.

---

## Next Steps for Implementation

1. **Download SocialGrep/Fraser/taivop immediately** — these are immediately available and constitute 1.4M baseline.
2. **Extract ChatGPT 25 jokes** — build reference corpus for 100% memorization check.
3. **Decide commercial licensing:** If commercialization likely, exclude taivop (research-only) and Oogiri (CC-BY-NC).
4. **De-duplicate:** Combine 1M Reddit sets and remove exact + fuzzy (n-gram/embedding) duplicates.
5. **Build n-gram + embedding lookups:** Use the combined corpus to score novelty in training loop.
6. **Evaluate diversity:** Use Oogiri-Corpus as reference for what "diverse, non-memorized" looks like.

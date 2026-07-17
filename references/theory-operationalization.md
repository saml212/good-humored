# Theory → Computable Quantity: An Operationalization Review

Owner's standing rule under test: every metric or reward term in this project must
trace to a theory construct; when a reward term is gameable, the fix should come
from theory, not from a patch. This file's mission is different from
`psychology.md` and `trajectory-grounding.md` — those establish *which theories
are real*; this one asks **who has already turned those theories into a number**,
so the benchmark and reward stack can reuse or deliberately improve on prior
computational work instead of inventing metrics from scratch.

Read first (not re-verified here, only extended): `README.md` ("Why humor" +
"Humor, honesty, and beauty"), `docs/BENCHMARK.md` §4, `references/README.md`,
`references/psychology.md`.

**Verification method:** every arXiv ID below was checked with
`curl -s -o /dev/null -w '%{http_code}' -L https://arxiv.org/abs/<id>` (200 unless
noted) and cross-read via direct WebFetch of the arXiv abstract page or a hosted
PDF/PMC full text where the fetch succeeded. Journal papers were checked against
their publisher DOI page and/or a PubMed/PMC record. Anything not independently
read at the primary-text level is flagged inline.

**Evidence-strength tags** (same convention as `humor-honesty-beauty.md`):
- **ESTABLISHED** — peer-reviewed, primary text read or formulas/numbers directly
  extracted, methodology sound for the claim made.
- **SUGGESTIVE** — real, verifiable paper/venue, but single-study, secondary-only
  read, or a narrower claim than the headline suggests.
- **SPECULATIVE** — the owner's synthesis or mine; plausible, not tested as stated.

Citations are **not yet added to `papers.bib`** — this file is self-contained
(full citation given inline at first mention), matching `humor-honesty-beauty.md`'s
convention rather than `psychology.md`'s cite-key convention. A future pass can
merge the confirmed entries into the shared bibliography.

---

## RQ1. Surprisal and incongruity as a computable, judge-free humor signal

This is the highest-priority question because a surprisal-based term would be
**verifiable and judge-free** — no LLM-judge reward-hacking surface at all. The
literature is real and multi-decade, but the honest finding, replicated twice
independently below, is a load-bearing caveat: **surprisal/incongruity measures
discriminate joke-from-non-joke well, but do not reliably discriminate
funnier-from-less-funny among things that are already jokes.** That distinction
is the single most important thing this section has to offer the reward-design
conversation.

### 1.1 Kao, Levy & Goodman (2016) — the foundational information-theoretic model

**Kao, J. T., Levy, R., & Goodman, N. D. (2016). A Computational Model of
Linguistic Humor in Puns.** *Cognitive Science*, 40(5), 1270–1287.
doi:10.1111/cogs.12269. (Preprint title differs: "The Funny Thing About
Incongruity: A Computational Model of Humor in Puns," escholarship.org/uc/item/04j190sw.)
**ESTABLISHED** — verified via direct PMC full-text fetch (PMC5042108).

- Two information-theoretic measures derived from a noisy-channel sentence
  comprehension model:
  - **Ambiguity**: `Amb(M) = −Σ P(mₖ|w→) log P(mₖ|w→)` — entropy over the
    distribution of possible meanings given the sentence.
  - **Distinctiveness**: `Dist(Fₐ,F_b) = Σᵢ ln[Fₐ(i)/F_b(i)]Fₐ(i) + ln[F_b(i)/Fₐ(i)]F_b(i)`
    — symmetrized KL divergence between the word distributions that support
    each of the two meanings.
- Across all 435 sentences (145 puns + 290 non-pun controls; N=100 raters on
  identical-homophone puns, N=160 on near-homophone puns; funniness rated 1–7,
  z-scored within participant): ambiguity regression coefficient = 1.915
  (p<.0001), distinctiveness coefficient = 0.264 (p<.0001), model R²=0.25,
  F(2,432)=74.07, p<.0001.
- **The load-bearing caveat**: *within puns only*, distinctiveness still
  predicts funniness (r=.28, p<.001) but **ambiguity does not** (r=.03,
  p=.697). Ambiguity is a strong pun-vs-non-pun discriminator but contributes
  ~nothing to ranking funnier vs. less-funny puns once you're already looking
  at puns.

**ACTIONABILITY:** (a) cheap verifiable reward term — distinctiveness-style KL
divergence between two candidate-meaning word distributions is computable from
any embedding model or LM without a judge; needs a way to identify the "two
readings" automatically for non-pun humor (the paper's method is pun-specific,
porting it is nontrivial). (c) analysis lens — the ambiguity/distinctiveness
split is a good post-hoc feature for auditing *why* a novelty-only reward
term might reward joke-ness without funniness.

### 1.2 Trott, Walker, Taylor & Coulson (2025) — the direct replication of the caveat with modern LLMs

**Trott, S., Walker, D. E., Taylor, S. M., & Coulson, S. (2025). Turing Jest:
Distributional Semantics and One-Line Jokes.** *Cognitive Science*.
doi:10.1111/cogs.70066. **ESTABLISHED** — verified via direct PMC full-text
fetch (PMC12093450).

- Surprisal = negative log-probability of the sentence's final word, computed
  via GPT-3 (`text-davinci-002`) plus open-source Llama-3 and Mixtral as
  robustness checks. Materials: 400 sentences (80 Expected, 160 Jokes, 160
  "Straight" counterparts), adapted from Coulson & Lovett (2004). Human N=167
  (Study 1) / 160 (Study 2).
- **The finding that independently replicates Kao et al.'s within-puns
  null result, on different materials and a different (LLM) surprisal
  estimator, a decade later**: a model with an Is-Joke × Surprisal
  interaction fits better than main effects alone, because "final-word
  Surprisal was associated with higher funniness ratings for nonjoke
  materials, but not for statements correctly classified as jokes." I.e.,
  **once something is a joke, how surprising its punchline is stops
  predicting how funny it is** — the opposite of the naive "more surprisal =
  funnier" assumption a reward-design conversation might reach for first.
- Separately (a different measurement, not surprisal): zero-shot GPT-3
  funniness ratings correlate with mean human ratings at **Pearson r=.47** —
  a modest LLM-judge/human agreement figure, useful as a baseline number for
  how far an LLM judge alone gets before any decomposition.

**ACTIONABILITY:** (a) — this is the strongest reason **not** to build a
naive "reward = punchline surprisal" term as the primary signal; two
independent, decade-apart, differently-instrumented studies say it won't
discriminate quality among jokes. It's better used as (c) a **joke-detection
gate** (is this even attempting incongruity) than as a funniness-magnitude
reward. (b) benchmark metric — r=.47 GPT-3-vs-human agreement is a citable
floor number for "how good is a bare LLM judge" to compare any decomposed
reward stack against.

### 1.3 Xie, Li & Pu (2021) — GPT-2 uncertainty+surprisal for humor *recognition*, not rating

**Xie, Y., Li, J., & Pu, P. (2021). Uncertainty and Surprisal Jointly Deliver
the Punchline: Exploiting Incongruity-Based Features for Humor Recognition.**
ACL-IJCNLP 2021 (ACL Anthology 2021.acl-short.6). arXiv:2012.12007 (HTTP 200).
**SUGGESTIVE** — abstract-level verified via WebFetch; full performance table
not independently re-extracted.

- Models jokes as setup (develops semantic **uncertainty**) + punchline
  (delivers **surprisal**, i.e., disrupts expectation), both computed via
  GPT-2 conditional probabilities. Evaluated on SemEval-2021 Task 7
  (classification: is this text a joke). This is a **detection task**, not a
  funniness-magnitude regression — important not to conflate with 1.1/1.2's
  rating-correlation framing when citing this paper.

**ACTIONABILITY:** (c) analysis lens / cheap pre-filter — a fast, judge-free
"does this look like it's attempting the two-stage setup→punchline
incongruity structure at all" gate, distinct from and complementary to a
funniness-magnitude signal (which 1.1/1.2 say this kind of feature can't
supply).

### 1.4 He, Peng & Liang (2019) — surprisal used to *generate*, not just detect

**He, H., Peng, N., & Liang, P. (2019). Pun Generation with Surprise.** NAACL
2019. arXiv:1904.06828 (HTTP 200). **ESTABLISHED** — abstract/method verified
via WebFetch.

- The "local-global surprisal principle": the pun word should associate
  strongly with the *distant* context while the alternative (non-pun) word
  associates with the *immediate* context, operationalized as a probability-
  ratio contrast measure from a language model, with a retrieve-and-edit
  generation procedure guided by a skip-gram model.
- Human eval: the retrieve-and-edit approach generates successful puns **31%
  of the time — roughly 3× a neural generation baseline's success rate.**

**ACTIONABILITY:** (a) — this is the clearest existing precedent for using a
surprisal-shaped signal as a **generation-guiding** mechanism rather than a
post-hoc correlational feature, i.e., precedent that this family of measure
can do more than describe — it can steer sampling/generation. Directly
relevant if the reward stack wants a surprisal-shaped *shaping* term rather
than a scalar reward (a distinction the RL literature draws between reward
and potential-based shaping).

### 1.5 Ma, Peng, Lyu, Zhang & Zhu (2026) — timing beats average surprisal, and it's the *peak*, not the average

**Ma, Y., Peng, Y., Lyu, J., Zhang, C., & Zhu, Y. (2026). Timing is
Everything: Temporal Scaffolding of Semantic Surprise in Humor.** CogSci 2026
(submitted 2026-04-30). arXiv:2605.00143 (HTTP 200). **SUGGESTIVE** — recent,
single study, abstract/method-level verified via WebFetch; exact effect sizes
not independently extracted (not given at abstract level).

- Dual Prediction Violation (DPV) framework on 828 professional Chinese
  stand-up performances (spoken, not text). Central finding: **temporal
  features (pause length before the punchline) substantially outweigh
  semantic incongruity magnitude** in predicting audience appreciation, and
  **peak** semantic violation (not average violation across the bit) is what
  matters — pauses strategically lengthen specifically before high-surprise
  punchlines.

**ACTIONABILITY:** (c), with a hard caveat. Two design lessons transfer even
though the underlying study is spoken/performed, not text: (1) **don't
average surprisal across a whole generation — score the peak/local spike at
the punchline span specifically**, matching the setup→punchline two-stage
structure this project already cites (Suls 1972); (2) a text-only chatbot
reward is structurally blind to the timing channel this paper says dominates
— worth stating explicitly as a known gap in Track 1 (single-turn joke) and
partially addressed by Track 2's turn-taking structure but not by anything
sub-turn (there's no text analog to "the pause before the punchline" unless
the model is allowed to control message chunking/delivery).

### 1.6 Word-level and caption-level corroboration (lower unit of analysis, still real)

- **Engelthaler, T., & Hills, T. T. (2018). Humor norms for 4,997 English
  words.** *Behavior Research Methods*, 50(3), 1116–1124.
  doi:10.3758/s13428-017-0930-6. PMID 28710716. **ESTABLISHED** (existence,
  method, and headline correlation cross-confirmed across PubMed/Springer/
  ResearchGate, not independently read at full-text level — tag reflects that
  the core numbers below are corroborated across independent secondary
  sources with matching specifics). N=821 raters, 4,997 words, ~35 ratings/word.
  Word funniness correlates with **inverse word frequency at r=−.42** (rarer
  words funnier) and with lexical decision time (processing difficulty) as
  the two strongest predictors; weak correlation with valence/arousal/
  concreteness. This is a surprisal-*adjacent* (frequency-based) signal at
  the single-word level, not the punchline level — a narrower and much
  cheaper-to-compute cousin of 1.1/1.2's setup→punchline surprisal.
- **Shahaf, D., Horvitz, E., & Mankoff, R. (2015). Inside Jokes: Identifying
  Humorous Cartoon Captions.** KDD 2015. doi:10.1145/2783258.2783388.
  **SUGGESTIVE** — existence, venue, and headline numbers confirmed via
  multiple independent secondary sources (Microsoft Research listing, ACM DOI
  page, search-engine synthesis of the abstract); the hosted PDF did not
  render as extractable text for this pass, so the *exact* weight/importance
  of the perplexity feature specifically (vs. sentiment/readability/keyword
  features) is not independently confirmed — re-verify from the PDF before
  citing a specific number. Headline: supervised funnier-of-pair classifier
  using perplexity + sentiment + readability + image-anomaly-keyword-overlap
  features reaches **64% accuracy vs. 55% for a bag-of-words baseline.**
  **Important disambiguation**: this is a *different* paper from Hurley,
  Dennett & Adams's *Inside Jokes* (MIT Press, 2011) already cited in
  `humor-honesty-beauty.md` — identical title, unrelated work. Do not
  conflate the two in any future citation.

**ACTIONABILITY (both):** (c) — cheap component features for a decomposed
reward or for post-hoc auditing, not standalone reward terms; both are
corpus-level correlational findings, not generation-time signals.

### Cross-cutting pattern for RQ1

Three independent lines (Kao et al. 2016 within-puns; Trott et al. 2025
within-jokes; and the general shape of "surprisal is a good detector feature,
not validated as a rating-magnitude feature" running through 1.3) converge on
the same caveat from different data and different eras. **A pure
punchline-surprisal reward term is well-motivated by incongruity theory as a
*joke-detector*, and there is no verified evidence it works as a
*funniness-magnitude* signal once you already have a joke.** Any reward-stack
design that uses surprisal should use it as a gate/filter term (is this
attempting the incongruity structure at all — 0 reward if not) rather than as
a monotonic "more surprisal = more reward" term.

---

## RQ2. Computational Benign Violation Theory — is there empirical basis for a multiplicative reward?

**Short answer: no direct test found, in either direction.** `psychology.md`
already establishes BVT and incongruity-resolution as real, dual-appraisal,
simultaneity-gated theories (McGraw & Warren 2010; Suls 1972) — that part is
settled and not re-verified here. What was searched for specifically and not
found: any paper that implements "violation score × benignness score" as an
explicit product, tests it against an additive/averaged combination, and
reports which fits human funniness judgments better.

**Search trail (all empty for the specific multiplicative-vs-additive test):**
"computational operationalize benign violation theory violation score
benignness score product LLM classifier", "benign violation theory
computational model multiplicative violation benignness" — both returned only
BVT's own theoretical statements and unrelated LLM-safety "benign vs.
violation" jailbreak papers (different, unrelated use of the word "benign").

### What exists instead: the one paper that tried to computationally combine humor theories chose additive, not multiplicative

**De Marez, V., Winters, T., & Rigouts Terryn, A. (2024). THInC: A
Theory-Driven Framework for Computational Humor Detection.** CREAI 2024
(workshop). arXiv:2409.01232 (HTTP 200). **SUGGESTIVE** — abstract/method
verified via WebFetch; which specific humor theories are ensembled (does it
include BVT specifically) not independently confirmed from the abstract
alone — flag before citing that detail further.

- Ensembles interpretable **GA2M** classifiers (generalized additive models
  with pairwise interactions — `g(E[y]) = β₀ + Σᵢfᵢ(xᵢ) + Σᵢ≠ⱼfᵢⱼ(xᵢ,xⱼ)`),
  each mapping engineered proxy features onto a specific humor theory, F1=0.85
  overall.
- **This is directly relevant, if indirect, evidence for RQ2**: the one
  published attempt to formally combine multiple humor-theory-derived signals
  into a single computational score picked an **additive-with-limited-
  pairwise-interactions** architecture (GA2M), not a clean multiplicative
  gate. GA2M's pairwise interaction terms *can* approximate multiplicative-
  like behavior for specific feature pairs, so this isn't a clean refutation
  of multiplicative gating — but it is evidence that the field's actual
  engineering choice, when someone built the thing, was additive-plus-
  interactions rather than a two-signal product.

### A third, structurally different unification attempt (neither additive nor multiplicative)

**Safron, A. (2019). Rapid Anxiety Reduction (RAR): A unified theory of
humor.** arXiv:1911.02364 (HTTP 200). **SPECULATIVE** — a single-author
theory-synthesis preprint (q-bio.NC), not an empirical test.

- Proposes humor as the *rate* of anxiety reduction, formalized as `−dA/dt`
  (negative derivative of anxiety w.r.t. time), explicitly drawing on BVT and
  "Cognitive Debugging Theory" (i.e., the Hurley/Dennett/Adams framing already
  in `humor-honesty-beauty.md`). Worth knowing this exists as a third
  candidate mathematical unification — a rate-based formulation is neither
  additive nor multiplicative, it's a different modeling choice entirely
  (dynamics of a single quantity over time, not a combination of two static
  signals). Not empirically validated against human data in what was
  reviewed here.

**ACTIONABILITY for RQ2 overall:** (a)/(b) — building a genuine violation × 
benignness multiplicative reward term, and testing it against an additive
baseline on a real funniness-rating dataset (e.g., Oogiri-GO or NYCC), **would
be a novel empirical contribution**, not a replication of existing work — the
search trail above is the documentation that nothing was found either
confirming or refuting the multiplicative hypothesis computationally. Needs:
a violation detector (norm-expectation classifier) and a benign/safety
detector (harm/severity classifier) as two separate reference signals, plus a
human-rated funniness dataset to fit/compare architectures against. **Flag
this explicitly as a NOVELTY CLAIM** if built — the theoretical convergence
(McGraw & Warren + Suls, already documented in `psychology.md`) motivates the
architecture, but no one has tested whether it's empirically better than
additive.

---

## RQ3. Economy, brevity, and punchline position as quantified predictors

Real, but thinner and lower-rigor than RQ1/RQ4 — the strongest quantitative
entry here is actually 1.5 (Ma et al. 2026, timing/peak-not-average), already
covered above and cross-referenced rather than repeated.

- **Hempelmann, C. F., Taylor Rayz, J., & Raskin, V. (2012). Tightening up
  Joke Structure: Not by Length Alone.** CogSci 2012 proceedings (existence
  confirmed via Semantic Scholar, escholarship.org/uc/item/5m55g73k listing,
  and the CogSci mindmodeling.org 2012 program; primary PDF returned HTTP 403
  on two fetch attempts during this pass — **not independently read,
  SUGGESTIVE/UNVERIFIED-SECONDARY for its specific argument**). Title alone
  indicates the paper's thesis runs *against* a naive "shorter = funnier"
  account, arguing (per Ontological Semantic Theory of Humor, the
  Raskin/Attardo lineage already noted in `psychology.md`'s GTVH entry) that
  structural script-opposition, not raw length, is what a minimal joke
  actually needs. **Re-verify from the primary PDF before citing a specific
  claim beyond the title's thesis.**
- **McGraw, Mankoff & Fernbach caption-brevity analysis** (Peter McGraw's Humor
  Research Lab / CU Boulder press coverage, 2011; not found as a peer-reviewed
  paper with an independent DOI in this search pass). Analysis of 5,291
  captions from New Yorker Caption Contest No. 281: finalist captions were
  reliably shorter, avoided question marks/commas/exclamation points, and
  favored words rare in the submission pool (novelty) — "be novel, be brief,
  avoid punctuation" per Fernbach's own summary. **Flagged as low-rigor,
  illustrative only** — same tier as `psychology.md`'s treatment of LaughLab:
  real, quotable, methodologically opaque (no controls, no significance
  tests visible in secondary reporting), not citable-grade for a paper, fine
  as a design heuristic.
- **Radev, D., Stent, A., Tetreault, J., Pappu, A., et al. (2016). Humor in
  Collective Discourse: Unsupervised Funniness Detection in the New Yorker
  Cartoon Caption Contest.** LREC 2016 (ACL Anthology L16-1076).
  arXiv:1506.08126 (HTTP 200). **SUGGESTIVE** — headline finding
  cross-confirmed via search synthesis, not independently read at primary-text
  level. Reports that **negative sentiment, human-centeredness, and lexical
  centrality** (proximity to the "average" caption in embedding space) most
  strongly separate funniest from non-funniest captions — a different (and
  more rigorously-sourced) feature set than the brevity claim above; length
  is not the headline feature in this peer-reviewed version, which is a
  useful check against over-trusting the press-covered brevity claim.

**ACTIONABILITY:** (b)/(c) — brevity/punctuation-avoidance are cheap,
computable style features usable as a minor auxiliary signal or as an
analysis lens (are the model's jokes needlessly padded compared to
human-authored ones in the memorized-joke corpus), but the evidence quality
here is the weakest of the six research questions — do not build a primary
reward term on the brevity claim specifically without first re-deriving it
from Radev et al.'s actual peer-reviewed feature-importance numbers (not yet
independently extracted here) or Hempelmann et al.'s primary text.

---

## RQ4. Semantic vs. surface novelty — what would survive a 2-word reskin

Directly motivated by the project's own finding (tonight) that the current
n-gram novelty penalty is evadable by minor reskins. Good news: the
creativity-measurement literature has already built, and partly validated,
exactly the class of paraphrase-robust measure this project needs — none of
it is humor-specific yet, which is the actionability gap to flag.

### 4.1 Divergent Semantic Integration (DSI) — the best-validated embedding novelty measure, but for narratives, not jokes

**Johnson, D. R., Kaufman, J. C., Baker, B. S., Patterson, J. D., Barbot, B.,
Green, A. E., van Hell, J., Kennedy, E., Sullivan, G. F., Taylor, C. L., Ward,
T., & Beaty, R. E. (2022). Divergent semantic integration (DSI): Extracting
creativity from narratives with distributional semantic modeling.**
*Behavior Research Methods*, 54(6). doi:10.3758/s13428-022-01986-2.
**ESTABLISHED** — verified via direct PMC full-text fetch (PMC10615993).

- DSI = mean pairwise semantic (cosine) distance between all word-embedding
  pairs in a text — the more a piece of text connects semantically distant
  concepts, the higher its DSI.
- **BERT-based DSI explained up to 72% of variance in human creativity
  ratings** (Study 2 latent-variable correlation r=.85); Study 1 correlation
  r=.77, approaching the human single-rater reliability ceiling of r=.84.
- Validated across nine studies, 27 prompts, 3,500+ narratives — but
  specifically on **short narratives (59–200 word averages)**; predictive
  power diminished on longer stories, and **the validation set is
  narrative-composition tasks, not jokes or one-liners.** This is an
  important scope caveat before assuming it ports.

**ACTIONABILITY:** (a)/(b) — this is the single best-validated embedding-
based novelty measure found in this search, and BERT-DSI is directly
computable (no judge, no reference corpus needed beyond an embedding model)
— makes it the strongest candidate to prototype as a novelty-penalty
*replacement or supplement*. Needs: validation specifically on short comedic
text before trusting the 72%-variance number to transfer (unvalidated
assumption otherwise) — this is exactly the missing piece a small internal
study could supply (score a set of Oogiri-GO or memorized-joke-corpus items
with DSI, correlate against their existing human ratings).

### 4.2 Forward Flow — the free-association-distance measure, weaker replication record

**Gray, K., Anderson, S., Chen, E. E., Kelly, J. M., Christian, M. S.,
Patrick, J., Kaufman, S. B., Benedek, M., & Runco, M. A. (2019). "Forward
Flow": A New Measure to Quantify Free Thought and Predict Creativity.**
*American Psychologist*. doi:10.1037/amp0000391. **SUGGESTIVE** — headline
method and findings cross-confirmed across HBS, Semantic Scholar, and
ResearchGate secondary sources, not independently read at full-text level.

- Uses latent semantic analysis over a chained free-association task (start
  with a cue word, keep producing the next word that comes to mind) to
  measure how far semantic distance travels over the chain. Predicts
  creativity in student and general-population samples, and predicts
  membership in creative professions (performers, actors, entrepreneurs) even
  controlling for standard divergent-thinking tests.
- **Important caveat found during this pass**: a published comment
  ("Forward flow — an alternative interpretation: Comment on Gray et al.
  2019," Charles Sturt University) exists specifically disputing the
  measure, and independent secondary reporting notes forward flow shows
  **only weak correlations with some standard creativity measures** — flag
  this as a real, if secondary-sourced, contested-replication caveat, not
  settled science, before leaning on forward flow over DSI.

**ACTIONABILITY:** (c) — a plausible secondary lens (does a model's *sequence*
of jokes under the rejection cascade travel further in semantic space over
time, direct methodological kinship to this project's own
patch-foraging/trajectory metrics in `trajectory-grounding.md`), but the
comment/critique above means it should not be the *primary* novelty metric —
DSI (4.1) has the stronger, less-disputed validation record.

### 4.3 The paraphrase-robustness problem, independently confirmed by others tonight

- **Lu, L.-C., Liu, M., Lu, P.-C., Tian, Y., Sun, S.-H., & Peng, N. (2026).
  Rethinking Creativity Evaluation: A Critical Analysis of Existing
  Creativity Evaluations.** EACL 2026. arXiv:2508.05470 (HTTP 200).
  **ESTABLISHED** — abstract/method verified via WebFetch. Directly
  corroborates, from an independent paper, tonight's own discovery: examining
  perplexity, LLM-as-Judge, the n-gram "Creativity Index," and syntactic
  templates across creative-writing/problem-solving/ideation tasks, finds
  "limited consistency both across domains and metrics" — the n-gram
  Creativity Index metric "primarily measur[es] lexical diversity, with high
  sensitivity to implementation choices" (i.e., exactly gameable-by-reskin
  the way this project already found), and perplexity "reflect[s] fluency
  rather than novelty" (a second, independent warning against the RQ1
  surprisal-as-novelty conflation). LLM-as-Judge shows inconsistent judgments
  under prompt variation, undercutting a judge-only novelty check too.
- **Davydov, P., Prabhu, A., Bethge, M., Nguyen, E., & Oh, S. J. (2025–2026).
  LLM generation novelty through the lens of semantic similarity.**
  arXiv:2510.27313 (HTTP 200). **SUGGESTIVE** — abstract/method verified via
  WebFetch; not yet peer-reviewed, but methodologically the most directly
  actionable candidate found: a three-stage semantic-retrieval pipeline
  (retrieve semantically similar samples, rerank at multiple subsequence
  lengths, calibrate against a human novelty reference) explicitly built to
  catch paraphrased reuse that lexical-overlap metrics miss, released with
  ~20TB of corpus/tooling artifacts.
- **Tan, M. S., Choy, Z. K. C., Alsagoff, S. A. R., Wangsajaya, N. Y.,
  Banerjee, M., Saikia, S. B., & Chan, A. (2026). Automated Creativity
  Evaluation of Language Models Across Open-Ended Tasks.** ACL 2026 (Main).
  arXiv:2606.11762 (HTTP 200). **SUGGESTIVE** — abstract/method verified via
  WebFetch; exact validation correlation numbers not obtainable at
  abstract-fetch level (flag before citing a specific figure). Proposes
  **semantic entropy** — map each generation to a semantic label via an LLM
  judge, then compute Shannon entropy over the label distribution — as a
  reference-free novelty/diversity metric, validated (per the paper's own
  framing) against human annotations, LLM-based novelty judgments, and
  baseline diversity measures. Tested on problem-solving (MacGyver), research
  ideation (HypoGen), and creative writing (BookMIA) — **not on humor/jokes.**
  Directly relevant to this project's `intra_group_diversity_reward` design:
  if an LLM-judge semantic clustering step is already budgeted for
  (comparable cost to the existing judge calls), entropy over cluster labels
  is a paraphrase-robust drop-in replacement for n-gram distinct-k counting.

**ACTIONABILITY (4.3 overall):** (a) — of everything reviewed in this whole
document, **the semantic-similarity-based novelty pipeline (Davydov et al.)
and semantic-entropy-over-LLM-judge-clusters (Tan et al.) are the two most
directly buildable replacements for the evadable n-gram novelty penalty**,
because both are explicitly paraphrase-robust by construction and both have
at least some human-judgment calibration built into their own validation.
Neither has been validated on humor specifically — that gap is this
project's to fill, not to assume away.

---

## RQ5. Audience adaptation / Theory-of-Mind-linked humor production

The negative result already in this corpus (`negative-results.md` §2b,
`zhou2025bridging`: persona-*prompting* failed to adapt caption rankings to
subgroup preferences) is the load-bearing finding here and is **not
re-verified, only extended** with two more precedents that sharpen the
methodological contrast.

- **Garimella, A., Banea, C., Hossain, N., & Mihalcea, R. (2020). "Judge me
  by my size (noun), do you?" YodaLib: A Demographic-Aware Humor Generation
  Framework.** COLING 2020 (aclanthology.org/2020.coling-main.253).
  arXiv:2006.00578 (HTTP 200). **ESTABLISHED** — abstract/method verified via
  WebFetch. Pre-LLM (BERT-era) system that *fine-tunes* on location-specific
  word-filling for Mad Libs-style stories to generate humor tailored to a
  target demographic; reported to outperform a prior semi-automated approach
  and to surpass human annotators on this specific narrow task, per the
  paper's own reporting (not independently re-derived from a
  results table here — **SUGGESTIVE tier for the specific "beats humans"
  claim**, ESTABLISHED for the paper's existence/method).
  **The useful methodological contrast this adds to `zhou2025bridging`**:
  YodaLib *trains in* demographic conditioning (fine-tuning on
  demographic-labeled data) and reportedly works on its narrow task, while
  `zhou2025bridging` shows *prompting* a model to adopt a persona fails to
  adapt to subgroup preference — consistent with this project's own
  `negative-results.md` §2b lesson that persona-prompting is the cheap thing
  everyone tries first and it doesn't work; the fix implied by contrasting
  these two papers is **train the conditioning in, don't prompt for it.**
- **Meaney, J. A., Wilson, S., Chiruzzo, L., & Magdy, W. (2022). Don't Take
  it Personally: Analyzing Gender and Age Differences in Ratings of Online
  Humor.** SocInfo 2022. arXiv:2208.10898 (HTTP 200). **ESTABLISHED** —
  abstract/method verified via WebFetch. Documents rater-level demographic
  effects on humor/offense judgments specifically (women give lower humor
  ratings and higher offense ratings than men for the same items; the
  humor-offense correlation strengthens with age). This is *audience
  variation in rating*, not *model adaptation to audience* — it grounds why
  a reward model blind to rater demographics is a design risk, complementing
  `psychology.md`'s `cao2023culture` (culture × social-distance) and
  `zhang2024diverging` (annotator disagreement taxonomy) entries already in
  this corpus, rather than adding a new claim about generation.

**No paper was found** that builds the specific thing Track 1b's
context-ablation design implies as a longer-term goal: a computational
harness that (1) generates the *same* joke/response conditioned on N
different explicit audience personas, (2) scores funniness per-persona, and
(3) correlates the resulting adaptation-sensitivity with an independent
ToM benchmark score (ToMBench, ToM-HCAT) for the same model. HumorBench and
ToM-HCAT/ToMBench (already verified in `humor-honesty-beauty.md` §3.3) test
*comprehension*, not *audience-conditioned production*; YodaLib and
`zhou2025bridging` test *adaptation*, but neither correlates it against an
independent ToM measure. **Search trail:** "audience-conditioned humor persona
theory of mind LLM joke generation same joke different audience",
"theory of mind humor production ability LLM benchmark measure mentalizing
joke generation", "audience adaptation computational humor rating different
demographic persona same joke rated differently NLP" — all returned
comprehension benchmarks, persona-prompting studies, or rater-disagreement
studies, never the three-part combination above in one paper.

**ACTIONABILITY:** (b) — the missing combination (production, per-audience
scored, correlated against an independent ToM score) is a genuinely open
benchmark-metric opportunity, closely related to but distinct from this
project's own Track 2 context-ablation delta (`docs/BENCHMARK.md` §1b), which
already measures context-sensitivity but not ToM-correlated *audience*
sensitivity specifically. Needs: a persona/audience-conditioning dataset
(demographic or explicit-preference-labeled), a held-out ToM benchmark score
per model, and a way to fine-tune (not merely prompt) the conditioning per
the YodaLib-vs-`zhou2025bridging` lesson above.

---

## RQ6. Foraging / Marginal Value Theorem on LLM *production* sequences

**No precedent found for applying the MVT patch-departure statistic itself to
LLM production sequences** (joke sequences, dialogue turns, or chain-of-thought
steps as the forager's own output, rather than using an LLM to *model human*
foraging data). This appears to be a clean, well-documented gap — exactly
consistent with what the cascade benchmark would be first to formalize as an
analysis lens on a model's own generated trajectory, which is the framing
`trajectory-grounding.md` already uses; this section's job was specifically
to check whether an LLM-production-sequence application already exists
elsewhere, and it does not appear to.

**Search trail (four distinct queries, all empty for the specific claim):**
1. "marginal value theorem patch departure LLM text generation semantic
   search" → returned classic foraging-ecology and human-semantic-search
   papers only (Hills, Jones & Todd 2012 lineage, already in
   `trajectory-grounding.md`).
2. "marginal value theorem LLM chain-of-thought reasoning steps patch
   foraging analysis exploration exploitation" → returned MVT
   biology/neuroscience papers (rat ACC decision variables, foraging-under-
   uncertainty Bayesian updating) with no LLM connection.
3. "optimal foraging theory applied to large language model outputs
   generation exploration exploitation tradeoff" → returned generic
   RL-for-LLM exploration-exploitation framing (multi-armed bandit problems,
   arXiv:2501.08925 "Disentangling Exploration of Large Language Models by
   Optimal Exploitation") — real and adjacent, but framed as bandit
   exploration, not MVT-specific patch-departure statistics, and not applied
   to a model's own multi-turn output sequence.
4. "'topic exhaustion' OR 'patch departure' language model generation
   sequence dialogue novel" → returned Stack Overflow topic-exhaustion
   studies (unrelated domain) and program-repair "patch generation" (a
   different, unrelated sense of "patch") — no hit on the intended sense.

**What does exist, and is already cited in `trajectory-grounding.md`, is
LLMs used to *model human* fluency/foraging data** (arXiv:2208.09719,
"Cognitive Modeling of Semantic Fluency Using Transformers"; arXiv:2511.12759,
"Optimal Foraging in Memory Retrieval: Evaluating Random Walks and
Metropolis-Hastings Sampling in Modern Semantic Spaces") — this is the
inverse direction (LLM as a model of a human forager) from what RQ6 asked
about (LLM's own output sequence treated as the forager).

**ACTIONABILITY:** (c), and **explicitly flag as a NOVELTY CLAIM if built**.
Treating a model's own topic-trajectory under the rejection cascade (or a
long multi-turn banter session) as an MVT-testable forager — comparing its
per-topic "patch residence" and departure timing against the running average
"yield" the way Hills et al. (2012) do for human verbal fluency — has no
found LLM-production precedent. This is exactly what `docs/BENCHMARK.md`'s
cascade already does at the metric-design level (`trajectory-grounding.md`
§6, metric #3); this section's contribution is confirming, with a documented
search trail, that the *novelty claim* the project would want to make about
this ("first to apply patch-departure MVT analysis to a model's own
production sequence") survives a dedicated check, not just an assumption.

---

## Summary table

| # | Finding | Evidence | RQ | Actionability |
|---|---|---|---|---|
| 1 | Ambiguity/distinctiveness (KL-divergence) predict pun funniness r²=.25 overall, but ambiguity ≈0 *within* puns | ESTABLISHED | 1 | (a) gate, not magnitude signal |
| 2 | GPT-3/Llama-3 surprisal predicts funniness for non-jokes, not for jokes (independent replication of #1's caveat) | ESTABLISHED | 1 | (a)/(b) same caveat, cross-validated |
| 3 | GPT-2 uncertainty+surprisal improve joke/non-joke *classification*, not rating | SUGGESTIVE | 1 | (c) pre-filter gate |
| 4 | Surprisal-shaped local-global contrast *generates* puns, 3× baseline success | ESTABLISHED | 1 | (a) shaping signal, not just scoring |
| 5 | Punchline *timing* (peak, not average, incongruity + pause length) outweighs semantic surprisal in predicting appreciation | SUGGESTIVE | 1/3 | (c) score peak not mean; text-only reward is blind to timing channel |
| 6 | Word-frequency (surprisal-adjacent) r=−.42 with word funniness | ESTABLISHED (word-level) | 1 | (c) cheap component feature |
| 7 | Perplexity + sentiment + readability features: 64% vs 55% baseline for funnier-caption classification | SUGGESTIVE | 1/3 | (c) component features |
| 8 | No computational test found of violation×benignness (multiplicative) vs. additive combination | Gap — search trail documented | 2 | (a)/(b) build + test — NOVEL if done |
| 9 | The one theory-ensembling humor classifier built (THInC) uses additive GA2M, not a multiplicative gate | SUGGESTIVE | 2 | (c) counter-evidence to assumed multiplicative default |
| 10 | RAR: humor as −dA/dt, a third (dynamical) unification, unvalidated | SPECULATIVE | 2 | none yet — theory only |
| 11 | Brevity/no-punctuation predicts NYCC finalist selection (press-reported, not peer-reviewed) | SUGGESTIVE, low-rigor | 3 | (c) weak auxiliary signal only |
| 12 | Negative sentiment / human-centeredness / lexical centrality (not length) separate funniest NYCC captions (peer-reviewed) | SUGGESTIVE | 3 | (c) — contradicts over-trusting the brevity claim |
| 13 | BERT-based DSI explains 72% of variance in narrative creativity ratings | ESTABLISHED (narratives, not jokes) | 4 | (a)/(b) best embedding-novelty candidate, needs humor validation |
| 14 | Forward flow predicts creativity/creative-profession membership, but disputed/weak-correlation critique exists | SUGGESTIVE, contested | 4 | (c) secondary lens only |
| 15 | n-gram Creativity Index is gameable/implementation-sensitive; perplexity measures fluency not novelty (independent confirmation of this project's own finding) | ESTABLISHED | 4 | validates need for a fix, doesn't supply one |
| 16 | Semantic-retrieval-based novelty pipeline, paraphrase-robust by design, human-calibrated | SUGGESTIVE (unpublished/2026) | 4 | (a) direct n-gram-penalty replacement candidate |
| 17 | Semantic entropy over LLM-judge clusters as reference-free novelty/diversity metric | SUGGESTIVE | 4 | (a) direct distinct-k replacement candidate |
| 18 | Persona *fine-tuning* (YodaLib) reportedly works on a narrow task; persona *prompting* (`zhou2025bridging`, already in corpus) fails | ESTABLISHED / already-corpus | 5 | (c) train the conditioning in, don't prompt for it |
| 19 | Rater gender/age shift humor & offense ratings on the same items | ESTABLISHED | 5 | (c) reward-model design risk, complements existing psychology.md entries |
| 20 | No paper found combining audience-conditioned production + ToM-benchmark correlation | Gap — search trail documented | 5 | (b) open benchmark-metric opportunity |
| 21 | No paper found applying MVT patch-departure analysis to LLM production sequences (only to LLM-as-model-of-human-forager) | Gap — search trail documented | 6 | (c) — supports project's own novelty claim |

---

## Ranked shortlist: what's most worth building for THIS project

Judged on verifiability (judge-free or cheaply judge-assisted), cheapness
(compute + engineering cost), and theory fidelity (traces to a real
construct, not a vibe).

1. **Peak-not-average punchline surprisal as a joke-detection GATE, not a
   funniness reward** (RQ1, findings #1/#2/#5). Cheapest possible signal — a
   reference LM's log-prob on the punchline span, nothing else needed — and
   the most rigorously *cross-validated negative result* in this whole
   review (two independent studies, decade apart) tells us exactly how not
   to misuse it. Build it as a zero/nonzero gate feeding into the existing
   comprehensibility term, not as a scalar to maximize.

2. **Semantic-entropy-over-LLM-judge-clusters as the novelty/diversity term**
   (RQ4, finding #17, secondarily #16). This is the most direct fix for
   tonight's documented reskin-evasion failure: it reuses the judge
   infrastructure the reward stack already pays for (an LLM call), is
   explicitly paraphrase-robust by construction (semantic clustering, not
   n-gram matching), and has at least a stated human-calibration claim.
   Needs a small validation pass on this project's own memorized-joke corpus
   before trusting it blind, since it hasn't been validated on humor
   specifically — but that validation pass is cheap (score a known set of
   reskinned duplicates, confirm they cluster together).

3. **BERT-based DSI as a secondary/audit novelty metric** (RQ4, finding #13).
   Second choice to #2 because it needs no LLM-judge call at all (pure
   embedding distance), making it strictly cheaper, but it is validated on
   narratives, not jokes, and the "does it transfer to short comedic text"
   question is open. Worth running in parallel with #2 as a cross-check
   rather than picking one — if they disagree systematically on the same
   corpus, that disagreement is itself informative about which failure mode
   each is catching.

4. **The BVT multiplicative-vs-additive reward architecture test** (RQ2,
   finding #8). Highest theory-fidelity of anything on this list — it
   directly tests the one specific, falsifiable, already-well-supported
   theoretical prediction (`psychology.md`'s existing McGraw & Warren / Suls
   entries) that this project's current *additive* reward stack design
   doesn't yet reflect. Costs more than #1–#3 (needs a violation detector, a
   benign/harm detector, and a real human-rated funniness dataset — Oogiri-GO
   is the obvious candidate, already in this corpus's Tier 1 datasets) but is
   the strongest candidate for an actual **novel empirical contribution**,
   not just an engineering fix, because the search trail above confirms no
   one has run this comparison.

5. **MVT patch-departure analysis of the rejection cascade's own topic
   trajectory** (RQ6). Already effectively the benchmark's own design
   (`trajectory-grounding.md` §6); listed here specifically because this
   review's job was to confirm the novelty claim survives a dedicated check,
   and it does — no LLM-production precedent was found despite four
   differently-worded searches. Lowest new-build cost of the five (the
   cascade transcripts already being collected are the only required input;
   this is an analysis added to data already being generated for other
   reasons), and it is the cleanest "we get to say this first, and we
   checked" claim of anything in this document.

**What did not make the cut, and why:** brevity/punctuation (RQ3) — evidence
too thin/press-level to build a reward term on without first re-deriving the
Radev et al. peer-reviewed numbers; forward flow (RQ4) — DSI is better-
validated and forward flow has an unresolved published critique; the
audience-adaptation × ToM-correlation combination (RQ5) — genuinely open and
theoretically clean, but it is a full benchmark-construction project on its
own scale (needs a persona-conditioned production dataset plus a ToM
benchmark run per model), not a near-term reward-stack addition.

---

## What contradicts or complicates something already in this project's corpus

- **Nothing found here contradicts an existing verified claim** in
  `psychology.md`, `negative-results.md`, or `humor-honesty-beauty.md`. Two
  findings **complicate** existing framing and should inform how those
  claims get used going forward:
  1. The RQ1 within-jokes surprisal-null-result (findings #1/#2) complicates
     any future temptation to treat "incongruity" as a single scalar the
     reward stack can just maximize — it's a detector, not a magnitude
     signal, a distinction `docs/BENCHMARK.md` §4 doesn't currently draw.
  2. THInC's additive-GA2M architecture (finding #9) is mild evidence *against*
     assuming the multiplicative BVT-inspired architecture is obviously
     correct just because the theory is dual-appraisal — worth stating
     explicitly if/when the project writes up the case for building #4 above,
     so the pitch doesn't overstate how settled the multiplicative case is.

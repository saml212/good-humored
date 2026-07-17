# Psychology / Sociology / Philosophy Grounding for the Rejection-Cascade Humor Benchmark

Research pass for: benchmark where a model is repeatedly asked for a joke, a rejector
says "not that topic, try another" for ~50 accumulating rounds, and we measure the
TOPIC TRAJECTORY (patch residence, switch timing, revisitation, exhaustion) under
forced switching. All citations below were checked via web search (existence of
title/venue/year/authors) and, for arXiv items, via
`curl -s -o /dev/null -w '%{http_code}' -L https://arxiv.org/abs/<id>` returning 200.
Nothing below is fabricated; anything I could not independently confirm is flagged
UNVERIFIED.

---

## 1. Semantic Foraging / Verbal Fluency (highest priority — direct human analog)

This is the closest existing human paradigm to our task: subjects asked to keep
producing items from a category (animals) until they run dry, forced to keep
switching sub-clusters as each one gets exhausted. Our benchmark replaces
"self-exhaustion of a category" with "externally forced exhaustion of a topic by a
rejector," which is actually a cleaner experimental manipulation than the classic
free-fluency task — closer to a constrained/adversarial foraging condition.

### 1.1 Core paper
**Hills, T. T., Jones, M. N., & Todd, P. M. (2012). Optimal foraging in semantic
memory. *Psychological Review*, 119(2), 431–440.** doi:10.1037/a0027373
- Verified: title/authors/venue/year/pages confirmed via multiple independent
  sources (PhilPapers, ResearchGate, MSU course PDF, Colorado course PDF hosting
  the full text).
- Core claim: in the verbal fluency task (name all the animals you can in N
  seconds), humans search memory the way animals forage in physical space —
  local search within a semantic "patch" (e.g., pets, farm animals, African
  safari animals) followed by a global jump to a new patch when the patch's
  yield rate drops. The timing of patch departure is well predicted by the
  **Marginal Value Theorem** (Charnov, 1976, ecology) applied to memory: leave a
  patch when its current (marginal) retrieval rate falls to the *average* rate
  across the whole task, not simply when it "feels empty."
- Directly gives us a principled null model: an LLM under forced switching should
  show patch-like clustering (runs of jokes on related sub-topics before a
  semantically distant jump), and we can ask whether patch length / switch
  timing looks like an MVT-consistent forager or like a memoryless random walk.

### 1.2 Original clustering/switching methodology
**Troyer, A. K., Moscovitch, M., & Winocur, G. (1997). Clustering and switching
as two components of verbal fluency: Evidence from younger and older healthy
adults. *Neuropsychology*, 11(1), 138–146.** doi:10.1037/0894-4105.11.1.138,
PMID: 9055277
- Verified via PubMed record and Google Scholar citation record (volume/issue/
  pages/DOI cross-checked).
- Decomposes fluency output into two scorable components: **clustering** (mean
  size of runs of semantically related items, e.g., consecutive "dog, cat,
  hamster" = one cluster of size 3) and **switching** (count of transitions
  between clusters). This is *the* standard clinical/cognitive metric pair for
  fluency trajectories (also used to differentiate frontal vs. temporal lesion
  patients, and Alzheimer's vs. Parkinson's populations — clustering/switching
  dissociate where raw word count does not).
- Related normative-data paper (large N, corrected for age/education/sex):
  **Troyer, A. K. (2000). Normative data for clustering and switching on verbal
  fluency tasks.** (PubMed record found; exact journal/volume not independently
  re-confirmed in this pass — flag as **UNVERIFIED-SECONDARY**, but existence of
  the Troyer-2000 norms is corroborated by multiple independent citing papers.)
- Follow-up dissociation/critique worth knowing about (shows the metric isn't
  unanimously settled — useful for our methods-limitations paragraph): a
  published comment, "On the dissociation between clustering and switching in
  verbal fluency: comment on Troyer, Moscovitch, Winocur, Alexander and Stuss,"
  *Neuropsychologia* (PubMed record confirmed, PMID 11749985; exact author/year
  not fully re-verified — **UNVERIFIED-SECONDARY**, cite cautiously or drop).

### 1.3 Marginal Value Theorem mechanics
- MVT (Charnov 1976, foraging ecology — pre-dates and is imported into the
  Hills et al. framework) prescribes: leave a patch when its instantaneous
  intake rate drops to the environment's average intake rate. Hills et al.
  (2012) empirically show human patch-departure timing in verbal fluency tracks
  this rule, and that people whose switching is *more* MVT-consistent recall
  *more* items overall — i.e., optimal foraging behavior is adaptive, not just
  descriptive.
- Competing/alternative account, useful to cite for balance: some later work
  argues similar cluster/switch *patterns* can emerge from a memoryless random
  walk on a semantic network without any strategic "decide to leave" computation
  (i.e., clustering could be an emergent property of network structure, not a
  foraging policy). This matters for us: if a model's topic trajectory *looks*
  like foraging, we should ask whether that's because of a policy-like
  computation or merely because embedding space has patch-like structure. Cite
  generically as an open debate in the memory-search literature rather than a
  single paper (avoids overclaiming a specific citation I didn't fully pin down
  — **UNVERIFIED-SECONDARY** as a single-paper claim, but the debate itself is
  well attested across multiple hits, e.g. work contrasting "optimal foraging"
  vs. "random walk on semantic network" accounts).

### 1.4 Very recent / directly relevant follow-on work (all arXiv-verified 200 OK)
These are recent papers extending the foraging framework toward exactly the kind
of trajectory analysis and even LLM comparison we need. Flagging as secondary
literature (not classic canon) but high relevance for method-porting.

- **Heineman, D., Koenen, R., & Varma, S. Towards a Path Dependent Account of
  Category Fluency.** arXiv:2405.06714. HTTP 200 verified.
  Argues fluency retrieval is better modeled with sequence-level (path-dependent)
  structure than isolated-response models, and proposes an **n-gram overlap
  metric** comparing generated fluency sequences to human ones — a directly
  portable idea for comparing model joke-topic sequences to human joke-cycle
  sequences.
- **Cognitive Modeling of Semantic Fluency Using Transformers.** arXiv:2208.09719.
  HTTP 200 verified. Uses transformer LMs to model the animal-fluency task
  itself — precedent for treating an LM's internal trajectory through a
  category as directly analogous to a human fluency trajectory.
- **Simple Search Algorithms on Semantic Networks Learned from Language Use.**
  arXiv:1602.03265. HTTP 200 verified. Foraging-style search algorithms over
  learned semantic networks (precedent for scoring model output against graph-
  theoretic patch structure).
- **Optimal Foraging in Memory Retrieval: Evaluating Random Walks and
  Metropolis-Hastings Sampling in Modern Semantic Spaces.** arXiv:2511.12759.
  HTTP 200 verified. Directly pits MVT-style foraging against random-walk /
  MCMC accounts in modern (presumably embedding-based) semantic spaces — good
  citation for our "is the model foraging or randomly walking" analysis section.
- **Ovando-Tellez, M., Vigreux, L., Kenett, Y. N., Benedek, M., Hills, T. T., et
  al. Switching, fast and slow: Deciphering the dynamics of memory search, its
  brain connectivity patterns, and its role in creativity.** *Imaging
  Neuroscience* (Cambridge), 2025. PMC12628016.
  Verified via PMC full text. Classifies fluency responses by inter-response-time
  ratio (IRTr, response latency relative to a subject's own mean latency) into
  fast-clustering vs. slow-switching (MVT-consistent) vs. fast-switching vs.
  slow-clustering (MVT-inconsistent) categories, and links **slow clustering**
  (prolonged patch exploitation) to divergent-thinking/creativity scores and
  **fast switching** to remote-associates-test performance. This is a strong,
  very recent (2025) result directly connecting foraging dynamics to creative
  ability — good for framing "does the model's switching style predict funnier
  output," and gives us a portable per-response metric (IRTr-style timing
  ratio, if we have per-token/per-response latency or an analogous cost proxy).

---

## 2. Humor Topic Sociology

### 2.1 Christie Davies — structural theory of joke targets
**Davies, C. (1990). Ethnic Humor Around the World: A Comparative Analysis.
Bloomington: Indiana University Press.** ISBN 0-253-31655-3 (cloth) /
0-253-21081-X (paper), 404 pp.
- Verified via Amazon listings, Google Books, and American Journal of Sociology
  review (Vol 96, No 6).
- Core thesis: ethnic jokes are not idiosyncratic — the *choice of target*
  follows structural regularities tied to the relative economic/social/
  political position of teller and target (a canonical target is a group seen
  as "peripheral," rustic, or recently urbanized relative to the joke-tellers,
  e.g., stupidity-joke targets across many different countries share a
  structural position rather than actual stupidity). This is the strongest
  citable grounding for "the space of joke topics has known non-random
  structure" — Davies built comparative, falsifiable, cross-national models
  rather than one-off cultural readings.

**Davies, C. (2011). Jokes and Targets. Bloomington: Indiana University Press.**
314 pp., ISBN 978-0-253-22302-9.
- Verified via Oxford Academic (Journal of Social History review), American
  Journal of Sociology review (Vol 118, No 2), and Érudit (Ethnologies journal
  review).
- Extends the 1990 book with three explicit models used together: a
  **center-periphery model** (1990), a **monopoly-competition model** (2009),
  and a **mind-over-matter model**, applied across five joke-target "species"
  (blondes/sex, Jewish jokes, jokes about male intimacy/homosexuality, US
  lawyer jokes, Soviet/post-Soviet jokes). Useful citation for the idea that
  joke *cycles* have a target-selection structure that recurs across otherwise
  unrelated topic domains — i.e., topic space in humor is not flat, it has
  identifiable basins.

### 2.2 Joke cycles: propagation and death
- **Encyclopedia of Humor Studies, "Joke Cycles" entry** (SAGE Reference).
  Verified as an existing reference-work entry via SAGE's own site
  (sk.sagepub.com/ency/edvol/encyclopedia-of-humor-studies/chpt/joke-cycles).
  Defines a joke cycle as a folk process where a group of short jokes (riddle
  or one-liner format) circulate and are told as a set rather than
  individually; topics are frequently ones socially taboo in ordinary
  conversation (ethnic/gender stereotypes, tragedy-adjacent "sick joke"
  cycles). Cycles typically emerge in the wake of a topical triggering event
  and most have short lifespans (weeks to months), with a minority persisting
  for years — direct precedent for treating "topic exhaustion / topic death"
  as an empirically studied phenomenon in human joke culture, which is exactly
  what our forced-switching benchmark artificially compresses into ~50 rounds.
- **Dundes, A. "The Dead Baby Joke Cycle."** Confirmed to exist (Semantic
  Scholar record: "The dead baby joke cycle," Dundes). Alan Dundes is the
  folklorist credited with founding the academic study of joke cycles from the
  1960s onward; exact journal/year not independently re-confirmed in this pass
  — flag **UNVERIFIED-SECONDARY** for full bibliographic details, but existence
  of the paper and Dundes's foundational role in joke-cycle scholarship is
  corroborated by the IU Libraries and folklore.ee sources found alongside it.

### 2.3 Gary Alan Fine — humor in small groups / idioculture
**Fine, G. A. (1979). Small Groups and Culture Creation: The Idioculture of
Little League Baseball Teams. *American Sociological Review*, 44(5), 733–745.**
- Verified via Wikidata record, Semantic Scholar record, and JSTOR volume
  listing (Vol. 44, No. 5, Oct 1979) all cross-matching.
- Coined "idioculture": a system of knowledge, beliefs, behaviors, and customs
  (including nicknames, jokes, stories, rules of conduct) shared by members of
  an interacting small group, generated bottom-up through interaction rather
  than inherited from the wider culture. Relevant framing device: humor topics
  in our benchmark are being generated by a single model with no group to
  co-create an idioculture with — worth a line noting this is a disanalogy
  (Fine's setting is inherently social/co-created; ours is single-agent under
  adversarial constraint).

**Fine, G. A., & DeSoucey, M. Joking Cultures: Humor Themes as Social
Regulation in Group Life.** Published in *Humor: International Journal of
Humor Research*. Verified via ResearchGate record
(publication/249929625_Joking_Cultures...). Exact volume/issue/year not
independently re-confirmed — **UNVERIFIED-SECONDARY** for those details, but
existence/authorship/journal confirmed.

---

## 3. Philosophy of Humor — four classical theories, focused on topic exhaustion / novelty

Primary secondary source used to cross-check all four: Stanford Encyclopedia of
Philosophy, "Philosophy of Humor" (multiple archived editions confirmed live,
e.g. plato.stanford.edu/entries/humor/, and Summer 2023 edition at
plato.stanford.edu/archives/sum2023/entries/humor/). Also cross-checked against
the Internet Encyclopedia of Philosophy's "Humor" entry (iep.utm.edu/humor/).

**3.1 Incongruity theory (Kant, Schopenhauer; modern: Suls, Shultz).**
Kant (*Critique of Judgment*, 1790): laughter is "an affection arising from the
sudden transformation of a strained expectation into nothing" — funniness lives
in the *collapse of an expectation*, not in the content itself. Schopenhauer
sharpened this into a cognitive-mismatch account: amusement arises when a
perceived object suddenly fails to fit the concept under which the mind had
subsumed it, and the magnitude of laughter scales with how great and how
*unexpected* the incongruity is. **Prediction for topic exhaustion:** incongruity
theory directly predicts decay with familiarity — once a listener (or rejector)
has seen the mismatch-resolution pattern for a given topic/frame, the "strained
expectation" can no longer be strained the same way, so re-use of a topic or
setup should mechanically reduce perceived funniness. This is the theory most
directly relevant to justifying *why* our benchmark should expect novelty
pressure to correlate with the incongruity a model can still generate as topics
accumulate.

**3.2 Superiority theory (Plato, Hobbes).**
Hobbes (*Leviathan* / *Human Nature*): laughter is "sudden glory," arising from
a sudden conception of some eminency in ourselves by comparison with the
infirmity of others (or our own former infirmity). Plato (*Philebus*, 48–50)
treats laughter as a species of malice/scorn at others' self-ignorance, and on
that basis wanted comedy tightly controlled in the ideal state. **Prediction
for topic exhaustion:** superiority theory says almost nothing about topic
familiarity per se — its predictions are about *target* selection (who is being
laughed at), which maps onto Davies's target-structure work in §2, not onto
novelty decay. This is useful to state explicitly: superiority theory predicts
which topics recur (targets with stable structural position) rather than how
fast a topic exhausts.

**3.3 Relief theory (Spencer, Freud).**
Herbert Spencer first framed laughter as discharge of surplus nervous energy;
Freud (*Jokes and Their Relation to the Unconscious*, 1905) gave the fullest
version — a joke's technique lets a forbidden thought/impulse past the
"censor" cheaply, and the psychic energy that had been spent suppressing it is
released as laughter once suppression becomes unnecessary. **Prediction for
topic exhaustion:** relief theory predicts topic-specific decay driven by
*taboo strength*, not familiarity per se — a topic stays funny only as long as
saying it costs inhibition to overcome; if a topic is normalized (or, in our
setup, forcibly re-approached many times), the "forbidden" charge that funded
the relief drains away, which is a topic-exhaustion mechanism distinct from
incongruity's expectation-collapse mechanism. Useful for distinguishing two
different reasons the same phenomenon (jokes about topic X get worse the more
times you return to X) might occur.

**3.4 Benign violation theory (McGraw & Warren, 2010) — modern synthesis.**
**McGraw, A. P., & Warren, C. (2010). Benign Violations: Making Immoral
Behavior Funny. *Psychological Science*, 21(8), 1141–1149.**
doi:10.1177/0956797610376073. Verified via SAGE Journals abstract page and the
authors' own hosted PDF (leeds-faculty.colorado.edu/mcgrawp/pdf/
mcgraw.warren.2010.pdf), which also gives the exact volume/issue/pages.
Something is funny iff it is simultaneously appraised as (1) a violation
(threatens how the world "should" be) and (2) benign (an alternative norm,
weak commitment to the violated norm, or psychological distance neutralizes
the threat). **Prediction for topic exhaustion:** this is the theory most
naturally suited to predicting a *forced-switching* dynamic specifically,
because it makes benignness a function of psychological distance and norm
commitment — both of which are plausibly perturbed by a rejector repeatedly
saying "not that." Repeated rejection could either (a) increase psychological
distance from the rejected topic (making subsequent violations attempted on it
feel safer/more benign, funnier) or (b) increase salience/commitment to the
"correct" norm the rejector is enforcing (making violations feel riskier, less
benign). BVT gives us a concrete, testable account of why forced rejection
might non-monotonically affect a model's willingness to keep attempting
violation-humor on adjacent topics, which is different from the pure
memory-decay account of incongruity theory.

---

## 4. Individual Differences — Humor Production Ability Testing

**Greengross, G., & Miller, G. (2011). Humor Ability Reveals Intelligence,
Predicts Mating Success, and Is Higher in Males. *Intelligence*, 39(4),
188–192.** doi:10.1016/j.intell.2011.03.006
- Verified via SciRP reference record (giving exact volume/pages/DOI) and
  corroborated by ResearchGate, Cogn-IQ.org summary, and a Psychology Today
  hosted PDF of the paper.
- Methodology ("the cap task"): ~400 undergraduates took IQ tests (Mill Hill
  Vocabulary + Raven's Progressive Matrices, i.e. crystallized + fluid g) and
  then wrote captions for New-Yorker-style wordless cartoons; captions were
  rated for funniness by independent judges blind to the writer's IQ. Humor
  production ability (mean funniness rating of a person's captions) correlated
  with general intelligence at **r ≈ .29–.40** (matches the correlation figure
  already in this project's CLAUDE.md), driven mostly by verbal/crystallized
  intelligence, and the g→humor latent correlation was larger in men (.67)
  than women (.51) in this sample.
- **Portability to model evaluation:** the cap-task design — fixed prompt
  (a cartoon / a topic), independent-judge funniness rating, aggregated over
  many trials per "subject" — is structurally close to what an LLM humor
  benchmark already does (funniness judged per generation), so the closest
  port isn't the task itself but the *scoring logic*: (a) using **multiple
  independent judges** rather than one judge/model to reduce single-rater
  reward-hacking risk (directly relevant to this project's documented
  LLM-judge-hacking failure mode), and (b) treating humor production ability
  as a **stable per-subject latent trait** measured across many independent
  items — i.e., our benchmark's ~50-round trajectory could, following this
  design, produce not just a trajectory-shape metric but a single scalar
  "humor-production-ability" estimate per model, correlated (as an external
  validity check) against the model's performance on general reasoning
  benchmarks, directly probing this project's Gap #1 (reverse transfer).
- Related/follow-on papers found and title-confirmed but not deeply reviewed
  (flag **UNVERIFIED-SECONDARY**, cite only if independently re-checked):
  "Clever People: Intelligence and Humor Production Ability" (Christensen et
  al., hosted PDF at Penn State beatylab.la.psu.edu); "Sex differences in
  humor production ability: A meta-analysis" (ResearchGate record); "Heritability
  of Humor Production Ability — A Twin Study," Gil Greengross et al., published
  in *Twin Research and Human Genetics* (Cambridge Core, PDF confirmed).

---

## 5. Adjacent, unrequested but directly relevant: Novelty Decay / Satiation Mechanisms

Found while researching incongruity theory's novelty prediction (§3.1) — worth
including because it gives a *second*, independent psychological mechanism for
topic exhaustion beyond Berlyne's arousal account already likely known to this
project (see CLAUDE.md's mode-collapse concern).

**Semantic satiation.** Coined by Leon Jakobovits James in his 1962 McGill PhD
dissertation; foundational empirical paper: **Jakobovits, L. A. (1967).
Semantic Satiation and Cognitive Dynamics.** *The Journal of Special
Education* (SAGE), 2(1). Verified via SAGE Journals abstract page
(doi:10.1177/002246696700200103). Rapid repetition (~15–30 reps) of a word
causes temporary, reversible loss of its meaning for the listener — explained
originally via reactive inhibition (repeated activation of the same neural
representation briefly suppresses its own responsiveness). Distinct mechanism
from incongruity-decay or relief-taboo-drain: this one is about *lexical/
conceptual* fatigue from raw repetition frequency, independent of whether the
repeated item was ever funny or taboo. **Port to our benchmark:** if a model
revisits a *near-identical* topic word/frame many times across the 50 rounds
(not just the same broad category but literal lexical repetition), semantic
satiation predicts a repetition-driven quality floor distinct from
category-level patch exhaustion — worth measuring as a separate metric from
patch/cluster size (see §6).

**Berlyne's arousal / two-factor wear-out model** (referenced for context,
already implicitly known to this project via the mode-collapse concern):
Berlyne's arousal theory treats humor appreciation as a function of arousal
overshoot-then-relief; repetition studies built on this (advertising "humor
wear-out" literature) model funniness decay under repetition as the sum of two
opposing factors — habituation (arousal response weakens) and tedium/boredom
(actively grows). Both factors trend the same direction (declining funniness)
but for different reasons, which again argues for measuring topic-repetition
effects as more than a single scalar.

---

## 6. Metrics We Can Port

A consolidated list of citable, human-validated metrics/methodologies directly
transferable to scoring model topic trajectories under forced switching:

1. **Cluster size (Troyer et al., 1997).** Mean run-length of consecutive jokes
   drawn from the same semantic sub-topic before a switch. Direct analog:
   segment the 50-round transcript into topic clusters (via embedding
   similarity or human/LLM-judge sub-category labeling) and report mean/median
   cluster size, exactly as in clinical fluency scoring.

2. **Switch count / switch rate (Troyer et al., 1997).** Number of transitions
   between clusters, normalized by total rounds. Already dissociates from
   raw output count in the clinical literature (i.e., a model could produce 50
   "good" jokes but with very low switch rate, indicating shallow topic
   exploration despite forced rejection — this alone would be a notable,
   citable finding).

3. **Marginal Value Theorem departure timing (Hills et al., 2012).** For each
   patch, compare local "yield" (e.g., judge-rated funniness of the last k
   jokes in the patch, or embedding novelty within-patch) against the
   running global average; test whether the model's (forced) departure point
   is earlier/later than an MVT-optimal forager would choose. Since our
   departures are *externally forced* by the rejector rather than
   self-chosen, the interesting measurement inverts: does the model's *choice
   of next topic after a forced exit* look like a forager re-entering a
   nearby unexploited patch (MVT-consistent global search) or a random jump?

4. **Inter-response "cost" ratio, IRTr-style (Ovando-Tellez et al., 2025).**
   The human study uses inter-response time relative to a subject's own mean
   latency to classify fast/slow clustering vs. switching, and links the
   fast-switching / slow-clustering axis to different creativity subtypes
   (remote-associates vs. divergent-thinking). For an LLM we lack "thinking
   time" in the human sense, but token-length of the reasoning/generation
   before each joke, or embedding-distance jumped per round, is a directly
   analogous per-response cost proxy — worth reporting a similar 2x2
   (fast/slow x cluster/switch) taxonomy of rounds.

5. **n-gram / sequence-overlap fidelity to human joke-cycle sequences
   (Heineman, Koenen & Varma, arXiv:2405.06714).** Their path-dependent
   category-fluency metric (comparing generated sequences to human sequences
   via n-gram overlap) ports directly to comparing model topic-sequences
   against corpora of real joke-cycle successions (Davies's documented target
   cycles) — i.e., does the model's forced walk through topic space resemble
   the *order* in which real joke cycles historically succeeded one another,
   or is it structurally different?

6. **Revisitation / semantic-satiation flag (Jakobovits, 1967, adapted).**
   Flag near-literal lexical repetition of a topic frame within a short
   window (not just same broad category) as a distinct signal from
   category-level patch exhaustion — predicts a different, repetition-driven
   quality floor than patch-switching metrics alone would show.

7. **Multi-judge funniness aggregation (Greengross & Miller, 2011 cap-task
   design).** Score each joke with multiple independent judges (human or
   diverse LLM judges), not a single judge/model — directly mitigates the
   documented LLM-judge reward-hacking failure mode this project already
   flags, and gives a principled per-subject scalar (mean rated funniness)
   analogous to their humor-production-ability score, usable as an external
   validity check against reasoning-benchmark performance for the reverse-
   transfer research question (Gap #1).

8. **Novelty-vs-memorized-corpus check (already a hard rule in this project's
   CLAUDE.md; grounded further by the "ChatGPT >90% of jokes = 25 templates"
   finding cross-verified during this pass).** The classical-theory grounding
   in §3.1 (incongruity theory) gives a *principled reason*, not just an
   empirical patch, for why mode collapse should be expected under repeated
   sampling: once a "strained expectation" has been used, re-deploying the
   identical setup mechanically can't re-strain the same expectation, so a
   memory-less generator has no in-context reason not to reuse it — novelty
   pressure has to be imposed externally (via the benchmark's scoring), which
   is exactly the corpus-diversity check the project already mandates.

---

## Verification Log (for auditability)

| Citation | Verification method | Result |
|---|---|---|
| Hills, Jones & Todd 2012, Psych Review 119(2):431-440 | Web search, cross-matched 4+ independent hosts | Confirmed |
| Troyer, Moscovitch & Winocur 1997, Neuropsychology 11(1):138-146 | PubMed (PMID 9055277) + Google Scholar record | Confirmed |
| Troyer 2000 normative data paper | PubMed record found | UNVERIFIED-SECONDARY (exact venue not re-confirmed) |
| McGraw & Warren 2010, Psych Science 21(8):1141-1149 | SAGE abstract page + authors' own hosted PDF | Confirmed |
| Davies 1990, Ethnic Humor Around the World, Indiana UP | Amazon (2 ISBNs), Google Books, AJS review | Confirmed |
| Davies 2011, Jokes and Targets, Indiana UP | Oxford Academic review, AJS review, Érudit review | Confirmed |
| Fine 1979, ASR 44(5):733-745 | Wikidata + Semantic Scholar + JSTOR volume listing | Confirmed |
| Fine & DeSoucey, "Joking Cultures," Humor journal | ResearchGate record | UNVERIFIED-SECONDARY (vol/issue/year not re-checked) |
| Dundes, "The Dead Baby Joke Cycle" | Semantic Scholar record | UNVERIFIED-SECONDARY (full bibliographic detail not re-checked) |
| Greengross & Miller 2011, Intelligence 39(4):188-192 | SciRP reference record + ResearchGate + hosted PDF | Confirmed |
| Jakobovits 1967, J. Special Education 2(1) | SAGE Journals abstract page + DOI | Confirmed |
| Kant, Critique of Judgment (incongruity) | Stanford Encyclopedia of Philosophy, IEP | Confirmed (canonical, pre-DOI era) |
| Schopenhauer (incongruity) | Stanford Encyclopedia of Philosophy | Confirmed (canonical) |
| Hobbes, Leviathan/Human Nature (superiority) | Stanford Encyclopedia of Philosophy, IEP | Confirmed (canonical) |
| Plato, Philebus 48-50 (superiority) | Stanford Encyclopedia of Philosophy | Confirmed (canonical) |
| Spencer / Freud, Jokes and Their Relation to the Unconscious (relief) | Stanford Encyclopedia of Philosophy, IEP | Confirmed (canonical) |
| arXiv:2405.06714 (path-dependent category fluency) | curl -L → 200; WebFetch abstract | Confirmed |
| arXiv:2208.09719 (transformer semantic fluency modeling) | curl -L → 200 | Confirmed (title only, not deep-read) |
| arXiv:1602.03265 (search algorithms on semantic networks) | curl -L → 200 | Confirmed (title only, not deep-read) |
| arXiv:2511.12759 (foraging vs random-walk/MCMC in modern semantic spaces) | curl -L → 200 | Confirmed (title only, not deep-read) |
| Ovando-Tellez et al. 2025, Imaging Neuroscience, PMC12628016 | PMC full text fetch | Confirmed |
| arXiv:2504.05228 (NoveltyBench) | curl -L → 200 | Confirmed (title only, not deep-read) |
| arXiv:2604.09629, 2606.00022, 2602.03545, 2508.11429 (humor-gen mode collapse cluster) | curl -L → 200 each | Confirmed (titles only, not deep-read) |
| "ChatGPT >90% of 1008 jokes = 25 templates" | Cross-verified via search result summarizing published finding; matches this project's own CLAUDE.md claim | Corroborated |

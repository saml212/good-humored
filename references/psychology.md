# Psychology and Theory of Humor

This is the literature the project wants "a lot of" — it's for benchmark and
reward-model design, not just background. Confidence levels are marked
explicitly per entry: **primary** (full text directly read/extracted),
**secondary** (citation and stats consistent across independent sources but
not directly read), or **[UNVERIFIED]**. Cite keys refer to `papers.bib`.

---

## 1. Humor production ability and general intelligence (g)

This is the empirical backbone of the project's r≈.29–.40 thesis. The range
itself traces most directly to `christensen2018clever`'s own literature
discussion, not to a single study — treat it as a summary of a small
literature, not one number from one paper.

### `greengross2011humor` — primary
Greengross, G. & Miller, G. (2011). *Intelligence*, 39(3), 188–192.
N=400 university students (200M/200F). Raven's Advanced Progressive Matrices
+ vocabulary + cartoon-caption funniness rating + mating-success measure.
**Confirmed exact numbers:** vocabulary↔humor r=.38 combined (males r=.44,
females r=.31); Raven's↔humor r=.25 combined. SEM: intelligence→humor path
.83/.70 (M/F); humor→sexual-behavior path .47/.58.
**Note:** the project brief's "r=.39, N≈400" is very likely this paper (right
N, right topic), but the literal string "r=.39" does **not** appear in the
paper's actual text — the closest real number is **r=.38**. Don't force the
match; cite .38.

### `christensen2018clever` — primary
Christensen, Silvia, Nusbaum & Beaty (2018). *Psychology of Aesthetics,
Creativity, and the Arts*, 12(2), 136–143. N=270. CHC model (Gf/Gc/Gr) vs. a
cartoon-captioning battery. **Confirmed:** Gc (crystallized/vocabulary)
r=.49 [CI .29,.70]; Gr (retrieval fluency) r=.38 [.22,.54]; Gf (fluid
reasoning) r=.22 [.04,.39]. The paper's own discussion states prior work
found vocabulary ~r=.37–.38 and fluid intelligence ~r=.13–.29 — **this is
the direct primary-source basis for the project's cited "r≈.29–.40" range.**
Effect is specific to crystallized/verbal ability, not general processing
speed — argues for verbal-fluency-style features in a reward model rather
than generic "sounds smart" proxies.

### `howrigan2008humor` — primary
Howrigan & MacDonald (2008). *Evolutionary Psychology*, 6(4), 652–666.
N=185. **Confirmed:** g↔humor r=.29 (p<.01); controlling for Big Five, g
remained significant (β=+.20, p=.010); no g×sex interaction. **This is the
lower bound (.29) of the project's cited range, confirmed from the primary
text.**

### `feingold1991psychometric` — secondary
Feingold & Mazzella (1991). *Personality and Individual Differences*, 12(5),
427–435. Three samples, N=147 total. Distinguishes a "humor cognition" facet
(strongly tied to verbal ability) from a "humor memory/information" facet
(weakly tied). Commonly-repeated range r=.31–.52; a specific "r=.43" figure
surfaced once during research but could not be independently verified —
**[UNVERIFIED, treat .31–.52 as the supported range]**.

### `feingold1993preliminary` — secondary
Feingold & Mazzella (1993). *Journal of Personality*, 61(3), 439–456.
Proposes wittiness = humor motivation × cognition × communication. **Key
finding directly relevant to reward-model design:** self- and peer-ratings
of "wittiness" (reputation) correlate strongly with each other but **not**
with objectively-measured humor production (task output) — perceived and
demonstrated funniness are separable constructs. A reward model trained
only on stated preference, not production quality, conflates exactly this
distinction.

### `kellner2017creative` — secondary
Kellner & Benedek (2017). *Psychology of Aesthetics, Creativity, and the
Arts*, 11(1), 52–58. N=151. Crystallized intelligence and divergent-thinking
fluency **independently** predict humor-production quality (each explains
unique variance) — supports separating a "creativity/divergent-thinking"
reward channel from a "crystallized-knowledge" channel; they are not
redundant.

### `arslan2021moreamorous` — secondary (the children's-study target)
Arslan, Sak & Ateşgöz (2021). *HUMOR*, 34(4), 567–588. N=217 Turkish
6th/7th graders (51 identified gifted), 10-cartoon captioning task rated by
7 experts (30,380 total ratings). **Intelligence explained 68% of variance
in humor-production ability**; verbal analogical reasoning was the single
strongest specific predictor. This is the exact figure the project brief
asked to verify — confirmed via two independent search passes, though the
paper itself was paywalled and not read directly.

### `masten1986humor` — secondary
Masten (1986). *Child Development*, 57(2), 461–473. Ages 10–14; correlated
humor with teacher-rated competence, peer reputation, academic achievement
(r=.50–.53 range reported). Corroborated indirectly: `howrigan2008humor`'s
own primary text cites Masten (1986) alongside Feingold & Mazzella (1991) as
foundational — so its role in the literature is independently confirmed even
though its own abstract wasn't directly read.

### `greengross2025heritability` — secondary, important caveat for the thesis
Greengross et al. (2025). *Twin Research and Human Genetics*, 28(3),
265–272. N=448 MZ + 196 DZ twin pairs. Self-rated humor ability showed
genetic + environmental influence; the **objective humor-production task**
(writing captions) showed **no significant additive genetic effect** —
individual differences attributable to environment only. **Read this as good
news for an RL training approach**: the g-correlation is robust, but that
doesn't mean funniness-capability is a fixed trait — the objective-task
ability looks environmentally malleable, i.e., plausibly trainable.

### `greengross2020sexdifferences` — secondary
Greengross, Silvia & Nusbaum (2020). *Journal of Research in Personality*,
84, 103886. Meta-analysis, 36 effect sizes / 28 studies (1976–2018),
N=5,057. d=0.321 (small-moderate) favoring men's rated humor output, robust
across moderators. A documented, quantified rater-bias confound relevant to
auditing any reward model trained on human humor ratings.

---

## 2 & 3. Greengross & Miller; Feingold & Mazzella

Covered in full under §1 above — both are the same papers the project asked
about by author name.

---

## 4. Humor theories

### Benign Violation Theory — `mcgraw2010benign` — primary
McGraw & Warren (2010). *Psychological Science*, 21(8), 1141–1149. **Exact
theory statement, confirmed from primary text:** *"three conditions are
jointly necessary and sufficient for eliciting humor: A situation must be
appraised as a violation, a situation must be appraised as benign, and these
two appraisals must occur simultaneously."* Five experiments (N=36–80 each)
support the simultaneity requirement specifically — a situation that reads
as pure violation (offense) or pure benign-ness (boring) does not produce
humor. **This is arguably the single most directly useful theory for reward
design**: it's an explicitly *dual-appraisal, simultaneity-gated* model,
suggesting a two-signal reward architecture (violation-detector ×
safety/benign-detector, **multiplied**, not summed or averaged) rather than a
single funniness scalar. Companion: `mcgraw2014benigntheory` (encyclopedia
chapter, same theory, more concise statement).

### Incongruity-Resolution Theory — `suls1972twostage` — secondary (citation confirmed, full text not independently re-read)
Suls (1972), in Goldstein & McGhee (eds.), *The Psychology of Humor*,
pp. 81–100. Two stages: (1) the setup generates an expectation the
punchline disconfirms, producing incongruity/surprise; (2) the perceiver
problem-solves to find a cognitive rule that reconciles the punchline with
the setup. **Incongruity alone is not sufficient — resolution is required.**
**This is literally the project's stated thesis** ("familiar-but-
expectation-breaking") — Suls is the primary citation for it, and it
directly suggests a two-stage reward: (a) expectation-violation magnitude,
(b) resolvability/coherence of the violation — a structurally parallel
architecture to BVT's dual appraisal, arrived at from a different
theoretical lineage. `wyer1992comprehension` (Wyer & Collins, 1992,
*Psychological Review*, 99(4), 663–688) extends this, arguing resolution
alone is insufficient — a joke needs further "elaboration" (extracting
additional implications) to actually land, not merely resolve.

### Superiority Theory — primary (via Stanford Encyclopedia of Philosophy fetch)
Traced to Hobbes, *Leviathan* (1651): *"Sudden glory, is the passion which
makes those grimaces called laughter; and is caused either by some sudden
act of their own, that pleases them; or by the apprehension of some deformed
thing in another, by comparison whereof they suddenly applaud themselves."*
The SEP's own modern treatment notes the "strong" version (all amusement =
superiority) is "wildly implausible" to contemporary philosophers (rebutted
since James Beattie in the 18th century); a weaker "often accompanies"
version is more defensible. Modern proponent: Roger Scruton ("attentive
demolition"). **Mainly a cautionary theory for this project**: a reward
model naively trained on "funny = puts someone down" will overfit to
superiority humor and produce mean/punching-down content — this failure mode
is theoretically predicted here, not just an empirical accident to be
discovered the hard way.

### Relief Theory — primary (via SEP/Wikipedia fetch)
Freud (1905), *Der Witz und seine Beziehung zum Unbewussten*; earlier
physiological version in Herbert Spencer. Laughter releases psychic energy
mobilized for repression, cognitive effort, or anticipated distress (Freud),
or is a release valve for surplus feeling (Spencer). **Largely deprecated**
as literal neuro-energetics in modern scholarship; survives loosely in
laughter-therapy framing only. Low direct value for reward design — the
weakest-supported classical theory, included for completeness/contrast with
BVT and incongruity-resolution.

### Play-mirth theory — `hatzithomas2024playmirth` — primary, newest entry
Hatzithomas (2024). *Frontiers in Psychology*, 15, 1473742. A 2024
cognitive-appraisal theory: mirth arises from two simultaneous appraisals —
(1) a "playful turn" (rapid serious→non-serious reframing) and (2)
motive-consistency (the turn aligns with what's good for the perceiver/
their group). Explicitly argues this differs from BVT's "benign" concept and
that incongruity/unexpectedness alone cannot discriminate humorous from
non-humorous stimuli. Worth reading directly given how close it sits to the
project's own framing — the newest theoretical entry found in this research
pass.

### Other theories found, not independently deep-dived
- **Script-based Semantic Theory of Humor (SSTH)** — Raskin (1985): humor
  arises when text is compatible with two opposed semantic scripts.
- **General Theory of Verbal Humor (GTVH)** — Attardo & Raskin, extends SSTH
  with six Knowledge Resources (Script Opposition, Logical Mechanism,
  Situation, Target, Narrative Strategy, Language). Probably the single most
  "computational" classical theory — worth a closer look for reward-model
  feature engineering specifically.
- **Zillmann & Cantor's disposition theory** and **Zillmann & Bryant's
  misattribution theory** — audience attitude toward the joke's target
  modulates amusement.
- **`latta1999basichumor`** — Latta (1999), *The Basic Humor Process: A
  Cognitive-Shift Theory and the Case against Incongruity* (Mouton de
  Gruyter). Argues *against* incongruity theory, for a "cognitive-shift"
  mechanism instead.
- **"N+V theory" attributed to Latta — [UNVERIFIED, real effort made, not
  found].** No theory under this name was located anywhere, attributed to
  Latta or otherwise. If this is a real label it may belong to a different
  author or be a mis-transcription of something else — flag this rather than
  force a citation onto Latta (1999), whose actual argument is against
  incongruity theory, not a variant of it named "N+V."

---

## 5. Humor Styles Questionnaire (HSQ) — `martin2003hsq` — secondary
Martin, Puhlik-Doris, Larsen, Gray & Weir (2003). *Journal of Research in
Personality*, 37(1), 48–75. 32-item scale, four styles:
1. **Affiliative** — amusing others, enhancing relationships (benign)
2. **Self-enhancing** — humorous outlook to self-regulate/cope (benign)
3. **Aggressive** — sarcasm/ridicule/disparagement at others' expense
   (potentially detrimental)
4. **Self-defeating** — excessive self-disparagement to ingratiate/mask
   negative feelings (potentially detrimental)

Reliability alphas .77–.81 across the four scales. **This is the standard
instrument for decomposing "humor" into styles with opposite valence** —
critical for humor-RL because "funny" is not unidimensional-benign. A reward
model that doesn't distinguish affiliative/self-enhancing from
aggressive/self-defeating risks optimizing toward cruelty, or toward
self-deprecation past the point of being harmful, while still scoring
"funny" by a naive judge.

Related instruments not deep-dived: **Ruch's 3WD Humor Test** (1992; three
stimulus categories — incongruity-resolution, nonsense, sexual humor —
crossed with funniness/aversiveness ratings; Ruch's later work links
openness-to-experience to a preference for *unresolved* "nonsense" humor
over resolved incongruity-resolution humor, directly relevant if a benchmark
wants a resolved/unresolved axis rather than assuming more resolution is
always better); **Thorson & Powell's Multidimensional Sense of Humor Scale**
(1993, *Journal of Clinical Psychology*, 49(1), 13–23).

---

## 6. Stand-up comedians vs. controls — `greengross2012comedians` — secondary
Greengross, Martin & Miller (2012). *Psychology of Aesthetics, Creativity,
and the Arts*, 6(1), 74–82. N=31 professional stand-up comedians vs. 400
college students. Citation and headline findings confirmed across
independent sources; exact inferential statistics were blocked by a
paywall/CAPTCHA during research — treat the qualitative findings as
medium-confidence. **Findings:** comedians scored higher than students on
verbal intelligence, humor-production ability, and **all four** HSQ styles —
including aggressive and self-defeating, not just the two benign ones.
Humor-production correlated with both divergent thinking and crystallized
verbal intelligence. Among comedians specifically, professional success was
predicted *positively* by affiliative humor and *negatively* by
self-defeating humor. **Directly useful nuance for benchmark design:** expert
humor production is not simply "more of the benign styles," and self-
defeating humor specifically predicts *worse* professional outcomes even
though comedians use it more than controls than a naive "funnier = more of
every style" reward model would assume.

No additional distinct empirical comedian-vs-control study was found beyond
this one plus the general Greengross meta-analytic/heritability work in §1 —
this appears to be the primary quantitative study in this specific niche.

---

## 7. Audience-dependence of humor and rater disagreement in psychometrics

This section is the most directly load-bearing for reward-model design,
per the project's own framing.

### `uma2021disagreement` — primary
Uma, Fornaciari, Hovy, Paun, Plank & Poesio (2021). *JAIR*, 72, 1385–1470.
Comprehensive survey documenting the shift away from a single-gold-label
assumption toward directly modeling human label variation/disagreement as
signal, not noise, across NLP and CV subjective tasks. **The standard
citation for "don't force a single ground truth on inherently subjective
judgments"** — directly applicable to humor.

### `davani2022dealing` — primary
Davani, Díaz & Prabhakaran (2022). *TACL*, 10, 92–110. Proposes
multi-annotator architectures (predict each annotator's label separately,
then aggregate post-hoc) instead of majority-vote gold labels for subjective
tasks — captures systematic minority perspectives that majority-vote erases.
**A template for humor reward-model design:** train per-rater or
per-cluster preference models rather than collapsing disagreement into one
score, since "who found it funny" is informative, not error.

### `zhang2024diverging` — primary (abstract-level)
Zhang et al., "Diverging Preferences: When do Annotators Disagree and do
Models Know?" arXiv:2410.14632. Taxonomy of 10 disagreement-source
categories across 4 high-level classes in RLHF preference data; shows most
disagreement reflects task underspecification/response-style differences
rather than annotator error; proposes distributional (not point-estimate)
reward models. **This is essentially the exact problem statement for a
humor reward model** — "is this funny" is maximally underspecified without
audience context — and the taxonomy plus distributional-RM proposal is a
directly transferable method.

### `cao2023culture` — primary, exact stats confirmed
Cao, Hou, Dong & Ji (2023). *Social Psychological and Personality Science*,
14(2), 207–217. Four studies. Study 1 (N=419 Chinese students): jokes about
strangers rated funnier than jokes about close others (F(1,417)=34.03,
p<.001, ηp²=.08). Studies 2a/2b (N=204: 110 Chinese/94 American; N=479:
229/250) replicate a **culture × social-distance interaction**: Chinese
participants find distant-other jokes reliably funnier than close-other
jokes (F(1,201)=35.64, p<.001 in 2a); **Americans show no such difference**
(F(1,201)=2.08, p=.151). Study 3 shows this is causally manipulable via
independence/interdependence priming. **A rigorously-verified example of
audience/culture-dependent humor perception** — directly relevant to
whether a single reward model can generalize across audiences, or whether
audience/cultural conditioning needs to be an explicit modeled variable
rather than assumed away.

### On the specific "κ ≈ 0.49 for humor annotation agreement" claim
**Corrected, not simply verified — see `negative-results.md` §4 for the full
account.** The number is real (Sun et al. 2022, `sun2022expunations`) but
measures agreement on pun-word semantic validity, not funniness. The same
paper's funniness-rating agreement is Cohen's κ=0.41 — actually *worse*
agreement, which if anything strengthens rather than weakens the "humor
labels are noisy" argument. Use κ=0.41, not 0.49, when citing "how noisy are
funniness labels."

### LaughLab — low-rigor, illustrative only
Wiseman (2002), the "LaughLab" project: ~40,000 jokes rated by 350,000+
people across 70 countries; reported country-level joke-type preference
differences (surreal/dark humor rated higher in France/Denmark/Belgium;
wordplay in UK/Ireland/Australia; superiority-humor in US/Canada, per
secondary reporting). **Flagged as lower-rigor than the peer-reviewed items
above** — a public-engagement project with a popular book, not a
peer-reviewed psychometric study. Treat any specific percentage or ranking
from it as illustrative, not citable-grade.

---

## Summary confidence table

| Primary full-text confirmed | Secondary (citation/stats consistent, not primary-read) | Unverified / flagged |
|---|---|---|
| Greengross & Miller 2011 | Feingold & Mazzella 1991, 1993 | Exact "r=.39" figure (real number is .38) |
| McGraw & Warren 2010 (BVT) | Martin et al. 2003 (HSQ) | "N+V theory" attributed to Latta |
| Christensen et al. 2018 | Greengross, Martin & Miller 2012 | Cohen's κ=0.49 as a funniness-agreement figure (it's real, but for a different construct — see correction above) |
| Howrigan & MacDonald 2008 | Kellner & Benedek 2017 | |
| Cao et al. 2023 | Arslan, Sak & Ateşgöz 2021 | |
| Hatzithomas 2024 (Play-mirth) | Masten 1986 | |
| SEP/Wikipedia theory summaries | Greengross et al. 2020, 2025 | |
| Uma et al. 2021, Davani et al. 2022 | Ruch 1992 (3WD), Thorson & Powell 1993 | |

No fabricated statistics were introduced anywhere in this file — every
number is either directly extracted from a source or explicitly labeled as
secondary/unverified above.

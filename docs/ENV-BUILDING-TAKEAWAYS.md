# Building RL Environments — Takeaways

Running log of what demonstrably works and fails while building the
conversational-humor RL environment. Each entry earned its place with
evidence from this repo (experiment IDs / run dirs where applicable).
Newest lessons appended at the bottom of each section.

## Measurement doctrine

1. **Separate load-bearing signals from diagnostics, in writing, before
   data arrives.** The reaction-logprob channel survived only because it
   was demoted to "diagnostic" with a pinned consequence before EXP-023c
   returned ρ=0.122. Signals without a pre-pinned role get promoted by
   wishful thinking after the fact.
2. **Certify instruments on blind, multi-author fixtures or not at all.**
   Every instrument we certified on same-author fixtures later failed a
   blind set (four separate lexical-paraphrase wall confirmations). The
   band-gate floor survives *because* it was certified cross-author.
3. **The final quality bar is a human read, not a number.** Curation
   scores rank; they do not certify "funny." The pipeline's terminal
   artifact is a top-5 full-transcript file for human reading, and the
   scoring code says so in its docstring. Any env whose reward is also
   its acceptance test is unfalsifiable.
4. **Score generation OFFLINE, never in-loop.** Generation and
   measurement stay separable (separate processes, separate files).
   This is what lets us re-score 20k banked sessions when an instrument
   changes — in-loop scoring would have welded a falsified metric into
   the data.
5. **Cheap API-scale pilots before GPU-scale anything.** Six instrument
   falsifications (<$20 total) happened before a single H100-hour was
   spent on them. The instrument graveyard is the cheapest part of the
   project and the reason the expensive part is trustworthy.

## Environment design

6. **Partner quality gates environment value.** A weak partner model
   collapses the whole rollout (8B partner echo-looped; 235B partner
   yes-ands). Ship conversational envs with a strong partner and treat
   the partner as part of the env spec, not an implementation detail.
7. **Directives go FIRST in the system prompt.** Small models ignore
   trailing instructions; the 8B echo-loop pathology (parroting when a
   directive was un-executable) went 0/200 after directive-first plus an
   explicit anti-echo line.
8. **Neutral policy prompts measure the right thing.** No "be funny"
   instruction anywhere in the policy prompt — the dataset's value is
   what wit policies produce UNPROMPTED and how they take provocation
   openings. Instructed humor is a different (and less sellable) task.
9. **Frontier partners sanitize provocations.** The 235B partner
   softens "swear" directives into mild frustration (observed in
   curated transcripts, 2026-08-07). Provocation realism needs either
   prompt strengthening or a partner chosen for compliance — check the
   provocation actually happened, don't trust the schedule.
10. **Seeded schedules ≠ reproducible transcripts under vLLM.** Client
    seeds reproduce task/provocation *schedules* exactly, but MoE
    serving is not bit-reproducible across runs (verified: 0/20
    identical turns for the same session across two runs). Design for
    schedule-level reproducibility; don't promise transcript-level.
11. **Generate deliberate negative-contrast data.** A dedicated
    weak-model (8B self-play) lane makes the curation distribution span
    bad-to-good instead of clustering at strong-model quality — reward
    models trained only on strong outputs never see the failure modes
    they exist to catch.
12. **Turn-0 mode collapse is real: engineer opening diversity.** The
    235B partner produced a near-verbatim identical opening line for
    the same task across batches AND temperatures ("Ugh, this box of
    old cables is a nightmare—" 3/3 in the top-5 read). Temperature
    does not buy first-turn diversity; a seeded opening-angle directive
    does. Check the FIRST turn of your rollouts, not just the middles.
13. **Measure directive compliance; never assume it.** The `swear`
    directive yielded actual profanity in only 17.1% of directed turns
    (542/3165, measured 2026-08-07) — the provocation *schedule* said
    one thing, the *data* did another. Any env whose conditions are
    prompt-induced needs a compliance metric per condition, or its
    condition labels are fiction.
14. **Uncapped top-N curation collapses to one trajectory family.**
    9/10 top curated sessions were the same high-affordance task
    (supply closet), partly riding one replayed opening. Diversity
    caps (max-per-task) in the human-read artifact, and a per-task
    top view in the master, or your "best of" is a monoculture — the
    selection-level twin of the mode-collapse failure this project
    exists to avoid.

## Operations (the part that actually loses the most time)

15. **Watchers must alert on failure AND success.** A success-only grep
    watcher wedged silently when the scorer crashed, contributing to
    7.5 idle GPU-hours. Silence must be impossible to confuse with
    progress.
16. **Expensive hardware needs box-side keeper loops, not agent-side
    attention.** Turn-gated agent attention cannot guarantee
    utilization. The fix that worked: perpetual keeper scripts in tmux
    (generate → score → curate, each its own keeper), stoppable by one
    flag file, with the agent reduced to a periodic consumer/iterator.
    GPUs went from a 7.5h idle gap to sustained 8/8 at 100%.
17. **Rotate configs inside the keeper; don't resample one point
    forever.** Perpetual generation at a single (temperature,
    provocation-rate, model) point buys redundancy, not information.
    The keeper cycles a config grid and the curation table aggregates
    per-config stats — env defaults get chosen by table, not vibes.
    It paid off in one night: 72 rotated batches (N=500/cell) showed
    provocation 0.5 > 0.35 > 0.25 on BOTH curation and audience
    reaction in BOTH policy lanes, temperature no clear effect; the
    losing cell was retired and its slot probes 0.65.
18. **Utilization holes hide in alternation — and in barriers.**
    Rotating two policy models through ONE lane left each model's GPU
    idle half the time; then the fix (both lanes in one loop with a
    `wait`) idled the FASTER lane's GPU every iteration while the
    slower finished. Two instances, one class: any synchronization
    point between unequal workloads is an idle generator. Fully
    independent loops per lane; check per-GPU utilization after every
    topology change.
19. **Failure markers, not retries-forever.** The score keeper writes a
    `.failed` marker for a batch that crashes scoring — loud in the
    log, but the loop never wedges on one bad batch.
20. **Verify "fixed" infrastructure actually landed.** A pre-download
    "fix" for the missing embedding model never reached the box cache;
    the scorer stayed dead for hours while its log path didn't even
    exist. After any remote fix: re-run the failing thing, don't trust
    the fix report.

## Discipline machinery that paid for itself

21. **Pre-registration + pinned consequences + blind calibration rows**
    (~52 closed) is why six falsifications were accepted instead of
    argued with. The paperwork is the product insurance.
22. **Adversarial pre-run audits catch real bugs** (multi-methodology
    contest-file merge, firewall divergence, cap truncation — all
    caught before running). The implementer must not review their own
    work.
23. **Close-out readings need explicit denominators.** "4/5" was
    misread as rejected when it meant leaked; validators now print
    `leaked=N/M`. Ambiguous summaries corrupt downstream decisions.
24. **Cross-session repetition needs its own instrument.** Per-session
    self-repetition scored clean while "Bermuda Triangle," the
    haunted-office register, and "time capsule" recurred across
    sessions and models — the documented 25-template regurgitation
    failure mode at motif level. Pool n-grams across a batch's policy
    turns (distinct-trigram ratio + top content bigrams) and track it
    per config; within-session metrics structurally cannot see it.
25. **Close predictions on schedule, against pre-named data.** All
    three v0.2 predictions (swear >=60%: hit at 64.2%; opening trigram
    <20%: hit at 5.1%; P=0.65 continues-or-plateaus: continued)
    resolved in one cycle because they named their thresholds AND
    their evaluation data (v0.2 batches only) in advance — no
    room to grade on the curve afterward.
26. **The read finds phenomena; only measurement assigns blame.** A
    stage-direction register drift (*asterisk action narration*)
    surfaced in ONE read transcript from the GLM lane — measurement
    showed the GLM lane at 0.4% and the OTHER lane at 27.1%. A
    single-transcript read that prompted the check had the culprit
    exactly backwards. Never attribute a pathology to a model or
    config from reads alone.
27. **Cap sweeps on construct validity, not on metric plateaus.** The
    provocation-rate monotone never plateaued (third continuation at
    P=0.80) and mechanically never would — more directives create
    more reaction opportunities. The sweep was capped at 0.80 because
    the construct being measured (UNPROMPTED wit) ceases to exist at
    the limit. If you wait for the metric to tell you when to stop,
    Goodhart decides for you.
28. **No audience is neutral; orderings can flip with the judge.**
    GLM-as-audience favored GLM's lane; 235B-as-audience favored the
    other lane; rank agreement between them was rho 0.53-0.72. And
    the "neutral" alternative had co-written half of every transcript
    it would judge. Within-judge comparisons stay valid; cross-model
    claims need judge-swapped verification, always.
29. **Test the flattering explanation before logging it.** When the
    register fix dropped the curation metric, the comfortable stories
    were "the artifact was inflating scores" and "the partner never
    needed the constraint." Both were falsified by five-minute
    measurements (asterisk turns scored WORSE in-era; the partner
    asterisked at 25.9%). The real account — the constraint taxes
    every turn's expressiveness — was less flattering and only
    reachable after killing the easy stories.
30. **Model attractors the env exposes are product, not bugs.** The
    policy agreement attractor (28-30% of strong-model turns open
    with yes-and; the WEAKER 8B agrees less) is trained-in
    conversational risk-aversion. Prompt-patching it would blind the
    env to exactly the behavior a buyer's RL is supposed to move.
    Characterize baselines; leave the neutral prompt alone; let the
    delta be the demo.
31. **Prompt revisions shift metric scales; window your comparisons.**
    Any change to system prompts moves the whole scoring
    distribution (here -0.07 to -0.13). Cross-version score
    comparisons are invalid by default; human-read artifacts and
    leaderboards must draw from a recent same-version window.
32. **Unequal sample sizes poison cross-population selection.** The
    weak-model contrast lane (50x the batches) landed a tail-luck
    session at #1 on the demo shortlist, above transcripts any human
    ranks higher. Selection across populations with different n
    finds the big population's outliers, not the best population's
    typical excellence. Scope shortlists per lane; compare lanes by
    distribution, never by shared top-N.
33. **Provocation density is an anti-sycophancy dial.** Agreement-
    opener rate falls monotonically as provocation rate rises (both
    strong models, all cells) — scheduled provocations force the
    policy out of yes-and register. Conversational envs can TUNE how
    much social risk the data demands; that dial is part of the env
    spec, not an accident of sampling.
34. **Provocation types have wildly different wit yields — and
    compliance is not productivity.** Mock/tease provocations elicit
    the best ripostes (2+ logits of audience reaction over
    unprovoked turns, identical ordering across models); swear — the
    type that took two cycles of compliance engineering — is the
    weakest elicitor. Rank your elicitors by downstream yield before
    polishing their execution. Weight the mix, but never drop types:
    the env must keep measuring the full space.
35. **Sampling-parameter response is model-specific; sweep per model.**
    Temperature moved every metric for one model (GLM, monotone to
    1.1) and nothing for the other (30B, flat at n=14/cell). A global
    temperature decision would have been wrong for one of them in
    whichever direction it went.

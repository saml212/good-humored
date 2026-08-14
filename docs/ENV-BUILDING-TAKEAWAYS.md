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
36. **Close sweeps deliberately, or they become treadmills.** Every
    config axis got an explicit terminal state: retired by evidence
    (P=0.25/0.35), capped on construct grounds (P=0.80), closed
    per-model with a guard (temperature), or weighted with types
    retained (provocation mix). An always-running optimization loop
    with no closure rule will happily tune knobs forever while the
    first-order lever (training on the banked data) waits.
37. **Generate the pitch from the pipeline, never hand-assemble it.**
    The demo pack (framing + characterization card + curated
    transcripts) is one command over live run data. Hand-picked
    demos rot and overclaim; a generated pack regenerates after
    training on identical machinery, making the pre/post delta — the
    actual product claim — reproducible on demand.
38. **Monitor throughput ratios, not just liveness.** Every keeper
    was alive and every health check green while the scorer fell 106
    batches behind a 3-lane generator. Pipelines diverge silently
    when producers outnumber consumers; the health check must compare
    produced-vs-consumed counts, and consumers must scale (lock-claim
    + second instance) rather than assume one worker suffices.
39. **Reads cannot catch sub-percent defects; instrument every
    defect class the moment a read discovers one instance.** A
    single CJK-leaked turn in one transcript led to a counter that
    exposed a 20x temperature-correlated defect gradient (1.1% at
    the top temp) and reversed a closed decision. The read's job is
    DISCOVERY of defect classes; only counters establish prevalence
    — and closed decisions must stay reversible when a new class
    gets instrumented, because pinned guards only ever cover the
    failure modes known at pin time.
40. **Unequal-rate populations break every selection layer they
    touch.** The fast contrast lane first polluted the top-N (cycle
    5), then silently emptied the recency window (global mtime went
    40/40 contrast). When one population produces 3-4x faster, every
    window, quota, and queue in the pipeline needs explicit
    per-population handling — audit them all at once, not as each
    one breaks.
41. **Servers upstream of a shared bottleneck starve in bursts —
    feed them dependency-free work.** Sessions bunch-synchronize on
    the slowest shared stage (the 235B partner), so the faster GLM
    server idled ~60% in bursts that instant-checks missed. Adding
    workers just deepens the bottleneck queue; the fix is work with
    no external dependency (a self-play lane on the starved server),
    interleaved by continuous batching. Duty-cycle SAMPLING (not
    instant snapshots) is what found it.
42. **Verify restarts ACROSS the boundary that killed them.** A
    keeper that crashes on its first loop iteration looks launched:
    the tmux session existed, the old log had plausible lines. The
    relaunch "verification" read a dying process's last output as
    fresh. After any restart, confirm the process completes one full
    unit AND starts the next — the boundary is where between-batch
    crashes live.
43. **Log tails are not liveness; encode health as an executable
    check.** Five consecutive cycles tailed a dead lane's log and
    read its restart note without alarm — humans (and agents)
    normalize a stale line they've seen before. Health = session
    existence + output freshness + produced/consumed ratios,
    encapsulated in a script that exits nonzero, run every cycle.
    If a check can't fail loudly, it isn't a check.
44. **Cross-lane score comparability requires judge independence.**
    Self-play sessions (judge = both participants) posted tail
    scores at all-time-#1 level that read a full tier lower. Every
    additional identity shared between judge and participants
    inflates further. Scores from different judge-participant
    configurations support within-lane trends only; demo channels
    must carry the product configuration exclusively.
45. **When a stream's marginal information saturates, sample it —
    don't scale scoring to match generation.** The contrast lane's
    scores stopped informing anything at ~180k sessions; its raw
    data stays banked, its scoring became 1-in-4 sampled, and the
    permanent backlog race ended by decision instead of capacity.
46. **Strict priority orders starve the bottom whenever the top
    saturates capacity.** The same lane monopolized both scanners
    under newest-first order, then starved to zero under
    policy-first order — two failures, one mechanism, opposite
    signs. Priorities need either capacity headroom or an explicit
    share for the lowest class (here: one scanner visits it FIRST,
    cheaply, then rejoins the priority order).

## The RL era (GRPO v1-v2) — what training taught about training

47. **A rising training reward is not learning until fresh contexts
    say so.** Two 200-step GRPO runs produced real training-curve
    gains (+0.07 and a late ~3-SE climb); both transferred ~nothing
    to held-out seeds (+0.008, t=0.41). At small-adapter scale the
    curve measures the policy's relationship to its own rollouts —
    v2's signature: score/max flat all run, mean rising late =
    mode-sharpening onto existing good modes, not new competence.
    The held-out matched-seed A/B is the only reward claim.
48. **Verify BOTH parities before any training step.** Reward-path
    parity (same transcripts through training-time and eval-time
    scoring) and generation parity (same model scores the same
    through the training pathway and the clean serving pathway,
    n>=200 matched pairs). v1 lacked both; its entire curve was
    uninterpretable. The one-hour gates are cheaper than one
    uninterpretable 13-hour run.
49. **Dump training transcripts always.** v1's post-mortem was
    archaeology because rollouts weren't saved; v2's 12,800 dumped
    sessions turned every hypothesis into a five-minute measurement.
    Per-worker files; the cost is megabytes.
50. **Power before mechanism.** Three successive n=30 pathway
    comparisons produced three contradictory mechanism stories
    (SE 0.05 vs a 0.06 effect); the n=200 gate settled it in one
    shot. Compute the SE the effect demands BEFORE concluding.
51. **Credit assignment eats sparse session rewards.** One scalar
    over 10 masked-turn groups gave gradients nothing to grip for
    110 steps, then sharpened modes instead of skill. Dense
    per-turn shaping (the reaction term exists per turn already)
    is the first lever to pull before buying capacity.
52. **Colocated RL memory budgets are set by the SYNC phase, not
    steady state** (FSDP shard + engine weights resident together);
    and the training/serving stacks disagree about templates,
    stop-token inclusion, and LoRA-on-MoE — every seam between them
    is a place the two distributions silently diverge.
53. **Watchers on block-buffered drivers need buffering-immune
    signals** (file mtime, terminal lines, server-side traffic) and
    trigger patterns validated against a healthy log first; liveness
    verdicts need paired stack dumps and movement, never snapshots.
54. **After a context compaction, verify watcher liveness against
    OUTPUT FILES, not the task list.** (Amended on new evidence:
    the "vanished" RL-C watcher was alive the whole time and fired
    on schedule — an empty task list after compaction is not
    evidence of death, and re-armed duplicates are harmless but
    conclusions drawn from the empty list are not.) Stale watcher
    outputs from PRIOR runs also linger and read like fresh results
    (a v2 step-200 curve was briefly misread as RL-C finishing
    +0.15 up) — match every output file to its run before believing
    it. Corollary:
    never watch a multi-hour remote job over ONE long-lived ssh
    connection — an idle session gets dropped by the remote host
    (broken pipe, exit 255). Poll with fresh short connections and
    alert on consecutive probe failures too, so "watcher dead" and
    "box unreachable" are events rather than silence.
55. **Tensor-diff the "trained" model against base BEFORE serving
    any A/B arm.** verl's model_merger does not fold LoRA — it
    exports base weights beside an unfolded lora_adapter/ dir, so
    two eval campaigns silently compared base against base and
    produced structurally guaranteed "nulls" (+0.008, −0.012). The
    gate is two cheap safetensors reads: a LoRA-target tensor must
    differ from base, a non-target must not. Inverted silver
    lining: a vacuous identity A/B is a perfect blind measurement
    of the eval instrument's noise floor (~±0.01 at n=500 paired
    here) — worth running ON PURPOSE once, but only on purpose.
56. **What RL learns FIRST from a product-shaped reward is defect
    avoidance, not the skill you care about.** The first valid
    positive A/B (+0.059, t 2.9) decomposed as ~2/3 screen-cliff
    avoidance and ~1/3 topical grounding; audience laughter moved
    zero. Dense penalties get learned before dense rewards at small
    adapter capacity — budget for a defect-cleanup phase before
    expecting taste to move, and always decompose the delta into
    components before celebrating the headline number.
57. **A flat training curve and a real effect coexist** whenever the
    effect size sits below curve visibility (here: +0.02–0.06 vs
    per-step smooth sd ~0.10 at 64 sessions/step). Weight motion
    can be 3e-5 max|delta| and still produce a detectable behavior
    change. The powered held-out A/B is the measurement; the curve
    only bounds what it can see.
58. **Matched seeds cancel seed effects, not DAY effects.** The base
    model on identical seeds scored +0.039 higher one day later —
    same weights, same servers, fresh sampling. Between-day
    variance of a paired-effect estimate (~±0.02 here) can equal
    the effect itself. Within-day arm-vs-arm deltas stay valid
    (roll arms minutes apart, never across days); any cross-day
    comparison of point estimates is confounded; report effect
    magnitudes as ranges across measurement days. Corollary: a
    success declared on one seed set inflates re-measurements on
    those seeds (winner's curse measured: +0.037 on declaration
    seeds vs +0.015 fresh).
59. **An LLM-audience taste term can resist RL even with everything
    stacked for it.** Dominant objective weight (0.6), dominant
    within-group advantage variance, instrument-matched framing,
    bait channel closed by construction, zero audience errors —
    and the held-out taste delta was +0.003 (t=0.6) after 200
    steps. Mid-training it rose ~+0.02 then faded (repeated looks
    found t=3.9 at the peak — trajectory snapshots are not
    results). The signal as constructed (next-token laughter mass
    of a frozen model over whole turns) appears too diffuse for
    r32/200-step credit assignment; redesign means denser
    attribution or a different construct, not more weight.
60. **An LLM audience roleplaying a participant echoes the OTHER
    participant.** Our audience predicts the partner's next token,
    so partner-emitted laughter primes laughter logprob (+0.102
    taste/turn, causally probed) — a channel the policy cannot
    steer (a within-group lottery) that we only noticed after
    stripping the POLICY side. Normalize every speaker's tokens in
    a judge's context, not just the trained one's; and remember an
    in-character judge's next-token distribution is about the
    CHARACTER's habits as much as the content.
61. **Check a logprob-derived reward for floor censoring before
    training on it.** 61% of 220k banked turns sat exactly at the
    laughter-mass floor — the construct's median turn carries zero
    gradient information, and a "neutral counterfactual" baseline
    is floored 95.7% of the time, so contrastive differencing
    cancels nothing. Also check sub-terms for internal conflict:
    our grounding gate anti-correlated with the taste term
    (beta -0.54) — the objective was fighting itself.
62. **Adversarial suites for embedding-based rewards must include
    word-shuffled positives.** MiniLM-class embeddings are near
    bag-of-words: a shuffled reply keeps its topical embedding
    while gaining "surprise," so any embedding-distance surprise
    gate scores syntax-destroyed text ABOVE real wit (measured:
    gate-pass 13/50 shuffle vs 12/50 witty). Historical anti-gaming
    tests only covered cross-topic incoherence — the attack that
    preserves the embedding and destroys the meaning is the one
    that matters. Pair every embedding gate with a fluency/NLL
    screen (chat-NLL separated shuffle at AUC 1.000).
63. **A candidate computed from policy tokens can still be a
    lottery.** The incongruity gate read only the policy's own
    text, yet its within-group variance was the same flat
    lottery shape as the audience construct (policy-observable
    R^2 0.026) — token provenance does not guarantee attributable
    signal. Run the variance-profile + feature-attribution check
    on ANY candidate reward before registering, regardless of how
    attributable it looks structurally.
64. **Fluency screens structurally fight taste.** On blind human
    pairs, the human preferred the HIGHER-NLL reply 31/50 — wit is
    unpredictable text, so an NLL screen's real-data action is a
    false-positive channel that eats exactly the turns you want
    (8/19 screened turns were witty; humans preferred the screened
    side 17/24 when it bound). Adversarial incoherence never
    occurred on the training distribution — the screen fixed a
    hack that only exists off-distribution, at the price of
    anti-correlating with taste on-distribution. Screens must be
    validated on the DEPLOYMENT distribution, not just the attack
    suite; and a hard penalty spliced into a reward can triple
    within-group advantage variance (Dark Room hazard) even when
    it fires on 4% of turns.

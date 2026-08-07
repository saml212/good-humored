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

## Operations (the part that actually loses the most time)

12. **Watchers must alert on failure AND success.** A success-only grep
    watcher wedged silently when the scorer crashed, contributing to
    7.5 idle GPU-hours. Silence must be impossible to confuse with
    progress.
13. **Expensive hardware needs box-side keeper loops, not agent-side
    attention.** Turn-gated agent attention cannot guarantee
    utilization. The fix that worked: perpetual keeper scripts in tmux
    (generate → score → curate, each its own keeper), stoppable by one
    flag file, with the agent reduced to a periodic consumer/iterator.
    GPUs went from a 7.5h idle gap to sustained 8/8 at 100%.
14. **Rotate configs inside the keeper; don't resample one point
    forever.** Perpetual generation at a single (temperature,
    provocation-rate, model) point buys redundancy, not information.
    The keeper cycles a config grid and the curation table aggregates
    per-config stats — env defaults get chosen by table, not vibes.
15. **Utilization holes hide in alternation.** Rotating two policy
    models through ONE lane left each model's GPU idle half the time;
    running both lanes concurrently against the shared partner closed
    it. Check per-GPU utilization after every topology change.
16. **Failure markers, not retries-forever.** The score keeper writes a
    `.failed` marker for a batch that crashes scoring — loud in the
    log, but the loop never wedges on one bad batch.
17. **Verify "fixed" infrastructure actually landed.** A pre-download
    "fix" for the missing embedding model never reached the box cache;
    the scorer stayed dead for hours while its log path didn't even
    exist. After any remote fix: re-run the failing thing, don't trust
    the fix report.

## Discipline machinery that paid for itself

18. **Pre-registration + pinned consequences + blind calibration rows**
    (~52 closed) is why six falsifications were accepted instead of
    argued with. The paperwork is the product insurance.
19. **Adversarial pre-run audits catch real bugs** (multi-methodology
    contest-file merge, firewall divergence, cap truncation — all
    caught before running). The implementer must not review their own
    work.
20. **Close-out readings need explicit denominators.** "4/5" was
    misread as rejected when it meant leaked; validators now print
    `leaked=N/M`. Ambiguous summaries corrupt downstream decisions.

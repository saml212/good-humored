# Rockie harness — owner's design intent and session findings

Recorded 2026-07-16 from a working session in which the harness was installed,
dogfooded on real work, and fixed. Sam Larson owns Rockie; these are his stated
intentions plus what dogfooding actually revealed. Kept here (not in
`rockie-claude`) because it mixes product direction with research context.

---

## 1. Design intent (Sam's words, paraphrased minimally)

**The CLI is how an agent acquires capability.** All ~297 catalog skills sitting
in context would be bloat. The intent is that an agent *browses* the catalog and
*imports* the right skills for its work. This is the product's central idea.

> "The second an agent comes in contact with the rockie harness — rockie claude,
> rockie codex, or rockie nugget — it should have all this information blasted at
> it. Same for downloading the rockie cli. So let's think how to be maximal about
> this and maximal about agent onboarding."

**Status:** the mechanism worked all along; nothing told the agent it existed.
`cli-guidance.md` was purely about nudging the *human* to install the CLI. Fixed
in rockie-claude#27 (five redundant channels), ported to codex#31 and nugget#22.

**Porting is not optional.** Every fix lands in rockie-claude, then rockie-codex
and rockie-nugget. But port *idiomatically* — nugget is Goose-based with
advisory-only hooks and no `SKILL.md` discovery; forcing Claude Code's
architecture onto it would be cargo-culting. A port that fights the host runtime
is worse than a smaller native one.

**Don't overburden the harness.** Harness ships the loop; the catalog ships
domains. See rockie-claude#30.

**`/onboard` must be a necessity, not a nudge.** See rockie-claude#31.

**The loop must actually be enforced.** See rockie-claude#32.

---

## 2. What dogfooding revealed (all fixed, or filed)

| Finding | Severity | Status |
|---|---|---|
| Budget gate **failed open** — a crashing `budget.py` exited 1, the hook only blocked on exit 2, so GPU ceilings were silently unenforced | Highest — an autonomous harness that fails open on spend is worse than no gate, because it looks like it's enforcing something | Fixed #20 |
| `/clean` catch-22 — the gate's own remediation command could never unblock the commit | Blocks every new user | Fixed #29 |
| `/clean` blocked the bootstrap `CLAUDE.md` the installer mandates | First thing a user does, blocked | Fixed #29 |
| `pre-train-gate` demanded a gradient check for a **lint script** | Trains reflexive bypassing | Fixed #29 |
| Worktree sentinel mismatch — `/clean` passes, gate blocks anyway | Hits `/deploy-team`, which runs agents in worktrees | Fixed #29 |
| Two unmerged private-skill backends — CLI-pushed skills structurally invisible to the catalog | Blocked the central design intent | Fixed platform-context#876 |
| Leaked `.git.backup-pre-split` on public main (178 files, competitive strategy doc + old codename) | Embarrassment, not breach — no credentials | Fixed #26 |

**The pattern:** *nudges do not work on agents.* Every failure above is a case
where the harness advised rather than enforced — and the most compliant agent
available ignored the advice. Gates work. Nudges are decoration.

**Test suite: 81 → 115 assertions** across today's work.

---

## 3. Architectural facts worth not rediscovering

- **Two skill layers.** Shared catalog (~297, auto-mounted, serves the cloud lab
  agent) + writable per-tenant overlay (`rockie skill push/pull/list`). A Rockie
  `SKILL.md` uses Claude Code's frontmatter format, so catalog skills pull
  straight into `.claude/skills/` and are immediately invocable.
- **`rockie skill pull` takes `catalog_id`, not `name`.** They differ for
  **39/297 entries (13%)** — e.g. `name: huggingface-accelerate` but
  `catalog_id: accelerate`. Pulling by name 404s.
- **`rockie skill catalog --search` is client-side substring matching.**
  `--search verl` returns *geopandas* ("o**verl**ay operations") and does not
  surface `/verl`.
- **15 of 28 harness skills are duplicated in the catalog** — two sources of
  truth that can drift. Which is canonical is undecided.
- **`main` rulesets** (rockie-claude): `pull_request` (0 approvals),
  `required_linear_history`, `non_fast_forward`. Direct push to main is
  impossible; a stale branch *must* be rebased — no merge-commit escape hatch.
- **Trailer conflict:** rockie-claude's `CONTRIBUTING.md` and platform-context's
  `commit-msg` hook both forbid `Co-Authored-By: Claude` trailers. platform-context
  actively blocks them. Unresolved inconsistency worth reconciling.

---

## 4. Commercial posture

- `good-humored` is public with **no license** — all rights reserved, which is the
  maximum commercial protection. The danger runs the other way: an OSS license is
  **irrevocable** for every version published under it, which is what would
  destroy the asset in an acquisition.
- If adoption is later wanted without giving up rights: BSL 1.1, FSL, PolyForm
  Shield, Elastic License 2.0 — source-available, not OSI open source.
- **The thing that actually decides saleability is ownership, not the license
  file.** Sam's email is `@pebbleml.com`; standard IP-assignment terms may mean
  the seller must be PebbleML rather than Sam personally. That is a question for
  counsel and worth settling while it's cheap to change.
- The `.claude/` harness copy in this repo is Apache-2.0 (Rockie's own license)
  and gitignored here; vendored MIT platform skills likewise. Only `humor-rl`,
  these docs, and `references/` are original IP.

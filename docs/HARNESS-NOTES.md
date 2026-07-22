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

## 4. Running ledger — what still needs fixing (updated 2026-07-16 night)

Found while doing real research work with the harness, as a user-agent would.
Status: OPEN unless noted.

### rockie-claude harness

| # | Finding | Status |
|---|---|---|
| L1 | `pre-train-gate` false-blocks read-only commands: extracts the LAST `.py` token in a compound command (grabbed a `find -name "journal.py"` pattern), and `[[:graph:]]` keeps quote chars → malformed path → hard error blocks the command | FIXED — rockie-claude #34, rockie-codex #32 (also fixed both repos' broken-main CI: smoke test ran real `git commit` with no local identity — passed locally via ambient global config, failed on runners) |
| L2 | `/clean` new-.md blocker fires on the exact files the harness's own template mandates (STATE.md, EXPERIMENT_LOG.md, README.md) — a new user's first real commit is blocked by the harness's own requirements | Fix PR in flight (agent, canonical) |
| L3 | `pre-commit-gate` blocks the ENTIRE compound command, so `git add X && git commit` dies with the add never having run — and the error doesn't say so. Cost me two phantom-debugging rounds. Error text should state "no part of the command ran" | OPEN |
| L4 | Audit-then-commit must be two separate Bash calls (PreToolUse evaluates before the in-command audit can write the sentinel). Fine as design, but nothing documents it; every new agent will trip it once | OPEN (docs) |
| L5 | `audit.py --scope staged` on an EMPTY staged set happily writes a sentinel (trivial zero-blocker pass). Harmless today (hash won't match once files stage) but a confusing state; should warn "nothing staged" | OPEN |
| L6 | `install.sh` rsynced ALL of project-harness/skills into every project — 10 catalog-duplicated domain skills bloating every session | FIXED — PR #33 (manifest) |
| L7 | Budget gate failed OPEN on a crashing budget.py | FIXED — PR #20 |
| L8 | Nested `claude -p` inherits project CLAUDE.md from cwd — any harness feature that shells out to a model as an *instrument* must use a neutral cwd. The harness itself does not document or guard this | OPEN (doc/guard) |
| L9 | STATE.md / EXPERIMENT_LOG.md are mandated by the template but not scaffolded by the installer — every new project recreates them from prose | OPEN |
| L10 | `[LEARN]` blocks: no feedback on whether the hook actually captured one (fire-and-forget from the agent's side) | OPEN (UX) |
| L11 | install.sh never prunes skills later moved to `excluded_from_overlay` — pre-policy installs carry dead-weight skills forever (rsync has no `--delete`, correctly, but no prune/notice path either) | OPEN — [#36](https://github.com/Rockielab/rockie-claude/issues/36) |
| L12 | Reinstall's idempotent schema.sql reapply churns workflow.db's checksum with zero data change — false data-loss scare for any before/after-checksum audit; needs a docs note (verified benign via schema diff + row counts) | OPEN — [#37](https://github.com/Rockielab/rockie-claude/issues/37) |
| L13 | pre-commit-gate fires on commits in UNRELATED repos/scratch clones (cwd-scoping doesn't exempt them) — forces CLEAN_BYPASS on legitimate upstream work, training agents to reach for the bypass | OPEN — [#39](https://github.com/Rockielab/rockie-claude/issues/39) |
| L14 | Skills-survive-compaction shipped (rockie-claude #38, rockie-codex #34, rockie-nugget #25 — all merged 2026-07-17). Cross-harness dogfood: skill paths diverge silently (`.claude/skills/` vs `.agents/skills/` vs `./skills/`); claude/codex get compaction re-surfacing free via unmatched SessionStart, Goose has NO compaction lifecycle event at all (rockie-nugget#24 tracks); commit-trailer rules differ per repo (informal vs DCO-required vs unstated) | SHIPPED + follow-ups filed |
| L15 | rockie CLI is unrunnable from agent context on this machine (2026-07-22): `~/.local/bin/rockie` execs node whose dylib chain resolves into the boot-offload SSD (`/Volumes/1TB_SSD/.../Cellar/node/26.3.0`), where `libnode.147.dylib` is missing at `bin/` and macOS denies the `lib/` path with "Operation not permitted" even for unsandboxed agent processes (TCC volume access is per-app; the agent's terminal lacks it). Agents cannot pull new skills; user's own shell works. Two product angles: (a) the boot-offload symlink strategy makes the CLI hostage to brew upgrades + volume TCC in a way plain installs aren't; (b) skills should be pullable via the MCP server too — `rockie_search` covers labs/chats/artifacts but there is no `rockie_skill_*` MCP surface, so agent-driven skill acquisition has no CLI-free fallback | OPEN — file rockie-cli issue + MCP skill-surface feature request |
| L16 | Same boot-offload dylib class, second victim (2026-07-22): the PDF toolchain — `pdftotext`/`pdftoppm` die on a `libpoppler.161.dylib` dyld error, so research agents cannot extract equations from arXiv PDFs (WebFetch's PDF path also choked); `pip install --user pypdf` is separately blocked by the macOS user-site permission denial (L15's TCC cousin). Research agents fell back to HTML mirrors/abstract APIs and correctly tagged the un-extractable citation SUGGESTIVE rather than citing unread math. Product angle: the boot-offload strategy needs a post-brew-upgrade relink check, or agents need a bundled minimal PDF reader | OPEN |

### rockie CLI

| # | Finding | Status |
|---|---|---|
| C1 | `skill pull` takes `catalog_id`, not `name`; they differ for 39/297 entries. Pull-by-name 404s with no hint | OPEN |
| C2 | `catalog --search` is client-side substring: `--search verl` returns geopandas ("o**verl**ay") | OPEN |
| C3 | `sota-delta` exists in the harness but was never pushed to the catalog (`tenant_visible_count: 296`) | OPEN |
| C4 | No `rockie skill push` dry-run; pushing writes to the live tenant overlay with no preview — agents were forbidden from using it this session for that reason | OPEN |
| C5 | Trailer policy conflict: rockie-claude CONTRIBUTING.md + platform-context commit-msg hook forbid `Co-Authored-By: Claude`, while Claude Code's defaults add it — every agent hits this once per repo | OPEN (reconcile) |
| C6 | `skill pull ml-paper-writing` fails 415 "file is not valid UTF-8" — one bad byte makes a whole catalog entry unpullable, and the error doesn't name the offending file | OPEN — [platform-skills#91](https://github.com/Rockielab/platform-skills/issues/91) |

## 5. Commercial posture

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

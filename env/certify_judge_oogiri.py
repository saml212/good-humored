#!/usr/bin/env python3
"""EXP-012 — certify the contained reward kernel's funniness judge against
human consensus (EXPERIMENT_LOG.md EXP-012, registered 2026-07-17 evening).

Design constraint this serves (Sam): the RL reward must be a CONTAINED
KERNEL -- model calls + pure computation, no humans at training time. Human
judgment data is used exactly once, offline, to CERTIFY the kernel's judged
components. This script is that certification: it scores every candidate in
a sample of Oogiri prompt groups with the kernel's naked funniness judge
(env/rewards.py's `JudgePreferenceReward` contract -- judge(prompt,
completion) -> float in [0, 1]) and computes, per prompt, the Spearman rank
correlation between the judge's scores and the human-consensus ranking
already present in the data (`RankedGroup.candidates[i].score`).

REGISTERED-DESIGN DEVIATION, flagged loudly (not silently substituted):
EXP-012's registration describes "the ~100-human consensus ranking that
already exists for each prompt" and CLAUDE.md's Research Direction section
cites Oogiri-Master's "~100 candidates per prompt, each rated by ~100
independent judges" as the clean, popularity-bias-free instrument. The only
loader that exists in this repo, `data_adapters/oogiri.py`, is Oogiri-GO
(`zhongshsh/CLoT-Oogiri-GO`), NOT Oogiri-Master -- that module's own
docstring states plainly that "Oogiri-Master has no public pre-built
release as of this writing." Oogiri-GO's per-candidate `score` is the
`star` field, an aggregate like/popularity count on ONE response
(data_adapters/schema.py's `Candidate.score` docstring), not an independent
~100-judge panel rating -- a WEAKER, popularity-adjacent human signal, by
that module's own explicit admission. Average group size in the real data
is ~6.3 candidates/prompt (6,343 candidates / 1,009 multi-candidate
groups), not ~100. This script proceeds anyway -- it is still a real
human-preference signal, still popularity-bias-free ACROSS prompts (the
schema's structural design: `Candidate.score` never leaves its own
`RankedGroup`), and it is the only Oogiri instrument this repo can actually
run today -- but the certificate this produces should be read as "vs.
Oogiri-GO popularity consensus, ~6 candidates/prompt," not "vs. a
~100-independent-judge panel." See `report.json`'s `data_source` field for
this same statement, so it travels with every run's output.

CONTAINED-KERNEL SCOPE: this script calls the judge exactly like
env/rewards.py's `JudgePreferenceReward` would at training time (a haiku
CLI call, normalized 1-10 -> [0,1]) but is itself an OFFLINE, one-time
certification run -- it is not part of the trained kernel and never runs
during RL training.

Mirrors benchmark/validate_banter_judge.py's instrument-validation shape
(fixture -> judge -> per-item metric -> aggregate + separation stats) and
reuses benchmark/relabel.py's `LabelCache` verbatim for a sha256-keyed,
resume-safe judge-score cache (see `cache_key` below for why the cache
key is `context + candidate`, not just candidate text -- the same reply
text can appear under different Oogiri prompts and a funniness judgment
is only meaningful in-context).

Usage:
  PYTHONNOUSERSITE=1 python3 -m env.certify_judge_oogiri \
      --n-prompts 30 --seed 20260717 \
      --out experiment-runs/2026-07-17-exp012-judge-certification/report.json
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Make the repo root importable regardless of how this script is invoked
# (same convention as env/smoke.py / env/validate_semantic_novelty.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.providers import get_provider  # noqa: E402
from benchmark.relabel import LabelCache  # noqa: E402
from benchmark.stats import bootstrap_ci  # noqa: E402
from data_adapters.oogiri import OogiriLoadStats, load_ranked_groups  # noqa: E402
from data_adapters.schema import RankedGroup  # noqa: E402

# ------------------------------------------------------------- the judge

JUDGE_PROMPT_VERSION = "exp012-v1"
# Cache/labeler namespace passed to LabelCache -- bump alongside
# JUDGE_PROMPT_VERSION if the prompt text below ever changes, so an old
# cache never silently mixes scores from two different rubrics.
JUDGE_LABELER_NAME = "oogiri-funniness-%s" % JUDGE_PROMPT_VERSION

# Written plainly per the task spec: rate funniness of this response to
# this prompt, 1-10, one number. No rubric decomposition, no persona --
# this IS the "naked" judge EXP-012 is certifying (structured/gated
# variants are EXP-013/014's job, registered as a follow-on comparison
# against this same certification).
FUNNINESS_JUDGE_PROMPT = """Rate the funniness of this response to this prompt.

Prompt: {context}

Response: {candidate}

On a scale from 1 (not funny at all) to 10 (extremely funny), how funny is this response? Output ONLY a single integer from 1 to 10, nothing else.
"""

_SCORE_RE = re.compile(r"(?<!\d)(10|[1-9])(?!\d)")


def _parse_score(raw: str) -> Optional[int]:
    """First standalone integer 1-10 on the first line. (?<!\\d)/(?!\\d)
    guards keep '100' or '23' from being misread as '10'/'2' -- same
    pattern as benchmark/banter.py's `_parse_score`."""
    if not raw or not raw.strip():
        return None
    first_line = raw.strip().splitlines()[0]
    m = _SCORE_RE.search(first_line)
    return int(m.group(1)) if m else None


def judge_once(context: str, candidate: str, complete: Callable[[str], str],
               retries: int = 1) -> Optional[int]:
    """Raw 1-10 integer funniness score, retried up to `retries` times on
    a parse failure. Returns None (an UNPARSEABLE-equivalent sentinel,
    mirroring rejector.py/banter.py's convention) on repeated failure --
    NEVER cached by the caller, so a future run gets a fresh attempt
    rather than being stuck with a frozen non-answer."""
    prompt = FUNNINESS_JUDGE_PROMPT.format(context=context, candidate=candidate)
    for _ in range(retries + 1):
        raw = complete(prompt)
        score = _parse_score(raw)
        if score is not None:
            return score
    return None


# ------------------------------------------------------------- budget guard


class BudgetExceeded(Exception):
    """Raised by the wrapped complete() once --max-calls underlying judge
    calls have been made. Caught by `run_certification` to stop the sweep
    cleanly and still emit a partial report -- see that function's
    docstring."""


def make_budgeted_complete(complete: Callable[[str], str], stats: Dict,
                           max_calls: int) -> Callable[[str], str]:
    """Wrap a provider's complete() so every ACTUAL call (including
    retries -- a retry is a real network/CLI call and a real budget cost)
    increments `stats["calls"]` and raises BudgetExceeded before making a
    call that would exceed `max_calls`. Cache hits never reach this
    wrapper at all (score_candidate checks the cache first), so cached
    answers are correctly free against the budget."""
    def wrapped(prompt: str) -> str:
        if stats["calls"] >= max_calls:
            raise BudgetExceeded(
                "certify_judge_oogiri: hit --max-calls=%d underlying judge "
                "calls" % max_calls)
        stats["calls"] += 1
        return complete(prompt)
    return wrapped


# --------------------------------------------------------------- caching

# Unit separator: context and candidate text may legitimately contain any
# printable character (including "|" or ":"), so a plain string-join could
# collide two distinct (context, candidate) pairs onto the same cache key.
# U+001F is vanishingly unlikely to appear in scraped joke/prompt text and
# is never itself content.
_CACHE_KEY_SEP = "\x1f"


def cache_key(context: str, candidate: str) -> str:
    return context + _CACHE_KEY_SEP + candidate


def score_candidate(context: str, candidate_text: str,
                    complete: Callable[[str], str], cache: LabelCache,
                    stats: Dict, retries: int = 1) -> Optional[float]:
    """Cache-checked judge score for one (context, candidate) pair,
    normalized to [0, 1] -- exactly the contract env/rewards.py's
    `JudgePreferenceReward` requires of any judge callable it wraps, so
    this cache doubles as a ready-made judge-score cache for that class
    later, not just a certification artifact.

    Reuses benchmark/relabel.py's `LabelCache` unmodified (sha256-keyed
    JSONL, flush-on-write, resume-safe) -- keyed by (JUDGE_LABELER_NAME,
    cache_key(context, candidate)), so identical (prompt, response) pairs
    across repeated invocations of this script are judged exactly once,
    ever, regardless of how many times the run is interrupted and resumed.

    Returns None (never cached) if the judge's output could not be parsed
    even after `retries` retries -- an unparseable candidate is dropped
    from that prompt's Spearman correlation, not treated as a 0 score.
    """
    key = cache_key(context, candidate_text)
    cached = cache.get(JUDGE_LABELER_NAME, key)
    if cached is not None:
        stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        return cached
    raw_score = judge_once(context, candidate_text, complete, retries=retries)
    if raw_score is None:
        stats["unparseable"] = stats.get("unparseable", 0) + 1
        return None
    normalized = raw_score / 10.0
    cache.put(JUDGE_LABELER_NAME, key, normalized)
    return normalized


# ------------------------------------------------------- rank correlation


def _ranks(values: Sequence[float]) -> List[float]:
    """Average (fractional) ranks, 1-indexed, standard tie-handling: tied
    values all receive the mean of the rank positions they jointly
    occupy. E.g. ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0] -- the
    two 20s jointly occupy positions 2 and 3, so both get rank 2.5. This
    is the tie-corrected definition Spearman's rho uses in every standard
    reference (and what scipy.stats.rankdata(method='average') computes),
    not a simplification of it."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with average-rank tie handling: Pearson
    correlation computed over `_ranks(xs)` / `_ranks(ys)` rather than the
    raw values -- the standard, algebraically-equivalent way to define
    tie-corrected Spearman's rho (mirrors benchmark/validate_banter_judge
    .py's `_pearson` helper, applied to ranks instead of raw scores).

    Returns None when undefined: fewer than 2 paired observations, or
    either series has zero rank variance (every value tied -- e.g. a
    prompt where the judge gave every candidate the same score, or every
    candidate has the same human `star` count). None is a real
    "inconclusive," not a fabricated 0.0 -- a 0.0 rho would misleadingly
    read as "measured, no correlation" rather than "undefined here."
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("spearman_rho: xs and ys must be the same length "
                         "(got %d and %d)" % (n, len(ys)))
    if n < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


# ------------------------------------------------------------ sampling


def sample_groups(groups: List[RankedGroup], n_prompts: int, seed: int,
                  min_candidates: int) -> List[RankedGroup]:
    """Deterministic sample of `n_prompts` RankedGroups with >= min_candidates
    candidates. Sorted by `source_id` FIRST (defensive -- data_adapters
    /oogiri.py already emits groups in a deterministic, question-sorted
    order with source_ids assigned in that same order, but sorting again
    here means this function's determinism does not silently depend on an
    upstream ordering guarantee it doesn't itself enforce) so that
    `random.Random(seed).sample(...)` over a fixed input order is
    reproducible byte-for-byte across machines and runs. Returns every
    eligible group (unsampled) if `n_prompts >= len(eligible)` --
    `random.sample` raises if asked to draw more than the population."""
    eligible = sorted(
        (g for g in groups if len(g.candidates) >= min_candidates),
        key=lambda g: g.source_id)
    if n_prompts >= len(eligible):
        return eligible
    return random.Random(seed).sample(eligible, n_prompts)


def load_and_sample(
    n_prompts: int, seed: int, allowed_licenses: List[str],
    min_candidates: int = 2,
    data_dir: Optional[str] = None,
    jsonl_path: Optional[str] = None,
) -> Tuple[List[RankedGroup], OogiriLoadStats]:
    """Load Oogiri-GO RankedGroups and deterministically sample
    `n_prompts` of them. `allowed_licenses` has NO default and is
    forwarded verbatim to `load_ranked_groups` -- data_adapters/firewall
    .py's license firewall -- so a caller of this function cannot admit
    research-only data any more silently than a direct caller of the
    adapter could (Oogiri-GO is hardcoded `research_only`; see
    data_adapters/oogiri.py's module docstring for the flagged,
    unresolved MIT-tag discrepancy this project's conservative default
    stance means it is NOT treated as commercial_safe)."""
    groups, load_stats = load_ranked_groups(
        allowed_licenses=allowed_licenses, data_dir=data_dir,
        jsonl_path=jsonl_path)
    sampled = sample_groups(groups, n_prompts, seed, min_candidates)
    return sampled, load_stats


# ------------------------------------------------------------ per-prompt


def certify_group(group: RankedGroup, complete: Callable[[str], str],
                  cache: LabelCache, stats: Dict, retries: int = 1) -> Dict:
    """Score every candidate in one RankedGroup with the judge and compute
    the Spearman rho between judge scores and the human-consensus `.score`
    field (Oogiri-GO's `star` count -- see module docstring's registered-
    design-deviation note for what this signal actually is).

    A candidate whose judge score is unparseable (None) is dropped from
    the rho computation entirely (not scored as 0) -- `n_scored` records
    how many of `n_candidates` actually contributed. `rho` is None if
    fewer than 2 candidates were scored, or if either the human or judge
    scores have zero variance across the scored candidates (see
    `spearman_rho`'s docstring).

    May raise BudgetExceeded partway through a group (propagated to the
    caller, uncaught here) -- `run_certification` is the boundary that
    catches it and decides what "partial" means for the run as a whole."""
    human_scores: List[float] = []
    judge_scores: List[float] = []
    n_unparseable = 0
    for c in group.candidates:
        s = score_candidate(group.context, c.text, complete, cache, stats,
                            retries=retries)
        if s is None:
            n_unparseable += 1
            continue
        human_scores.append(c.score)
        judge_scores.append(s)

    rho = spearman_rho(human_scores, judge_scores) if len(human_scores) >= 2 else None
    return {
        "source_id": group.source_id,
        "context": group.context,
        "n_candidates": len(group.candidates),
        "n_scored": len(human_scores),
        "n_unparseable": n_unparseable,
        "rho": rho,
        "human_scores": human_scores,
        "judge_scores": judge_scores,
    }


def run_certification(
    groups: List[RankedGroup], complete: Callable[[str], str],
    cache: LabelCache, max_calls: int, retries: int = 1,
    raw_log=None, bootstrap_seed: int = 0,
) -> Dict:
    """Certify the judge over every group in `groups`, in order, stopping
    cleanly (no exception escapes this function) if --max-calls underlying
    judge calls are exhausted partway through. The group in progress when
    the budget is hit is dropped from the output entirely (its partial
    candidate scores stay in the on-disk cache and are reused free on the
    next invocation -- see score_candidate's docstring -- but the group's
    rho is not computed from an incomplete candidate set this run).

    Returns a dict with the per-prompt results, the aggregate mean rho +
    bootstrap CI (benchmark/stats.py's `bootstrap_ci`, reused unmodified),
    and call/cache/budget statistics -- everything `main()` needs to write
    report.json, whether the run completed or stopped on budget.
    """
    stats: Dict = {"calls": 0, "cache_hits": 0, "unparseable": 0}
    budgeted = make_budgeted_complete(complete, stats, max_calls)

    per_prompt: List[Dict] = []
    budget_exhausted = False
    for group in groups:
        try:
            result = certify_group(group, budgeted, cache, stats,
                                   retries=retries)
        except BudgetExceeded:
            budget_exhausted = True
            break
        per_prompt.append(result)
        if raw_log is not None:
            raw_log.write(json.dumps(result) + "\n")
            raw_log.flush()

    rhos = [p["rho"] for p in per_prompt if p["rho"] is not None]
    mean_rho = sum(rhos) / len(rhos) if rhos else None
    ci = bootstrap_ci(rhos, seed=bootstrap_seed) if len(rhos) >= 2 else None

    return {
        "per_prompt": per_prompt,
        "n_prompts_requested": len(groups),
        "n_prompts_scored": len(per_prompt),
        "n_prompts_with_valid_rho": len(rhos),
        "n_prompts_degenerate": len(per_prompt) - len(rhos),
        "mean_rho": mean_rho,
        "mean_rho_bootstrap_ci": ci,
        "call_stats": {**stats, "max_calls": max_calls,
                      "budget_exhausted": budget_exhausted},
    }


# --------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=30,
                    help="number of Oogiri prompt groups to certify "
                         "against (registered default: 30)")
    ap.add_argument("--seed", type=int, default=20260717,
                    help="deterministic sample seed (registered: 20260717)")
    ap.add_argument("--judge", default="claude:haiku",
                    help="provider spec, resolved ONLY through "
                         "get_provider() (benchmark/providers.py) so the "
                         "neutral-cwd guard applies -- see that module for "
                         "why a rejector/judge must never run from a cwd "
                         "with this repo's own CLAUDE.md/AGENTS.md on disk.")
    ap.add_argument("--min-candidates", type=int, default=2,
                    help="skip prompt groups with fewer candidates than "
                         "this (2 matches data_adapters/oogiri.py's own "
                         "MIN_GROUP_SIZE -- the adapter has already "
                         "dropped every group below this floor).")
    ap.add_argument("--max-calls", type=int, default=3500,
                    help="hard cap on underlying judge calls (including "
                         "retries); stop cleanly with a partial report if "
                         "hit (registered budget: ~3,500)")
    ap.add_argument("--retries", type=int, default=1,
                    help="parse-failure retries per candidate judge call")
    ap.add_argument("--allowed-licenses", nargs="+", default=["research_only"],
                    help="license classes admitted from the firewall "
                         "(data_adapters/firewall.py). Oogiri-GO is "
                         "hardcoded research_only (data_adapters/oogiri.py) "
                         "-- passed explicitly here, never a silent "
                         "default inside the loader itself. Sam's final "
                         "commercial-safe/research-only license call on "
                         "this dataset is still pending (EXPERIMENT_LOG.md "
                         "EXP-012) -- this flag is the visible, overridable "
                         "seam for that decision, not a hidden constant.")
    ap.add_argument("--data-dir", default=None,
                    help="override GOOD_HUMORED_DATA_DIR (default: "
                         "~/Experiments/good-humored-data)")
    ap.add_argument("--out", required=True,
                    help="path to write report.json (parent dir also "
                         "holds label_cache.jsonl and the raw per-prompt "
                         "log)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    command = ("python3 -m env.certify_judge_oogiri --n-prompts %d --seed %d "
              "--judge %s --min-candidates %d --max-calls %d --retries %d "
              "--allowed-licenses %s --out %s" %
              (args.n_prompts, args.seed, args.judge, args.min_candidates,
               args.max_calls, args.retries, " ".join(args.allowed_licenses),
               args.out))
    print("command: %s" % command)

    groups, load_stats = load_and_sample(
        args.n_prompts, args.seed, args.allowed_licenses,
        min_candidates=args.min_candidates, data_dir=args.data_dir)
    print("sampled %d prompts (min_candidates=%d) from %d Oogiri-GO ranked "
          "groups available (%d T2T rows, %d total rows scanned)" %
          (len(groups), args.min_candidates, load_stats.ranked_groups_emitted,
           load_stats.t2t_rows, load_stats.total_rows))
    if groups:
        sizes = [len(g.candidates) for g in groups]
        print("candidates/prompt in this sample: min=%d max=%d mean=%.2f" %
              (min(sizes), max(sizes), sum(sizes) / len(sizes)))

    judge_complete = get_provider(args.judge)
    cache = LabelCache(out_path.parent / "label_cache.jsonl")

    run_stamp = time.strftime("%Y%m%dT%H%M%S")
    raw_path = out_path.parent / ("raw_per_prompt.%s.jsonl" % run_stamp)
    t0 = time.time()
    try:
        with open(raw_path, "a") as raw_log:
            result = run_certification(
                groups, judge_complete, cache, max_calls=args.max_calls,
                retries=args.retries, raw_log=raw_log,
                bootstrap_seed=args.seed)
    finally:
        cache.close()
    wall = time.time() - t0

    report = {
        "experiment": "EXP-012-judge-certification-oogiri",
        "command": command,
        "run_stamp": run_stamp,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt": FUNNINESS_JUDGE_PROMPT,
        "judge_provider": args.judge,
        "seed": args.seed,
        "n_prompts_requested_cli": args.n_prompts,
        "min_candidates": args.min_candidates,
        "allowed_licenses": args.allowed_licenses,
        "data_source": (
            "oogiri-go (data_adapters/oogiri.py), research_only. "
            "REGISTERED-DESIGN DEVIATION: EXP-012's registration and "
            "CLAUDE.md's Research Direction section describe Oogiri-"
            "Master's ~100-candidates/~100-independent-judges methodology; "
            "Oogiri-Master has no public pre-built release (oogiri.py "
            "module docstring), so this certification runs against "
            "Oogiri-GO instead -- candidates are scored by the 'star' "
            "aggregate like/popularity count (a WEAKER human signal per "
            "that module's own docstring), averaging ~6.3 candidates/"
            "prompt, not ~100. See this script's module docstring for the "
            "full statement."),
        "load_stats": vars(load_stats),
        "wall_seconds": round(wall, 1),
        "cache_path": str(out_path.parent / "label_cache.jsonl"),
        "raw_log_path": str(raw_path),
        "predicted_mean_rho_registered": 0.40,
        "floor_mean_rho_registered": 0.10,
        **result,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=== EXP-012 judge certification (%s, %d prompts sampled) ===" %
          (args.judge, len(groups)))
    print("scored %d/%d prompts (budget_exhausted=%s)" %
          (result["n_prompts_scored"], result["n_prompts_requested"],
           result["call_stats"]["budget_exhausted"]))
    print("mean rho = %s over %d/%d scored prompts with a valid (non-"
          "degenerate) rho" %
          ("None" if result["mean_rho"] is None else "%.4f" % result["mean_rho"],
           result["n_prompts_with_valid_rho"], result["n_prompts_scored"]))
    if result["mean_rho_bootstrap_ci"]:
        ci = result["mean_rho_bootstrap_ci"]
        print("bootstrap 95%% CI: [%.4f, %.4f] (%s, n_boot=%d)" %
              (ci["lo"], ci["hi"], ci["method"], ci["n_boot"]))
    print("calls=%d cache_hits=%d unparseable_candidates=%d wall=%.1fs" %
          (result["call_stats"]["calls"], result["call_stats"]["cache_hits"],
           result["call_stats"]["unparseable"], wall))
    print("report: %s" % out_path)


if __name__ == "__main__":
    main()

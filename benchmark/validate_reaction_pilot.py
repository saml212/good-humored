#!/usr/bin/env python3
"""EXP-023a -- API Monte-Carlo reaction pilot (registered 2026-07-30).

Samples k replies from API chat models prompted as a texting friend
(NO evaluative instruction) reacting to NYCC caption entries;
reaction_rate = fraction of replies opening with a laughter marker;
metric = mean within-contest Spearman rho(reaction_rate, mean rating).
The sampled estimate of EXP-023's logprob quantity -- the cheap
de-risk before GPU time. Design pinned in EXPERIMENT_LOG.md.

Usage:
  python3 -m benchmark.validate_reaction_pilot \
      --providers deepseek,qwen,grok \
      --out experiment-runs/2026-07-30-exp023a-reaction-pilot
"""

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.providers import make_openai_compat  # noqa: E402

DATA_ROOT = Path.home() / "Experiments" / "good-humored-data" / "raw" / "nycc-full"
SUMMARIES = DATA_ROOT / "raw" / "nextml-summaries" / "summaries"
SPLITS = Path("/Volumes/1TB_SSD/good-humored-data/nycc/splits.json")
DESCRIPTIONS = Path("/Volumes/1TB_SSD/good-humored-data/nycc/contest_descriptions.json")

N_CONTESTS = 12
CAPTIONS_PER_CONTEST = 24  # top 8 / middle 8 / bottom 8 by mean rating
K_SAMPLES = 8
BASE_RATE_BAND = (0.02, 0.80)  # outside -> provider degenerate (pinned)
MIN_CONTESTS_SCORED = 8   # audit #2: fewer non-degenerate contests ->
                          # provider UNDERPOWERED, excluded from primary
MAX_ERROR_FRACTION = 0.20  # audit #4: more failed calls -> INCOMPLETE

LAUGHTER_MARKERS = ("haha", "ha ha", "hehe", "heh", "lol", "lolol",
                    "lmao", "lmfao", "dead", "dying",
                    "\U0001F602", "\U0001F923", "\U0001F480")
ONSET_WINDOW = 12  # first N normalized chars searched for a marker

SYSTEM_PROMPT = ("You are texting with your close friend. Reply with a "
                 "single short, natural text message.")
USER_TEMPLATE = ("Your friend is reading New Yorker caption contest entries out "
                 "loud from the internet. The cartoon: {description}\n\nThey "
                 "read this one: \"{caption}\"")


def is_laughter_onset(reply: str) -> bool:
    head = reply.strip().strip('"“”').lower()[:ONSET_WINDOW]
    return any(m in head for m in LAUGHTER_MARKERS)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rho with average ranks for ties. None when either side
    is constant (rho undefined -- happens when reaction_rate is all-0
    for a contest; treated upstream as a degenerate contest)."""
    def avg_ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            rank = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                ranks[order[t]] = rank
            i = j + 1
        return ranks

    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("spearman needs matched sequences of length >= 3")
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = avg_ranks(xs), avg_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def load_pilot_set() -> List[Dict]:
    """The pinned deterministic pilot set: first N_CONTESTS val contests
    with descriptions; 24 stratified captions each."""
    splits = json.loads(SPLITS.read_text())
    descriptions = json.loads(DESCRIPTIONS.read_text())
    val_ids = [c for c in sorted(splits["contest_ids"]["val"])
               if str(c) in descriptions or c in descriptions]
    contests = []
    for cid in val_ids:
        if len(contests) >= N_CONTESTS:
            break
        rows = _load_contest_rows(cid)
        if rows is None or len(rows) < CAPTIONS_PER_CONTEST:
            continue
        desc = descriptions.get(str(cid)) or descriptions.get(cid)
        contests.append({"contest_id": cid, "description": desc,
                         "captions": _stratify(rows)})
    if len(contests) < N_CONTESTS:
        raise RuntimeError("only %d usable contests found" % len(contests))
    return contests


def _load_contest_rows(cid: int) -> Optional[List[Dict]]:
    """Merge ALL of a contest's summary files (some contests were run
    under two NEXT sampling algorithms, e.g. LilUCB + RoundRobin, whose
    per-caption means only agree at rho~0.2 — audit #1). Votes-weighted
    mean across files uses all the data instead of an alphabetic
    accident picking one methodology. Files used are recorded."""
    matches = sorted(SUMMARIES.glob("%d*.csv" % cid))
    # Exact-id match only: contest 510 must not pick up 5100.csv.
    matches = [m for m in matches
               if m.stem == str(cid) or m.stem.startswith("%d_" % cid)]
    if not matches:
        return None
    acc: Dict[str, Dict] = {}
    for path in matches:
        with open(path) as f:
            for row in csv.DictReader(f):
                caption = row["caption"].strip()
                if not caption:
                    continue
                try:
                    mean, votes = float(row["mean"]), int(row["votes"])
                except (KeyError, ValueError):
                    continue
                key = caption.lower()
                if key in acc:
                    a = acc[key]
                    total = a["votes"] + votes
                    if total:
                        a["mean"] = (a["mean"] * a["votes"]
                                     + mean * votes) / total
                    a["votes"] = total
                else:
                    acc[key] = {"caption": caption, "mean": mean,
                                "votes": votes}
    rows = list(acc.values())
    for r in rows:
        r["source_files"] = [m.name for m in matches]
    return rows


def _stratify(rows: List[Dict]) -> List[Dict]:
    rows = sorted(rows, key=lambda r: -r["mean"])
    n = len(rows)
    third = CAPTIONS_PER_CONTEST // 3
    mid_start = (n - third) // 2
    return (rows[:third] + rows[mid_start:mid_start + third]
            + rows[-third:])


def run_provider(provider: str, contests: List[Dict], cache_path: Path,
                 workers: int) -> Dict:
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if not line.strip():
                continue
            try:  # a hard-kill mid-write can truncate the last line
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[rec["key"]] = rec

    complete = make_openai_compat(provider, temperature=1.0)
    jobs = []
    for c in contests:
        for ci, cap in enumerate(c["captions"]):
            prompt = (SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
                description=c["description"], caption=cap["caption"]))
            for s in range(K_SAMPLES):
                key = "%s|%d|%d|%d" % (provider, c["contest_id"], ci, s)
                if key not in cache:
                    jobs.append((key, prompt))

    def call(job):
        key, prompt = job
        try:
            reply = complete(prompt)
            return {"key": key, "reply": reply[:200],
                    "laugh": is_laughter_onset(reply)}
        except Exception as e:  # recorded, not silently retried further
            return {"key": key, "reply": None, "error": str(e)[:200],
                    "laugh": None}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, \
            open(cache_path, "a") as fh:
        for rec in pool.map(call, jobs):
            cache[rec["key"]] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 100 == 0:
                print("[%s] %d/%d calls" % (provider, done, len(jobs)),
                      flush=True)

    per_contest = []
    all_rates, errors = [], 0
    for c in contests:
        rates, means = [], []
        for ci, cap in enumerate(c["captions"]):
            laughs, valid = 0, 0
            for s in range(K_SAMPLES):
                rec = cache.get("%s|%d|%d|%d"
                                % (provider, c["contest_id"], ci, s))
                if rec is None or rec.get("laugh") is None:
                    errors += 1
                    continue
                valid += 1
                laughs += 1 if rec["laugh"] else 0
            if valid:
                rates.append(laughs / valid)
                means.append(cap["mean"])
        rho = spearman(rates, means) if len(rates) >= 3 else None
        per_contest.append({"contest_id": c["contest_id"], "rho": rho,
                            "n_captions": len(rates)})
        all_rates.extend(rates)

    rhos = [p["rho"] for p in per_contest if p["rho"] is not None]
    base_rate = sum(all_rates) / len(all_rates) if all_rates else 0.0
    degenerate = not (BASE_RATE_BAND[0] <= base_rate <= BASE_RATE_BAND[1])
    expected_calls = len(contests) * CAPTIONS_PER_CONTEST * K_SAMPLES
    incomplete = errors > MAX_ERROR_FRACTION * expected_calls
    underpowered = len(rhos) < MIN_CONTESTS_SCORED
    return {
        "provider": provider,
        "mean_rho": sum(rhos) / len(rhos) if rhos else None,
        "counts_for_primary": not (degenerate or incomplete or underpowered),
        "n_contests_scored": len(rhos),
        "n_degenerate_contests": len(per_contest) - len(rhos),
        "base_laughter_rate": base_rate,
        "degenerate": degenerate,
        "underpowered": underpowered,
        "incomplete": incomplete,
        "errors": errors,
        "per_contest": per_contest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--providers", default="deepseek,qwen,grok")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contests = load_pilot_set()
    print("pilot set: %d contests, %d captions each, k=%d"
          % (len(contests), CAPTIONS_PER_CONTEST, K_SAMPLES))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_set.json").write_text(json.dumps(
        [{"contest_id": c["contest_id"],
          "captions": [cap["caption"] for cap in c["captions"]]}
         for c in contests], indent=2))

    t0 = time.time()
    results = {"k": K_SAMPLES, "n_contests": len(contests),
               "markers": LAUGHTER_MARKERS, "providers": {}}
    for provider in args.providers.split(","):
        provider = provider.strip()
        print("== %s ==" % provider, flush=True)
        results["providers"][provider] = run_provider(
            provider, contests, args.out / ("cache_%s.jsonl" % provider),
            args.workers)
    results["wall_seconds"] = round(time.time() - t0, 1)

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    shutil.copy2(__file__, args.out / Path(__file__).name)

    for name, r in results["providers"].items():
        print("%-10s mean_rho=%s  base_rate=%.3f%s  contests=%d  errors=%d"
              % (name,
                 "%.3f" % r["mean_rho"] if r["mean_rho"] is not None else "n/a",
                 r["base_laughter_rate"],
                 "  DEGENERATE" if r["degenerate"] else "",
                 r["n_contests_scored"], r["errors"]))
    print("results -> %s" % (args.out / "results.json"))


if __name__ == "__main__":
    main()

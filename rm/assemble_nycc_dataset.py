#!/usr/bin/env python3
"""Assemble the emulator-v1 training set from NYCC-Zhang ranked groups.

Emits train/val jsonl with one row per caption:
  {"contest": id, "context": description-or-empty, "text": caption,
   "target": within-contest percentile in [0,1], "score": raw mean,
   "raters": n}

The percentile target IS the quantile-emulator objective: rank
position within the contest, not absolute funniness — popularity-bias
free by construction. License class research_only is asserted via the
firewall (this emulator is for the INTERNAL research loop; a
buyer-facing emulator retrains on the commercial-safe bucket).

Run locally (data lives on this machine), rsync the jsonl to the box.
"""

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_adapters.nycc_zhang import load_ranked_groups  # noqa: E402


def percentiles(cands):
    """Average-rank percentile of each candidate's score, in [0,1]."""
    order = sorted(range(len(cands)), key=lambda i: cands[i].score)
    ranks = [0.0] * len(cands)
    i = 0
    while i < len(order):
        j = i
        while (j + 1 < len(order)
               and cands[order[j + 1]].score == cands[order[i]].score):
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    denom = max(len(cands) - 1, 1)
    return [r / denom for r in ranks]


def emit(split, out_path, cap):
    groups, stats = load_ranked_groups(
        allowed_license_classes=["research_only"], split=split,
        max_candidates_per_group=cap)
    n = 0
    with open(out_path, "w") as f:
        for g in groups:
            desc = (g.context or "").strip()
            pcts = percentiles(g.candidates)
            for c, p in zip(g.candidates, pcts):
                f.write(json.dumps({
                    "contest": g.group_id, "context": desc[:400],
                    "text": c.text, "target": round(p, 5),
                    "score": round(c.score, 4), "raters": c.rater_count,
                }) + "\n")
                n += 1
    print("%s: %d rows, %d contests -> %s" % (split, n, len(groups), out_path))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-cap", type=int, default=2000,
                    help="max candidates per training contest (balance)")
    ap.add_argument("--val-cap", type=int, default=1000,
                    help="max candidates per val contest (eval cost)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emit("train", out / "nycc_train.jsonl", args.train_cap)
    emit("val", out / "nycc_val.jsonl", args.val_cap)


if __name__ == "__main__":
    main()

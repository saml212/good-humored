#!/usr/bin/env python3
"""Per-model environment characterization report card.

Pivots curation_master batch_stats (recent same-version window) into
one row per (model, provocation_rate) and one summary row per model:
anchoredness floor, self-repetition, audience reaction (diagnostic,
within-audience validity only), cross-session trigram diversity,
agreement-opener rate (sycophancy attractor), RP-asterisk register
rate. This is the artifact a buyer sees first, and the pre/post-RL
delta table the product story rests on.

Usage:
  python3 -m env.report_card --master /data/good-humored/runs/curation_master.json \
      --min-batch 66 [--out report.json]
"""

import argparse
import collections
import json


COLS = ("curation", "floor", "react", "tridiv", "agree", "asterisk")


def cell_rows(master, min_batch):
    cells = collections.defaultdict(list)
    for b, st in master["batch_stats"].items():
        num = int(b.split("_")[-1])
        if num < min_batch or "policy_agreement_rate" not in st:
            continue
        c = st["config"]
        cells[(c["model"], c["provocation_rate"])].append({
            "curation": st["mean_curation"],
            "floor": st["mean_floor_pass"],
            "react": st["mean_reaction_L"],
            "tridiv": st["motifs"]["trigram_diversity"],
            "agree": st["policy_agreement_rate"],
            "asterisk": st.get("policy_asterisk_rate", 0.0),
        })
    return cells


def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--min-batch", type=int, default=66,
                    help="first batch of the current prompt version "
                         "(cross-version scores are not comparable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    master = json.loads(open(args.master).read())
    cells = cell_rows(master, args.min_batch)
    by_model = collections.defaultdict(list)
    for (model, p), rows in cells.items():
        by_model[model].extend(rows)

    hdr = "%-16s %-5s %8s %7s %7s %7s %7s %9s %4s"
    print(hdr % ("model", "P", "curation", "floor", "react",
                 "tridiv", "agree", "asterisk", "n"))
    report = {"min_batch": args.min_batch, "cells": [], "models": {}}
    for (model, p), rows in sorted(cells.items()):
        vals = {k: round(mean(rows, k), 3) for k in COLS}
        print(hdr % (model[:16], p, vals["curation"], vals["floor"],
                     vals["react"], vals["tridiv"], vals["agree"],
                     vals["asterisk"], len(rows)))
        report["cells"].append({"model": model, "provocation_rate": p,
                                **vals, "n_batches": len(rows)})
    print("-" * 78)
    for model, rows in sorted(by_model.items()):
        vals = {k: round(mean(rows, k), 3) for k in COLS}
        print(hdr % (model[:16], "all", vals["curation"], vals["floor"],
                     vals["react"], vals["tridiv"], vals["agree"],
                     vals["asterisk"], len(rows)))
        report["models"][model] = {**vals, "n_batches": len(rows)}
    print("\nnotes: reaction is diagnostic + within-audience only; "
          "agreement is a range metric (both extremes pathological); "
          "scores comparable only within a prompt version.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("report ->", args.out)


if __name__ == "__main__":
    main()

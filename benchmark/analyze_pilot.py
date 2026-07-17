"""Merge cascade-pilot lanes and compute the cross-model analysis.

The pilot runs in parallel lanes (one process per provider family), so
no single summary.json sees every model. This script pools the lanes'
RAW paths (primary, per EXP-004 pre-registration) and reports:

  - cross-model overlap over all models (raw + semantic-alongside)
  - per-model divergence / degradation table
  - opening-topic census (does every model start at the same well?)
  - shared-topic census (which topics do N models all visit?)

Usage:
  python3 -m benchmark.analyze_pilot \
      --pilot experiment-runs/2026-07-17-cascade-pilot --json analysis.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .label_space import LabelSpace
from .metrics import (cluster_switch_stats, cross_model_overlap,
                      depth_to_degradation, path_divergence)


def load_lanes(pilot_dir: Path) -> Dict[str, List[List[str]]]:
    """Pool raw paths from every lane's summary.json under pilot_dir.

    MERGES across lanes when the same model key appears in more than one
    lane's per_model — a "-fill-" lane topping up runs that failed in the
    main lane for that provider family is keyed by the same model string
    (e.g. "api:glm" in both lane-api and lane-api-fill-glm), and a plain
    overwrite here would silently drop whichever lane's summary.json
    sorts first, losing real completed runs from the final analysis. Run
    lists are concatenated (never deduped/reconciled beyond that): every
    metric downstream (path_divergence, cross_model_overlap) is an
    unordered function over the run list, so concatenation order is not
    load-bearing.
    """
    paths: Dict[str, List[List[str]]] = {}
    failures: List[Dict] = []
    for summary_file in sorted(pilot_dir.glob("*/summary.json")):
        with open(summary_file) as f:
            s = json.load(f)
        failures.extend(s.get("failures", []))
        for model, pm in s.get("per_model", {}).items():
            # MERGE, not overwrite — RAW paths, primary per EXP-004.
            paths.setdefault(model, []).extend(pm["paths"])
    return paths, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True,
                    help="parent dir containing lane subdirs")
    ap.add_argument("--json", default=None, help="write full analysis here")
    args = ap.parse_args()

    pilot = Path(args.pilot)
    paths, failures = load_lanes(pilot)
    if len(paths) < 2:
        raise SystemExit("fewer than 2 models with >=2 runs — nothing to "
                         "compare (found: %s)" % sorted(paths))

    # One label space fit on the POOLED vocabulary — the per-lane fits in
    # each summary.json saw only their own lane's labels, so their
    # semantic views are not comparable across lanes. Semantic stays a
    # secondary view here too.
    ls = LabelSpace().fit([t for ps in paths.values() for p in ps for t in p])
    canon = {m: ls.canonize_paths(ps) for m, ps in paths.items()}

    analysis: Dict = {
        "n_models": len(paths),
        "failures": failures,
        "cross_model": cross_model_overlap(paths),
        "cross_model_semantic": cross_model_overlap(canon),
        "per_model": {},
    }

    for m, ps in sorted(paths.items()):
        analysis["per_model"][m] = {
            "n_runs": len(ps),
            "divergence": path_divergence(ps),
            "degradation": [depth_to_degradation(p) for p in ps],
            "cluster_switch_semantic": [
                cluster_switch_stats(p, ls.canon) for p in ps],
            "opening_topics": [p[0] for p in ps if p],
        }

    # Census: which topics does the ecosystem share? (semantic canon here
    # on purpose — 'flying' and 'airplanes' are one shared well; labeled
    # as the semantic view it is.)
    model_topics = {m: {ls.canon(t) for p in ps for t in p}
                    for m, ps in paths.items()}
    census: Counter = Counter()
    for topics in model_topics.values():
        for t in topics:
            census[t] += 1
    analysis["shared_topic_census_semantic"] = [
        {"topic": t, "n_models": c}
        for t, c in census.most_common(40) if c >= 2]

    openings = Counter(t for pm in analysis["per_model"].values()
                       for t in pm["opening_topics"])
    analysis["opening_topic_census_raw"] = [
        {"topic": t, "n_runs": c} for t, c in openings.most_common(15)]

    if args.json:
        with open(args.json, "w") as f:
            json.dump(analysis, f, indent=2)

    # Human-readable digest
    cm = analysis["cross_model"]
    print("=== cascade pilot: %d models ===" % len(paths))
    print("cross-model mean jaccard (RAW, primary): %.3f" %
          cm["mean_cross_jaccard"])
    print("cross-model mean jaccard (semantic):     %.3f" %
          analysis["cross_model_semantic"]["mean_cross_jaccard"])
    print("cross-model mean prefix depth (RAW):     %.2f" %
          cm["mean_cross_prefix_depth"])
    print()
    print("%-14s %7s %8s %9s  %s" %
          ("model", "runs", "self-jac", "deg-depth", "openings"))
    for m, pm in sorted(analysis["per_model"].items()):
        degs = [d["depth"] for d in pm["degradation"]]
        deg_str = ",".join("-" if d is None else str(d) for d in degs)
        print("%-14s %7d %8.3f %9s  %s" %
              (m, pm["n_runs"], pm["divergence"]["set_jaccard"],
               deg_str, ",".join(pm["opening_topics"])))
    print()
    print("top shared topics (semantic): %s" % ", ".join(
        "%s(%d)" % (e["topic"], e["n_models"])
        for e in analysis["shared_topic_census_semantic"][:12]))
    if failures:
        print("\nFAILURES (%d): %s" % (len(failures), failures))


if __name__ == "__main__":
    main()

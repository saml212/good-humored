"""Sampling-based diversity metrics — the ones the claim says are fake-able.

docs/BENCHMARK.md §1: "sampling-based diversity metrics can be bought with
temperature ... a model reading down a memorized list at temperature 1.0
looks 'diverse' to any sampling-based metric." This module computes
exactly those metrics, over the JOKE TEXT (not the topic path), so EXP-007
can show them moving with temperature while `benchmark.metrics.
path_divergence` — computed over the same run-set's TOPIC labels — barely
moves. The two lenses over one run-set is the whole demonstration; see
`aggregate_run_set` below.

Trigram machinery (normalize -> trigram set -> Jaccard) reuses
`benchmark.joke_novelty`'s implementation rather than re-deriving it, so
"trigram" means the same thing everywhere in this repo. Pure stdlib,
Python 3.9 compatible, no network/model calls.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from .joke_novelty import norm, trigram_jaccard, trigrams
from .metrics import path_divergence

# --------------------------------------------------------------- distinct-n


def _tokens(text: str) -> List[str]:
    return norm(text).split()


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Corpus-level distinct-n (Li et al. 2016): unique n-grams / total
    n-grams, pooled across every text in the set — a normalized
    type/token ratio, not a raw count. 1.0 = every n-gram emitted was
    unique (maximally diverse by this lens); low = heavy repetition.

    This is exactly the kind of number temperature can buy: rewording the
    same underlying idea in fresh vocabulary raises distinct-n even when
    the model is reading down a fixed list of topics (the cascade's
    path-divergence metric, over the same texts' topic labels, would not
    move at all in that case).

    0.0 for an empty input, or one with no text long enough to contain an
    n-gram of the requested length (never a ZeroDivisionError).
    """
    total = 0
    seen = set()
    for t in texts:
        grams = _ngrams(_tokens(t), n)
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


# --------------------------------------------------- pairwise trigram overlap


def mean_pairwise_trigram_jaccard(texts: Sequence[str]) -> float:
    """Mean trigram-Jaccard similarity over every pair of texts.

    HIGH = texts share a lot of surface n-gram structure (low surface
    diversity); temperature is expected to push this DOWN sharply even
    when the model is walking an identical topic path underneath —
    that gap (surface diversity drops, path divergence doesn't) is the
    empirical signature EXP-007 is registered to look for.
    """
    if len(texts) < 2:
        raise ValueError(
            "need >= 2 texts to measure pairwise trigram similarity "
            "(got %d)" % len(texts))
    grams = [trigrams(t) for t in texts]
    vals = [trigram_jaccard(grams[i], grams[j])
            for i in range(len(grams)) for j in range(i + 1, len(grams))]
    return sum(vals) / len(vals)


def sampling_diversity(texts: Sequence[str]) -> Dict[str, float]:
    """The full sampling-diversity bundle for one set of joke texts:
    distinct-1, distinct-2, and mean pairwise trigram-Jaccard. Requires
    >= 2 texts (mirrors `benchmark.metrics.path_divergence`'s bar for the
    same reason — a single text has no pairwise structure to measure)."""
    if len(texts) < 2:
        raise ValueError(
            "need >= 2 texts to measure sampling diversity (got %d)"
            % len(texts))
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "mean_pairwise_trigram_jaccard": mean_pairwise_trigram_jaccard(texts),
        "n_texts": float(len(texts)),
    }


# ------------------------------------------------------- run-set aggregation


def load_run_set(turns_files: Sequence[Path]) -> Dict[str, list]:
    """Load one model's run-set from its turns-*.jsonl files (one file
    per run, cascade.run_cascade's per-turn log). Returns:
      paths — list of topic-label lists, one per run (what
              `path_divergence` scores)
      jokes — every joke text pooled across all runs in the set (what
              `sampling_diversity` scores)
    Same run-set, two lenses — that pairing is the point.
    """
    paths: List[List[str]] = []
    jokes: List[str] = []
    for f in turns_files:
        path: List[str] = []
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                path.append(rec["topic"])
                jokes.append(rec["joke"])
        paths.append(path)
    return {"paths": paths, "jokes": jokes}


def aggregate_run_set(turns_files: Sequence[Path]) -> Dict:
    """The central EXP-007 comparison for one model's run-set: sampling
    diversity over the JOKES vs. path divergence over the TOPICS, from
    the exact same runs. `None` for either sub-metric when the run-set is
    too small to compute it (< 2 runs for path_divergence, < 2 pooled
    jokes for sampling_diversity) rather than raising — a run-set report
    spans many models/temperatures and one thin group should not abort
    the rest."""
    loaded = load_run_set(turns_files)
    paths, jokes = loaded["paths"], loaded["jokes"]
    out: Dict = {"n_runs": len(paths), "n_jokes": len(jokes)}
    out["sampling_diversity"] = (
        sampling_diversity(jokes) if len(jokes) >= 2 else None)
    out["path_divergence"] = (
        path_divergence(paths) if len(paths) >= 2 else None)
    return out


def group_turns_files_by_model(pilot_dir: Path) -> Dict[str, List[Path]]:
    """Group a pilot output dir's turns-<model>-r<NN>.jsonl files by
    model — same filename convention `joke_novelty.py` groups by."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(pilot_dir.rglob("turns-*.jsonl")):
        model = f.stem.replace("turns-", "").rsplit("-r", 1)[0]
        groups[model].append(f)
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True,
                    help="cascade run-set dir (or dir of dirs) containing "
                         "turns-<model>-r<NN>.jsonl files")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    groups = group_turns_files_by_model(Path(args.pilot))
    report = {m: aggregate_run_set(files) for m, files in groups.items()}

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)

    print("%-16s %6s %7s %10s %10s %10s %10s" %
          ("model", "runs", "jokes", "distinct1", "distinct2",
           "trigramJ", "set_jacc"))
    for m, v in sorted(report.items()):
        sd = v["sampling_diversity"] or {}
        pd = v["path_divergence"] or {}
        print("%-16s %6d %7d %10s %10s %10s %10s" % (
            m, v["n_runs"], v["n_jokes"],
            "%.3f" % sd["distinct_1"] if sd else "n/a",
            "%.3f" % sd["distinct_2"] if sd else "n/a",
            "%.3f" % sd["mean_pairwise_trigram_jaccard"] if sd else "n/a",
            "%.3f" % pd["set_jaccard"] if pd else "n/a"))


if __name__ == "__main__":
    main()

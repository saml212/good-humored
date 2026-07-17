"""Field-coverage checker for the topic labeler (EXP-008 addendum's
field-failure finding + the adversarial audit's two-tier redesign,
EXPERIMENT_LOG.md).

EXP-008 validated LABEL_PROMPT_V3 at paper grade ON THE FIXTURE
(invariance 1.000, ari_vs_gold 0.9237) -- but the fixture only ever
contains in-vocabulary topics, so *coverage* of the wild joke
distribution was untested by construction. A first fix just grew the
closed vocabulary and kept a catch-all `other` entry; the adversarial
audit rejected that design (finding B1): no closed vocabulary can cover
an unbounded long tail (13.6% of wild turns fall below even a 4-
occurrence threshold), and worse, a catch-all MERGES distinct rare
topics into one label, manufacturing false repeats -- exactly the
collapse-bias this benchmark cannot afford.

v4 (rejector.py: label_topic_v4) is therefore TWO-TIER: a joke either
matches a closed-vocabulary entry (tier "canon") or falls through to a
free, v2-style specific label (tier "free") -- never a category
placeholder. This module's job changed to match: it no longer reports a
catch-all rate against a pass/fail bar. It reports the ESCAPE RATE
(fraction of turns that landed in the free tier) and the UNPARSEABLE
rate (shape-guard parse failures), plus a full label histogram split by
tier. There is deliberately NO pass/fail baked in here -- EXP-010's
bars live in the experiment that consumes this report, not in the tool
(a high escape rate is not necessarily bad: it means the free tier is
doing its documented job of splitting rather than merging rare topics).

Generalized over any of the rejector's labelers (v2/v3/v4) so a future
v5 slots in as one more LABELERS entry:
  - v4 is genuinely two-tier (canon/free/unparseable, all three occur).
  - v3 is closed (no free tier exists in its prompt), so every non-
    UNPARSEABLE v3 label -- including `other` itself -- is reported
    under tier "canon"; its histogram_canon will show the old
    catch-all pattern directly (a large `other` bucket), which is a
    fair, honest way to show v3's failure mode side by side with v4's.
  - v2 is fully free (no vocabulary at all), so every non-UNPARSEABLE
    v2 label is tier "free" by definition.

Two ways to point this at data:
  1. A RELABEL dir already written by `benchmark.relabel` for the exact
     labeler being checked (lane-*/turns-*.relabel-<labeler>.jsonl) --
     the label is already on disk; this module reads it and derives the
     tier from the label + that labeler's vocabulary (no separate tier
     field needs to exist on disk -- tier is a deterministic function of
     (labeler, label), see `infer_tier`). Zero calls, zero cache, works
     with --dry-run or without it.
  2. A PILOT dir of raw generation output (lane-*/turns-*.jsonl, the
     `cascade.py` run_pilot shape) or a relabel dir written for a
     DIFFERENT labeler (whose files still carry the original joke text,
     just not the label being checked) -- the joke text is read, and a
     label is resolved per joke from, in order: (a) an already-relabeled
     file's matching field if one happens to be present, (b) the label
     cache (`benchmark.relabel.LabelCache` -- same on-disk format,
     reused directly so a v4 relabel run and a v4 field-coverage run
     share one cache file; only the label is cached, tier is always
     re-derived), (c) [only when --dry-run is NOT set] a live labeler
     call via `benchmark.providers.get_provider`, cached immediately
     after.

`--dry-run` NEVER calls a provider -- every joke without an already-known
label (from a relabel file or the cache) is counted as "missing" and
excluded from the rate math, not silently treated as canon or free.

Usage:
  # reproduce v3's already-measured catch-all failure, zero calls:
  python3 -m benchmark.field_coverage \\
      --dir experiment-runs/2026-07-17-cascade-pilot-v3-relabel \\
      --labeler v3 --dry-run

  # check v4 dry (cache-only; expect mostly "missing" before any real run):
  python3 -m benchmark.field_coverage \\
      --dir experiment-runs/2026-07-17-cascade-pilot-v3-relabel \\
      --labeler v4 --dry-run --cache experiment-runs-scratch/v4_cache.jsonl

  # actually label the wild corpus under v4 (real calls, cached):
  python3 -m benchmark.field_coverage \\
      --dir experiment-runs/2026-07-17-cascade-pilot-v3-relabel \\
      --labeler v4 --provider claude:haiku \\
      --cache experiment-runs-scratch/v4_cache.jsonl \\
      --out experiment-runs-scratch/v4_field_coverage_report.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .providers import get_provider
from .relabel import LabelCache, model_of
from .rejector import (LABEL_PROMPT_VERSION, LABEL_PROMPT_VERSION_V3,
                       LABEL_PROMPT_VERSION_V4, TIER_CANON, TIER_FREE,
                       TIER_UNPARSEABLE, UNPARSEABLE, VOCABULARY_PATH,
                       VOCABULARY_PATH_V4, label_topic, label_topic_v3,
                       label_topic_v4, load_vocabulary, normalize_label)

# labeler name -> closed-vocabulary set (normalized), or None if the
# labeler has no vocabulary at all (v2 is fully free-tier by
# definition). Used only by infer_tier -- deliberately a SEPARATE
# registry from benchmark.relabel's, so this module never needs to
# import or edit relabel.py to support a new labeler.
_VOCAB_SETS: Dict[str, Optional[set]] = {
    "v2": None,
    "v3": {normalize_label(v) for v in load_vocabulary(VOCABULARY_PATH)},
    "v4": {normalize_label(v) for v in load_vocabulary(VOCABULARY_PATH_V4)},
}


def infer_tier(labeler: str, label: Optional[str]) -> Optional[str]:
    """Tier is a pure function of (labeler, label): UNPARSEABLE is
    always its own tier; otherwise canon iff the labeler has a
    vocabulary AND the label is in it, free otherwise. Works uniformly
    whether `label` just came fresh off a labeler_fn call, out of the
    LabelCache (which only ever stores the label string), or out of an
    already-written relabel-<labeler>.jsonl file (whose schema, per
    benchmark.relabel, also only ever has the label, not a tier) --
    tier never needs its own on-disk field."""
    if label is None:
        return None
    if label == UNPARSEABLE:
        return TIER_UNPARSEABLE
    vocab = _VOCAB_SETS.get(labeler)
    if vocab is None:
        return TIER_FREE
    return TIER_CANON if label in vocab else TIER_FREE


def _wrap_v2(joke: str, complete: Callable[[str], str]) -> Tuple[str, str]:
    label = label_topic(joke, complete)
    return label, infer_tier("v2", label)


def _wrap_v3(joke: str, complete: Callable[[str], str]) -> Tuple[str, str]:
    label = label_topic_v3(joke, complete)
    return label, infer_tier("v3", label)


# labeler name -> (labeling fn returning (label, tier), version string
# recorded in the report). label_topic_v4 already returns (label, tier)
# natively (that IS its two-tier contract); v2/v3 are wrapped so every
# entry here has the same uniform shape.
LABELERS: Dict[str, Tuple[Callable[[str, Callable], Tuple[str, str]], str]] = {
    "v2": (_wrap_v2, LABEL_PROMPT_VERSION),
    "v3": (_wrap_v3, LABEL_PROMPT_VERSION_V3),
    "v4": (label_topic_v4, LABEL_PROMPT_VERSION_V4),
}


def _relabel_field(labeler: str) -> str:
    """Matches benchmark.relabel.new_field_name() exactly -- must stay in
    sync so this module can read files relabel.py wrote."""
    return "topic_%s" % labeler


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _model_of_safe(run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    try:
        return model_of(run_id)
    except ValueError:
        return None  # malformed run_id -- keep the turn, just unattributed


def find_raw_turn_files(root: Path) -> List[Path]:
    """Raw (un-relabeled) turns-*.jsonl files, one level of lane-*/ dirs
    deep or directly in root. Explicitly excludes anything already
    stamped '.relabel-' -- a naive 'turns-*.jsonl' glob would otherwise
    also match every relabel output file, since they share the prefix."""
    candidates = list(root.glob("lane-*/turns-*.jsonl"))
    if not candidates:
        candidates = list(root.glob("turns-*.jsonl"))
    return sorted(p for p in candidates if ".relabel-" not in p.name)


def find_relabel_files(root: Path, labeler: Optional[str] = None) -> List[Path]:
    """Already-relabeled turns files. `labeler=None` matches ANY labeler's
    relabel output (used as a joke-text source when the target labeler
    hasn't been run over this data yet); a given `labeler` narrows to
    exactly that one (used when its label is already known)."""
    pattern = "turns-*.relabel-%s.jsonl" % (labeler if labeler else "*")
    candidates = list(root.glob("lane-*/%s" % pattern))
    if not candidates:
        candidates = list(root.glob(pattern))
    return sorted(candidates)


def collect_records(root: Path, labeler: str) -> List[Dict]:
    """One dict per turn: run_id, model, turn, joke, label (str or None
    if not yet known), tier (derived from label via infer_tier, or None
    alongside a None label), source_file. Resolution order, per this
    module's docstring: exact relabel-<labeler> files > raw pilot files
    > any other labeler's relabel files (joke text only, label/tier
    left None)."""
    exact = find_relabel_files(root, labeler)
    if exact:
        field = _relabel_field(labeler)
        return _records_from_files(exact, labeler, label_field=field)

    raw = find_raw_turn_files(root)
    if raw:
        return _records_from_files(raw, labeler, label_field=None)

    other = find_relabel_files(root)
    if other:
        return _records_from_files(other, labeler, label_field=None)

    raise SystemExit(
        "no turns-*.jsonl or turns-*.relabel-*.jsonl files found under %s "
        "(expected a pilot dir or a benchmark.relabel output dir)" % root)


def _records_from_files(files: List[Path], labeler: str,
                        label_field: Optional[str]) -> List[Dict]:
    records = []
    for p in files:
        for rec in _load_jsonl(p):
            joke = rec.get("joke")
            if joke is None:
                continue  # malformed/legacy line, same skip as relabel.py
            run_id = rec.get("run_id")
            label = rec.get(label_field) if label_field else None
            records.append({
                "run_id": run_id,
                "model": _model_of_safe(run_id),
                "turn": rec.get("turn"),
                "joke": joke,
                "label": label,
                "tier": infer_tier(labeler, label),
                "source_file": str(p),
            })
    return records


def resolve_labels(records: List[Dict], labeler: str,
                   labeler_fn: Callable[[str, Callable], Tuple[str, str]],
                   complete: Optional[Callable[[str], str]],
                   cache: LabelCache, dry_run: bool,
                   stats: Dict[str, int]) -> None:
    """Fill in every record's `label`/`tier` in place. Already-known
    labels (from an exact relabel file) are left untouched -- their tier
    was already set by collect_records. For the rest: a cache hit always
    resolves for free, regardless of --dry-run (tier re-derived via
    infer_tier, since the cache only ever stores the label string); a
    cache miss is a live call ONLY when dry_run is False, otherwise it is
    counted as `missing` and the provider is never touched."""
    for rec in records:
        if rec["label"] is not None:
            continue
        cached = cache.get(labeler, rec["joke"])
        if cached is not None:
            rec["label"] = cached
            rec["tier"] = infer_tier(labeler, cached)
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1
            continue
        if dry_run:
            stats["missing"] = stats.get("missing", 0) + 1
            continue
        assert complete is not None, "non-dry-run requires a provider"
        label, tier = labeler_fn(rec["joke"], complete)
        cache.put(labeler, rec["joke"], label)  # only the label is cached;
        rec["label"] = label                    # tier is always cheap to
        rec["tier"] = tier                       # re-derive, see infer_tier
        stats["calls"] = stats.get("calls", 0) + 1


def build_report(records: List[Dict]) -> Dict:
    """No pass/fail here on purpose (audit finding on the prior design's
    baked-in bar): reports escape_rate (free-tier fraction) and
    unparseable_rate, plus canon_rate for completeness, each split into
    a per-label histogram and a per-model breakdown. EXP-010 applies
    whatever bar it pre-registers to these numbers itself."""
    labeled = [r for r in records if r["label"] is not None]
    missing = [r for r in records if r["label"] is None]
    n = len(labeled)

    canon = [r for r in labeled if r["tier"] == TIER_CANON]
    free = [r for r in labeled if r["tier"] == TIER_FREE]
    unparseable = [r for r in labeled if r["tier"] == TIER_UNPARSEABLE]

    def rate(count: int) -> Optional[float]:
        return (count / n) if n else None

    per_model_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, TIER_CANON: 0, TIER_FREE: 0, TIER_UNPARSEABLE: 0})
    for r in labeled:
        d = per_model_counts[r["model"] or "unknown"]
        d["n"] += 1
        d[r["tier"]] += 1
    per_model = {}
    for model, d in per_model_counts.items():
        total = d["n"]
        per_model[model] = {
            "n": total,
            "canon_rate": (d[TIER_CANON] / total) if total else None,
            "escape_rate": (d[TIER_FREE] / total) if total else None,
            "unparseable_rate": (d[TIER_UNPARSEABLE] / total) if total else None,
        }

    return {
        "total_turns": len(records),
        "labeled_turns": n,
        "missing_turns": len(missing),
        "canon_count": len(canon),
        "canon_rate": rate(len(canon)),
        "escape_count": len(free),
        "escape_rate": rate(len(free)),
        "unparseable_count": len(unparseable),
        "unparseable_rate": rate(len(unparseable)),
        "histogram_canon": dict(Counter(r["label"] for r in canon).most_common()),
        "histogram_free": dict(Counter(r["label"] for r in free).most_common()),
        "per_model": per_model,
    }


# ------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="pilot dir (lane-*/turns-*.jsonl) or a "
                         "benchmark.relabel output dir "
                         "(lane-*/turns-*.relabel-<labeler>.jsonl)")
    ap.add_argument("--labeler", choices=sorted(LABELERS), default="v4")
    ap.add_argument("--provider", default="claude:haiku",
                    help="resolved ONLY through get_provider() (never "
                         "invoked directly), so the neutral-cwd guard "
                         "applies. Only used on a cache miss when "
                         "--dry-run is NOT set.")
    ap.add_argument("--cache", default=None,
                    help="label cache jsonl path, same on-disk format as "
                         "benchmark.relabel.LabelCache (joke-hash -> "
                         "label, keyed by labeler name; tier is never "
                         "stored, only ever re-derived). Default: "
                         "field_coverage_cache/<labeler>.jsonl under the "
                         "current directory -- deliberately never "
                         "defaults to a path under --dir, so this tool "
                         "never writes into a pilot or relabel dir "
                         "(e.g. anything under experiment-runs/) unless "
                         "explicitly pointed there.")
    ap.add_argument("--sample", type=int, default=None,
                    help="cap the number of turns examined; sampling is "
                         "deterministic (sorted by source file, then "
                         "turn index) so repeated runs are reproducible")
    ap.add_argument("--dry-run", action="store_true",
                    help="never call the provider; use only labels "
                         "already on disk or already in the cache")
    ap.add_argument("--out", default=None,
                    help="write the JSON report here in addition to "
                         "printing it")
    args = ap.parse_args()

    root = Path(args.dir)
    labeler_fn, version = LABELERS[args.labeler]

    records = collect_records(root, args.labeler)
    records.sort(key=lambda r: (r["source_file"],
                                r["turn"] if r["turn"] is not None else -1))
    if args.sample is not None:
        records = records[:args.sample]

    cache_path = (Path(args.cache) if args.cache
                 else Path("field_coverage_cache") / ("%s.jsonl" % args.labeler))
    cache = LabelCache(cache_path)
    stats: Dict[str, int] = {"cache_hits": 0, "calls": 0, "missing": 0}
    complete = None if args.dry_run else get_provider(args.provider)
    try:
        resolve_labels(records, args.labeler, labeler_fn, complete, cache,
                       args.dry_run, stats)
    finally:
        cache.close()

    report = build_report(records)
    report["labeler"] = args.labeler
    report["label_prompt_version"] = version
    report["source_dir"] = str(root)
    report["dry_run"] = args.dry_run
    report["stats"] = stats

    print(json.dumps(report, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print("report written to %s" % out_path)


if __name__ == "__main__":
    main()

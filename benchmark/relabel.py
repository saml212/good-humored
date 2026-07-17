"""Instrument-robustness relabeler for the cascade pilot (EXP-004 x EXP-008).

EXP-004's 12-model cascade pilot ran with LABEL_PROMPT v2 (free
vocabulary) because that was the validated instrument at the time. EXP-008
validated LABEL_PROMPT v3 (closed vocabulary) decisively ahead of it
(reworded_invariance 1.000 vs 0.800, ari_vs_gold 0.9237 vs 0.837) the same
night. The findings writeup needs every headline number reported under
BOTH instruments: if a conclusion holds under v2 AND v3, no reviewer can
wave it away as a labeler artifact.

This script does NOT re-run any cascade. It reads the JOKE TEXT already on
disk from a completed (or in-flight) run_pilot.py sweep and relabels it
through a different labeler, then recomputes the pilot's headline metrics
on the new labels using the exact same functions run_pilot.py and
analyze_pilot.py already call (path_divergence / depth_to_degradation /
cross_model_overlap / LabelSpace) -- this file adds zero new metric math.

Turn-record fields, locked against the real pilot's stored JSONL (not just
cascade.py's current source -- the on-disk EXP-004 data predates the
`temperature` field cascade.py now writes, so it is read defensively):
  run_id, turn, joke, topic, refusal, ts, and optionally temperature.
`joke` is the text to relabel; `topic` is the ORIGINAL (v2) label,
preserved in the output rather than overwritten (see build_output_record).

Directory contract:
  --pilot <dir>/lane-*/turns-<run_id>.jsonl   (input; read-only)
  --out   <dir>/lane-*/turns-<run_id>.relabel-<labeler>.jsonl  (per-turn)
          <dir>/lane-*/summary.json           (run_pilot-shaped, v3 paths)
          <dir>/label_cache.jsonl             (joke-hash -> label cache)
          <dir>/instrument_agreement.json     (v2-vs-v3 agreement report)

Usage:
  python3 -m benchmark.relabel \
      --pilot experiment-runs/2026-07-17-cascade-pilot \
      --labeler v3 \
      --out experiment-runs/2026-07-17-cascade-pilot-v3-relabel
"""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .label_space import LabelSpace
from .metrics import (cross_model_overlap, depth_to_degradation,
                      normalize_label, path_divergence)
from .providers import get_provider
from .rejector import (LABEL_PROMPT_VERSION, LABEL_PROMPT_VERSION_V3,
                       label_topic, label_topic_v3)

# labeler name -> (labeling function, version string recorded in output).
# Mirrors validate_rejector.py's PROMPT_VERSIONS registry on purpose: "v3"
# is the only labeler this relabel is ever run with today, but the CLI and
# the cache are keyed by labeler name throughout, not hardcoded to "v3" --
# a future v2-vs-v3 relabel A/B (or a v4) is a new dict entry, not a
# rewrite of this file.
LABELERS: Dict[str, Tuple[Callable[[str, Callable], str], str]] = {
    "v2": (label_topic, LABEL_PROMPT_VERSION),
    "v3": (label_topic_v3, LABEL_PROMPT_VERSION_V3),
}

# The field the ORIGINAL (as-stored) label is renamed to in every output
# record. Only relevant note: if --labeler is ever "v2" (relabeling v2
# data with v2 again), the new label's field name would collide with this
# one -- build_output_record detects that and renames the original to
# "topic_v2_original" instead, so no run ever silently overwrites data.
ORIGINAL_LABEL_FIELD = "topic_v2"

_RUN_SUFFIX = re.compile(r"-r(\d+)$")


def model_of(run_id: str) -> str:
    """Recover the model name from a run_id by stripping the trailing
    '-rNN' run index. Anchored on the suffix, not a naive split on '-',
    because run_ids themselves contain colons and dots that could look
    like separators (api:deepseek-r00, codex:5.4-r00)."""
    m = _RUN_SUFFIX.search(run_id)
    if not m:
        raise ValueError("run_id %r does not end in -rNN" % run_id)
    return run_id[:m.start()]


def joke_hash(joke: str) -> str:
    """Exact-text hash -- deliberately NOT normalized. Two jokes that
    differ by even one character (a rewording) are, correctly, different
    cache entries; only byte-identical repeats (the degradation signal
    this whole pilot is about) collapse to one call."""
    return hashlib.sha256(joke.encode("utf-8")).hexdigest()


class LabelCache:
    """Persistent joke-text -> label cache, JSONL on disk, keyed by
    (labeler name, exact-joke-text hash).

    Two guarantees this class exists to provide:
      1. Identical jokes label identically. Every lookup for the same
         (labeler, joke) pair returns the same cached string -- the
         labeler is never called twice for the same joke, and repeats
         (the degradation signal) can never drift to two different v3
         labels by chance of which call landed first.
      2. An interrupted run never re-spends. Every new label is appended
         and flushed to disk BEFORE the caller moves on, so a crash right
         after the call that produced it still leaves it on disk; the
         next invocation loads this file first and skips any joke it
         already has an answer for.

    Keyed by labeler name (not just hash) so a future v2 AND v3 relabel
    can share one cache file without collision -- the same joke text can
    legitimately have different correct labels under different labelers.
    """

    def __init__(self, path: Path):
        self.path = path
        self._map: Dict[Tuple[str, str], str] = {}
        self._f = None
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._map[(rec["labeler"], rec["joke_hash"])] = rec["topic"]

    def get(self, labeler: str, joke: str) -> Optional[str]:
        return self._map.get((labeler, joke_hash(joke)))

    def put(self, labeler: str, joke: str, topic: str) -> None:
        h = joke_hash(joke)
        self._map[(labeler, h)] = topic
        if self._f is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "a")
        self._f.write(json.dumps(
            {"labeler": labeler, "joke_hash": h, "topic": topic}) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __len__(self) -> int:
        return len(self._map)


def label_cached(joke: str, labeler_name: str,
                 labeler_fn: Callable[[str, Callable], str],
                 complete: Callable[[str], str], cache: LabelCache,
                 stats: Dict[str, int]) -> str:
    """Cache-checked relabel of one joke. `labeler_fn` is label_topic or
    label_topic_v3 (or a fake standing in for either in tests); `complete`
    is the underlying provider callable it will invoke on a cache miss."""
    cached = cache.get(labeler_name, joke)
    if cached is not None:
        stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        return cached
    topic = labeler_fn(joke, complete)
    cache.put(labeler_name, joke, topic)
    stats["calls"] = stats.get("calls", 0) + 1
    return topic


def new_field_name(labeler_name: str) -> str:
    """Output field name for the new label, dynamic on labeler name so a
    future --labeler v2 run cannot collide with ORIGINAL_LABEL_FIELD."""
    return "topic_%s" % labeler_name


def build_output_record(rec: Dict, labeler_name: str, new_label: str) -> Dict:
    """Original turn record + both labels, original preserved for
    diffing. Never mutates `rec`."""
    out_rec = dict(rec)
    new_field = new_field_name(labeler_name)
    orig_field = ORIGINAL_LABEL_FIELD
    if new_field == orig_field:  # only possible if --labeler v2 vs v2
        orig_field = "topic_v2_original"
    out_rec[orig_field] = rec.get("topic")
    out_rec[new_field] = new_label
    out_rec.pop("topic", None)
    out_rec["relabeled_with"] = labeler_name
    return out_rec


def find_turn_files(lane_dir: Path) -> List[Path]:
    return sorted(lane_dir.glob("turns-*.jsonl"))


def relabeled_out_path(lane_out: Path, turns_path: Path, labeler_name: str) -> Path:
    return lane_out / ("%s.relabel-%s.jsonl" % (turns_path.stem, labeler_name))


def already_done(turns_path: Path, out_path: Path) -> bool:
    """True if out_path already holds one relabeled line per non-blank
    line of turns_path -- lets a resumed invocation skip re-reading and
    re-writing a whole run file. NOT what prevents re-spending calls --
    the LabelCache does that at the per-joke level regardless of this
    check; this is purely a cheap short-circuit on top."""
    if not out_path.exists():
        return False
    with open(turns_path) as f:
        n_in = sum(1 for l in f if l.strip())
    with open(out_path) as f:
        n_out = sum(1 for l in f if l.strip())
    return n_in > 0 and n_in == n_out


def relabel_run(turns_path: Path, out_path: Path, labeler_name: str,
                labeler_fn: Callable[[str, Callable], str],
                complete: Callable[[str], str], cache: LabelCache,
                stats: Dict[str, int]) -> List[str]:
    """Relabel one run's stored turns, in turn order. Writes the output
    JSONL incrementally, flushing after every line -- a crash mid-run
    preserves every turn relabeled so far (same discipline as cascade.py's
    own per-turn log_f.write + flush). Returns the v3 (or whichever
    labeler) path in turn order, for the metrics pass."""
    records = []
    with open(turns_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("turn", 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    path_new: List[str] = []
    with open(out_path, "w") as out_f:
        for rec in records:
            joke = rec.get("joke")
            if joke is None:
                continue  # malformed/legacy line with no joke text -- skip
            label = label_cached(joke, labeler_name, labeler_fn, complete,
                                 cache, stats)
            out_f.write(json.dumps(
                build_output_record(rec, labeler_name, label)) + "\n")
            out_f.flush()
            path_new.append(label)
    return path_new


def read_relabeled_path(out_path: Path, labeler_name: str) -> List[str]:
    """Recover the v3 (or whichever labeler) path from an already-written
    output file, for the resume-skip branch (no relabeling work needed)."""
    field = new_field_name(labeler_name)
    with open(out_path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    recs.sort(key=lambda r: r.get("turn", 0))
    return [r[field] for r in recs]


def process_lane(lane_dir: Path, out_dir: Path, labeler_name: str,
                 labeler_fn: Callable[[str, Callable], str],
                 complete: Callable[[str], str], cache: LabelCache,
                 stats: Dict[str, int], failures: List[Dict]
                 ) -> Dict[str, List[List[str]]]:
    """Relabel every run file in one lane. Returns {model: [path, ...]}
    (new-labeler paths, turn order) -- the same shape run_pilot.py builds
    internally while a sweep is live, one lane's worth of it. Each run
    file is individually fenced (same W5 discipline as run_pilot.py's own
    sweep loop): one bad run's exception is recorded in `failures` and the
    lane keeps going, rather than losing every other run's already-cached
    labels."""
    paths: Dict[str, List[List[str]]] = {}
    lane_out = out_dir / lane_dir.name
    for turns_path in find_turn_files(lane_dir):
        with open(turns_path) as f:
            first_line = f.readline()
        if not first_line.strip():
            continue  # empty file, nothing to relabel
        try:
            run_id = json.loads(first_line)["run_id"]
            model = model_of(run_id)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            failures.append({"run_id": turns_path.name, "error": repr(e)})
            print("FAILED %s: %r" % (turns_path.name, e))
            continue

        out_path = relabeled_out_path(lane_out, turns_path, labeler_name)
        try:
            if already_done(turns_path, out_path):
                path_new = read_relabeled_path(out_path, labeler_name)
            else:
                path_new = relabel_run(turns_path, out_path, labeler_name,
                                       labeler_fn, complete, cache, stats)
        except Exception as e:  # fence per run, keep sweeping (like run_pilot W5)
            failures.append({"run_id": run_id, "error": repr(e)})
            print("FAILED %s: %r" % (run_id, e))
            continue
        paths.setdefault(model, []).append(path_new)
    return paths


def build_summary(paths: Dict[str, List[List[str]]], failures: List[Dict],
                  labeler_name: str, version: str, provider_spec: str,
                  source_lane: str) -> Dict:
    """Same shape run_pilot.py's summary.json has, computed on the new
    (v3, by default) paths -- reuses path_divergence / depth_to_degradation
    / cross_model_overlap / LabelSpace exactly as run_pilot.py calls them;
    no metric here is reimplemented."""
    label_space = LabelSpace().fit(
        [t for ps in paths.values() for p in ps for t in p])
    canon_paths = {m: label_space.canonize_paths(ps) for m, ps in paths.items()}

    summary: Dict = {
        "models": sorted(paths),
        "runs_per_model": {m: len(ps) for m, ps in paths.items()},
        "labeler": labeler_name,
        "label_prompt_version": version,
        "relabel_provider": provider_spec,
        "source_lane": source_lane,
        "label_space_degraded": label_space.degraded,
        "failures": failures,
        "per_model": {},
    }
    for model, ps in paths.items():
        if len(ps) >= 2:
            cps = canon_paths[model]
            summary["per_model"][model] = {
                "divergence": path_divergence(ps),
                "degradation": [depth_to_degradation(p) for p in ps],
                "divergence_semantic": path_divergence(cps),
                "degradation_semantic": [depth_to_degradation(p) for p in cps],
                "paths": ps,
                "paths_semantic": cps,
            }
    survivors = {m: ps for m, ps in paths.items() if ps}
    if len(survivors) >= 2:
        summary["cross_model"] = cross_model_overlap(survivors)
        summary["cross_model_semantic"] = cross_model_overlap(
            {m: canon_paths[m] for m in survivors})
    return summary


# ------------------------------------------------------ instrument agreement


def build_agreement_report(out_dir: Path, lanes: List[Path],
                           labeler_name: str) -> Dict:
    """Per-turn v2-vs-new-labeler agreement (after normalize_label),
    confusion pairs sorted by frequency, and per-model agreement rate --
    reads back the relabeled output files this run just wrote (or that a
    prior invocation wrote and this one resumed from)."""
    new_field = new_field_name(labeler_name)
    orig_field = ("topic_v2_original" if new_field == ORIGINAL_LABEL_FIELD
                  else ORIGINAL_LABEL_FIELD)

    n_turns = 0
    n_agree = 0
    confusion: Counter = Counter()
    per_model: Dict[str, Dict[str, int]] = {}

    for lane_dir in lanes:
        lane_out = out_dir / lane_dir.name
        if not lane_out.is_dir():
            continue
        for path in sorted(lane_out.glob("*.relabel-%s.jsonl" % labeler_name)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    v2 = rec.get(orig_field)
                    v_new = rec.get(new_field)
                    run_id = rec.get("run_id")
                    if v2 is None or v_new is None or run_id is None:
                        continue
                    try:
                        model = model_of(run_id)
                    except ValueError:
                        continue
                    v2n, newn = normalize_label(v2), normalize_label(v_new)
                    n_turns += 1
                    m = per_model.setdefault(model, {"n_turns": 0, "n_agree": 0})
                    m["n_turns"] += 1
                    if v2n == newn:
                        n_agree += 1
                        m["n_agree"] += 1
                    else:
                        confusion[(v2n, newn)] += 1

    per_model_out = {
        m: {"n_turns": d["n_turns"], "n_agree": d["n_agree"],
            "agreement_rate": (d["n_agree"] / d["n_turns"]
                              if d["n_turns"] else None)}
        for m, d in per_model.items()
    }
    confusion_pairs = [
        {"topic_v2": v2, "topic_%s" % labeler_name: newv, "count": c}
        for (v2, newv), c in confusion.most_common()
    ]
    return {
        "labeler": labeler_name,
        "overall": {
            "n_turns": n_turns,
            "n_agree": n_agree,
            "agreement_rate": (n_agree / n_turns) if n_turns else None,
        },
        "per_model": per_model_out,
        "confusion_pairs": confusion_pairs,
    }


# ------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True,
                    help="pilot dir containing lane-*/ subdirs of "
                         "turns-*.jsonl written by a run_pilot.py sweep")
    ap.add_argument("--labeler", choices=sorted(LABELERS), default="v3",
                    help="labeler to relabel every stored joke with "
                         "(default: v3, EXP-008's validated constrained-"
                         "vocabulary instrument). 'v2' is wired too so a "
                         "future v2-vs-v3 relabel A/B is a flag, not a "
                         "rewrite -- not exercised by this pilot's "
                         "writeup, which only needs the v2-stored data "
                         "relabeled under v3.")
    ap.add_argument("--provider", default="claude:haiku",
                    help="provider spec for the labeler backend, resolved "
                         "ONLY through get_provider() (never invoked "
                         "directly) so the neutral-cwd guard applies. "
                         "Default matches EXP-008's validated instrument.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir = Path(args.pilot)
    lanes = sorted((d for d in pilot_dir.glob("lane-*") if d.is_dir()),
                  key=lambda p: p.name)
    if not lanes:
        raise SystemExit("no lane-*/ subdirs found under %s" % pilot_dir)

    labeler_fn, version = LABELERS[args.labeler]
    complete = get_provider(args.provider)
    cache = LabelCache(out_dir / "label_cache.jsonl")
    stats: Dict[str, int] = {"calls": 0, "cache_hits": 0}

    t0 = time.time()
    try:
        for lane_dir in lanes:
            failures: List[Dict] = []
            paths = process_lane(lane_dir, out_dir, args.labeler, labeler_fn,
                                 complete, cache, stats, failures)
            summary = build_summary(paths, failures, args.labeler, version,
                                    args.provider, str(lane_dir))
            lane_out = out_dir / lane_dir.name
            lane_out.mkdir(parents=True, exist_ok=True)
            with open(lane_out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print("%s: %d models relabeled -> %s" %
                  (lane_dir.name, len(paths), lane_out / "summary.json"))
    finally:
        cache.close()

    agreement = build_agreement_report(out_dir, lanes, args.labeler)
    with open(out_dir / "instrument_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)

    overall = agreement["overall"]
    print("relabel done in %.0fs: %d calls, %d cache hits (%d unique "
          "jokes cached total)" %
          (time.time() - t0, stats["calls"], stats["cache_hits"], len(cache)))
    if overall["agreement_rate"] is not None:
        print("instrument agreement (v2 vs %s): %.3f (%d/%d turns) -> %s" %
              (args.labeler, overall["agreement_rate"], overall["n_agree"],
               overall["n_turns"], out_dir / "instrument_agreement.json"))


if __name__ == "__main__":
    main()

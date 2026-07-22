"""Callback-transformation validation — EXP-016 (EXPERIMENT_LOG.md).

Runs `benchmark.callback_transform.callback_transformation_score` (the
NEW, gated+transformation-scored detector) AND the OLD
`benchmark.banter.detect_callback` (read-only — the pre-existing,
bag-of-words, no-transformation-requirement detector this experiment
replaces) over the SAME 40-item hand-built fixture
(`benchmark/fixtures/callback_transform_fixture.jsonl`), producing:

  1. per-class mean NEW score, and the three pre-registered bars
     (EXPERIMENT_LOG.md's EXP-016 entry, registered BEFORE this script
     ever ran):
       - margin = mean(genuine_callback) - mean(trivial_paraphrase UNION
         verbatim_repeat), bar >= 0.50 (predicted ~0.50);
       - coincidental_word_reuse mean <= 0.10 (the false-positive bar —
         the >=2-shared-content-word gate must not fire on a single
         coincidentally-shared word the way the OLD detector does);
       - no_callback mean == 0.0 EXACTLY (an exact-equality bar, not a
         ceiling — the gate must never manufacture a callback from
         nothing).
  2. the OLD-vs-NEW before/after table this experiment exists to
     produce: OLD's per-class "callback fired" rate (`detect_callback`
     returns non-`None`), reframed as a comparable [0, 1] "old score"
     (1.0 if fired, matching `env/banter_env.py`'s own flat
     `callback_weight` bonus semantics — see
     `benchmark/callback_transform.py`'s module docstring for the traced
     consumption path this quantifies) — this is what makes
     `verbatim_repeat` (old=1.0, new=0.0) and `coincidental_word_reuse`
     (old fires, new=0.0) the exact bug/fix pairs EXP-016 registered this
     experiment to quantify.

Pure local compute — no judge, no network call, ever. The embedding
OR-branch of the detection gate is exercised here with `embed_fn=None`
(word-overlap gate only) by DEFAULT: every fixture item is, by
construction, decidable via the content-word gate alone (see
`benchmark/callback_transform.py`'s `DEFAULT_EMBED_SIM_FLOOR` docstring
for why the embedding path is not relied on for any committed item).
Pass `--embed` to additionally score with a real, LOCAL, OFFLINE
`all-MiniLM-L6-v2` `SentenceTransformer` encoder (same model + same
lazy-import discipline as `env/semantic_novelty.py` /
`env/incongruity_gate.py`) as a robustness check that enabling the
OR-branch does not change any bar's PASS/FAIL outcome.

Usage:
  python3 -m benchmark.validate_callback_transform \
      --out experiment-runs/2026-07-22-exp016-callback
  python3 -m benchmark.validate_callback_transform --embed \
      --out experiment-runs/2026-07-22-exp016-callback
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .banter import detect_callback
from .callback_transform import callback_transformation_score

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "callback_transform_fixture.jsonl"
GOLD_CLASSES = ("genuine_callback", "trivial_paraphrase",
               "coincidental_word_reuse", "verbatim_repeat", "no_callback")
MIN_GAP = 3  # matches both detectors' shared default; stated once for the CLI printout


def load_fixture() -> List[Dict]:
    items = []
    with open(FIXTURE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError("duplicate fixture ids: %s" % dupes)
    for it in items:
        if it["gold_class"] not in GOLD_CLASSES:
            raise ValueError("item %r has unknown gold_class %r" %
                             (it["id"], it["gold_class"]))
        if not (0 <= it["current_turn_idx"] < len(it["turns"])):
            raise ValueError("item %r: current_turn_idx out of range" % it["id"])
    return items


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def score_new(items: List[Dict],
             embed_fn: Optional[Callable] = None) -> Dict[str, Dict]:
    """New detector's score for every item, per `callback_transformation_score`.
    `embed_fn=None` (default) runs the content-word gate only, exactly the
    operating mode every committed fixture item is decidable under."""
    out = {}
    for it in items:
        r = callback_transformation_score(
            it["turns"], it["current_turn_idx"], min_gap=MIN_GAP,
            embed_fn=embed_fn)
        out[it["id"]] = r
    return out


def score_old(items: List[Dict]) -> Dict[str, Dict]:
    """OLD `detect_callback`'s result for every item (read-only import —
    `benchmark/banter.py` is not modified anywhere in EXP-016). Reports
    the raw matched word (or `None`) AND a comparable `old_score` in
    [0, 1]: 1.0 if `detect_callback` fired at all, matching
    `env/banter_env.py::BanterEnv.step`'s own flat `callback_weight`
    bonus semantics (a fired callback earns the FULL bonus regardless of
    transformation) — see `benchmark/callback_transform.py`'s module
    docstring for that traced consumption path."""
    out = {}
    for it in items:
        turns = it["turns"]
        idx = it["current_turn_idx"]
        matched_word = detect_callback(turns[idx], turns[:idx], min_gap=MIN_GAP)
        out[it["id"]] = {"matched_word": matched_word,
                         "old_score": 1.0 if matched_word is not None else 0.0}
    return out


def per_class_means(items: List[Dict], id_to_value: Dict[str, float]) -> Dict[str, Dict]:
    out = {}
    for cls in GOLD_CLASSES:
        ids = [it["id"] for it in items if it["gold_class"] == cls]
        vals = [id_to_value[i] for i in ids]
        out[cls] = {"n": len(ids), "mean": _mean(vals), "min": min(vals),
                   "max": max(vals)}
    return out


def build_bars(new_per_class: Dict[str, Dict]) -> List[Dict]:
    """The three EXP-016 pre-registered bars (EXPERIMENT_LOG.md), each
    computed ONLY from the NEW detector's per-class means."""
    genuine = new_per_class["genuine_callback"]["mean"]
    trivial_n = new_per_class["trivial_paraphrase"]["n"]
    trivial_mean = new_per_class["trivial_paraphrase"]["mean"]
    verbatim_n = new_per_class["verbatim_repeat"]["n"]
    verbatim_mean = new_per_class["verbatim_repeat"]["mean"]
    pooled_n = trivial_n + verbatim_n
    pooled_mean = ((trivial_mean * trivial_n + verbatim_mean * verbatim_n) / pooled_n
                  if pooled_n else 0.0)
    margin = genuine - pooled_mean

    coincidental_mean = new_per_class["coincidental_word_reuse"]["mean"]
    no_callback_mean = new_per_class["no_callback"]["mean"]

    def bar(name, value, op, threshold):
        if op == ">=":
            passed = value >= threshold
        elif op == "<=":
            passed = value <= threshold
        else:  # "=="
            passed = value == threshold
        return {"name": name, "value": value, "op": op, "threshold": threshold,
               "passed": bool(passed)}

    return [
        bar("exp016_callback_margin_genuine_minus_trivial_union_verbatim",
           margin, ">=", 0.50),
        bar("exp016_coincidental_word_reuse_mean", coincidental_mean, "<=", 0.10),
        bar("exp016_no_callback_mean_exactly_zero", no_callback_mean, "==", 0.0),
    ], margin, pooled_mean


def make_embed_fn():
    """Lazily construct a real, local, offline `all-MiniLM-L6-v2` embed_fn
    -- same model + same lazy-import-inside-a-function discipline
    `env/semantic_novelty.py`/`env/incongruity_gate.py` use, so this
    script stays importable (and every default-mode run stays provably
    dependency-free of sentence_transformers) unless `--embed` is passed."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(texts):
        return model.encode(list(texts), normalize_embeddings=True,
                            show_progress_bar=False)
    return _embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed", action="store_true",
                    help="also score the new detector with a real local "
                         "all-MiniLM-L6-v2 embed_fn enabled (robustness "
                         "check on the OR-branch; default is word-gate-"
                         "only, which every fixture item is decidable "
                         "under -- see module docstring).")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = load_fixture()
    t0 = time.time()

    new_results = score_new(items, embed_fn=None)
    old_results = score_old(items)

    embed_results = None
    if args.embed:
        embed_fn = make_embed_fn()
        embed_results = score_new(items, embed_fn=embed_fn)

    wall = time.time() - t0

    new_scores = {i: r["score"] for i, r in new_results.items()}
    old_scores = {i: r["old_score"] for i, r in old_results.items()}

    new_per_class = per_class_means(items, new_scores)
    old_per_class = per_class_means(items, old_scores)
    bars, margin, pooled_mean = build_bars(new_per_class)

    embed_per_class = None
    embed_bars = None
    if embed_results is not None:
        embed_scores = {i: r["score"] for i, r in embed_results.items()}
        embed_per_class = per_class_means(items, embed_scores)
        embed_bars, _, _ = build_bars(embed_per_class)

    old_vs_new = {
        it["id"]: {
            "gold_class": it["gold_class"],
            "old_matched_word": old_results[it["id"]]["matched_word"],
            "old_score": old_results[it["id"]]["old_score"],
            "new_matched_turn_idx": new_results[it["id"]]["matched_turn_idx"],
            "new_detection_reasons": new_results[it["id"]]["detection_reasons"],
            "new_trigram_similarity": new_results[it["id"]]["trigram_similarity"],
            "new_score": new_results[it["id"]]["score"],
        }
        for it in items
    }

    report = {
        "experiment": "EXP-016-callback-transform-validation",
        "run_stamp": time.strftime("%Y%m%dT%H%M%S"),
        "n_items": len(items),
        "min_gap": MIN_GAP,
        "wall_seconds": round(wall, 3),
        "embed_mode_run": args.embed,
        "new_per_class": new_per_class,
        "old_per_class": old_per_class,
        "margin_genuine_minus_trivial_union_verbatim": margin,
        "predicted_margin_registered": 0.50,
        "pooled_trivial_verbatim_mean": pooled_mean,
        "bars": bars,
        "all_bars_pass": all(b["passed"] for b in bars),
        "embed_mode_per_class": embed_per_class,
        "embed_mode_bars": embed_bars,
        "embed_mode_all_bars_pass": (all(b["passed"] for b in embed_bars)
                                     if embed_bars is not None else None),
        "old_vs_new": old_vs_new,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== EXP-016 callback-transform validation (%d items) ===" %
         len(items))
    print()
    print("--- NEW detector (word-gate only) vs OLD detect_callback, per class ---")
    print("%-26s %6s  %-10s %-10s  %-10s" %
         ("class", "n", "new_mean", "old_mean", "old_fire_n"))
    for cls in GOLD_CLASSES:
        new_pc = new_per_class[cls]
        old_pc = old_per_class[cls]
        old_fire_n = sum(1 for it in items if it["gold_class"] == cls
                         and old_results[it["id"]]["matched_word"] is not None)
        print("%-26s %6d  %-10.4f %-10.4f  %-10d" %
             (cls, new_pc["n"], new_pc["mean"], old_pc["mean"], old_fire_n))
    print()
    print("margin (genuine - pooled trivial∪verbatim) = %.4f (bar >= 0.50, "
         "predicted ~0.50)" % margin)
    print()
    print("=== bars ===")
    for b in bars:
        status = "PASS" if b["passed"] else "FAIL"
        print("[%s] %-55s value=%.4f %s %s" %
             (status, b["name"], b["value"], b["op"], b["threshold"]))
    print()
    if embed_bars is not None:
        print("=== --embed robustness check: bars with OR-branch enabled ===")
        for b in embed_bars:
            status = "PASS" if b["passed"] else "FAIL"
            print("[%s] %-55s value=%.4f %s %s" %
                 (status, b["name"], b["value"], b["op"], b["threshold"]))
        print()
    print("all_bars_pass = %s" % report["all_bars_pass"])
    print("wall=%.3fs" % wall)
    print("report: %s" % (out / "report.json"))


if __name__ == "__main__":
    main()

"""Threshold calibration for LabelSpace (see label_space.py).

Sweeps cosine-similarity threshold 0.50-0.95 (step 0.05) over a hand-built
gold fixture of must-merge / must-not-merge label pairs
(fixtures/label_equivalence_pairs.jsonl), reports precision/recall/F0.5
per threshold, and prints the threshold that maximizes F0.5.

Why precision-weighted, not F1: a FALSE MERGE (two genuinely different
joke topics collapsed into one canonical label) makes a model's
trajectory look more repetitive than it really is — it directly inflates
this benchmark's headline "mode collapse" number. A false SPLIT just
costs some statistical power (divergence reads slightly higher than it
should, topics that are really the same look like two). Both are errors,
but false merges bias the benchmark toward the more sensational, harder-
to-defend result — the direction that is most dangerous for this
project's credibility if a reviewer starts pulling threads. So the
threshold is picked to maximize F-beta with beta=0.5, which weights
precision 4x recall in the harmonic mean, rather than F1's even split.

This script is a one-off instrument-calibration tool, not part of the
package's runtime path — like validate_rejector.py, it is meant to be
run once (or after any fixture edit) and its output hand-copied into
label_space.py's DEFAULT_THRESHOLD comment as evidence.

Usage: python3 -m benchmark.calibrate_label_space
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .metrics import normalize_label

FIXTURES = Path(__file__).parent / "fixtures" / "label_equivalence_pairs.jsonl"
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95
BETA = 0.5  # precision weighted 4x recall -- see module docstring


def load_pairs() -> List[Dict]:
    with open(FIXTURES) as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine_sims(pairs: List[Dict], model) -> List[float]:
    """Cosine similarity per pair, computed the same way LabelSpace does:
    normalize_label first, then embed, then dot product of unit vectors."""
    a_text = [normalize_label(p["a"]) for p in pairs]
    b_text = [normalize_label(p["b"]) for p in pairs]
    ea = model.encode(a_text, normalize_embeddings=True)
    eb = model.encode(b_text, normalize_embeddings=True)
    return [float((ea[i] * eb[i]).sum()) for i in range(len(pairs))]


def fbeta(precision: float, recall: float, beta: float = BETA) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def score_at_threshold(
    pairs: List[Dict], sims: List[float], t: float
) -> Tuple[float, float, int, int, int, int]:
    tp = fp = fn = tn = 0
    for p, s in zip(pairs, sims):
        pred_merge = s >= t
        gold_merge = p["should_merge"]
        if pred_merge and gold_merge:
            tp += 1
        elif pred_merge and not gold_merge:
            fp += 1
        elif (not pred_merge) and gold_merge:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall, tp, fp, fn, tn


def main() -> None:
    pairs = load_pairs()
    n_merge = sum(1 for p in pairs if p["should_merge"])
    print("fixture: %d pairs (%d must-merge, %d must-not-merge)" %
          (len(pairs), n_merge, len(pairs) - n_merge))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("sentence_transformers not installed (%r) -- cannot "
              "calibrate. Install with: "
              "pip3 install --user sentence-transformers" % (e,))
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sims = cosine_sims(pairs, model)

    print()
    print("%6s  %9s  %6s  %6s  %6s  %10s  %14s" %
          ("thresh", "precision", "recall", "F0.5", "merged",
           "false_merges", "missed_merges"))
    best = None  # (f, t, precision, recall, fp_ids, fn_ids)
    for t in THRESHOLDS:
        precision, recall, tp, fp, fn, tn = score_at_threshold(pairs, sims, t)
        f = fbeta(precision, recall)
        print("%6.2f  %9.3f  %6.3f  %6.3f  %6d  %12d  %14d" %
              (t, precision, recall, f, tp + fp, fp, fn))
        if best is None or f > best[0]:
            fp_ids = [(p["a"], p["b"]) for p, s in zip(pairs, sims)
                      if s >= t and not p["should_merge"]]
            fn_ids = [(p["a"], p["b"]) for p, s in zip(pairs, sims)
                      if s < t and p["should_merge"]]
            best = (f, t, precision, recall, fp_ids, fn_ids)

    f, t, precision, recall, fp_ids, fn_ids = best
    print()
    print("best threshold by F0.5 (precision-weighted): %.2f "
          "(precision=%.3f recall=%.3f F0.5=%.3f)" % (t, precision, recall, f))
    if fp_ids:
        print("false merges at best threshold: %s" % fp_ids)
    if fn_ids:
        print("missed merges at best threshold: %s" % fn_ids)


if __name__ == "__main__":
    main()

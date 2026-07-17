"""Memorized-joke check for cascade outputs (CLAUDE.md hard rule: every
generation eval includes a novelty check against the memorized corpus —
mode collapse onto memorized jokes is THE documented failure mode).

Two tiers, cheap by design:
  1. exact-normalized match against the full corpus (~1.2M jokes,
     hash-set membership) — catches verbatim retelling;
  2. trigram-Jaccard similarity against the 25 ChatGPT templates
     (Jentzsch & Kersting) — catches light rewording of the known
     worst offenders.

Usage:
  python3 -m benchmark.joke_novelty \
      --pilot experiment-runs/2026-07-17-cascade-pilot \
      --data ~/Experiments/good-humored-data/corpus
"""

import argparse
import json
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

_PUNCT = str.maketrans("", "", string.punctuation)


def norm(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())


def trigrams(text: str) -> Set[str]:
    words = norm(text).split()
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def trigram_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_corpus_hashes(corpus_dir: Path) -> Set[int]:
    """Hash-set of normalized jokes across all corpus jsonl files.
    Python hash of str is fine here — same-process membership only."""
    hashes: Set[int] = set()
    for f in corpus_dir.rglob("jokes.jsonl"):
        with open(f) as fh:
            for line in fh:
                try:
                    hashes.add(hash(norm(json.loads(line)["text"])))
                except (json.JSONDecodeError, KeyError):
                    continue
    return hashes


def load_templates(corpus_dir: Path) -> List[Dict]:
    out = []
    tf = corpus_dir / "chatgpt-25-templates.jsonl"
    if tf.exists():
        with open(tf) as fh:
            for line in fh:
                rec = json.loads(line)
                rec["_trigrams"] = trigrams(rec["text"])
                out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--data", required=True,
                    help="corpus dir (commercial-safe/, research-only/, "
                         "chatgpt-25-templates.jsonl)")
    ap.add_argument("--template-threshold", type=float, default=0.5,
                    help="trigram-jaccard above this counts as a "
                         "template retelling")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    corpus_dir = Path(args.data).expanduser()
    corpus = load_corpus_hashes(corpus_dir)
    templates = load_templates(corpus_dir)
    print("corpus hashes: %d, templates: %d" % (len(corpus), len(templates)))

    per_model: Dict[str, Dict] = defaultdict(
        lambda: {"n_jokes": 0, "exact_corpus_hits": 0,
                 "template_hits": 0, "examples": []})

    for turns_file in sorted(Path(args.pilot).rglob("turns-*.jsonl")):
        model = turns_file.stem.replace("turns-", "").rsplit("-r", 1)[0]
        with open(turns_file) as fh:
            for line in fh:
                rec = json.loads(line)
                joke = rec["joke"]
                stats = per_model[model]
                stats["n_jokes"] += 1
                exact = hash(norm(joke)) in corpus
                tmpl_best, tmpl_sim = None, 0.0
                jt = trigrams(joke)
                for t in templates:
                    s = trigram_jaccard(jt, t["_trigrams"])
                    if s > tmpl_sim:
                        tmpl_best, tmpl_sim = t.get("id", "?"), s
                if exact:
                    stats["exact_corpus_hits"] += 1
                if tmpl_sim >= args.template_threshold:
                    stats["template_hits"] += 1
                    if len(stats["examples"]) < 5:
                        stats["examples"].append(
                            {"joke": joke[:120], "template": tmpl_best,
                             "sim": round(tmpl_sim, 2), "exact": exact})

    report = {m: dict(v) for m, v in sorted(per_model.items())}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)

    print("%-16s %7s %12s %14s" %
          ("model", "jokes", "exact-corpus", "template>=%.1f"
           % args.template_threshold))
    for m, v in report.items():
        print("%-16s %7d %12d %14d" %
              (m, v["n_jokes"], v["exact_corpus_hits"], v["template_hits"]))


if __name__ == "__main__":
    main()

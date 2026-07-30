#!/usr/bin/env python3
"""EXP-021 -- generative resolution probe (judge-free port of EXP-014 gate-2).

The policy model itself is the predictor: for each item it samples K
continuation guesses under EXP-014's verbatim COLD prompt (ordinary
continuation) and K under the PRIMED prompt (twist expected). Each
condition's guesses reduce to a centroid embedding (EXP-014b's
registered reduction); the score is

    d_cold      = 1 - cos(centroid_cold,   punchline_embedding)
    d_primed    = 1 - cos(centroid_primed, punchline_embedding)
    resolution  = d_cold - d_primed

A real joke's punchline is reachable by a twist-primed guesser (a
mechanism connects setup to punchline -> resolution > 0); an unrelated
punchline is not (resolution ~ 0). Registered in EXPERIMENT_LOG.md
(EXP-021, 2026-07-29) with the pooled two-author fixture and the
d_cold-alone baseline as the built-in simpler-baseline disproof.

Usage:
  python3 -m env.validate_generative_resolution \
      --fixtures env/tests/fixtures/pivot_differential_heldout.jsonl,env/tests/fixtures/surprisal_novel_fixture.jsonl \
      --out experiment-runs/2026-07-29-exp021-generative-resolution
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from env.incongruity_gate import (PREDICT_COLD_PROMPT,  # noqa: E402
                                  PREDICT_PRIMED_PROMPT)
from env.validate_pivot_differential import load_fixture, rank_auc  # noqa: E402

# Author A's and author B's non-sequitur classes pool for the primary.
NONSEQ_POOL = ("setup_nonsequitur", "nonseq_fluent")


def stable_seed(item_id: str, condition: str, k: int) -> int:
    digest = hashlib.sha256(
        ("%s|%s|%d" % (item_id, condition, k)).encode()).hexdigest()
    return int(digest[:8], 16)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero-norm embedding")
    return dot / (na * nb)


def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


def first_sentence(text: str) -> str:
    """Model output cleanup, pinned at registration: strip, drop
    surrounding quotes, keep the first non-empty line -- SKIPPING
    preamble lines that end with a colon ("Sure, here's my guess:"),
    which would otherwise silently replace the real guess (audit #17)."""
    for line in text.strip().splitlines():
        line = line.strip().strip('"“”').strip()
        if line and not line.endswith(":"):
            return line
    return ""


_HEDGE_STARTS = ("sure", "okay", "certainly", "of course", "here", "i'm",
                 "i cannot", "i can't", "as an")


def is_suspicious(guess: str) -> bool:
    """Refusal/hedge/fragment detector for the audit-#17 diagnostic:
    guesses this flags are COUNTED, not dropped -- the counter makes a
    corrupted subset visible instead of silently averaged in."""
    if len(guess.split()) < 3:
        return True
    return guess.lower().startswith(_HEDGE_STARTS)


def token_overlap(guess: str, punchline: str) -> float:
    """Fraction of punchline content words (>=5 chars) present in the
    guess -- the embedding-gaming diagnostic."""
    punch_words = {w.lower().strip(".,!?;:'\"") for w in punchline.split()
                   if len(w.strip(".,!?;:'\"")) >= 5}
    if not punch_words:
        return 0.0
    guess_words = {w.lower().strip(".,!?;:'\"") for w in guess.split()}
    return len(punch_words & guess_words) / len(punch_words)


class HFGenerator:
    """Chat-template sampling wrapper around a HF causal LM."""

    def __init__(self, model_name: str, device: str, dtype: str,
                 temperature: float, top_p: float, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, seed: int) -> str:
        torch = self.torch
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=self.temperature,
                top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return first_sentence(self.tokenizer.decode(
            new_tokens, skip_special_tokens=True))


def default_embed_fn() -> Callable[[List[str]], List[List[float]]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in model.encode(texts)]
    return embed


def score_item(generate: Callable[[str, int], str],
               embed: Callable[[List[str]], List[List[float]]],
               item_id: str, setup: str, punchline: str, k: int) -> Dict:
    cold = [generate(PREDICT_COLD_PROMPT.format(setup=setup),
                     stable_seed(item_id, "cold", i)) for i in range(k)]
    primed = [generate(PREDICT_PRIMED_PROMPT.format(setup=setup),
                       stable_seed(item_id, "primed", i)) for i in range(k)]
    cold = [g for g in cold if g]
    primed = [g for g in primed if g]
    if not cold or not primed:
        raise ValueError("empty guesses for item %s" % item_id)
    vectors = embed([punchline] + cold + primed)
    punch_e = vectors[0]
    cold_e, primed_e = vectors[1:1 + len(cold)], vectors[1 + len(cold):]
    d_cold = 1.0 - _cos(_centroid(cold_e), punch_e)
    d_primed = 1.0 - _cos(_centroid(primed_e), punch_e)

    def dispersion(embs, centroid):
        return sum(1.0 - _cos(e, centroid) for e in embs) / len(embs)

    return {
        "d_cold": d_cold,
        "d_primed": d_primed,
        "resolution": d_cold - d_primed,
        "overlap_primed_max": max(token_overlap(g, punchline) for g in primed),
        "cold_dispersion": dispersion(cold_e, _centroid(cold_e)),
        "primed_dispersion": dispersion(primed_e, _centroid(primed_e)),
        "n_cold": len(cold), "n_primed": len(primed),
        "n_suspicious": sum(1 for g in cold + primed if is_suspicious(g)),
        "guesses_cold": cold, "guesses_primed": primed,
    }


def run_validation(rows: List[Dict], generate, embed, k: int,
                   progress_cb: Callable[[Dict, int, int], None] = None) -> Dict:
    per_item = []
    for idx, row in enumerate(rows):
        scores = score_item(generate, embed, row["id"], row["setup"],
                            row["punchline"], k)
        item = {"id": row["id"], "gold_class": row["gold_class"],
                "author": row.get("author", "?"), **scores}
        per_item.append(item)
        if progress_cb:
            progress_cb(item, idx, len(rows))

    def vals(items: List[Dict], key: str) -> List[float]:
        return [i[key] for i in items]

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    by_class: Dict[str, List[Dict]] = {}
    for item in per_item:
        by_class.setdefault(item["gold_class"], []).append(item)
    jokes = by_class.get("real_joke", [])
    nonseq_pooled = [i for i in per_item if i["gold_class"] in NONSEQ_POOL]

    summary: Dict = {
        "n_items": len(per_item),
        "class_means": {cls: {kk: mean(vals(items, kk))
                              for kk in ("d_cold", "d_primed", "resolution",
                                         "overlap_primed_max")}
                        for cls, items in by_class.items()},
    }
    if jokes and nonseq_pooled:
        summary["auc_resolution_joke_vs_nonseq_pooled"] = rank_auc(
            vals(jokes, "resolution"), vals(nonseq_pooled, "resolution"))
        summary["auc_dcold_joke_vs_nonseq_pooled"] = rank_auc(
            vals(jokes, "d_cold"), vals(nonseq_pooled, "d_cold"))
        summary["auc_gap_resolution_minus_dcold"] = (
            summary["auc_resolution_joke_vs_nonseq_pooled"]
            - summary["auc_dcold_joke_vs_nonseq_pooled"])
        # Per-author primary (fixture-author-variance [LEARN]).
        for author in sorted({i["author"] for i in per_item}):
            a_jokes = [i for i in jokes if i["author"] == author]
            a_nonseq = [i for i in nonseq_pooled if i["author"] == author]
            if a_jokes and a_nonseq:
                summary["auc_resolution_joke_vs_nonseq_author_%s" % author] = \
                    rank_auc(vals(a_jokes, "resolution"),
                             vals(a_nonseq, "resolution"))
    if jokes:
        for cls, items in by_class.items():
            if cls == "real_joke" or cls in NONSEQ_POOL:
                continue
            summary["auc_resolution_joke_vs_%s" % cls] = rank_auc(
                vals(jokes, "resolution"), vals(items, "resolution"))
            # d_cold-alone control per guard (audit #15): a guard "pass"
            # where d_cold already separates carries no resolution signal.
            summary["auc_dcold_joke_vs_%s" % cls] = rank_auc(
                vals(jokes, "d_cold"), vals(items, "d_cold"))
    summary["n_suspicious_guesses"] = sum(i["n_suspicious"] for i in per_item)
    return {"per_item": per_item, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16",
                        choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--fixtures", required=True,
                        help="comma-separated fixture paths; author label = "
                             "A, B, ... by position")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke-test on the first N items per fixture")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: List[Dict] = []
    for idx, path in enumerate(args.fixtures.split(",")):
        author = chr(ord("A") + idx)
        fixture_rows = load_fixture(Path(path))
        if args.limit:
            fixture_rows = fixture_rows[:args.limit]
        for row in fixture_rows:
            row["author"] = author
        rows.extend(fixture_rows)

    t0 = time.time()
    gen = HFGenerator(args.model, args.device, args.dtype,
                      args.temperature, args.top_p, args.max_new_tokens)
    args.out.mkdir(parents=True, exist_ok=True)
    partial_path = args.out / "partial_items.jsonl"
    partial_path.write_text("")

    def progress(item: Dict, idx: int, total: int) -> None:
        # Incremental persistence + heartbeat (audit #21): a crash at item
        # 55 costs one item, not the run; a hang is visible on stdout.
        with partial_path.open("a") as fh:
            fh.write(json.dumps(item) + "\n")
        print("[%d/%d] %s resolution=%+.3f suspicious=%d"
              % (idx + 1, total, item["id"], item["resolution"],
                 item["n_suspicious"]), flush=True)

    results = {"model": args.model, "device": args.device,
               "dtype": args.dtype, "k": args.k,
               "temperature": args.temperature, "top_p": args.top_p,
               "max_new_tokens": args.max_new_tokens,
               "fixtures": args.fixtures,
               **run_validation(rows, gen.generate, default_embed_fn(),
                                args.k, progress_cb=progress)}
    results["wall_seconds"] = round(time.time() - t0, 1)

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    shutil.copy2(__file__, args.out / Path(__file__).name)

    s = results["summary"]
    print("== EXP-021 (n=%d) ==" % s["n_items"])
    for cls, m in sorted(s["class_means"].items()):
        print("  %-28s d_cold=%.3f  d_primed=%.3f  resolution=%+.3f  overlap=%.2f"
              % (cls, m["d_cold"], m["d_primed"], m["resolution"],
                 m["overlap_primed_max"]))
    for key in sorted(kk for kk in s if kk.startswith("auc_")):
        print("  %-48s %.3f" % (key, s[key]))
    print("results -> %s" % (args.out / "results.json"))


if __name__ == "__main__":
    main()

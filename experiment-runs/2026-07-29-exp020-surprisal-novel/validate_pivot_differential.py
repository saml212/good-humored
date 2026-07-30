#!/usr/bin/env python3
"""EXP-019 -- policy-native pivot surprisal-resolution differential.

Computes, from a locally-loaded causal LM's own teacher-forced logits
(no judge, no API, no sampling noise):

    S(cont|ctx)  = -mean per-token logprob of the continuation
    S_cold       = S(punchline | setup)
    S_primed     = S(punchline | setup + CUE)   # CUE is item-independent
    deltaS       = S_cold - S_primed

The registered hypothesis (EXPERIMENT_LOG.md, EXP-019, 2026-07-29):
deltaS separates real_joke from setup_nonsequitur -- both classes SPIKE
(high S_cold), but only a real joke's punchline becomes more predictable
when the model is told to expect a clever twist (resolution exists), so
only real jokes should see a large primed drop. Baselines computed in
the same pass, per the registration's built-in simpler-baseline
disproof: S_cold alone (Xie 2021 surprisal) and mean next-token entropy
over punchline positions (Xie 2021 uncertainty).

Boundary handling: the registration pins the CONCATENATED texts
(cold = fixture's own `text` field: setup + " " + punchline). For
teacher-forced scoring the space is attached to the CONTINUATION side
(ctx=setup, cont=" "+punchline) so BPE never merges a context token
across the seam; a prefix-match assertion verifies ctx tokens survive
re-tokenization of ctx+cont, and any item where they don't is scored
from the first divergent position and counted in `seam_mismatches`
(diagnostic -- 0 expected for this fixture).

Mode-primacy is per-experiment: EXP-019 registered raw-text as primary
(chat-template diagnostic); EXP-020 registered CHAT-TEMPLATE as primary
(raw-text diagnostic). The runner computes both under --chat-template;
which block is load-bearing is pinned in each EXPERIMENT_LOG
registration, not here.

Usage:
  python3 -m env.validate_pivot_differential \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --out experiment-runs/2026-07-29-exp019-pivot-differential
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "incongruity_gate_fixture.jsonl"
HELD_OUT = Path(__file__).parent / "tests" / "fixtures" / "pivot_differential_heldout.jsonl"
CLASSES = ("real_joke", "setup_nonsequitur", "boring_expected",
           "vague_abstract_gaming_probe")

# Registered verbatim in EXPERIMENT_LOG.md (EXP-019). Item-independent by
# construction -- the leakage-free analogue of EXP-014's generic
# "clever twist" primer.
CUE = "And here comes the clever twist that reinterprets the setup:"


def load_fixture(path: Path) -> List[Dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        for field in ("id", "gold_class", "setup", "punchline"):
            if field not in row:
                raise ValueError("fixture row missing %r: %s" % (field, row))
    return rows


def rank_auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5*P(pos == neg), over all pairs."""
    if not positives or not negatives:
        raise ValueError("rank_auc needs non-empty groups")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def extract_continuation_ids(tokenize: Callable[[str], List[int]],
                             ctx: str, cont: str) -> Tuple[List[int], int, bool]:
    """Token ids for ctx+cont, the index where scoring starts, and whether
    the ctx prefix survived re-tokenization intact (seam check).

    When BPE merges across the seam, scoring starts at the first index
    where full and ctx-alone tokenizations diverge -- every token from
    there on is (conservatively) treated as continuation.
    """
    ctx_ids = tokenize(ctx)
    full_ids = tokenize(ctx + cont)
    seam_ok = full_ids[:len(ctx_ids)] == ctx_ids
    if seam_ok:
        start = len(ctx_ids)
    else:
        start = 0
        for a, b in zip(ctx_ids, full_ids):
            if a != b:
                break
            start += 1
    if start >= len(full_ids):
        raise ValueError("continuation produced no scoreable tokens for ctx=%r" % ctx[:40])
    if start == 0:
        # scoring j=0 would index logits[-1] -- a silent wraparound to the
        # LAST position (audit finding #4). No valid ctx yields this.
        raise ValueError("no context tokens survived tokenization for ctx=%r" % ctx[:40])
    return full_ids, start, seam_ok


class HFScorer:
    """Teacher-forced scorer around a HF causal LM. Loaded once, reused."""

    def __init__(self, model_name: str, device: str, dtype: str):
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

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def score(self, ctx: str, cont: str) -> Dict:
        """NLL stats for cont given ctx (all logprob math in float32)."""
        torch = self.torch
        full_ids, start, seam_ok = extract_continuation_ids(self.tokenize, ctx, cont)
        input_ids = torch.tensor([full_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0].float()
        # logits[i] predicts token i+1: continuation token j (j >= start)
        # is predicted by logits[j-1].
        log_probs = torch.log_softmax(logits, dim=-1)
        cont_positions = list(range(start, len(full_ids)))
        token_nlls = [
            -log_probs[j - 1, full_ids[j]].item() for j in cont_positions
        ]
        probs = torch.softmax(logits, dim=-1)
        entropies = [
            -(probs[j - 1] * log_probs[j - 1]).sum().item() for j in cont_positions
        ]
        bad = [x for x in token_nlls + entropies
               if math.isnan(x) or math.isinf(x)]
        if bad:
            # fp16-on-MPS forward passes can silently produce NaN/Inf; a NaN
            # compares False in rank_auc and would deflate an AUC without an
            # error (audit finding #16).
            raise ValueError("non-finite NLL/entropy for ctx=%r" % ctx[:40])
        return {
            "mean_nll": sum(token_nlls) / len(token_nlls),
            "first3_nll": sum(token_nlls[:3]) / len(token_nlls[:3]),
            "max_nll": max(token_nlls),
            "mean_entropy": sum(entropies) / len(entropies),
            "n_tokens": len(token_nlls),
            "seam_ok": seam_ok,
        }


def score_item(scorer, setup: str, punchline: str, chat_template: bool = False) -> Dict:
    """Cold + primed scores and the differential for one fixture item."""
    cont = " " + punchline
    if chat_template:
        # Diagnostic variant only (registration): wrap the context as a
        # user turn with generation prompt; the punchline is scored as
        # the beginning of the assistant reply.
        def wrap(user_text: str) -> str:
            return scorer.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True)
        cold_ctx = wrap(setup)
        primed_ctx = wrap(setup + " " + CUE)
        cont = punchline  # assistant text starts unspaced after the prompt
    else:
        cold_ctx = setup
        primed_ctx = setup + " " + CUE
    cold = scorer.score(cold_ctx, cont)
    primed = scorer.score(primed_ctx, cont)
    return {
        "s_cold": cold["mean_nll"],
        "s_primed": primed["mean_nll"],
        "delta_s": cold["mean_nll"] - primed["mean_nll"],
        "entropy_cold": cold["mean_entropy"],
        "first3_cold": cold["first3_nll"],
        "max_cold": cold["max_nll"],
        "n_tokens": cold["n_tokens"],
        "seam_ok": cold["seam_ok"] and primed["seam_ok"],
    }


def run_validation(rows: List[Dict], scorer, chat_template: bool = False) -> Dict:
    per_item: List[Dict] = []
    for row in rows:
        scores = score_item(scorer, row["setup"], row["punchline"], chat_template)
        per_item.append({"id": row["id"], "gold_class": row["gold_class"], **scores})

    by_class: Dict[str, List[Dict]] = {}
    for item in per_item:
        by_class.setdefault(item["gold_class"], []).append(item)

    def vals(cls: str, key: str) -> List[float]:
        return [i[key] for i in by_class.get(cls, [])]

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "class_means": {
            cls: {k: mean(vals(cls, k))
                  for k in ("s_cold", "s_primed", "delta_s", "entropy_cold")}
            for cls in by_class
        },
        "seam_mismatches": sum(1 for i in per_item if not i["seam_ok"]),
        "n_items": len(per_item),
    }

    # Registered primary + baselines: real_joke vs setup_nonsequitur.
    if by_class.get("real_joke") and by_class.get("setup_nonsequitur"):
        jokes, nonseq = "real_joke", "setup_nonsequitur"
        summary["auc_deltaS_joke_vs_nonseq"] = rank_auc(
            vals(jokes, "delta_s"), vals(nonseq, "delta_s"))
        summary["auc_surprisal_joke_vs_nonseq"] = rank_auc(
            vals(jokes, "s_cold"), vals(nonseq, "s_cold"))
        summary["auc_entropy_joke_vs_nonseq"] = rank_auc(
            vals(jokes, "entropy_cold"), vals(nonseq, "entropy_cold"))
        summary["auc_gap_deltaS_minus_surprisal"] = (
            summary["auc_deltaS_joke_vs_nonseq"]
            - summary["auc_surprisal_joke_vs_nonseq"])
    # Registered guard: the vague-probe anti-gaming bar.
    if by_class.get("real_joke") and by_class.get("vague_abstract_gaming_probe"):
        summary["auc_deltaS_joke_vs_vague"] = rank_auc(
            vals("real_joke", "delta_s"),
            vals("vague_abstract_gaming_probe", "delta_s"))
    # Generic per-class AUCs (EXP-020: fixture classes vary; every
    # non-joke class gets a surprisal AUC and a deltaS AUC vs real_joke).
    if by_class.get("real_joke"):
        for cls in by_class:
            if cls == "real_joke":
                continue
            summary["auc_scold_joke_vs_%s" % cls] = rank_auc(
                vals("real_joke", "s_cold"), vals(cls, "s_cold"))
            summary["auc_deltaS_joke_vs_%s" % cls] = rank_auc(
                vals("real_joke", "delta_s"), vals(cls, "delta_s"))
    return {"per_item": per_item, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16",
                        choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--held-out", type=Path, default=HELD_OUT,
                        help="auditor-authored blind set; scored iff the file exists")
    parser.add_argument("--chat-template", action="store_true",
                        help="ALSO run the chat-templated diagnostic variant")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke-test on the first N fixture items only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_fixture(args.fixture)
    if args.limit:
        rows = rows[:args.limit]

    t0 = time.time()
    scorer = HFScorer(args.model, args.device, args.dtype)
    results = {"model": args.model, "device": args.device, "dtype": args.dtype,
               "cue": CUE, "fixture": str(args.fixture),
               "committed": run_validation(rows, scorer)}
    if args.chat_template:
        results["committed_chat_template"] = run_validation(
            rows, scorer, chat_template=True)
    if args.held_out.exists():
        held_rows = load_fixture(args.held_out)
        results["held_out"] = run_validation(held_rows, scorer)
        if args.chat_template:
            results["held_out_chat_template"] = run_validation(
                held_rows, scorer, chat_template=True)
    else:
        results["held_out"] = None
        print("WARNING: held-out fixture %s missing -- this run is "
              "DIAGNOSTIC ONLY and cannot close EXP-019 (registration "
              "requires committed + blind held-out, both or neither)."
              % args.held_out, file=sys.stderr)
    results["wall_seconds"] = round(time.time() - t0, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    shutil.copy2(__file__, args.out / Path(__file__).name)

    for split in ("committed", "held_out",
                  "committed_chat_template", "held_out_chat_template"):
        block = results.get(split)
        if not block:
            continue
        s = block["summary"]
        print("== %s (n=%d, seam_mismatches=%d) ==" %
              (split, s["n_items"], s["seam_mismatches"]))
        for cls, m in sorted(s["class_means"].items()):
            print("  %-28s s_cold=%.3f  s_primed=%.3f  deltaS=%+.3f" %
                  (cls, m["s_cold"], m["s_primed"], m["delta_s"]))
        for key in sorted(k for k in s if k.startswith("auc_")):
            print("  %-44s %.3f" % (key, s[key]))
    print("results -> %s" % (args.out / "results.json"))


if __name__ == "__main__":
    main()

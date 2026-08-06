#!/usr/bin/env python3
"""S5 throughput bench -- aggregate generation tok/s vs concurrency
against a local OpenAI-compatible server. Measurement only (no
calibration rows, per the screen registration: S5 is a table).

Usage:
  python3 -m benchmark.bench_throughput --base-url http://127.0.0.1:8001/v1 \
      --model qwen3-8b --out /data/good-humored/runs/bench_qwen3-8b.json
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("You're bantering with a friend about a broken office printer. "
          "Reply with one witty sentence, then keep riffing for a short "
          "paragraph about where the situation could go next.")


def one_call(base_url: str, model: str, max_tokens: int, seed: int) -> int:
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 1.0,
               "seed": seed,
               "messages": [{"role": "user", "content": PROMPT}],
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer gh-local"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    return body.get("usage", {}).get("completion_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--calls-per-level", type=int, default=None,
                    help="default: 4x the concurrency level")
    ap.add_argument("--levels", default="1,8,32,128")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {"model": args.model, "max_tokens": args.max_tokens,
               "levels": {}}
    for level in [int(x) for x in args.levels.split(",")]:
        n_calls = args.calls_per_level or 4 * level
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=level) as pool:
            toks = list(pool.map(
                lambda i: one_call(args.base_url, args.model,
                                   args.max_tokens, seed=i),
                range(n_calls)))
        wall = time.time() - t0
        total = sum(toks)
        results["levels"][str(level)] = {
            "n_calls": n_calls, "completion_tokens": total,
            "wall_s": round(wall, 1),
            "agg_tok_per_s": round(total / wall, 1)}
        print("concurrency=%-4d calls=%-4d tok=%-7d wall=%6.1fs  agg=%8.1f tok/s"
              % (level, n_calls, total, wall, total / wall), flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("results ->", args.out)


if __name__ == "__main__":
    main()

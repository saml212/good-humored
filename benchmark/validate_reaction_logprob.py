#!/usr/bin/env python3
"""EXP-023c -- API logprob-magnitude reaction arm (registered 2026-07-30).

One call per caption: read top_logprobs of the FIRST generated token of
the audience's reply and compute the laughter-class probability mass.
The continuous version of 023a/b's sampled binary -- readable at
ceiling, no binomial noise. Deepseek-only (capability-probed).

Usage:
  python3 -m benchmark.validate_reaction_logprob \
      --out experiment-runs/2026-07-30-exp023c-reaction-logprob
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.providers import (_FLEET_SECRETS_DIR,  # noqa: E402
                                 _OPENAI_COMPAT_REGISTRY, parse_env_file)
from benchmark.validate_reaction_pilot import (SYSTEM_PROMPT,  # noqa: E402
                                               USER_TEMPLATE, load_pilot_set,
                                               spearman)

PROVIDER = "deepseek"
TOP_LOGPROBS = 20
ZERO_MASS_FLOOR = 1e-8
REPEAT_SUBSET = 24  # captions re-called once for the determinism diagnostic

STRICT_PREFIXES = ("haha", "hah", "hehe", "heh", "lol", "lmao", "lmfao",
                   "laugh", "\U0001F602", "\U0001F923", "\U0001F480")
STRICT_EXACT = ("ha",)
LOOSE_EXTRA = ("l", "*l")  # ambiguous stubs (*laughs? looks?) -- bound only


def normalize_token(token: str) -> str:
    return token.lstrip("*\"'“” `").strip().lower()


def classify_token(token: str) -> Optional[str]:
    """'strict' / 'loose' / None for one top-logprob token."""
    raw = token.strip().lower()
    norm = normalize_token(token)
    if norm in STRICT_EXACT or any(norm.startswith(p) for p in STRICT_PREFIXES):
        return "strict"
    if raw in LOOSE_EXTRA or norm in LOOSE_EXTRA:
        return "loose"
    return None


def laughter_mass(top_logprobs: List[Dict]) -> Dict[str, float]:
    strict = ambiguous = 0.0
    for t in top_logprobs:
        cls = classify_token(t["token"])
        if cls == "strict":
            strict += math.exp(t["logprob"])
        elif cls == "loose":
            ambiguous += math.exp(t["logprob"])
    return {"L_strict": math.log(max(strict, ZERO_MASS_FLOOR)),
            "L_loose": math.log(max(strict + ambiguous, ZERO_MASS_FLOOR)),
            "strict_mass": strict}


def make_logprob_call():
    spec = _OPENAI_COMPAT_REGISTRY[PROVIDER]
    env = parse_env_file(os.path.join(_FLEET_SECRETS_DIR, spec["env_file"]))
    key = env[spec["key_var"]]
    url = (spec["base_url"] or env[spec["base_url_var"]]).rstrip("/") \
        + "/chat/completions"

    def call(prompt: str) -> List[Dict]:
        payload = {"model": spec["model"],
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 2, "temperature": 1.0,
                   "logprobs": True, "top_logprobs": TOP_LOGPROBS}
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})
        last: Exception = RuntimeError("unreachable")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read())
                return body["choices"][0]["logprobs"]["content"][0][
                    "top_logprobs"]
            except Exception as e:  # bounded retry, then loud
                last = e
                time.sleep(3 * 2 ** attempt)
        raise last
    return call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contests = load_pilot_set()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "cache_logprob.jsonl"
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                cache[rec["key"]] = rec
            except json.JSONDecodeError:
                continue

    call = make_logprob_call()
    jobs = []
    for c in contests:
        for ci, cap in enumerate(c["captions"]):
            for rep in (0, 1):
                if rep == 1 and not (c is contests[0] and ci < REPEAT_SUBSET):
                    continue  # repeats only for the determinism subset
                key = "%d|%d|%d" % (c["contest_id"], ci, rep)
                if key not in cache:
                    prompt = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
                        description=c["description"], caption=cap["caption"])
                    jobs.append((key, prompt))

    def worker(job):
        key, prompt = job
        try:
            tops = call(prompt)
            return {"key": key, "tops": [(t["token"], t["logprob"])
                                         for t in tops],
                    **laughter_mass(tops)}
        except Exception as e:
            return {"key": key, "error": str(e)[:200]}

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            open(cache_path, "a") as fh:
        for rec in pool.map(worker, jobs):
            cache[rec["key"]] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 50 == 0:
                print("%d/%d calls" % (done, len(jobs)), flush=True)

    per_contest, errors = [], 0
    masses = []
    for c in contests:
        ls, means = [], []
        for ci, cap in enumerate(c["captions"]):
            rec = cache.get("%d|%d|0" % (c["contest_id"], ci))
            if rec is None or "error" in rec:
                errors += 1
                continue
            ls.append(rec["L_strict"])
            means.append(cap["mean"])
            masses.append(rec["strict_mass"])
        rho = spearman(ls, means) if len(ls) >= 3 else None
        per_contest.append({"contest_id": c["contest_id"], "rho": rho,
                            "n": len(ls)})
    rhos = [p["rho"] for p in per_contest if p["rho"] is not None]

    # determinism diagnostic: |L_strict(rep0) - L_strict(rep1)| on subset
    diffs = []
    c0 = contests[0]
    for ci in range(min(REPEAT_SUBSET, len(c0["captions"]))):
        a = cache.get("%d|%d|0" % (c0["contest_id"], ci))
        b = cache.get("%d|%d|1" % (c0["contest_id"], ci))
        if a and b and "error" not in a and "error" not in b:
            diffs.append(abs(a["L_strict"] - b["L_strict"]))

    results = {
        "provider": PROVIDER,
        "mean_within_contest_rho_strict":
            sum(rhos) / len(rhos) if rhos else None,
        "n_contests_scored": len(rhos),
        "mean_strict_mass": sum(masses) / len(masses) if masses else None,
        "determinism_mean_abs_diff":
            sum(diffs) / len(diffs) if diffs else None,
        "errors": errors,
        "per_contest": per_contest,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    shutil.copy2(__file__, args.out / Path(__file__).name)
    for k in ("mean_within_contest_rho_strict", "n_contests_scored",
              "mean_strict_mass", "determinism_mean_abs_diff", "errors"):
        print("%-34s %s" % (k, results[k]))
    print("results -> %s" % (args.out / "results.json"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the verl prompt dataset for banter GRPO.

Each row is a session OPENER: the neutral policy system prompt for a
task (the partner's actual turns come from the frozen partner server
inside the agent loop). extra_info carries the session seed so the
loop reproduces the same seeded schedule machinery as the banked
data. Format per the installed-verl recon: parquet, prompt = messages
list, reward_model placeholder required by the dataset path.

Usage:
  python3 -m env.verl_bridge.build_session_dataset \
      --n-train 4096 --n-val 128 --out-dir /data/good-humored/data
"""

import argparse
import random
from pathlib import Path

from env.banter_rollout import POLICY_SYSTEM, TASKS


def rows(n, seed_base):
    rng = random.Random(seed_base)
    out = []
    for i in range(n):
        task = rng.choice(TASKS)
        out.append({
            "prompt": [{"role": "system",
                        "content": POLICY_SYSTEM.format(task=task)}],
            "data_source": "banter_session",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"task": task,
                           "session_seed": seed_base + i},
            "agent_name": "humor_session_agent",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import pandas as pd
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # seed bases far from every banked lane's offset space
    pd.DataFrame(rows(args.n_train, 2_000_000)).to_parquet(
        out / "sessions_train.parquet")
    pd.DataFrame(rows(args.n_val, 3_000_000)).to_parquet(
        out / "sessions_val.parquet")
    print("train=%d val=%d -> %s" % (args.n_train, args.n_val, out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Standalone smoke for HumorSessionAgentLoop: everything except the
verl runtime is REAL — live partner (:8004), audience (:8003), and a
'policy' served by the live 30B (:8002) standing in for verl's
rollout server; real MiniLM gate; real reward stack. Verl internals
(server_manager, apply_chat_template, rollout_config) are stubbed
with the same shapes the recon documented, so this burns down every
VERIFY-ON-SMOKE marker except the true verl token plumbing (which
only the GRPO smoke can confirm).

Usage (box, venv-verl):
  python3 -m env.verl_bridge.driver_smoke --n-sessions 2
"""

import argparse
import asyncio
import json
import sys
import types

sys.path.insert(0, "/data/good-humored/repo")

from env.verl_bridge.humor_session_loop import HumorSessionAgentLoop  # noqa


def build_stub(loop_cls, tokenizer, policy_url, policy_model):
    import httpx

    obj = loop_cls.__new__(loop_cls)  # bypass verl-runtime __init__
    obj.partner_base_url = "http://127.0.0.1:8004/v1"
    obj.partner_model = "qwen3-235b-a22b"
    obj.audience_base_url = "http://127.0.0.1:8003/v1"
    obj.audience_model = "glm-4.5-air"
    obj.num_rounds = 10
    obj.provocation_rate = 0.5
    obj.partner_max_tokens = 90
    obj._http = httpx.AsyncClient(timeout=300)
    obj._gate = None
    obj.tokenizer = tokenizer
    obj.rollout_config = types.SimpleNamespace(response_length=8192)

    async def apply_chat_template(messages, remove_system_prompt=False):
        # version-proof: render text, then tokenize (transformers 5.x
        # changed apply_chat_template's tokenize-return shape)
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False)
        return list(tokenizer(text, add_special_tokens=False).input_ids)

    obj.apply_chat_template = apply_chat_template

    class StubServerManager:
        async def generate(self, request_id, prompt_ids, sampling_params):
            # stand-in for verl's rollout server: live 30B server
            text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            async with httpx.AsyncClient(timeout=300) as c:
                r = await c.post(
                    policy_url + "/completions",
                    json={"model": policy_model, "prompt": text,
                          "max_tokens": 90, "temperature": 1.0},
                    headers={"Authorization": "Bearer gh-local"})
                r.raise_for_status()
                out_text = r.json()["choices"][0]["text"]
            ids = tokenizer(out_text, add_special_tokens=False).input_ids
            return types.SimpleNamespace(token_ids=list(ids),
                                         log_probs=None,
                                         stop_reason="stop")

    obj.server_manager = StubServerManager()
    return obj


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=2)
    ap.add_argument("--dump", default=None,
                    help="jsonl path: dump session turns + loop-path reward "
                         "components (for reward-path parity checks)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
    loop = build_stub(HumorSessionAgentLoop, tokenizer,
                      "http://127.0.0.1:8002/v1", "qwen3-30b-base")
    from env.banter_rollout import POLICY_SYSTEM, TASKS
    for i in range(args.n_sessions):
        task = TASKS[i % len(TASKS)]
        out = await loop.run(
            {"temperature": 1.0, "max_tokens": 90},
            raw_prompt=[{"role": "system",
                         "content": POLICY_SYSTEM.format(task=task)}],
            extra_info={"task": task, "session_seed": 5_000_000 + i},
            index=i)
        mask = out.response_mask
        print(json.dumps({
            "session": i, "task": task,
            "reward": out.reward_score,
            "num_turns": out.num_turns,
            "response_tokens": len(out.response_ids),
            "policy_tokens": sum(mask),
            "partner_tokens": len(mask) - sum(mask),
            "components": out.extra_fields.get("reward_components"),
        }, indent=2))
        if args.dump:
            with open(args.dump, "a") as f:
                f.write(json.dumps({
                    "session_id": 5_100_000 + i, "task": task,
                    "turns": out.extra_fields.get("session_turns"),
                    "loop_reward": out.reward_score,
                    "loop_components": out.extra_fields.get(
                        "reward_components")}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""verl AgentLoop for banter sessions: policy trains, a frozen partner
server produces the other side, the adjudicated reward stack scores
the finished session in-loop.

Built against the INSTALLED verl 0.8.0 surface (recon 2026-08-10,
file:line evidence in EXPERIMENT_LOG):
- AgentLoopBase.run(sampling_params, **kwargs) -> AgentLoopOutput is
  the only abstract method (agent_loop.py:360).
- response_mask: 1 = policy token (trained), 0 = partner token
  (excluded from loss) -- the ToolAgentLoop delta pattern
  (tool_agent_loop.py:380-403).
- Setting AgentLoopOutput.reward_score skips the reward manager
  (agent_loop.py:843) -- required, because the manager path only
  sees a flat decoded string and our reward needs session structure.
- Registration via agent_loop.yaml (_target_ this class) +
  rollout.agent.agent_loop_config_path.

Session mechanics mirror env/banter_rollout.py (same partner system
prompt, opening angles, weighted provocation scheduler -- imported,
not duplicated) so trained-on distribution == banked-data
distribution. VERIFY-ON-SMOKE markers flag the API calls that must
be confirmed the first time this actually runs.
"""

import asyncio
import json as _json
import os
import random

try:  # box venv-verl dependency; guarded so the module (and its
    # scheduler/reward logic) stays importable and testable locally
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from env.banter_rollout import (OPENING_ANGLES, PARTNER_SYSTEM,
                                PROVOCATIONS, TASKS, _seed)
from env.reward_stack import session_reward

try:  # importable only inside the verl venv; keep module testable outside
    from verl.experimental.agent_loop.agent_loop import (AgentLoopBase,
                                                         AgentLoopOutput)
except ImportError:  # pragma: no cover
    AgentLoopBase = object
    AgentLoopOutput = None

PROVOCATION_WEIGHTS = {"mock": 0.28, "joke": 0.24, "frustration": 0.16,
                       "observation": 0.16, "swear": 0.16}


class HumorSessionAgentLoop(AgentLoopBase):
    """10-round banter rollout against the frozen partner server."""

    def __init__(self, *args, partner_base_url, partner_model,
                 audience_base_url=None, audience_model=None,
                 num_rounds=10, provocation_rate=0.5,
                 partner_max_tokens=90, **kwargs):
        super().__init__(*args, **kwargs)
        self.partner_base_url = partner_base_url.rstrip("/")
        self.partner_model = partner_model
        self.audience_base_url = (audience_base_url or "").rstrip("/") or None
        self.audience_model = audience_model
        self.num_rounds = num_rounds
        self.provocation_rate = provocation_rate
        self.partner_max_tokens = partner_max_tokens
        if httpx is None:  # pragma: no cover
            raise RuntimeError("httpx required at runtime (venv-verl)")
        self._http = httpx.AsyncClient(timeout=300)
        self._gate = None  # lazy: MiniLM loads once per worker process

    def _ensure_gate(self):
        if self._gate is None:
            from env.band_term import BandGate
            from sentence_transformers import SentenceTransformer
            # CPU-pinned: every AgentLoop worker loads this; on GPU it
            # parked ~1.4GB x 8 workers of CUDA context on GPU 0 (98%
            # peak in the smoke -- debug-agent flag)
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

            def embed(texts):
                return [list(map(float, v)) for v in
                        model.encode(list(texts), normalize_embeddings=True,
                                     show_progress_bar=False)]
            self._gate = BandGate(embed)
        return self._gate

    async def _chat(self, base_url, model, system, messages, seed,
                    max_tokens):
        payload = {"model": model, "temperature": 1.0, "seed": seed,
                   "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}]
                   + messages,
                   "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(3):
            try:
                r = await self._http.post(
                    base_url + "/chat/completions", json=payload,
                    headers={"Authorization": "Bearer gh-local"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * 2 ** attempt)

    async def _reaction(self, transcript_msgs):
        """laughter_mass of the audience's next-token distribution."""
        from benchmark.validate_reaction_logprob import laughter_mass
        payload = {"model": self.audience_model, "max_tokens": 2,
                   "temperature": 1.0, "logprobs": True, "top_logprobs": 20,
                   "messages": transcript_msgs,
                   "chat_template_kwargs": {"enable_thinking": False}}
        r = await self._http.post(
            self.audience_base_url + "/chat/completions", json=payload,
            headers={"Authorization": "Bearer gh-local"})
        r.raise_for_status()
        tops = r.json()["choices"][0]["logprobs"]["content"][0][
            "top_logprobs"]
        return laughter_mass([{"token": t["token"], "logprob": t["logprob"]}
                              for t in tops])["L_strict"]

    async def run(self, sampling_params, **kwargs):
        extra = kwargs.get("extra_info") or {}
        session_id = int(extra.get("session_seed", kwargs.get("index", 0)))
        rng = random.Random(_seed(session_id, "schedule"))
        task = extra.get("task") or rng.choice(TASKS)
        opening_angle = rng.choice(OPENING_ANGLES)
        types = sorted(PROVOCATIONS)
        schedule = {}
        for turn in range(1, self.num_rounds):
            if rng.random() < self.provocation_rate:
                schedule[turn] = rng.choices(
                    types, weights=[PROVOCATION_WEIGHTS[t] for t in types])[0]

        partner_sys = PARTNER_SYSTEM.format(task=task)
        response_mask, turns = [], []
        partner_view = []

        async def partner_turn(rnd):
            p_sys = partner_sys
            if rnd == 0:
                p_sys = ("TO OPEN THIS CONVERSATION: " + opening_angle
                         + " ") + partner_sys
            elif schedule.get(rnd):
                p_sys = ("THIS TURN, before anything else: "
                         + PROVOCATIONS[schedule[rnd]] + " ") + partner_sys
            text = await self._chat(
                self.partner_base_url, self.partner_model, p_sys,
                partner_view, _seed(session_id, "partner%d" % rnd),
                self.partner_max_tokens)
            partner_view.append({"role": "assistant", "content": text})
            turns.append({"role": "partner", "turn": rnd, "text": text,
                          "provocation": schedule.get(rnd) if rnd else None})
            return text

        # GRPO-V1 post-mortem fix: the FIRST partner turn joins the
        # INITIAL render (system + user together, ToolAgentLoop
        # pattern). The old system-only initial render left a dangling
        # empty assistant block (verl's helper hardcodes
        # add_generation_prompt=True) -- the malformation behind the
        # depressed training scores and the held-out null.
        first_partner = await partner_turn(0)
        messages = list(kwargs["raw_prompt"]) + [
            {"role": "user", "content": first_partner}]
        prompt_ids = await self.apply_chat_template(messages)
        prompt_len = len(prompt_ids)

        for rnd in range(self.num_rounds):
            if rnd > 0:
                partner_text = await partner_turn(rnd)
                delta = await self.apply_chat_template(
                    [{"role": "user", "content": partner_text}],
                    remove_system_prompt=True)
                prompt_ids += delta
                response_mask += [0] * len(delta)
                if len(response_mask) >= self.rollout_config.response_length:
                    break

            # policy turn (mask 1, trained)
            out = await self.server_manager.generate(
                request_id="%s-%d" % (kwargs.get("index", session_id), rnd),
                prompt_ids=prompt_ids, sampling_params=sampling_params)
            prompt_ids += out.token_ids
            response_mask += [1] * len(out.token_ids)
            policy_text = self.tokenizer.decode(out.token_ids,
                                                skip_special_tokens=True)
            partner_view.append({"role": "user", "content": policy_text})
            turns.append({"role": "policy", "turn": rnd,
                          "text": policy_text, "provocation": None})
            if len(response_mask) >= self.rollout_config.response_length:
                break

        session = {"session_id": session_id, "task": task, "turns": turns}
        # v2: training transcript dumps (per-worker files -- concurrent
        # processes must not interleave one file). The missing evidence
        # that forced the v1 archaeology.
        dump_dir = os.environ.get("GH_SESSION_DUMP_DIR")
        if dump_dir:
            with open(os.path.join(
                    dump_dir, "sessions.%d.jsonl" % os.getpid()), "a") as f:
                f.write(_json.dumps(session) + "\n")
        reaction_fn = None
        if self.audience_base_url:
            # precompute reactions ASYNC here (session_reward is sync and
            # cannot await; a run_until_complete bridge would deadlock the
            # running loop) -- keyed by transcript length, same msgs shape
            # session_reward builds
            precomputed = {}
            for i, t in enumerate(turns):
                if t["role"] != "policy":
                    continue
                msgs = [{"role": ("assistant" if x["role"] == "partner"
                                  else "user"), "content": x["text"]}
                        for x in turns[:i + 1]]
                try:
                    precomputed[len(msgs)] = await self._reaction(msgs)
                except Exception:
                    precomputed[len(msgs)] = -18.4  # floor on audience error

            def reaction_fn(msgs):
                return precomputed.get(len(msgs), -18.4)
        r = session_reward(session, self._ensure_gate(),
                           reaction_fn=reaction_fn)

        n = min(len(response_mask), self.rollout_config.response_length)
        return AgentLoopOutput(
            prompt_ids=prompt_ids[:prompt_len],
            response_ids=prompt_ids[prompt_len:prompt_len + n],
            response_mask=response_mask[:n],
            reward_score=r["total"],
            num_turns=len(turns),
            metrics={},
            extra_fields={"reward_components":
                          {k: v for k, v in r.items() if k != "total"},
                          "session_turns": turns,
                          "task": task})

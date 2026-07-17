"""The rejection cascade runner.

One cascade = one conversation: the model under test tells a joke, the
rejector labels its topic and rejects it (accumulating), repeat to
`depth`. Output is a JSONL of turn records; the topic path is the
measurement (metrics.py).
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .metrics import looks_like_refusal
from .providers import transcript_prompt
from .rejector import (LABEL_PROMPT_VERSION, OPENING_PROMPT, label_topic,
                       rejection_message)


def run_cascade(
    model_complete: Callable[[str], str],
    rejector_complete: Callable[[str], str],
    depth: int,
    run_id: str,
    log_path: Optional[Path] = None,
    temperature: Optional[float] = None,
) -> Dict:
    """Run one cascade. Returns the run record; optionally logs JSONL
    per turn so a crashed run is still data.

    `temperature` is metadata ONLY — this function never constructs the
    request that used it (that already happened inside model_complete's
    closure, see providers.make_openai_compat). It is recorded here
    purely so every turn record and the run record carry the number a
    run was made at: a temperature we can't recover from the logs later
    is a protocol violation for EXP-007 (which manipulates it). None
    means "not overridden" (provider default), recorded as such, not
    coerced to 0.
    """
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": OPENING_PROMPT}]
    path: List[str] = []
    refusal_turns: List[int] = []
    turns: List[Dict] = []

    log_f = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a")

    try:
        for turn in range(depth):
            joke = model_complete(transcript_prompt(messages))
            messages.append({"role": "assistant", "content": joke})

            if looks_like_refusal(joke):
                refusal_turns.append(turn)

            topic = label_topic(joke, rejector_complete)
            rejection = rejection_message(topic, path)
            path.append(topic)
            messages.append({"role": "user", "content": rejection})

            rec = {"run_id": run_id, "turn": turn, "joke": joke,
                   "topic": topic, "refusal": turn in refusal_turns,
                   "temperature": temperature, "ts": time.time()}
            turns.append(rec)
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
    finally:
        if log_f:
            log_f.close()

    return {
        "run_id": run_id,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "temperature": temperature,
        "depth_requested": depth,
        "depth_completed": len(path),
        "path": path,
        "refusal_turns": refusal_turns,
        "turns": turns,
    }

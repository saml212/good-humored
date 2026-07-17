"""Model providers for the cascade benchmark.

v0 uses the `claude` CLI in print mode so experiments can run on a dev
box with no API keys. The provider interface is a plain callable
`complete(prompt: str) -> str`, so swapping in real APIs (Anthropic /
OpenAI / OpenRouter, true multi-turn) later touches nothing else.

Fidelity caveat, stated up front: `claude -p` is single-shot, so
multi-turn cascades are encoded as a transcript inside one prompt. Fine
for the pilot; the paper-grade run must use real multi-turn APIs with
temperature control. The rejector is single-turn by design, so for the
rejector this wrapper is fully faithful.
"""

import subprocess
import tempfile
import time
from typing import Callable, Dict, List

# Alias -> full model id, so run logs record exactly what ran.
CLAUDE_MODELS: Dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4-8",
}


def make_claude_cli(model: str, timeout_s: int = 120,
                    retries: int = 2) -> Callable[[str], str]:
    """Return a complete(prompt) callable backed by `claude -p`.

    Runs in a NEUTRAL empty cwd on purpose: `claude` loads the project
    CLAUDE.md and hooks from wherever it is invoked, and a rejector that
    has read this repo's research docs is a contaminated instrument
    (discovered live — a smoke call replied "I've read the context").
    """
    model_id = CLAUDE_MODELS.get(model, model)
    neutral_cwd = tempfile.mkdtemp(prefix="gh-neutral-")

    def complete(prompt: str) -> str:
        last_err: Exception = RuntimeError("unreachable")
        for attempt in range(retries + 1):
            try:
                proc = subprocess.run(
                    ["claude", "-p", "--model", model_id, prompt],
                    capture_output=True, text=True, timeout=timeout_s,
                    cwd=neutral_cwd,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
                last_err = RuntimeError(
                    "claude -p rc=%d stderr=%s"
                    % (proc.returncode, proc.stderr.strip()[:200]))
            except FileNotFoundError:
                # Retrying can't fix a missing binary — fail fast, actionably.
                raise RuntimeError(
                    "`claude` CLI not found on the subprocess PATH. Install "
                    "it or run from a shell where `claude` resolves.") from None
            except subprocess.TimeoutExpired as e:
                last_err = e
            time.sleep(3 * 2 ** attempt)  # 3s, 6s, 12s — rate-limit friendly
        raise last_err

    complete.__name__ = "claude_cli_%s" % model  # shows up in run logs
    return complete


def transcript_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten a conversation into one prompt for single-shot providers.

    Roles: 'user' is the rejector-side, 'assistant' is the model under
    test. The final line instructs the model to reply as the assistant.
    """
    lines = ["This is an ongoing conversation. Continue it as the "
             "Comedian. Reply with ONLY the Comedian's next message.", ""]
    role_name = {"user": "Audience", "assistant": "Comedian"}
    for m in messages:
        lines.append("%s: %s" % (role_name[m["role"]], m["content"]))
    lines.append("Comedian:")
    return "\n".join(lines)

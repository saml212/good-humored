"""Model providers for the cascade benchmark.

v0 uses the `claude` CLI in print mode so experiments can run on a dev
box with no API keys. The provider interface is a plain callable
`complete(prompt: str) -> str`, so swapping in real APIs (Anthropic /
OpenAI / OpenRouter, true multi-turn) later touches nothing else.
`codex exec` (v1) and direct OpenAI-compatible HTTP APIs (v2) follow the
same contract; `get_provider` at the bottom dispatches "prefix:alias"
specs to whichever factory backs that prefix.

Fidelity caveat, stated up front: `claude -p` / `codex exec` are
single-shot, so multi-turn cascades are encoded as a transcript inside
one prompt. Fine for the pilot; the paper-grade run must use real
multi-turn APIs with temperature control. The rejector is single-turn by
design, so for the rejector this wrapper is fully faithful.
"""

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

# Alias -> full model id, so run logs record exactly what ran.
CLAUDE_MODELS: Dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4-8",
    "fable": "claude-fable-5",
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
                # Audit WARN-2: claude puts its diagnostic on STDOUT on
                # failure ("There's an issue with the selected model...");
                # stderr is typically empty. Prefer whichever has content.
                last_err = RuntimeError(
                    "claude -p rc=%d: %s"
                    % (proc.returncode,
                       (proc.stderr.strip() or proc.stdout.strip())[:200]))
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


# codex-cli 0.144.4, run on the owner's ChatGPT subscription — GPT models
# as cascade subjects with no API key. Alias -> full model slug, cross
# -checked against `codex --help` and `~/.codex/models_cache.json`; sol/
# mini/5.4 verified live with "Say OK and nothing else" (~4-6s each).
CODEX_MODELS: Dict[str, str] = {
    "sol": "gpt-5.6-sol",      # default frontier agentic model (verified)
    "mini": "gpt-5.4-mini",    # cheap/fast (verified)
    "5.4": "gpt-5.4",          # verified
    "5.5": "gpt-5.5",
    "terra": "gpt-5.6-terra",
    "luna": "gpt-5.6-luna",
    "spark": "gpt-5.3-codex-spark",
}


def make_codex_cli(model: str, timeout_s: int = 120,
                   retries: int = 2) -> Callable[[str], str]:
    """Return a complete(prompt) callable backed by `codex exec`.

    Neutral cwd for the same reason as make_claude_cli, verified
    independently for codex: a planted AGENTS.md in cwd got read into
    context and hijacked a diagnostic prompt (asked to quote it or say
    NONE; got the file's injected string back instead) even though a
    plain "say OK" instruction was obeyed over it in the same directory —
    contamination is real even when it isn't visible in the easy case.
    `--skip-git-repo-check` is required: a fresh tempdir is never a
    trusted git repo and `codex exec` refuses to start without it.
    `-s read-only` blocks the model from writing or executing anything
    regardless of cwd (empirically confirmed: asked it to write a file,
    got BLOCKED and an empty directory back) — it runs as a pure text
    model. `-o <file>` sidesteps stdout entirely: stdout carries a
    version/workdir/model/session-id header, hook fire/complete lines,
    and a token count, none of which is the answer; -o writes exactly the
    agent's final message and nothing else.
    """
    model_id = CODEX_MODELS.get(model, model)
    neutral_cwd = tempfile.mkdtemp(prefix="gh-neutral-")

    def complete(prompt: str) -> str:
        last_err: Exception = RuntimeError("unreachable")
        for attempt in range(retries + 1):
            out_fd, out_path = tempfile.mkstemp(prefix="codex-out-",
                                                 suffix=".txt")
            os.close(out_fd)
            try:
                proc = subprocess.run(
                    ["codex", "exec", "--skip-git-repo-check",
                     "-s", "read-only", "-m", model_id, "-o", out_path,
                     prompt],
                    capture_output=True, text=True, timeout=timeout_s,
                    cwd=neutral_cwd, stdin=subprocess.DEVNULL,
                )
                if proc.returncode == 0:
                    with open(out_path) as f:
                        text = f.read().strip()
                    if text:
                        return text
                last_err = RuntimeError(
                    "codex exec rc=%d: %s"
                    # Audit WARN-2: codex buries the real ERROR line at the
                    # END of stderr behind a boilerplate header — take the
                    # tail, not the head.
                    % (proc.returncode, proc.stderr.strip()[-400:]))
            except FileNotFoundError:
                # Retrying can't fix a missing binary — fail fast, actionably.
                raise RuntimeError(
                    "`codex` CLI not found on the subprocess PATH. Install "
                    "it or run from a shell where `codex` resolves.") from None
            except subprocess.TimeoutExpired as e:
                last_err = e
            finally:
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            time.sleep(3 * 2 ** attempt)  # 3s, 6s, 12s — rate-limit friendly
        raise last_err

    complete.__name__ = "codex_cli_%s" % model  # shows up in run logs
    return complete


def parse_env_file(path: str) -> Dict[str, str]:
    """Read KEY=VALUE lines from a hand-written .env file. No shell
    quoting/escaping semantics — just `#` comments, blank lines, and
    optional surrounding quotes on the value. Callers must never print or
    log the returned dict; these are live API keys."""
    env: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


_FLEET_SECRETS_DIR = os.path.expanduser("~/agents/.fleet-secrets")

# provider name -> env file (under _FLEET_SECRETS_DIR), the api-key var in
# that file, base url (fixed, or read from base_url_var when a provider
# needs a per-account endpoint), and the default chat model. Open-weight
# models via their native OpenAI-compatible endpoints — no local GPU.
_OPENAI_COMPAT_REGISTRY: Dict[str, Dict[str, Optional[str]]] = {
    "deepseek": {
        "env_file": "deepseek.env", "key_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com", "base_url_var": None,
        "model": "deepseek-chat",
    },
    "qwen": {
        # Generic aliases like "qwen-plus" 403 insufficient_quota on this
        # account — only versioned model codes work. qwen-turbo is the
        # cheap fallback; qwen3-max also confirmed working.
        "env_file": "qwen.env", "key_var": "QWEN_API_KEY",
        "base_url": None, "base_url_var": "QWEN_BASE_URL",
        "model": "qwen-plus-2025-07-28",
    },
    "kimi": {
        # Key is for Moonshot's INTERNATIONAL platform — api.moonshot.ai,
        # not .cn (separate platforms, keys don't cross; the .cn 401 with
        # a valid key cost a debugging round). kimi-k3 is the current
        # flagship per GET /v1/models on this account.
        # kimi-k3 (flagship) was persistently 429 engine-overloaded at
        # pilot time; kimi-k2.5 answered in 1.8s. Prefer the model that
        # exists over the model that's ideal.
        "env_file": "kimi.env", "key_var": "KIMI_API_KEY",
        "base_url": "https://api.moonshot.ai/v1", "base_url_var": None,
        "model": "kimi-k2.5",
    },
    "grok": {
        # xAI, OpenAI-compatible. grok-4.5 confirmed on this account via
        # GET /v1/models (also present: 4.3, 4.20 variants). Grok's
        # humor-as-brand positioning makes it the single most interesting
        # cascade subject — it topped LOL Arena.
        "env_file": "xai.env", "key_var": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1", "base_url_var": None,
        "model": "grok-4.5",
    },
    "glm": {
        # glm-4-flash 400s "model does not exist" on this account — the
        # catalog has moved on (checked GET /v4/models: glm-4.5, -4.5-air,
        # -4.6, -4.7, -5, -5-turbo, -5.1, -5.2 are current). glm-4.5-air is
        # the fast/cheap tier and the closest surviving equivalent to the
        # old "flash" naming; glm-5-turbo also works but is ~2.5x slower.
        "env_file": "glm.env", "key_var": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "base_url_var": None, "model": "glm-4.5-air",
    },
    # minimax skipped: API shape diverges from OpenAI-compat enough that
    # it isn't a trivial registry entry (see benchmark/smoke_providers.py
    # run notes for the observed failure mode).
}


def make_openai_compat(provider_name: str, timeout_s: int = 120,
                       retries: int = 2,
                       temperature: Optional[float] = 1.0
                       ) -> Callable[[str], str]:
    """Return a complete(prompt) callable backed by a direct
    OpenAI-compatible /chat/completions endpoint (DeepSeek, Qwen, Kimi,
    GLM) — open-weight cascade subjects via their own API, no local GPU.

    The key is read from ~/agents/.fleet-secrets/<provider>.env at call
    time and held only in this closure; it is never returned, printed, or
    written into the repo (the repo is public). stdlib-only (urllib), no
    pip installs.

    `temperature` is the EXP-007 hook (docs/BENCHMARK.md §1): sampling
    diversity is bought with temperature while cascade path divergence
    is not, so the temperature actually sent has to be a first-class,
    recorded number, never an implicit default baked silently into the
    request.
      - default (1.0) — unchanged from this function's original hardcoded
        behavior, so any caller that doesn't ask for temperature control
        (i.e. every pre-existing spec, resolved through get_provider()
        without the new kwarg) builds a byte-identical request body to
        before this parameter existed.
      - a float (including 0.0) — sent verbatim as "temperature" in the
        request body. Checked with `is not None`, NOT truthiness: 0.0 is
        a legitimate, fully-greedy decoding setting and must not be
        confused with "no override."
      - None — omits the "temperature" key from the body entirely, so
        the provider's own server-side default applies (which may not be
        1.0). This is a deliberate, different thing from "0.0".
    """
    spec = _OPENAI_COMPAT_REGISTRY.get(provider_name)
    if spec is None:
        raise ValueError(
            "unknown api provider %r (known: %s)"
            % (provider_name, ", ".join(sorted(_OPENAI_COMPAT_REGISTRY))))

    env_path = os.path.join(_FLEET_SECRETS_DIR, spec["env_file"])
    try:
        env = parse_env_file(env_path)
    except FileNotFoundError:
        raise RuntimeError(
            "missing secrets file %s for api:%s — expected a %s=... line."
            % (env_path, provider_name, spec["key_var"])) from None

    api_key = env.get(spec["key_var"])
    if not api_key:
        raise RuntimeError(
            "%s has no %s set for api:%s"
            % (env_path, spec["key_var"], provider_name))

    base_url = spec["base_url"]
    if base_url is None:
        base_url = env.get(spec["base_url_var"])
    if not base_url:
        raise RuntimeError(
            "%s has no %s set for api:%s"
            % (env_path, spec["base_url_var"], provider_name))

    model = spec["model"]
    url = base_url.rstrip("/") + "/chat/completions"

    def complete(prompt: str) -> str:
        # Field order matches the pre-temperature-kwarg body exactly when
        # temperature takes its default (1.0), so the default path is
        # byte-identical to what this function sent before.
        payload = {"model": model,
                  "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            payload["temperature"] = temperature
        payload["max_tokens"] = 400
        body = json.dumps(payload).encode("utf-8")
        last_err: Exception = RuntimeError("unreachable")
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json",
                        "Authorization": "Bearer %s" % api_key})
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"]
                if text and text.strip():
                    return text.strip()
                last_err = RuntimeError(
                    "api:%s empty response" % provider_name)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                # Audit WARN-1: DeepSeek's 401 body echoes the tail of the
                # submitted key ("Your api key: ****cdef is invalid").
                # Scrub any key fragment before it can reach failures[] /
                # summary.json / stdout.
                for frag in (api_key[-8:], api_key[-6:], api_key[-4:]):
                    if frag:
                        detail = detail.replace(frag, "[REDACTED]")
                last_err = RuntimeError(
                    "api:%s HTTP %d: %s" % (provider_name, e.code, detail))
                if e.code != 429 and e.code < 500:
                    # Permanent client error (bad key, bad model, bad
                    # request) — retrying sends the identical request and
                    # gets the identical rejection. Only 429 (rate limit)
                    # and 5xx (transient) are worth a retry.
                    raise last_err from None
            except (urllib.error.URLError, OSError) as e:
                last_err = e
            time.sleep(3 * 2 ** attempt)  # 3s, 6s, 12s — rate-limit friendly
        raise last_err

    complete.__name__ = "api_%s" % provider_name  # shows up in run logs
    return complete


# Prefix -> factory. No prefix in a spec means "claude", so pre-existing
# --models values like "haiku,sonnet" resolve exactly as before.
PROVIDER_FACTORIES: Dict[str, Callable[..., Callable[[str], str]]] = {
    "claude": make_claude_cli,
    "codex": make_codex_cli,
    "api": make_openai_compat,
}


def get_provider(spec: str, timeout_s: int = 120, retries: int = 2,
                 temperature: Optional[float] = None) -> Callable[[str], str]:
    """Parse a "prefix:alias" provider spec (e.g. "codex:mini",
    "api:deepseek", or bare "haiku") into a complete(prompt) callable.

    `temperature`, if given, is forwarded to the underlying factory as a
    keyword — but ONLY for "api:" (OpenAI-compatible HTTP) specs, which
    are the only providers with real temperature control. Requesting it
    for "claude:"/"codex:" (or bare, which defaults to "claude") specs
    raises immediately: those are single-shot CLI wrappers with no
    sampling-parameter control, a documented confound (providers.py
    module docstring; docs/BENCHMARK.md), and silently ignoring the
    request would let a caller believe a manipulation happened when it
    didn't. Leaving `temperature` at its default (None) does not touch
    the factory call at all — every pre-existing call site (that never
    passed this kwarg) resolves through the exact same code path as
    before it existed, so old specs stay byte-identical."""
    if ":" in spec:
        prefix, alias = spec.split(":", 1)
    else:
        prefix, alias = "claude", spec
    try:
        factory = PROVIDER_FACTORIES[prefix]
    except KeyError:
        raise ValueError(
            "unknown provider prefix %r in spec %r (known: %s)"
            % (prefix, spec, ", ".join(sorted(PROVIDER_FACTORIES)))) from None
    if temperature is not None:
        if prefix != "api":
            raise ValueError(
                "temperature override requested for spec %r (prefix %r), "
                "but only 'api:' (OpenAI-compatible HTTP) providers support "
                "temperature control — CLI providers ('claude:'/'codex:', "
                "and bare aliases, which default to 'claude:') are a "
                "documented confound and must be left at their single "
                "fixed decoding setting." % (spec, prefix))
        return factory(alias, timeout_s=timeout_s, retries=retries,
                       temperature=temperature)
    return factory(alias, timeout_s=timeout_s, retries=retries)


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

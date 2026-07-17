# Compiled Rules

_Auto-generated from `~/.claude/memory/memory.db`. Do not edit by hand._
_Last compiled: 2026-07-17T12:18:49Z_

## Repo-local: good-humored

### experiment-validity

- **A nested `claude -p` call inherits the project's CLAUDE.md and hooks from its cwd — any LLM-as-instrument (rejector, judge) invoked via CLI must run from a neutral empty directory.** _(×2)_
  - *Mistake:* The rejector smoke test replied "OK. I've read the context" — it had loaded this repo's research docs, including the benchmark spec it was supposed to be a blind instrument for.
  - *Correction:* `providers.make_claude_cli` now creates a `tempfile.mkdtemp` neutral cwd per provider; verified the same call from neutral cwd returns a clean "OK".

### provider-design

- **A `temperature=None`-vs-`0.0`-vs-unset distinction must be checked with `is not None`, never truthiness, when a numeric sampling parameter can legitimately be zero.**
  - *Mistake:* `if temperature:` would silently drop an explicit greedy-decoding request (0.0) and misreport it as provider default.
  - *Correction:* gate optional numeric request fields on `is not None`; unit-test that 0.0 is not treated as omit.
- **Reasoning models (kimi-k2.5, glm-4.5-air) burn small max_tokens budgets entirely on reasoning_content and return empty content with finish_reason=length.**
  - *Mistake:* a 400-token cap that works for standard chat models produced 100% empty-response failure on kimi and 50% on glm at cascade prompt lengths — looking like an API outage when it's actually a token-budget starvation.
  - *Correction:* probe `message` keys and `finish_reason` on any new OpenAI-compatible provider before batch runs; reasoning models need max_tokens ≥ ~2k or thinking disabled; make max_tokens a per-provider registry field.


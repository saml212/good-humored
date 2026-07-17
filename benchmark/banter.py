"""Track 2 — contextual banter (docs/BENCHMARK.md §1b).

Track 1 (cascade.py) is the sterile diagnostic: a joke slot, rejected by
topic. Track 2 measures *conversational* humor: an episode is a whole
conversation, and the reward signal is whether a reply lands specifically
in the conversation it was said in — a canned joke should score the same
whether the surrounding conversation is real or swapped for a stranger's;
genuine banter should not.

Three pieces:
  - run_banter_episode  — drives the conversation (partner + model under
    test), logs every exchange.
  - context_ablation_score — the quantified version of "humor is
    contextualized": score(reply | true context) - score(reply | swapped
    context). Judge-hacking mitigation: the SAME judge scores both
    contexts with an IDENTICAL rubric, so a judge's absolute-scale bias
    (e.g. "always says 7") cancels in the subtraction. It does NOT cancel
    a judge whose bias is itself context-dependent (residual risk — see
    the doc note appended to docs/BENCHMARK.md §1b).
  - detect_callback — a deliberately crude bag-of-words heuristic for the
    other half of banter (reusing an earlier entity for a later payoff).
    It is not an entity linker or a novelty check; see its docstring for
    exactly what it will get wrong.

score_episode composes the two into a per-episode summary.

Design mirrors cascade.py/rejector.py: pure orchestration over injected
`complete(prompt) -> str` callables (benchmark/providers.py's contract),
so the whole module runs against fake providers with no network access.
"""

import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ------------------------------------------------------------- scenarios

# Small, fixed, deterministic-order list of everyday-situation openers.
# The TOPICS are built-in and scripted on purpose (docs/BENCHMARK.md
# §1b) — comparability across episodes/models requires that the sequence
# of subjects a conversation visits is controlled, the same way the
# cascade controls turn structure via the rejector. `partner_complete`
# only supplies natural phrasing and reacts to what's been said so far;
# it does not choose what happens next.
SCENARIO_OPENERS: List[str] = [
    "the barista got your coffee order wrong again this morning",
    "your neighbor's dog got loose in the yard while you were gardening",
    "traffic on the highway was backed up for an hour because of construction",
    "you and your sibling are arguing over what to cook for the holiday",
    "you forgot your umbrella and got caught in the rain on the walk home",
    "your landlord finally fixed the leaky faucet after three months of asking",
    "the gym replaced all the treadmills with ones nobody can figure out",
    "a coworker microwaved fish in the break room again",
]

_ROLE_NAME = {"partner": "Friend", "model": "You"}


def _format_turns(messages: List[Dict[str, str]]) -> str:
    return "\n".join(
        "%s: %s" % (_ROLE_NAME[m["role"]], m["content"]) for m in messages)


def _partner_prompt(scenario: str, messages: List[Dict[str, str]]) -> str:
    lines = [
        "You are chatting casually with a friend. In your own words, make "
        "ONE ordinary conversational remark about the following everyday "
        "situation, as if it just happened to you. Reply with ONLY your "
        "remark — no preamble.",
        "",
        "Situation: %s" % scenario,
    ]
    if messages:
        lines += ["", "Conversation so far:", _format_turns(messages)]
    return "\n".join(lines)


def _model_prompt(messages: List[Dict[str, str]]) -> str:
    lines = [
        "This is an ongoing casual conversation. You are 'You'. Reply to "
        "your friend's last remark wittily and IN CONTEXT — react "
        "specifically to what was just said, not with a generic or "
        "canned joke. Reply with ONLY your response.",
        "",
        _format_turns(messages),
        "You:",
    ]
    return "\n".join(lines)


def run_banter_episode(
    model_complete: Callable[[str], str],
    partner_complete: Callable[[str], str],
    n_turns: int,
    seed_topic: int,
    log_path: Optional[Path] = None,
) -> Dict:
    """Run one banter episode: `n_turns` exchanges of partner-statement +
    model-reply.

    The partner draws its everyday-situation openers from
    SCENARIO_OPENERS in a fixed rotation starting at `seed_topic`
    (`(seed_topic + turn) % len(SCENARIO_OPENERS)`), so two episodes with
    different `seed_topic` values walk different, fully deterministic
    topic sequences — this is what makes the swap pairing in
    `swap_partner` well-defined (episode A's context at turn i is never
    accidentally the same subject as episode B's).

    Returns the episode record; optionally logs JSONL per turn so a
    crashed run is still data (mirrors cascade.py's run_cascade)."""
    n_scenarios = len(SCENARIO_OPENERS)
    messages: List[Dict[str, str]] = []
    exchanges: List[Dict] = []

    log_f = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a")

    try:
        for turn in range(n_turns):
            scenario = SCENARIO_OPENERS[(seed_topic + turn) % n_scenarios]
            partner_line = partner_complete(_partner_prompt(scenario, messages))
            messages.append({"role": "partner", "content": partner_line})

            reply = model_complete(_model_prompt(messages))
            messages.append({"role": "model", "content": reply})

            rec = {"turn": turn, "scenario": scenario, "partner": partner_line,
                   "reply": reply, "ts": time.time()}
            exchanges.append(rec)
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
    finally:
        if log_f:
            log_f.close()

    return {"seed_topic": seed_topic, "n_turns": n_turns,
            "exchanges": exchanges, "messages": messages}


def swap_partner(episode_index: int, n_episodes: int) -> int:
    """Deterministic swap-context pairing rule: episode `episode_index`'s
    context-ablation swap source is episode `(episode_index + 1) %
    n_episodes` (the last pairs back to the first). Fixed and
    order-based on purpose — a random swap would make a pilot run
    non-reproducible, and this repo's whole thesis is that measurements
    must be re-derivable from logged data (docs/BENCHMARK.md)."""
    if n_episodes < 2:
        raise ValueError("need >= 2 episodes to form a swap pair")
    return (episode_index + 1) % n_episodes


# ------------------------------------------------------- ablation judge

JUDGE_PROMPT_VERSION = "v1"

JUDGE_PROMPT = """You are scoring one reply from a casual conversation.

Rate how well the reply FITS as witty banter IN THIS SPECIFIC CONVERSATION
— whether it responds specifically to what was just said — NOT whether it
would be funny read on its own as a standalone joke. A generic joke that
could follow any remark should score LOW here even if it is well-crafted.

Conversation context:
{context}

Reply to score:
{reply}

Output ONLY an integer from 1 (does not fit this conversation at all) to
10 (fits and lands perfectly here) on the first line.
"""

_SCORE_RE = re.compile(r"(?<!\d)(10|[1-9])(?!\d)")


def _parse_score(raw: str) -> Optional[int]:
    """First standalone integer 1-10 on the first line. (?<!\\d)/(?!\\d)
    guards keep '100' or '23' from being misread as '10'/'2'."""
    if not raw or not raw.strip():
        return None
    first_line = raw.strip().splitlines()[0]
    m = _SCORE_RE.search(first_line)
    return int(m.group(1)) if m else None


def _judge_once(reply: str, context: str,
                judge_complete: Callable[[str], str]) -> Optional[int]:
    """One context's score, retried once on a parse failure. Sentinel
    None on repeated failure — mirrors rejector.py's UNPARSEABLE pattern
    (a typed sentinel here since the metric is numeric, not a label)."""
    prompt = JUDGE_PROMPT.format(context=context, reply=reply)
    for _ in range(2):
        raw = judge_complete(prompt)
        score = _parse_score(raw)
        if score is not None:
            return score
    return None


def context_ablation_score(
    reply: str,
    true_context: str,
    swapped_context: str,
    judge_complete: Callable[[str], str],
) -> Dict:
    """delta = score(reply | true_context) - score(reply | swapped_context).

    A canned, context-blind joke should score about the same either way
    (delta ~ 0). Genuinely in-context banter should score lower once the
    context it was actually responding to is pulled out from under it.
    Both calls use the identical JUDGE_PROMPT rubric, differing only in
    which context block is shown, so a judge's absolute-scale bias
    cancels in the subtraction — see the module docstring for what does
    NOT cancel.

    Returns {"true_score", "swapped_score", "delta", "judge_prompt_version"}.
    true_score/swapped_score/delta are None if the judge's output could
    not be parsed as a score even after one retry."""
    true_score = _judge_once(reply, true_context, judge_complete)
    swapped_score = _judge_once(reply, swapped_context, judge_complete)
    delta = (true_score - swapped_score
             if true_score is not None and swapped_score is not None
             else None)
    return {"true_score": true_score, "swapped_score": swapped_score,
            "delta": delta, "judge_prompt_version": JUDGE_PROMPT_VERSION}


# ------------------------------------------------------ callback detector

# Function words long enough (>=5 chars) to otherwise slip past the
# length filter and pollute matches with coincidental, non-substantive
# reuse ("about", "which", "after", ...). Not exhaustive by design — see
# detect_callback's docstring for the honest limits of this whole check.
_CALLBACK_STOPWORDS = set(
    "about which there their would could should really something anyone "
    "someone anything nothing everything because before after while "
    "still every always never though these those where doesn didn wasn "
    "isnt arent havent hasn hadn wouldn couldn shouldn going gotten "
    "little pretty actually totally completely probably maybe another "
    "around between during without within".split()
)


def _content_words(text: str, min_len: int = 5) -> set:
    """Lowercased alpha-only tokens >= min_len, minus stopwords. Splitting
    on anything non-alphabetic (so apostrophes are separators, not part
    of a token) means a possessive like "neighbor's" contributes the
    plain word "neighbor" rather than a token that can never match a
    bare "neighbor" elsewhere — a deliberate simplification, not full
    contraction handling (see detect_callback's docstring)."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) >= min_len and w not in _CALLBACK_STOPWORDS}


def detect_callback(reply: str, earlier_turns: List[str],
                    min_gap: int = 3) -> Optional[str]:
    """Cheap callback detector: does `reply` bring back a content word
    from a turn far enough in the past, without that word having simply
    been re-mentioned in between (which would make it topical continuity,
    not a callback)?

    A word in `earlier_turns[i]` qualifies as a candidate origin only if
    it is at least `min_gap` turns before `reply` (i.e. `len(earlier_turns)
    - i >= min_gap`). Candidates are excluded if they also appear in any
    of the more-recent ("intervening") turns inside the gap. The first
    surviving candidate is returned, scanning from the nearest qualifying
    turn backward (so a callback to a closer setup is preferred when
    several match).

    Document its crudeness honestly: this is a bag-of-words check, not an
    entity linker or a novelty check.
      - It cannot tell "the neighbor's dog" apart from an unrelated
        second dog — any shared 5+-letter word registers as a "callback"
        whether or not it is the same referent.
      - No lemmatization beyond nothing: "gardening" and "garden" are
        different tokens to it.
      - It says nothing about whether the payoff is actually funny or
        novel, only that a word came back after a gap.
      - Generic-but-long filler ("morning", "today") is not in the
        stopword list and can produce a false-positive "callback" if it
        happens to repeat — a known, accepted false-positive source
        rather than a solved problem.
    Treat its output as a cheap signal to sample-check by hand, not as
    ground truth about banter quality."""
    n = len(earlier_turns)
    if n < min_gap:
        return None
    boundary = n - min_gap + 1  # old turns: indices [0, boundary)
    old_turns = earlier_turns[:boundary]
    intervening_turns = earlier_turns[boundary:]

    reply_words = _content_words(reply)
    intervening_words: set = set()
    for t in intervening_turns:
        intervening_words |= _content_words(t)

    for t in reversed(old_turns):
        for w in _content_words(t):
            if w in reply_words and w not in intervening_words:
                return w
    return None


# --------------------------------------------------------------- scoring


def _context_up_to(episode: Dict, turn: int) -> List[Dict[str, str]]:
    """Messages up to and including the partner's line for `turn`,
    excluding the model's reply at that turn (the thing being judged)."""
    return episode["messages"][: 2 * turn + 1]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_episode(
    episode: Dict,
    swap_episode: Dict,
    judge_complete: Callable[[str], str],
    min_gap: int = 3,
) -> Dict:
    """Score one banter episode: per-reply context-ablation delta
    (against `swap_episode`'s context at the same turn index — see
    `swap_partner` for how that pairing is chosen across a pilot's
    episode set) plus callback detection over the partner's turns.

    If `swap_episode` is shorter than `episode`, later turns clamp to its
    last available context rather than erroring — a pilot may run
    episodes of uneven length across models/failures (cascade.py's
    per-run failure fencing is the precedent).

    Returns {"per_turn": [...], "summary": {...}}. Summary's
    mean_delta averages only turns where the judge parsed on both sides;
    n_unparseable counts the rest."""
    exchanges = episode["exchanges"]
    swap_exchanges = swap_episode["exchanges"]
    per_turn: List[Dict] = []
    partner_so_far: List[str] = []
    deltas: List[float] = []
    n_callbacks = 0
    n_unparseable = 0

    for i, exch in enumerate(exchanges):
        reply = exch["reply"]
        true_ctx = _format_turns(_context_up_to(episode, i))
        swap_idx = min(i, len(swap_exchanges) - 1) if swap_exchanges else 0
        swapped_ctx = _format_turns(_context_up_to(swap_episode, swap_idx))

        ablation = context_ablation_score(reply, true_ctx, swapped_ctx,
                                          judge_complete)
        if ablation["delta"] is None:
            n_unparseable += 1
        else:
            deltas.append(ablation["delta"])

        callback = detect_callback(reply, partner_so_far, min_gap=min_gap)
        if callback is not None:
            n_callbacks += 1

        per_turn.append({"turn": i, "reply": reply, "callback": callback,
                         **ablation})
        partner_so_far.append(exch["partner"])

    summary = {
        "n_turns": len(exchanges),
        "n_callbacks": n_callbacks,
        "n_unparseable": n_unparseable,
        "mean_delta": _mean(deltas),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }
    return {"per_turn": per_turn, "summary": summary}

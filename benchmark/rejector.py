"""The rejector: the cascade's measurement instrument.

Two jobs, deliberately separated:
  1. LABEL the topic of a joke (1-3 words). The trajectory is built from
     these labels, so labeling quality IS the benchmark's validity —
     validate_rejector.py must pass before any cascade number is real.
  2. REJECT by topic, never by joke. If the rejector critiques the joke,
     the model under test rewords the same joke and the cascade measures
     nothing (docs/BENCHMARK.md, "load-bearing risk").
"""

from typing import Callable, List

from .metrics import normalize_label

# The labeling prompt is versioned: any change invalidates prior runs.
# Bump LABEL_PROMPT_VERSION and re-run validation when editing.
LABEL_PROMPT_VERSION = "v1"

LABEL_PROMPT = """You label joke topics for a research benchmark.

Output ONLY the topic of the joke below: the subject matter the joke is
about, in 1-3 lowercase words. Not the punchline mechanism, not a
category of humor, not an opinion. If the joke is about a cat sitting on
a laptop, the topic is "cats", not "puns" or "technology".

Examples:
Joke: "I told my suitcase there'd be no vacation this year. Now I'm dealing with emotional baggage."
Topic: travel

Joke: "Parallel lines have so much in common. Too bad they'll never meet."
Topic: math

Joke: {joke}
Topic:"""


def label_topic(joke: str, complete: Callable[[str], str]) -> str:
    """Label one joke's topic. Returns the normalized label."""
    raw = complete(LABEL_PROMPT.format(joke=joke))
    # Take the first line only — cheap models sometimes elaborate.
    return normalize_label(raw.splitlines()[0])


def rejection_message(topic: str, rejected_topics: List[str]) -> str:
    """The accumulating rejection. Names every dead topic so the model
    cannot loop back (a sliding window would hide exactly the collapse
    we measure)."""
    if rejected_topics:
        seen = ", ".join(rejected_topics)
        return (
            "I don't find jokes about %s funny. And I already told you "
            "I don't like jokes about %s — none of those topics either. "
            "Tell me a joke about something completely different." % (topic, seen)
        )
    return (
        "I don't find jokes about %s funny. Tell me a joke about "
        "something completely different." % topic
    )


OPENING_PROMPT = "Tell me a short joke."

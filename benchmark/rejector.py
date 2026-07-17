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
#
# v1 -> v2 (EXP-001 findings): joke is now delimited (a joke containing a
# colon broke v1's "Topic:" format once); output is pushed toward ONE
# most-generic domain noun (v1 scattered fitness/exercise/gym across
# repeats — correct labels, split partitions); two few-shots added that
# demonstrate generalizing up from the joke's surface object.
LABEL_PROMPT_VERSION = "v2"

LABEL_PROMPT = """You label joke topics for a research benchmark.

The joke appears between <joke> tags below. Output ONLY its topic: the
subject matter the joke is about. Rules:
- ONE lowercase word whenever possible (two only if truly unavoidable).
- Use the most GENERIC everyday word for the domain: a treadmill joke is
  "exercise" (not "treadmill" or "gym"); an airline joke is "travel"
  (not "luggage" or "flying").
- The subject matter, never the punchline mechanism, never a humor
  category, never an opinion.
- Same joke in different words = same topic word.

Examples:
<joke>I told my suitcase there'd be no vacation this year. Now I'm dealing with emotional baggage.</joke>
travel

<joke>Parallel lines have so much in common. Too bad they'll never meet.</joke>
math

<joke>My blender refuses to work unless I push its buttons. We're in couples counseling now.</joke>
appliances

<joke>{joke}</joke>
"""


UNPARSEABLE = "unparseable"


def label_topic(joke: str, complete: Callable[[str], str]) -> str:
    """Label one joke's topic. Returns the normalized label.

    Shape guard (audit WARN-3): on unusual inputs the labeler sometimes
    emits a sentence of meta-commentary instead of a topic (~1/2 on an
    adversarial probe). A >4-word "topic" is retried once, then becomes
    the UNPARSEABLE sentinel — flaggable downstream — rather than
    silently entering the metric pipeline as a Jaccard-singleton.
    """
    for _ in range(2):
        raw = complete(LABEL_PROMPT.format(joke=joke))
        # First line only — cheap models sometimes elaborate.
        label = normalize_label(raw.splitlines()[0])
        if label and len(label.split()) <= 4:
            return label
    return UNPARSEABLE


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

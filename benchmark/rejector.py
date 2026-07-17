"""The rejector: the cascade's measurement instrument.

Two jobs, deliberately separated:
  1. LABEL the topic of a joke (1-3 words). The trajectory is built from
     these labels, so labeling quality IS the benchmark's validity —
     validate_rejector.py must pass before any cascade number is real.
  2. REJECT by topic, never by joke. If the rejector critiques the joke,
     the model under test rewords the same joke and the cascade measures
     nothing (docs/BENCHMARK.md, "load-bearing risk").
"""

from pathlib import Path
from typing import Callable, List, Optional

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


# ======================================================================
# v3 — constrained vocabulary (EXP-008; additive, does not touch the v2
# constants/behavior above — a live experiment (EXP-004's cascade pilot)
# still runs on v2).
#
# EXP-002's verdict named two specific, structural free-vocabulary
# failures: granularity jitter (same joke labeled "cat" once, "pet"
# another time — semantically fine, but it splits the ARI partition and
# fails reworded invariance on a STRING-equality metric) and generalize-up
# overshoot (a flamingo-impression marriage joke consistently labeled
# "animal" — invariant, but clustered against the wrong gold group). Both
# are symptoms of the labeler being free to invent whatever word it wants.
# v3's fix: give it a closed list instead. It cannot invent "pet" as an
# alternative to "cat" if "pet" is not one of the choices.
LABEL_PROMPT_VERSION_V3 = "v3-constrained"

VOCABULARY_PATH = Path(__file__).parent / "fixtures" / "topic_vocabulary.txt"


def load_vocabulary(path: Path = VOCABULARY_PATH) -> List[str]:
    """Load the constrained topic vocabulary: one entry per non-blank,
    non-comment line (`#`-prefixed lines are provenance notes, see the
    file header), in file order. Every entry is normalize_label-idempotent
    by construction (benchmark/tests/test_rejector.py checks this against
    the actual file, not just this docstring's claim)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    return entries


_VOCABULARY = load_vocabulary()
# Compact, comma-separated — deliberately not one-per-line or numbered:
# a comma-joined list of ~110 short entries costs a few hundred tokens.
# Measured (chars/4 heuristic, one fixture joke, no real tokenizer run):
# LABEL_PROMPT_V3 all-in ~500 tokens vs. LABEL_PROMPT (v2) ~240 tokens —
# roughly +260 tokens/call. At Haiku pricing that's fractions of a cent
# even across the 96-call EXP-008 validation run; call this an order-of-
# magnitude estimate, not a billed number. Trivial against Haiku's context
# window either way. The real tradeoff is not context length; it
# is that every additional entry is one more way for a straddling item
# (see LABEL_PROMPT_V3's "names what the joke is actually ABOUT, not a
# word it merely mentions along the way" rule below) to be forced onto
# the wrong side of a real ambiguity. Bigger vocabulary != safer instrument;
# see this file's provenance comment for why entries were deliberately
# left OUT (pet, health/medicine, programmer, gym/fitness) rather than
# added alongside their near-synonym.
_VOCABULARY_TEXT = ", ".join(_VOCABULARY)

LABEL_PROMPT_V3 = """You label joke topics for a research benchmark, choosing
from a FIXED list. Do not invent a word that is not on the list, and do not
alter an entry's wording.

Topic list (pick exactly one, copied verbatim):
%s

The joke appears between <joke> tags below. Rules:
- Answer with ONLY one entry from the list above, exactly as written
  there, and nothing else.
- Pick the entry that names what the joke is actually ABOUT, not a word
  it merely mentions along the way.
- Same joke in different words = same list entry.
- If truly nothing on the list fits, answer "other".

Examples:
<joke>My grandmother knits a sweater shaped like each grandchild's least favorite vegetable. Family dinner is now a guessing contest.</joke>
family

<joke>The building's resident ghost filed a noise complaint against the OTHER resident ghost.</joke>
other

<joke>I asked the barista for a coffee that matches my personality. She handed me an empty cup.</joke>
coffee

<joke>{joke}</joke>
""" % _VOCABULARY_TEXT


def label_topic_v3(joke: str, complete: Callable[[str], str],
                   vocabulary: Optional[List[str]] = None) -> str:
    """Constrained-vocabulary labeling (LABEL_PROMPT_VERSION_V3).

    Same two-strikes-then-UNPARSEABLE shape as label_topic (v2), but a
    stricter validation layer: v2 accepts anything short enough to look
    like a topic; v3 additionally requires the reply to normalize to an
    entry actually ON the list. A reply that isn't in the vocabulary
    (model ignored the instruction, or answered with a near-miss like a
    plural/synonym) is retried once with the identical prompt; if the
    retry also misses, the sentinel is returned rather than letting an
    out-of-vocabulary string silently enter the metric pipeline (same
    reasoning as v2's shape guard, audit WARN-3, just with a tighter
    membership test instead of a word-count heuristic).

    `vocabulary` defaults to the module-level list loaded from
    topic_vocabulary.txt at import time; a caller may pass an explicit
    list (e.g. in tests) so the validation layer can be exercised without
    touching the fixtures file.
    """
    vocab = vocabulary if vocabulary is not None else _VOCABULARY
    normalized_vocab = {normalize_label(v) for v in vocab}
    for _ in range(2):
        raw = complete(LABEL_PROMPT_V3.format(joke=joke))
        # First line only — cheap models sometimes elaborate, same as v2.
        label = normalize_label(raw.splitlines()[0])
        if label in normalized_vocab:
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

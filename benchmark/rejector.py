"""The rejector: the cascade's measurement instrument.

Two jobs, deliberately separated:
  1. LABEL the topic of a joke (1-3 words). The trajectory is built from
     these labels, so labeling quality IS the benchmark's validity —
     validate_rejector.py must pass before any cascade number is real.
  2. REJECT by topic, never by joke. If the rejector critiques the joke,
     the model under test rewords the same joke and the cascade measures
     nothing (docs/BENCHMARK.md, "load-bearing risk").
"""

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


# ======================================================================
# v4 — TWO-TIER labeling (EXP-008 addendum "v3 FAILS in the field" +
# adversarial-audit NO-GO on the first v4 draft, EXPERIMENT_LOG.md;
# additive, does not touch the v2 or v3 constants/behavior above — v3
# stays hash-locked and untouched).
#
# EXP-008 validated v3's constrained vocabulary at paper grade ON THE
# FIXTURE (invariance 1.000, ari_vs_gold 0.9237) but the fixture only
# ever contained in-vocabulary topics, so field coverage was untested by
# construction. The full-pilot v3 relabel (1,532 wild turns) found v3
# maps 42.6% of turns to the catch-all `other`.
#
# The first v4 draft just grew the closed vocabulary (110 -> 128
# entries) and kept `other` as the escape hatch. Adversarial audit
# finding B1: that bar is arithmetically unreachable (the wild long
# tail below 4 occurrences is 13.6% of turns alone) AND, more
# importantly, `other` is the wrong fix regardless of vocabulary size —
# a catch-all MERGES distinct rare topics into one label, manufacturing
# false repeats, which is the one bias this benchmark cannot afford
# (CLAUDE.md; EXP-003's `pet` finding, at vocabulary scale instead of
# pair scale). So v4 is now TWO-TIER instead of closed:
#   - CANON tier: the joke matches a topic_vocabulary_v4.txt entry.
#   - FREE tier: nothing on the list fits, so the model instead answers
#     with its own specific common noun for the topic (v2-style — a
#     real word, never "other"/"misc"/a category placeholder). A free
#     label is a genuine, specific answer, not a parse failure.
# `other` is REMOVED from topic_vocabulary_v4.txt entirely — structurally
# impossible, not just discouraged, because the retry logic below no
# longer has any reason to reject an out-of-vocabulary answer (see
# label_topic_v4's docstring for the mechanics). Downstream, a free-tier
# label is treated as an ordinary distinct topic (string equality, same
# as v2's untiered labels) — free-tier wording jitter on a rare topic
# SPLITS a repeat rather than merging two different rare topics into
# one, which is the conservative direction for any collapse claim
# (documented, not just asserted — this is the same argument EXP-006's
# noise-bias-direction simulation made for v2's free-vocabulary jitter
# in general).
LABEL_PROMPT_VERSION_V4 = "v4-two-tier"

VOCABULARY_PATH_V4 = Path(__file__).parent / "fixtures" / "topic_vocabulary_v4.txt"

_VOCABULARY_V4 = load_vocabulary(VOCABULARY_PATH_V4)
_VOCABULARY_V4_TEXT = ", ".join(_VOCABULARY_V4)

TIER_CANON = "canon"
TIER_FREE = "free"
TIER_UNPARSEABLE = "unparseable"

# ---------------------------------------------------------------- aliases
#
# Re-audit MAJOR finding: topic_vocabulary_v4.txt's own comments and
# test_rejector_v4.py's DECISIONS table document folds ("humor" ->
# `comedy`, "bike" -> `bicycle`, ...) but label_topic_v4 originally did
# exact vocabulary-membership matching only -- a model answering
# "humor" (a real, documented synonym) escaped to the free tier as its
# own distinct label instead of resolving to the canonical `comedy`.
# 105/1532 wild turns were on the line for exactly this reason (mostly
# the comedy cluster). ALIASES_PATH_V4 is a SEPARATE file with its own
# parser (load_aliases below) -- deliberately NOT a change to
# load_vocabulary()'s format, so v3's shared loader (and
# topic_vocabulary.txt, hash-locked) is byte-untouched.
ALIASES_PATH_V4 = (Path(__file__).parent / "fixtures"
                   / "topic_vocabulary_v4_aliases.txt")


def load_aliases(path: Path = ALIASES_PATH_V4) -> Dict[str, str]:
    """Load the v4 alias table: `canonical: alias1, alias2` lines (blank
    and `#`-prefixed lines are comments, same convention load_vocabulary
    uses for topic_vocabulary_v4.txt -- but this is its own file and its
    own parser, not a load_vocabulary format change). Returns
    {normalized_alias: normalized_canonical}, one entry per alias (the
    canonical maps to itself implicitly via the ordinary vocabulary
    membership check, so it is never a key here)."""
    aliases: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            canonical, _, alias_list = line.partition(":")
            canonical = normalize_label(canonical)
            for alias in alias_list.split(","):
                alias = normalize_label(alias)
                if alias:
                    aliases[alias] = canonical
    return aliases


_ALIASES_V4 = load_aliases()

# ------------------------------------------------------ unicode normalization
#
# Re-audit MAJOR finding: normalize_label (benchmark/metrics.py, shared
# with v2/v3) strips ASCII punctuation only, so a reply like "cat! 🐱"
# normalizes to "cat 🐱" -- the emoji survives as its own whitespace-
# separated token, so the label never matches the vocabulary's plain
# "cat" entry and permanently escapes to the free tier as "cat 🐱". v3
# never hit this in practice because its retry-on-out-of-vocab loop
# (the very thing v4 correctly removed for B1) happened to give a
# second chance; v4 has no such safety net, so this needs its own fix.
#
# Deliberately NOT folded into the shared normalize_label: v2/v3 are
# hash-locked and must never see a behavior change, even one that looks
# harmless. This is a v4-only layer, applied on top of (never instead
# of) normalize_label.
_V4_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_V4_WHITESPACE_RE = re.compile(r"\s+")


def v4_normalize_label(raw_line: str) -> str:
    """v4's normalization: normalize_label's ASCII-punctuation strip +
    lowercase + naive singularization, THEN strip anything that is not
    a unicode word character or whitespace (emoji, symbols, stray
    combining marks) -- unicode-aware via Python's `\\w`, which matches
    letters/digits/underscore in any script but NOT emoji, so accented
    words (e.g. "café") round-trip untouched while "cat! 🐱" -> "cat"
    and "🥚 egg" -> "egg". Internal spaces are preserved (collapsed to
    one) so multi-word entries ("farm animal") still match; a reply
    that is ONLY symbols (e.g. just an emoji) normalizes to the empty
    string, which the caller's existing shape guard already treats as
    a retry-then-UNPARSEABLE case."""
    label = normalize_label(raw_line)
    label = _V4_NON_WORD_RE.sub("", label)
    label = _V4_WHITESPACE_RE.sub(" ", label).strip()
    return label

LABEL_PROMPT_V4 = """You label joke topics for a research benchmark.

A list of common topics is given below. Rules:
- If one of the listed topics is what the joke is actually ABOUT (not a
  word it merely mentions along the way), answer with that entry,
  copied verbatim, and nothing else.
- If NONE of the listed topics fit, do NOT force one and do NOT answer
  with a category placeholder like "other," "misc," "general," or
  "miscellaneous" -- instead answer with your own specific, concrete,
  lowercase common noun for the joke's actual subject: ONE word
  whenever possible, two only if truly unavoidable, the most generic
  everyday word for the domain, never the punchline mechanism and never
  a mood or opinion (the same style rule LABEL_PROMPT v2 uses for
  free-vocabulary labeling).
- Same joke in different words = same answer, whether that answer comes
  from the list or not.
- A few list entries are close but not interchangeable -- get these
  right:
    * `egg` is for jokes specifically about eggs; `food` is for other
      food/eating/cooking jokes not about egg or another specific
      listed food (pizza, coffee, ...).
    * `skeleton` is for Halloween-style bone wordplay ("they don't have
      the guts") -- this is NEVER `death`, even though skeletons are
      technically about the dead: the joke's actual subject is bone
      anatomy wordplay, not mortality. `death` is reserved for jokes
      actually about dying, funerals, or grief -- never for a skeleton
      pun just because a skeleton is a dead thing.
    * `farming` is for crops, scarecrows, fields, or agriculture in
      general, INCLUDING "outstanding in his/her/its field"-style
      scarecrow jokes, even though the punchline uses a work-adjacent
      word like "award." Those scarecrow jokes are NOT `work` (jobs,
      offices, careers, promotions -- no farm setting), NOT
      `farm animal` (livestock -- a scarecrow is not an animal), and
      NOT `farmer` (a joke centered on an actual farmer character,
      which a scarecrow joke is not).
    * `hair` is for a person's own hair or facial hair; `hairdresser` is
      for a joke centered on the hairdressing PROFESSION, not anyone's
      actual hair.
    * `comedy` is for jokes about joke-telling, comedians as a generic
      figure, or performing in general -- this is NOT `artist`, which
      is for visual/creative-arts professions (a painter, a sculptor),
      never stand-up comedy.
    * `censorship` is for jokes or in-character remarks specifically
      about a topic being banned, forbidden, or rejected -- narrower
      than `comedy`'s broader joke-about-jokes territory.
    * chores like laundry, cleaning, sweeping, and non-electric cleaning
      supplies (soap, brooms, paper towels) are `chore` -- not their own
      answer and not `appliance`. Electric household appliances
      (refrigerators, vacuums, lamps) ARE `appliance`.
    * ordinary job/workplace/promotion wordplay is `work`, even without
      an office explicitly appearing in the joke.
    * trip/vehicle/luggage/journey jokes are `travel`, even when the
      specific mode of transport (train, plane, suitcase) is the only
      thing mentioned.
    * genuine wordplay about grammar, vocabulary, or language itself is
      `language`; purely self-referential paradox jokes about jokes,
      nothingness, or concepts-referring-to-themselves are `comedy`,
      not `language`.

Topics:
%s

The joke appears between <joke> tags below.

Examples:
<joke>My grandmother knits a sweater shaped like each grandchild's least favorite vegetable. Family dinner is now a guessing contest.</joke>
family

<joke>I asked the barista for a coffee that matches my personality. She handed me an empty cup.</joke>
coffee

<joke>Why don't skeletons fight each other? They don't have the guts.</joke>
skeleton

<joke>My grandmother's tombstone reads: "I told you I was sick."</joke>
death

<joke>Why did the scarecrow win an award? Because he was outstanding in his field.</joke>
farming

<joke>Why did the punchline go to therapy? It was tired of being the butt of every joke.</joke>
comedy

<joke>You've banned cats, dogs, and weather. Fine -- here's a joke about how there's nothing left to joke about.</joke>
censorship

<joke>Why don't eggs tell jokes? They'd crack each other up.</joke>
egg

<joke>I'm on an all-food diet. I see food, and I eat it.</joke>
food

<joke>The building's resident ghost filed a noise complaint against the OTHER resident ghost.</joke>
ghost

<joke>{joke}</joke>
""" % _VOCABULARY_V4_TEXT


def label_topic_v4(joke: str, complete: Callable[[str], str],
                   vocabulary: Optional[List[str]] = None,
                   aliases: Optional[Dict[str, str]] = None
                   ) -> Tuple[str, str]:
    """Two-tier labeling (LABEL_PROMPT_VERSION_V4). Returns (label, tier).

    tier is one of:
      TIER_CANON        — the label matched a topic_vocabulary_v4.txt
                           entry (after normalize_label).
      TIER_FREE          — the label did not match the vocabulary, but
                           was otherwise shape-valid: KEPT as the
                           model's own specific word, never coerced into
                           a catch-all. This is the audit's B1 fix — v3
                           retried (and eventually UNPARSEABLE'd) an
                           out-of-vocabulary reply; v4 does not, because
                           there is no vocabulary-membership failure
                           mode left to retry on. Downstream, a
                           TIER_FREE label is an ordinary distinct topic
                           string, scored exactly like a v2 label.
      TIER_UNPARSEABLE   — the reply itself was malformed (empty, or
                           more than 4 words after normalization — the
                           same shape guard label_topic (v2) uses,
                           audit WARN-3) even after one retry. This is
                           the ONLY remaining retry trigger; unlike v3,
                           being out-of-vocabulary never triggers a
                           retry by itself.
    When tier is TIER_UNPARSEABLE, label == UNPARSEABLE (the same
    sentinel v2/v3 use) — exactly as before this two-tier change, just
    now paired with an explicit tier instead of being the only signal.

    Two re-audit fixes applied in order, both BEFORE the vocabulary
    membership check: (1) v4_normalize_label strips leftover unicode
    symbols (emoji) that the shared normalize_label leaves behind, so
    "cat! 🐱" still resolves to canon `cat` instead of permanently
    escaping to the free tier as "cat 🐱"; (2) an alias-table lookup
    (`aliases`, default the module-level table loaded from
    topic_vocabulary_v4_aliases.txt) resolves a documented synonym
    ("humor") to its canonical (`comedy`) BEFORE membership is checked
    — the returned label is always the canonical, never the raw alias,
    and tier is TIER_CANON in that case (an alias hit IS a vocabulary
    hit, one level of indirection earlier).

    `vocabulary` defaults to the module-level list loaded from
    topic_vocabulary_v4.txt at import time; a caller may pass an
    explicit list (e.g. in tests) so the tier logic can be exercised
    without touching the fixtures file. `aliases` defaults similarly to
    the module-level alias table; pass an explicit dict in tests for
    the same reason.
    """
    vocab = vocabulary if vocabulary is not None else _VOCABULARY_V4
    normalized_vocab = {normalize_label(v) for v in vocab}
    alias_table = aliases if aliases is not None else _ALIASES_V4
    for _ in range(2):
        raw = complete(LABEL_PROMPT_V4.format(joke=joke))
        # First line only — cheap models sometimes elaborate, same as
        # v2/v3. Guard against a truly empty reply (splitlines() -> [])
        # rather than indexing into it directly — a latent crash v2/v3
        # never hit in practice but this new code path shouldn't risk.
        lines = raw.splitlines()
        label = v4_normalize_label(lines[0]) if lines else ""
        if label and len(label.split()) <= 4:
            label = alias_table.get(label, label)
            tier = TIER_CANON if label in normalized_vocab else TIER_FREE
            return label, tier
    return UNPARSEABLE, TIER_UNPARSEABLE


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

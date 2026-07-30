"""Callback-as-transformation — EXP-016 (EXPERIMENT_LOG.md).

THE BUG THIS FIXES. `benchmark/banter.py`'s `detect_callback` is a
deliberately crude bag-of-words heuristic: it fires the instant a reply
shares ANY single 5+-char content word with a turn far enough in the past
(gated only by `min_gap`, minus a same-word "refreshed by an intervening
turn" exclusion — see that function's own docstring for its documented,
accepted crudeness). It has NO transformation requirement at all — a
model that repeats its own earlier line VERBATIM shares 100% of that
line's content words with itself and fires exactly like a genuine,
transformed callback would. `env/banter_env.py::BanterEnv.step` then pays
a FLAT bonus (`callback_weight`, default +0.5) the instant `detect_callback`
returns anything other than `None`:

    callback = detect_callback(reply_text, self._partner_history, min_gap=self.min_gap)
    callback_term = self.callback_weight if callback is not None else 0.0

(env/banter_env.py, in `BanterEnv.step` — traced, not modified; see this
module's own docstring further down for the full consumption-path note.)
So today, in the shipped reward path, a literal repeat of an earlier line
scores the SAME full callback bonus as a genuinely transformed callback.
Norrick's reincorporation construct (the craft-consensus term for a
callback, cited throughout this project's banter design) requires return
WITH TRANSFORMATION — bringing an earlier bit back reframed, escalated,
or recontextualized, not restated. `docs/THEORY-MAP.md` §6 ("a joke heard
twice is dead") is the same novelty-decay chain this mirrors on the
callback side: a VERBATIM callback should decay to zero exactly like any
other repeated joke, not earn a bonus for repetition. This is the
self-repetition reskin bug's MIRROR IMAGE, found in reward-path code
before it ever shaped a training run (EXPERIMENT_LOG.md's EXP-016
registration).

THE FIX, exactly the two-stage shape `env/rewards.py`'s own penalty terms
already use (a gate, then a ramped/computed severity — see
`SelfRepetitionPenalty`'s docstring for the sign-flipped mirror of this
same idea): (1) a DETECTION GATE, improved over `detect_callback`'s bare
"any one shared content word" test — see `find_callback_match`'s own
docstring for the exact two-tier design; (2) a TRANSFORMATION SCORE,
computed ONLY once the gate fires, following EXP-016's literal spec:

    score = 1 - trigram_jaccard(current_turn, matched_earlier_turn)

with a hard floor for near-verbatim reuse (similarity >= 0.8 -> score
EXACTLY 0, not a small residual credit) — see `VERBATIM_FLOOR`'s comment
for why this is a cliff, not a gentle ramp, and why that is the right
choice here specifically (mirrors THEORY-MAP §6's "a joke heard twice is
dead," not `SelfRepetitionPenalty`'s smoother above-threshold ramp).

Reuses this repo's own established primitives instead of reimplementing
them (same reuse-over-duplication discipline `env/banter_env.py`'s own
docstring states explicitly for its imports from `benchmark.banter`):
  - `benchmark.banter._content_words` / `_CALLBACK_STOPWORDS` — the exact
    tokenizer+stopword-filter `detect_callback` itself uses, so the two
    detectors tokenize identically and any per-class difference in the
    before/after comparison (`benchmark/validate_callback_transform.py`)
    is attributable to the GATE LOGIC, not to a silently different
    tokenizer.
  - `benchmark.joke_novelty.trigrams` / `trigram_jaccard` — the exact
    trigram machinery `env/rewards.py`'s `CorpusNoveltyPenalty` and
    `benchmark/joke_novelty.py`'s own novelty check use elsewhere in this
    project, not a fresh reimplementation. NOTE the same tokenization
    quirk documented on `env/rewards.py::_window_token_spans`: `norm()`
    (which `trigrams()` is built on) DELETES punctuation via
    `str.translate` rather than treating it as a separator, so e.g.
    `"Priya's"` normalizes to the fused token `"priyas"`, distinct from
    a bare `"Priya"` elsewhere in the text. This module inherits that
    behavior by reusing `trigrams()` unchanged (deliberate — see
    `env/rewards.py`'s own "ALSO NOTE" on why a similar divergence there
    was left alone rather than patched) — fixture text avoids possessive
    forms of a shared callback word for exactly this reason (see the
    fixture's own construction notes).
  - The `embed_fn: Callable[[Sequence[str]], <unit-normalized vectors>]`
    contract is IDENTICAL to `env/semantic_novelty.py`'s and
    `env/incongruity_gate.py`'s: vectors are assumed already
    L2-normalized, so cosine similarity reduces to a plain dot product —
    same convention, so a caller wiring a real `SentenceTransformer`
    encoder in for one of those modules can hand the exact same callable
    to this one.

Pure computation only, by design (EXP-016's constraint): NO judge, NO API
call anywhere in this file. The embedding path is an OPTIONAL OR-branch of
the detection gate (see `find_callback_match`) — omit `embed_fn` (the
default) and this module never touches an embedding model at all; the
content-word gate alone is fully sufficient to run this module standalone.

WHAT THIS MODULE DOES NOT DO (stated up front, this project's own
documentation habit — see e.g. `env/semantic_novelty.py`'s "RESIDUAL
RISKS" sections): it does not wire into `env/banter_env.py` at all.
`CallbackTransformationReward` below is a STUB — a documented target
shape for a LATER, separately-audited change, not a live reward term.
`env/banter_env.py` is read-only for this module's purposes (EXP-016's own
hard constraint) and is not imported, subclassed, or modified here.
"""

from typing import Callable, Dict, List, Optional, Sequence

from benchmark.banter import _content_words
from benchmark.joke_novelty import trigram_jaccard, trigrams

# --------------------------------------------------------------- constants
#
# Every constant below is deliberately named and documented individually
# (EXP-016's own instruction: "document every constant") rather than
# inlined, so a future calibration pass has one place to change each
# number and one place to read why it was chosen.

DEFAULT_MIN_GAP = 3
"""How many turns back an earlier turn must sit before it can qualify as
a callback ORIGIN for the CURRENT turn being scored. Mirrors
`benchmark.banter.detect_callback`'s own `min_gap` default and EXP-016's
registered design language verbatim: "a SPECIFIC earlier turn >= 3 turns
back." A turn at index `j` qualifies as a candidate origin for
`current_turn_idx` iff `current_turn_idx - j >= DEFAULT_MIN_GAP` — turns
closer than that are "recent" (topical continuity, not a callback), the
identical distinction `detect_callback`'s own `old_turns` /
`intervening_turns` split draws."""

DEFAULT_MIN_SHARED_CONTENT_WORDS = 2
"""The core fix. `detect_callback` fires on ANY single shared 5+-char
content word (minus its own stopword list) — EXPERIMENT_LOG.md's EXP-016
registration names this explicitly: "bag-of-words ... with NO
transformation requirement." Requiring >= 2 INDEPENDENTLY-shared content
words (not >= 1) is the cheapest lexical-only tightening that closes the
single-coincidental-word false positive `detect_callback`'s own docstring
documents as an accepted, unsolved gap ("Generic-but-long filler ... can
produce a false-positive 'callback' if it happens to repeat") — see
`coincidental_word_reuse` in the fixture, and
`benchmark/validate_callback_transform.py`'s old-vs-new comparison table,
for exactly this case quantified: one coincidentally-shared word now
correctly fails the gate that used to let it straight through. Chosen as
2, not 3+: two independently-shared distinctive content words (e.g. a
name plus a specific noun) is already a strong, cheap signal that the
SAME earlier material is being referenced, per this module's own fixture
design (see `benchmark/fixtures/callback_transform_fixture.jsonl`'s
construction notes); demanding 3+ would also reject genuine short
callbacks that reuse only two words of a longer earlier turn."""

DEFAULT_EMBED_SIM_FLOOR = 0.35
"""Cosine-similarity floor for the OPTIONAL embedding OR-branch of the
detection gate (only consulted when a caller supplies `embed_fn` — see
module docstring). EXP-016b RECALIBRATION (fix 3): the original 0.6 sat
ABOVE the measured genuine-coreference band (0.31-0.48), so the branch
could never catch its target case. 0.35 sits above the measured
unrelated/coincidental band's top (0.32) and inside the genuine band —
it catches upper-band genuine zero-lexical-overlap coreference while
staying out of the bulk of the unrelated range. The band overlap at
0.31-0.32 is real (see the honest finding below); the additional
EXP-016b freshness guard in `find_callback_match` (embedding fires only
when the ORIGIN's similarity exceeds every INTERVENING turn's) is what
keeps topical-continuity noise from exploiting the lowered floor.
Original spot-check data (real `all-MiniLM-L6-v2`, this project's
standard embedding instrument) over short casual conversational
one-liners, NOT a literature value:

    genuinely unrelated topics:                  cosine ~0.06 - 0.32
    coincidental single-shared-word, diff topic: cosine ~0.06 - 0.17
    light 1-2-word synonym paraphrase (same
      sentence, near-full lexical overlap):       cosine ~0.82 - 0.90
    hand-written "same referent, zero shared
      words" coreference pairs (e.g. "the new
      barista" / "that same chaos-loving coffee
      person"):                                   cosine ~0.31 - 0.48

HONEST FINDING from that spot-check, stated plainly rather than papered
over (this project's own documentation habit — see e.g.
`env/semantic_novelty.py`'s extensive residual-risk sections): MiniLM
sentence embeddings do NOT cleanly separate genuine same-referent
coreference from topically-unrelated short conversational turns in this
length/register regime — the two ranges overlap (0.31-0.48 vs 0.06-0.32,
with no clean gap). The 0.31-0.32 overlap zone means a floor of 0.35
still admits the TOP of the unrelated band in principle; the branch
therefore remains OFF by default (`embed_fn=None`) and its EXP-016b
validation numbers are reported as a diagnostic, not a certified
operating point."""

WORDSET_FLOOR = 0.7
"""EXP-016b fix 1. Content-word-set Jaccard similarity at or above which
the transformation score is forced to EXACTLY 0, alongside (OR) the
trigram `VERBATIM_FLOOR`. Closes the two escapes EXP-016's blind
held-out set quantified: (a) the CORRELATED DEAD ZONE — clause-reorder
paraphrase destroys trigrams (sim drops below both this module's old
floor AND SelfRepetitionPenalty's 0.5 threshold) but preserves the word
multiset, so a word-set similarity catches exactly what trigrams lose to
reordering; (b) the HYPHEN ESCAPE — a punctuation-level edit fuses/
splits `norm()` tokens and collapsed trigram similarity to 0.35 on a
functionally verbatim repeat, but `_wordset_words` splits hyphens before
tokenizing, so the punctuation-edit variant measures ~1.0 here and
floors. 0.7 (below the trigram floor's 0.8): a reorder-only or
punctuation-edit repeat shares essentially ALL content words (Jaccard
near 1.0); genuine transformed callbacks reuse 2-3 words of an earlier
turn while ADDING new material (measured band well under 0.5); 0.7
splits those with margin on both sides. Synonym-heavy paraphrase
(3+ content words swapped) can still land between the floors and earn
partial credit — recorded as the residual known gap, now shrunk by the
score's `1 - max(trigram, wordset)` basis (EXP-016b registration)."""

VERBATIM_FLOOR = 0.8
"""Trigram-Jaccard similarity at or above which the transformation score
is forced to EXACTLY 0, regardless of how close to 1.0 the raw `1 -
similarity` would otherwise land — EXP-016's own literal spec ("a
near-verbatim floor... the verbatim-decay rule"). Deliberately a HARD
CLIFF, not `SelfRepetitionPenalty`'s smoother above-threshold ramp
(env/rewards.py) — that class ramps CONTINUOUSLY from a small floor up
to full penalty because it is scoring "how similar is this to something
already said" as a matter of degree; this term is instead answering a
near-binary question paralleling THEORY-MAP.md §6's "a joke heard twice
is dead" — once a callback is close enough to a word-for-word repeat that
a listener would just hear the ORIGINAL line again, it has produced zero
new transformation, not "a little" transformation. 0.8 (not 1.0 exactly)
gives a small tolerance band for trivial capitalization/whitespace
differences that would otherwise dodge an exact-match check while still
being, functionally, the same sentence."""

_FALSE_POSITIVE_WORDS = frozenset({"morning", "today"})
"""`benchmark.banter.detect_callback`'s own docstring names these two
words EXACTLY as its documented, accepted false-positive source:
"Generic-but-long filler ('morning', 'today') is not in the stopword
list and can produce a false-positive 'callback' if it happens to
repeat -- a known, accepted false-positive source rather than a solved
problem." Neither word is in `_CALLBACK_STOPWORDS` (verified by direct
inspection of that set in `benchmark/banter.py`) — this module excludes
them ADDITIONALLY, on top of `_content_words`'s existing stopword
filtering, so the improved detector does not inherit this
already-diagnosed gap. Not a general solution to generic-but-long filler
words (`detect_callback`'s docstring is explicit that this is a known,
unsolved category, not just these two words) — only the two SPECIFIC
words that module documents by name."""

DEFAULT_NGRAM = 3
"""Trigram window size for the transformation-score comparison — matches
`benchmark.joke_novelty.trigrams`' own hardcoded n=3 and every other
n-gram tier in this project (`env/rewards.py`'s `_ngrams`,
`CorpusNoveltyPenalty`, `SelfRepetitionPenalty` all default to n=3).
Exposed as a parameter for completeness / future sweeps, not because a
different value has been validated."""


def _gate_content_words(text: str) -> set:
    """Content words for the DETECTION GATE specifically: `_content_words`
    (banter.py's own 5+-char, `_CALLBACK_STOPWORDS`-filtered tokenizer)
    minus `_FALSE_POSITIVE_WORDS`. Kept as a thin, separately-named
    wrapper (not inlined at every call site) so it is obvious this is the
    one place the false-positive exclusion is applied — the
    transformation-score half of this module (trigram similarity) does
    NOT use this function; it scores the full, unfiltered text."""
    return _content_words(text) - _FALSE_POSITIVE_WORDS


def _wordset_words(text: str) -> set:
    """Content-word set for the WORDSET similarity (EXP-016b fix 1):
    `_gate_content_words` over HYPHEN-SPLIT text, so `chaos-loving` and
    `chaos loving` produce the same tokens — the punctuation-edit
    verbatim escape (`norm()` fuses hyphenated words into one token,
    shifting every trigram) does not exist at the word-set level."""
    return _gate_content_words(text.replace("-", " "))


def wordset_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity of the two texts' hyphen-safe content-word
    sets. 0.0 when either side has no content words (nothing to compare
    — the safe direction for a floor test that ZEROES scores)."""
    a, b = _wordset_words(text_a), _wordset_words(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain dot product, pure Python (no numpy dependency in this
    module — same "provably pure-stdlib" discipline `env/rewards.py`'s
    own module docstring states for its default import path). Assumes
    both vectors are already unit-normalized, exactly the `embed_fn`
    contract documented on `DEFAULT_EMBED_SIM_FLOOR` above, so this dot
    product IS cosine similarity — `env/semantic_novelty.py`'s and
    `env/incongruity_gate.py`'s identical convention."""
    return float(sum(x * y for x, y in zip(a, b)))


def find_callback_match(
    turns: List[str],
    current_turn_idx: int,
    min_gap: int = DEFAULT_MIN_GAP,
    min_shared_content_words: int = DEFAULT_MIN_SHARED_CONTENT_WORDS,
    embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
    embed_sim_floor: float = DEFAULT_EMBED_SIM_FLOOR,
) -> Dict:
    """The DETECTION GATE: does `turns[current_turn_idx]` genuinely call
    back to a SPECIFIC earlier turn, not just echo generic vocabulary?

    A turn at index `j` qualifies as a CANDIDATE origin iff `j >= 0` and
    `current_turn_idx - j >= min_gap` (see `DEFAULT_MIN_GAP`'s docstring).
    Candidates are scanned NEAREST-TO-FARTHEST (highest `j` first) —
    mirrors `detect_callback`'s own "a callback to a closer setup is
    preferred when several match" rule — and the gate fires on the first
    candidate satisfying EITHER:

      (a) CONTENT-WORD TIER (always checked, no dependency needed): the
          current turn and the candidate turn share
          `>= min_shared_content_words` content words, per
          `_gate_content_words` (banter.py's tokenizer, minus the two
          documented false-positive words). This alone is a strict
          tightening of `detect_callback`'s own ">= 1 shared word" gate
          — see `DEFAULT_MIN_SHARED_CONTENT_WORDS`'s docstring.

      (b) EMBEDDING TIER (only consulted if `embed_fn` is supplied): the
          current turn's embedding has cosine similarity
          `>= embed_sim_floor` to the candidate turn's embedding. OFF by
          default (`embed_fn=None`) — this module never imports or calls
          an embedding backend on its own; the caller owns that decision
          entirely, same discipline as `env/semantic_novelty.py`'s
          injectable `embed_fn`. When supplied, ALL of `turns` is
          embedded in ONE batched `embed_fn` call (never one call per
          candidate) — same "one call per unit of work" discipline
          `env/semantic_novelty.py`'s own docstring states for its
          windowed mode.

    Returns a dict with:
      `matched_turn_idx`   — the qualifying turn's index, or `None`.
      `detection_reasons`  — list, subset of `["content_words",
                             "embedding"]`, empty if no match.
      `shared_content_words` — sorted list of the words that satisfied
                             tier (a), `[]` if tier (a) did not fire at
                             this turn (even if tier (b) alone matched).
      `embed_similarity`   — the cosine similarity at the matched turn
                             (float), or `None` if `embed_fn` was not
                             supplied.

    No match (gate does not fire) when `turns` has no qualifying
    candidate at all (too few earlier turns for `min_gap`), or no
    candidate satisfies either tier: `matched_turn_idx=None`,
    `detection_reasons=[]`, `shared_content_words=[]`,
    `embed_similarity=None`.
    """
    if not (0 <= current_turn_idx < len(turns)):
        raise ValueError(
            "find_callback_match: current_turn_idx=%r out of range for "
            "%d turns." % (current_turn_idx, len(turns)))
    if min_shared_content_words < 1:
        raise ValueError(
            "find_callback_match: min_shared_content_words must be >= 1 "
            "(got %r)." % (min_shared_content_words,))

    latest_candidate = current_turn_idx - min_gap
    candidates = list(range(0, latest_candidate + 1))  # oldest..nearest, index order
    no_match = {"matched_turn_idx": None, "detection_reasons": [],
               "shared_content_words": [], "embed_similarity": None}
    if not candidates:
        return no_match

    current_words = _gate_content_words(turns[current_turn_idx])

    embed_sims: Optional[Dict[int, float]] = None
    if embed_fn is not None:
        vectors = embed_fn(turns)
        cur_vec = vectors[current_turn_idx]
        embed_sims = {j: _dot(cur_vec, vectors[j]) for j in candidates}

    for j in sorted(candidates, reverse=True):  # nearest first
        shared = current_words & _gate_content_words(turns[j])
        # EXP-016b fix 2 (intervening-exclusion, ported from
        # detect_callback's documented guard): a shared word that also
        # appears in ANY intervening turn never left the conversation —
        # that is topical continuity, not a callback. Only FRESH words
        # (absent from every turn between origin and current) count.
        intervening_words: set = set()
        for k in range(j + 1, current_turn_idx):
            intervening_words |= _gate_content_words(turns[k])
        fresh = shared - intervening_words
        reasons = []
        if len(fresh) >= min_shared_content_words:
            reasons.append("content_words")
        sim = embed_sims[j] if embed_sims is not None else None
        if sim is not None and sim >= embed_sim_floor:
            # EXP-016b fix 3 (freshness guard, the embedding analogue of
            # fix 2): the origin must be MORE similar to the current turn
            # than every intervening turn is — otherwise the "match" is
            # just the ongoing topic, which the lowered 0.35 floor would
            # otherwise admit.
            max_intervening_sim = max(
                (_dot(vectors[current_turn_idx], vectors[k])
                 for k in range(j + 1, current_turn_idx)), default=float("-inf"))
            if sim > max_intervening_sim:
                reasons.append("embedding")
        if reasons:
            return {"matched_turn_idx": j, "detection_reasons": reasons,
                    "shared_content_words": sorted(fresh), "embed_similarity": sim}

    return no_match


def callback_transformation_score(
    turns: List[str],
    current_turn_idx: int,
    min_gap: int = DEFAULT_MIN_GAP,
    min_shared_content_words: int = DEFAULT_MIN_SHARED_CONTENT_WORDS,
    embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
    embed_sim_floor: float = DEFAULT_EMBED_SIM_FLOOR,
    verbatim_floor: float = VERBATIM_FLOOR,
    wordset_floor: float = WORDSET_FLOOR,
    ngram: int = DEFAULT_NGRAM,
) -> Dict:
    """Detection gate (`find_callback_match`) THEN transformation score.

    `score` is 0.0 whenever the gate does not fire (no genuine callback
    detected — nothing to score). When the gate fires, `score = 1 -
    trigram_jaccard(turns[current_turn_idx], turns[matched_turn_idx])`,
    EXCEPT that a trigram-Jaccard similarity `>= verbatim_floor` forces
    `score` to EXACTLY 0.0 (the verbatim-decay rule — see
    `VERBATIM_FLOOR`'s docstring). Both tiers use the SAME
    `matched_turn_idx` chosen by the gate — the gate decides WHETHER a
    callback exists and WHICH earlier turn it targets; this function
    never re-searches for a "better" match once the gate has committed
    to one.

    Returns a dict extending `find_callback_match`'s own with two more
    keys: `trigram_similarity` (float in [0, 1], or `None` if the gate
    did not fire) and `score` (float in [0, 1], always 0.0 when
    `matched_turn_idx` is `None`).
    """
    match = find_callback_match(
        turns, current_turn_idx, min_gap=min_gap,
        min_shared_content_words=min_shared_content_words,
        embed_fn=embed_fn, embed_sim_floor=embed_sim_floor)

    if match["matched_turn_idx"] is None:
        return {**match, "trigram_similarity": None,
                "wordset_similarity": None, "score": 0.0}

    current_trigrams = trigrams(turns[current_turn_idx])
    matched_trigrams = trigrams(turns[match["matched_turn_idx"]])
    tri_sim = trigram_jaccard(current_trigrams, matched_trigrams)
    ws_sim = wordset_jaccard(turns[current_turn_idx],
                             turns[match["matched_turn_idx"]])
    # EXP-016b fix 1: transformation must be novel in BOTH sequence
    # (trigrams) and vocabulary (word set) — `1 - max(...)` scores
    # against whichever axis the reuse hides in, and either floor
    # zeroes the score outright (verbatim/reorder/punctuation-edit).
    if tri_sim >= verbatim_floor or ws_sim >= wordset_floor:
        score = 0.0
    else:
        score = 1.0 - max(tri_sim, ws_sim)

    return {**match, "trigram_similarity": tri_sim,
            "wordset_similarity": ws_sim, "score": score}


# --------------------------------------------------------- reward-term stub


class CallbackTransformationReward:
    """STUB SIGNATURE — documented, NOT WIRED into `env/banter_env.py`.

    Wiring this in is explicitly a LATER, separately-audited change
    (EXP-016's own hard constraint: "do not change env/banter_env.py").
    This class is the drop-in TARGET shape for that future change, not a
    live reward term — `env/banter_env.py` does not import this module.

    THE CONSUMPTION PATH THIS REPLACES (traced, not modified — see
    `env/banter_env.py::BanterEnv.step`, reproduced here for reference):

        callback = detect_callback(reply_text, self._partner_history,
                                   min_gap=self.min_gap)
        callback_term = self.callback_weight if callback is not None else 0.0
        ...
        reward = callback_term + ablation_term

    i.e. today: a FLAT `callback_weight` bonus (default +0.5) the instant
    `detect_callback` returns non-`None`, with no transformation
    requirement at all — exactly the bug this whole module exists to fix
    (see module docstring). The wiring this class targets is a one-line
    swap of that block for:

        callback_term = self(turns, current_turn_idx)

    where `turns` is `[m["content"] for m in self.messages]` (flattening
    `BanterEnv`'s own `messages` list of `{"role", "content"}` dicts,
    partner AND model turns interleaved in conversation order — the shape
    `callback_transformation_score` is specified against, per EXP-016)
    and `current_turn_idx = len(self.messages) - 1` (the just-appended
    model reply, `self.messages[-1]`, is the turn being scored). NOTE the
    shape difference from today's call: `detect_callback` takes the reply
    text SEPARATELY from `earlier_turns` (partner turns only); this
    class's `turns` includes the turn being scored AT `current_turn_idx`,
    and spans BOTH speakers — a deliberate broadening (Norrick's
    reincorporation construct does not require the callback's origin to
    be the OTHER speaker's line; a model can call back to its own earlier
    bit too), not an oversight. Actually performing that wiring, updating
    `env/banter_env.py`'s docstring/tests, and re-validating
    `BanterEnv`'s own reward-magnitude assumptions (its module docstring's
    "MAGNITUDE FIX" note) are ALL separately-audited follow-up work, not
    done by this class's mere existence.

    Sign-managed like `env/rewards.py::RewardConfig`'s BONUS fields
    (`intra_group_diversity_weight`, `comprehensibility_weight`,
    ...): this is a bonus term (reward genuine transformation, never
    penalize its absence beyond withholding the bonus), so `weight < 0`
    is rejected at construction — the same "a flipped sign silently
    rewards the opposite of what this term encourages" guard those
    fields carry.
    """

    __name__ = "callback_transformation_reward"

    def __init__(
        self,
        weight: float = 0.5,
        min_gap: int = DEFAULT_MIN_GAP,
        min_shared_content_words: int = DEFAULT_MIN_SHARED_CONTENT_WORDS,
        embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
        embed_sim_floor: float = DEFAULT_EMBED_SIM_FLOOR,
        verbatim_floor: float = VERBATIM_FLOOR,
        wordset_floor: float = WORDSET_FLOOR,
    ):
        if weight < 0:
            raise ValueError(
                "CallbackTransformationReward.weight must be >= 0 (got "
                "%r). This is a BONUS term (reward genuine, transformed "
                "callbacks) -- a negative weight silently rewards the "
                "OPPOSITE of what it exists to encourage, mirroring "
                "env/rewards.py's RewardConfig sign guard for its bonus "
                "fields." % (weight,))
        self.weight = weight
        self.min_gap = min_gap
        self.min_shared_content_words = min_shared_content_words
        self.embed_fn = embed_fn
        self.embed_sim_floor = embed_sim_floor
        self.verbatim_floor = verbatim_floor
        self.wordset_floor = wordset_floor

    def __call__(self, turns: List[str], current_turn_idx: int) -> float:
        """`weight * transformation_score`, `0.0` if the detection gate
        does not fire. Pure computation -- no judge, no network call,
        ever (this module's own hard constraint, inherited unchanged)."""
        result = callback_transformation_score(
            turns, current_turn_idx, min_gap=self.min_gap,
            min_shared_content_words=self.min_shared_content_words,
            embed_fn=self.embed_fn, embed_sim_floor=self.embed_sim_floor,
            verbatim_floor=self.verbatim_floor,
            wordset_floor=self.wordset_floor)
        return self.weight * result["score"]

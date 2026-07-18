#!/usr/bin/env python3
"""THE VALIDATION for EXP-009: does embedding-based (semantic) novelty
detection actually close CorpusNoveltyPenalty's documented 2-word-reskin
evasion, and at what threshold/corpus-cap tradeoff?

Builds an adversarial test set from the REAL 25 ChatGPT memorized-joke
templates (~/Experiments/good-humored-data/corpus/chatgpt-25-templates.jsonl,
read-only -- NOT the tiny env/tests/fixtures copy, which exists only for
zero-network unit tests):

  POSITIVES = each of the 25 templates, reskinned at 3 deterministic
              content-word edit depths (1, 2, 4 -- see `generate_reskin`)
              + 5 hand-written full paraphrases (same joke, completely
              reworded -- the hard case n-gram approaches cannot see at
              all and embedding approaches are supposed to catch).
  NEGATIVES = ~100 deterministically-sampled, genuinely different jokes
              from the commercial-safe corpus (real different content,
              not derived from any template), with any exact-text overlap
              against the embedded corpus set filtered and redrawn --
              see `build_negatives`'s `exclude_texts` for why: an
              un-excluded negative can independently reservoir-sample the
              SAME row that also lands in the corpus-embedding sample
              (both draw from the same commercial-safe pool), which is a
              held-out-set leak, not a real false positive.

Scores every item with BOTH:
  (a) the EXISTING trigram-Jaccard scorer (`benchmark.joke_novelty`,
      i.e. what `CorpusNoveltyPenalty` already ships) -- to reproduce and
      quantify tonight's documented evasion, not just assert it. This
      scorer's reference set is the 25 templates ONLY (`best_ngram_score`
      loops over `templates`, never the general corpus).
  (b) the NEW semantic (cosine-similarity, all-MiniLM-L6-v2) scorer this
      experiment is validating -- ALSO scored against the 25 templates
      ONLY for threshold calibration (`rec["semantic_score"]`), mirroring
      (a)'s reference set exactly so the two are apples-to-apples. A
      SEPARATE, purely informational `general_corpus_max_sim` field
      additionally records max similarity against templates + the capped
      general-corpus sample -- this field must NEVER feed
      `pick_threshold`: an EXP-009 corrected-validation pass (2026-07-17)
      found the general corpus is pervasively internally near-duplicated
      (recycled/reworded reposts unrelated to the 25 templates; see
      `~/Experiments/good-humored-data/corpus/_dedup_stats.json`, which
      documents only EXACT-string dedup was ever done), so a "negative"
      matching some OTHER unrelated corpus joke closely is real signal
      about corpus redundancy, not evidence the scorer confuses novel
      content with a memorized-template paraphrase. The original
      (buggy) validation pass conflated the two, calibrated
      DEFAULT_THRESHOLD=0.94 against the confounded general-corpus
      reference set, and got 0.0 detection at every depth as a result --
      see experiment-runs/2026-07-17-semantic-novelty-validation/
      report.json's git history / EXPERIMENT_LOG.md for that artifact.

...then sweeps thresholds for both (templates-only reference), reports
per-edit-depth detection rate + false-positive rate on the negatives, and
picks the semantic threshold to ship as
`env.semantic_novelty.DEFAULT_THRESHOLD`.

Also times corpus scanning and embedding separately (real numbers, not
estimates) to answer the training-time feasibility question directly:
embedding the full ~1.2M-row corpus vs. the capped/sampled default.

Uses sentence_transformers DIRECTLY (not via SemanticNoveltyPenalty) so
this script controls its own timing phases precisely; the class's actual
scoring/ramping logic is exercised separately by
env/tests/test_semantic_novelty.py's fake-embedder unit tests -- this
script's job is producing the real NUMBERS, not re-testing the class's
plumbing.

Usage: python3 -m env.validate_semantic_novelty
   or: python3 env/validate_semantic_novelty.py
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Make the repo root importable regardless of how this script is invoked
# (same convention as env/smoke.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.joke_novelty import (load_templates, trigram_jaccard,  # noqa: E402
                                    trigrams)
from env.rewards import _window_token_spans  # noqa: E402
from env.semantic_novelty import (SemanticNoveltyPenalty,  # noqa: E402
                                  _reservoir_sample_jokes)

REAL_CORPUS_DIR = Path("~/Experiments/good-humored-data/corpus").expanduser()
OUTPUT_DIR = _REPO_ROOT / "experiment-runs" / "2026-07-17-semantic-novelty-validation"

# EXP-011 (windowed re-calibration) reads this run's own operating point
# for bar 4 ("paraphrase detection not below EXP-009's whole-text
# operating point on unpadded inputs") -- see run_windowed_validation.
WHOLE_TEXT_REPORT_PATH = OUTPUT_DIR / "report.json"

EDIT_DEPTHS = (1, 2, 4)
RESKIN_SEED = 20260717          # see generate_reskin's docstring
NEGATIVE_SAMPLE_SEED = 20260717
NEGATIVE_SAMPLE_N = 100
NGRAM_SHIPPED_THRESHOLD = 0.35  # CorpusNoveltyPenalty's actual default
CORPUS_CAP_FOR_TIMING = 50_000  # env.semantic_novelty.DEFAULT_CORPUS_CAP
THRESHOLD_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05..0.95
TARGET_FPR = 0.05               # CLAUDE.md-style success criterion

# ------------------------------------------------------- EXP-011 (windowed)

# The registered design (EXPERIMENT_LOG.md's "Windowed novelty tiers" entry)
# and the auditor's refinement to it, both implemented in
# `run_windowed_validation` below:
#
#   POSITIVES = this module's own reskins/paraphrases (build_positive_records,
#               reused unmodified) PLUS a new VERBATIM class -- the raw 25
#               templates, unmodified -- the exact shape of the audited
#               dilution exploit (see env/semantic_novelty.py's module
#               docstring and env/tests/test_semantic_novelty.py's
#               TestWindowedSemanticDilution, which pins the fixture-scale
#               version of this same exploit down as a regression test).
#               Every positive is embedded inside VERBATIM_PADDING_REPS
#               (--padding-reps) repetitions of WINDOW_PADDING_FILLER.
#   NEGATIVES  = this module's own 100 novel negatives (build_negatives,
#               reused unpadded, for continuity with EXP-009) PLUS a new
#               STRADDLING class: multi-sentence novel completions --
#               concatenations of 2-5 genuinely different corpus jokes plus
#               non-joke prose -- constructed at boundary-token lengths that
#               straddle (below/at/above) EVERY window-ladder level's width.
#               The auditor's refinement: a negative shorter than every
#               window width never triggers windowing at all (it degenerates
#               to whole-text-only scoring, see SemanticNoveltyPenalty.
#               _window_texts's "n <= width: continue" branch) and would
#               flatter the threshold by simply never exercising the new
#               scoring path.
#
# Scoring reuses `SemanticNoveltyPenalty._window_texts` and
# `_template_embeddings` directly (see windowed_semantic_scores) -- this
# script must never re-implement windowing separately from the class it is
# calibrating; that was exactly the kind of drift EXP-009's own corrected
# pass was written to eliminate (see module docstring above).
WINDOW_PADDING_FILLER = "here is some unrelated filler text to pad this completion"
DEFAULT_PADDING_REPS = "0,5,20,50"
VERBATIM_KIND = "verbatim"

# Multi-sentence negative construction (the auditor's refinement).
STRADDLE_POOL_N = 250            # distinct corpus jokes available to build from
STRADDLE_POOL_SEED = NEGATIVE_SAMPLE_SEED + 1  # deliberately != the
                                  # baseline-100 negatives' own seed, so the
                                  # two negative classes are independent
                                  # deterministic draws, not overlapping
                                  # reuses of the exact same 100 rows.
STRADDLE_REPS_PER_LENGTH = 15     # replicate negatives per target length
STRADDLE_DELTAS = (-4, 0, 6)      # below / at / above each ladder width
MIN_JOKES_PER_NEGATIVE = 2
MAX_JOKES_PER_NEGATIVE = 5

# Neutral, non-joke, non-template prose used as (a) length-padding filler
# when a straddling negative's joke+prose draw doesn't reach its target
# length, and (b) part of the draw itself ("+ non-joke prose" per the
# registered design). Deliberately dull/expository -- the whole point is
# content with no punchline structure at all, orthogonal to both the humor
# templates and the sampled jokes.
NONJOKE_PROSE = [
    "The quarterly maintenance report noted a modest increase in regional "
    "rainfall this season.",
    "Local officials confirmed the bridge repairs would continue steadily "
    "through the autumn months.",
    "The museum's new wing features artifacts recovered during last year's "
    "harbor excavation project.",
    "Researchers spent a full year tracking the migratory patterns of the "
    "coastal shorebird population.",
    "The zoning committee reviewed several proposed changes to the "
    "downtown parking ordinance at length.",
    "A new ferry route between the two islands is expected to open "
    "sometime early next spring.",
    "The library's reading room will remain closed for renovations until "
    "the middle of next month.",
    "City engineers are studying whether the old water tower can be "
    "safely repurposed as a landmark.",
]

# Registered bars (EXPERIMENT_LOG.md's "Windowed novelty tiers" entry).
BAR_VERBATIM_PADDED_DETECTION_MIN = 0.95
BAR_PADDING_INVARIANCE_MAX_DELTA = 0.02  # 2 percentage points, between the
                                          # 5-rep and 50-rep padding levels

# --------------------------------------------------------- reskin generator

# Small, fixed lookup covering the 25 real templates' OWN vocabulary --
# NOT a real POS tagger (see generate_reskin's docstring for the honest
# limitation this implies). Anything not listed defaults to NOUN, which
# is right for the overwhelming majority of this template set's
# substitutable content words.
_KNOWN_POS: Dict[str, str] = {
    # T1
    "win": "VERB", "award": "NOUN", "outstanding": "ADJ", "field": "NOUN",
    "scarecrow": "NOUN",
    # T2
    "tomato": "NOUN", "turn": "VERB", "red": "ADJ", "saw": "VERB",
    "salad": "NOUN", "dressing": "NOUN",
    # T3
    "math": "NOUN", "book": "NOUN", "sad": "ADJ", "problems": "NOUN",
    # T4
    "scientists": "NOUN", "trust": "VERB", "atoms": "NOUN", "make": "VERB",
    "everything": "NOUN",
    # T5
    "cookie": "NOUN", "doctor": "NOUN", "feeling": "VERB", "crumbly": "ADJ",
    # T6
    "bicycle": "NOUN", "stand": "VERB", "tired": "ADJ",
    # T7
    "frog": "NOUN", "call": "VERB", "insurance": "NOUN", "company": "NOUN",
    "jump": "NOUN", "car": "NOUN",
    # T8
    "chicken": "NOUN", "cross": "VERB", "playground": "NOUN", "slide": "NOUN",
    # T9
    "computer": "NOUN", "cold": "ADJ", "left": "VERB", "windows": "NOUN",
    "open": "ADJ",
    # T10
    "hipster": "NOUN", "burn": "VERB", "tongue": "NOUN", "drank": "VERB",
    "coffee": "NOUN", "cool": "ADJ",
    # T11
    "oysters": "NOUN", "give": "VERB", "charity": "NOUN", "shellfish": "NOUN",
    # T12
    "virus": "NOUN",
    # T13
    "banana": "NOUN", "go": "VERB", "peeling": "VERB", "well": "ADJ",
    # T14
    "file": "VERB", "police": "NOUN", "report": "NOUN", "mugged": "VERB",
    # T15
    "golfer": "NOUN", "bring": "VERB", "pairs": "NOUN", "pants": "NOUN",
    "hole": "NOUN",
    # T16
    "man": "NOUN", "put": "VERB", "money": "NOUN", "freezer": "NOUN",
    "wanted": "VERB", "hard": "ADJ", "cash": "NOUN",
    # T17
    "seagulls": "NOUN", "fly": "VERB", "bay": "NOUN", "bagels": "NOUN",
    # T18
    "seance": "NOUN", "talk": "VERB", "side": "NOUN",
    # T19
    "belt": "NOUN", "sent": "VERB", "jail": "NOUN", "held": "VERB",
    "pair": "NOUN",
    # T20
    "road": "NOUN",
    # T21
    "byte": "NOUN",
    # T22
    "cow": "NOUN", "outer": "ADJ", "space": "NOUN", "moooon": "NOUN",
    "see": "VERB",
    # T23
    "blender": "NOUN", "liquid": "ADJ", "assets": "NOUN",
    # T24
    "skeletons": "NOUN", "fight": "VERB", "guts": "NOUN",
    # T25
    "alligator": "NOUN", "vest": "NOUN", "investigator": "NOUN",
}

_SWAP_POOLS: Dict[str, List[str]] = {
    "NOUN": ["kangaroo", "umbrella", "volcano", "trombone", "spreadsheet",
            "lighthouse", "pretzel", "accordion", "waffle", "hedgehog",
            "flamingo", "mechanic", "librarian", "astronaut", "gorilla",
            "walrus", "otter", "beaver", "cucumber", "tractor", "trumpet",
            "helicopter", "bulldozer", "peacock", "octopus", "chipmunk",
            "wizard", "pirate", "dentist", "plumber"],
    "VERB": ["juggle", "whisper", "stumble", "wobble", "vanish", "sparkle",
            "wander", "giggle", "tumble", "hover", "waddle", "shimmer",
            "tiptoe", "snooze", "fidget"],
    "ADJ": ["purple", "soggy", "enormous", "invisible", "ancient",
           "electric", "fragile", "noisy", "glowing", "crooked", "squishy",
           "lopsided", "gigantic", "shimmering", "wobbly"],
}

# Function words + numbers excluded from substitution candidacy -- keeps
# the reskin generator away from grammar-critical scaffolding and from
# turning "two pairs of pants" into a semantically-nonsensical numeric
# edit ("kangaroo pairs of pants").
STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "and",
    "or", "but", "because", "why", "did", "do", "does", "he", "she", "it",
    "its", "they", "them", "his", "her", "their", "that", "this", "what",
    "who", "whom", "up", "out", "over", "by", "from", "into", "than",
    "then", "so", "if", "not", "no", "was", "were", "is", "are", "be",
    "been", "being", "had", "has", "have", "get", "got", "you", "other",
    "each", "one", "two", "first", "second", "before", "case", "too",
    "many", "itself",
    "don't", "didn't", "doesn't", "couldn't", "wasn't", "weren't",
    "hasn't", "hadn't", "isn't", "aren't", "they're", "he's", "she's",
    "it's", "they'd",
}

# The real 25-template file (unlike this repo's tiny fixture copy) uses
# the Unicode right single quotation mark (U+2019, "'") for every
# contraction, not the ASCII apostrophe -- confirmed by scanning all 25
# rows for non-ASCII characters (exactly one: U+2019). BOTH are included
# in the word-character class so "don’t"/"don't" tokenize as ONE
# word either way, not split at the apostrophe into "don" + "t" -- the
# bug this comment is here to prevent a regression of: an early version
# of this generator used ASCII-only `[A-Za-z']+`, which silently split
# "don’t" into "don" (an eligible 3-letter, non-stopword candidate!)
# + a discarded 1-letter "t", so the FIRST "content word" substituted in
# depth=1 reskins was routinely the mangled stub of a contraction (e.g.
# "don’t" -> "walrus’t"), corrupting every template that uses
# a contraction near its start. Caught by manually inspecting a reskin's
# actual output text, not by any automated check -- see
# env/tests/test_validate_semantic_novelty.py's curly-apostrophe
# regression tests for the pinned-down fix.
_WORD_PATTERN = re.compile(r"[A-Za-z’']+")


def _normalize_apostrophe(word: str) -> str:
    """Curly -> straight apostrophe, lowercased -- so STOPWORDS only needs
    ONE spelling per contraction regardless of which quote style the
    source text uses."""
    return word.lower().replace("’", "'")


def _content_candidates(text: str) -> List[Tuple[int, int, str]]:
    """Ordered (start, end, word) spans eligible for substitution:
    alphabetic tokens >= 3 chars, not in STOPWORDS, in order of first
    appearance. Order is load-bearing: edit depth N substitutes the FIRST
    N candidates, so depths NEST (a depth-4 reskin's edits are a superset
    of its own depth-2 reskin's edits) -- this is what makes "detection
    rate by edit depth" a genuine monotonic-degradation curve rather than
    3 unrelated random edits."""
    out = []
    for m in _WORD_PATTERN.finditer(text):
        w = m.group(0)
        if len(w) < 3 or _normalize_apostrophe(w) in STOPWORDS:
            continue
        out.append((m.start(), m.end(), w))
    return out


def _substitute_for(word: str, template_id: str, seed: int) -> str:
    """Stable (CRC32, NOT Python's randomized-per-process hash()) pick
    from the same-POS-ish pool, keyed by (seed, template_id, word) so
    re-running this script always produces byte-identical reskins."""
    pos = _KNOWN_POS.get(word.lower(), "NOUN")
    pool = _SWAP_POOLS[pos]
    key = "%d:%s:%s" % (seed, template_id, word.lower())
    idx = zlib.crc32(key.encode("utf-8")) % len(pool)
    substitute = pool[idx]
    if substitute.lower() == word.lower():  # guarantee an actual change
        substitute = pool[(idx + 1) % len(pool)]
    # Crude plurality match -- if the original looked plural, keep the
    # substitute plural-shaped too (heuristic, not real morphology).
    if word.lower().endswith("s") and len(word) > 3 and not substitute.endswith("s"):
        substitute = substitute + "s"
    if word[0].isupper():  # match sentence-initial capitalization
        substitute = substitute[0].upper() + substitute[1:]
    return substitute


def generate_reskin(template_id: str, text: str, depth: int,
                    seed: int = RESKIN_SEED) -> Tuple[str, int]:
    """Deterministic word-substitution reskin of `text`. Returns
    `(reskinned_text, actual_depth)` -- `actual_depth <= depth` when the
    template has fewer eligible candidates than requested (e.g.
    chatgpt25-T24, "Why don't skeletons fight each other? They don't have
    the guts.", has only 3 candidates -- skeletons/fight/guts -- so a
    depth=4 request clamps to actual_depth=3). Callers MUST report
    `actual_depth`, not the requested `depth` -- mislabeling a 3-word edit
    as "depth 4" would be exactly the kind of unlabeled measurement error
    this project's own culture (CLAUDE.md's falsifiable-registration
    discipline) exists to prevent.

    KNOWN ROUGH EDGE, stated plainly: `_KNOWN_POS` is a fixed lookup over
    these 25 templates' own vocabulary, not a real POS tagger. Two
    templates (T5 "was feeling", T13 "wasn't peeling") have a candidate in
    a gerund/participle slot that a base-form verb substitute doesn't
    grammatically match ("was giggle"). The result reads as a rough
    machine edit, not gibberish -- adequate for testing "does a novelty
    scorer detect this is the same joke, reworded," which is this
    script's actual question, not "is this reskin fluent prose."
    """
    candidates = _content_candidates(text)
    chosen = candidates[:depth]
    out, last_end = [], 0
    for start, end, word in chosen:
        out.append(text[last_end:start])
        out.append(_substitute_for(word, template_id, seed))
        last_end = end
    out.append(text[last_end:])
    return "".join(out), len(chosen)


# Hand-written FULL paraphrases -- the hard case. Same joke/punchline
# logic as the named real template, completely reworded (heavy
# vocabulary + syntax changes, not a handful of word swaps). n-gram
# overlap with the original is trivially near-zero for all five; the
# question this validation asks is whether the SEMANTIC scorer still
# recognizes "this is a reworded version of a memorized joke."
HAND_PARAPHRASES = [
    {"template_id": "chatgpt25-T4",
     "text": "Physicists have serious trust issues with tiny particles, "
            "and honestly, everything you can see or touch is put "
            "together out of them."},
    {"template_id": "chatgpt25-T20",
     "text": "What made the hen dash across the street? So she could "
            "reach the opposite curb."},
    {"template_id": "chatgpt25-T6",
     "text": "How come the bike kept toppling over on its own? It was "
            "simply exhausted, two tires' worth, in fact."},
    {"template_id": "chatgpt25-T3",
     "text": "What made the algebra textbook feel down? It was weighed "
            "down by an overwhelming number of exercises to solve."},
    {"template_id": "chatgpt25-T17",
     "text": "How come the gulls avoid soaring above the inlet? If they "
            "did, they would technically become breakfast pastries."},
]


# ------------------------------------------------------------- test set


def build_positive_records(templates: List[Dict], seed: int) -> List[Dict]:
    records = []
    for t in templates:
        for depth in EDIT_DEPTHS:
            reskinned, actual_depth = generate_reskin(t["id"], t["text"],
                                                      depth, seed)
            records.append({
                "template_id": t["id"], "kind": "reskin",
                "requested_depth": depth, "actual_depth": actual_depth,
                "text": reskinned,
            })
    for p in HAND_PARAPHRASES:
        records.append({
            "template_id": p["template_id"], "kind": "paraphrase",
            "requested_depth": None, "actual_depth": None,
            "text": p["text"],
        })
    return records


def build_negatives(
    corpus_dir: Path, n: int, seed: int,
    exclude_texts: Optional[Iterable[str]] = None,
) -> Tuple[List[str], int, int]:
    """~`n` genuinely different jokes from the commercial-safe corpus,
    deterministic seed. Over-samples then filters out (a) degenerate
    very-short entries (a handful of real corpus rows are single-word
    non-jokes like "lol") and (b) any row whose EXACT text is in
    `exclude_texts`, before truncating to exactly `n`.

    `exclude_texts` closes a held-out-set LEAK found in the 2026-07-17
    validation run: the corpus-embedding sample and this negative sample
    are both deterministic reservoir draws over the same underlying
    ~887K-row commercial-safe pool, so without exclusion a "negative" can
    independently land on the EXACT SAME row that also got sampled into
    the embedded corpus -- 3/100 negatives did, in the original run,
    each scoring a trivial self-match similarity of 1.0 and corrupting
    the FPR-based threshold pick. Callers validating a corpus-comparison
    scorer MUST pass the corpus's own embedded/compared-against text set
    here (typically `set(corpus_texts)`, i.e. templates + the sampled
    general-corpus rows) -- a negative is only a genuine held-out
    negative if it cannot also be a corpus member being compared against.
    Returns the count of rows dropped for this reason (`n_excluded_leaks`,
    from the FINAL/returned draw only, matching the adaptive buffer's own
    "fresh draw, not more of the same" semantics below) so the report can
    surface it rather than silently absorbing it into the oversample
    ratio.

    Adaptive buffer, not a fixed multiplier: starts at a 3x oversample and
    DOUBLES (re-drawing the reservoir sample at the larger cap -- note
    this is a fresh deterministic draw at that cap, not simply "more of
    the same draw," since Algorithm R's replacement behavior depends on
    cap itself; still fully reproducible for a fixed corpus+seed+starting
    multiplier) until either `n` valid entries survive the filter or the
    cap reaches the full scanned population (nothing more to gain from
    growing further). A fixed 3x buffer silently under-delivers if the
    corpus's degenerate-entry OR leak rate is higher than expected --
    this doesn't."""
    exclude = set(exclude_texts) if exclude_texts is not None else set()
    multiplier = 3
    while True:
        cap = n * multiplier
        oversample, n_scanned = _reservoir_sample_jokes(
            corpus_dir / "commercial-safe", cap=cap, seed=seed)
        n_excluded_leaks = sum(1 for t in oversample if t in exclude)
        filtered = [t for t in oversample
                   if len(t.split()) >= 3 and t not in exclude]
        if len(filtered) >= n or cap >= n_scanned:
            return filtered[:n], n_scanned, n_excluded_leaks
        multiplier *= 2


# ------------------------------------------------------------ scoring


def best_ngram_score(text: str, templates: List[Dict]) -> float:
    jt = trigrams(text)
    return max((trigram_jaccard(jt, t["_trigrams"]) for t in templates),
              default=0.0)


def semantic_scores(
    texts: List[str], embed_fn, template_embeddings, corpus_embeddings,
) -> Tuple[List[float], List[float]]:
    """Returns (template_only_scores, general_corpus_scores) for `texts`,
    both derived from a SINGLE `embed_fn(texts)` call.

    `template_only_scores` (max cosine similarity vs. `template_embeddings`
    ALONE, the 25 known-memorized templates) is the CALIBRATION reference
    -- the only one `main()` ever passes to `sweep()`/`pick_threshold()`.
    This mirrors `best_ngram_score`'s own reference set exactly (that
    function loops over `templates`, never the general corpus) -- making
    the n-gram baseline and this semantic scorer apples-to-apples for the
    first time. See module docstring for why the ORIGINAL validation
    (which calibrated against `corpus_embeddings` -- templates + a 50K
    general-corpus sample -- and produced DEFAULT_THRESHOLD=0.94 /
    detection=0.0 at every depth) was wrong.

    `general_corpus_scores` (vs. `corpus_embeddings`, which is templates +
    the capped general-corpus sample) is informational ONLY -- callers
    must never pass it to `pick_threshold`. The general corpus is
    pervasively internally near-duplicated (recycled/reworded reposts
    unrelated to the 25 templates; `_dedup_stats.json` alongside the real
    corpus confirms only exact-string dedup was ever done), so a high
    general-corpus similarity is evidence about corpus redundancy, not
    evidence this scorer confuses novel content with a memorized-template
    paraphrase.

    A standalone, top-level function (not a closure over `main()`'s local
    `model`) specifically so `env/tests/test_validate_semantic_novelty.py`
    can unit-test the templates-only/general-corpus separation directly
    by injecting a fake `embed_fn` -- the exact same dependency-injection
    pattern `env/semantic_novelty.py`'s `SemanticNoveltyPenalty` uses for
    the same reason (see that class's docstring). `numpy` is imported
    locally, inside this function, not at module scope -- consistent with
    this whole module's "ML imports stay inside function/method bodies,
    never at import time" convention (see module docstring). Both
    `template_embeddings` and `corpus_embeddings` are assumed already
    unit-normalized numpy arrays (as `main()` constructs them via
    `corpus_embeddings[:len(templates)]` and `model.encode(...,
    normalize_embeddings=True)`), and `embed_fn`'s output is assumed
    unit-normalized too -- exactly `SemanticNoveltyPenalty`'s own
    `embed_fn` contract, so cosine similarity reduces to a plain dot
    product on both sides.
    """
    import numpy as np
    emb = np.asarray(embed_fn(texts))
    template_sims = emb @ template_embeddings.T
    corpus_sims = emb @ corpus_embeddings.T
    return (template_sims.max(axis=1).tolist(),
           corpus_sims.max(axis=1).tolist())


def sweep(name: str, scores_by_group: Dict[str, List[float]],
         negative_scores: List[float], thresholds: List[float]) -> List[Dict]:
    rows = []
    for t in thresholds:
        detection = {
            g: (sum(1 for s in scores if s > t) / len(scores) if scores else None)
            for g, scores in scores_by_group.items()
        }
        fpr = (sum(1 for s in negative_scores if s > t) / len(negative_scores)
              if negative_scores else None)
        rows.append({"threshold": t, "detection": detection, "fpr": fpr})
    return rows


def pick_threshold(rows: List[Dict], target_fpr: float) -> Dict:
    """Lowest threshold clearing `fpr <= target_fpr` -- maximizes
    detection subject to the FPR ceiling, since both detection and FPR
    are (empirically, for a well-separated scorer) monotonically
    non-increasing in threshold."""
    candidates = [r for r in rows if r["fpr"] is not None and r["fpr"] <= target_fpr]
    if not candidates:
        return rows[-1]  # nothing clears the target -- report the strictest available
    return min(candidates, key=lambda r: r["threshold"])


# ------------------------------------------------------- EXP-011 (windowed)


def _parse_padding_reps(s: str) -> List[int]:
    """Parses `--padding-reps` ('0,5,20,50') into a sorted, deduped list of
    non-negative ints. Raises ValueError with a clear message on a
    malformed value rather than an opaque int() traceback three frames
    down."""
    try:
        reps = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    except ValueError as e:
        raise ValueError(
            "validate_semantic_novelty: --padding-reps must be a "
            "comma-separated list of non-negative integers (e.g. "
            "'0,5,20,50'), got %r" % (s,)) from e
    if not reps or any(r < 0 for r in reps):
        raise ValueError(
            "validate_semantic_novelty: --padding-reps must contain at "
            "least one non-negative integer, got %r" % (s,))
    return reps


def pad_with_filler(text: str, reps: int) -> str:
    """`reps` repetitions of WINDOW_PADDING_FILLER prepended before `text`
    -- the audited dilution-exploit shape (env/semantic_novelty.py's module
    docstring; env/tests/test_semantic_novelty.py's
    TestWindowedSemanticDilution._exploit). `reps<=0` returns `text`
    unchanged (the whole-text-equivalent, unpadded case)."""
    if reps <= 0:
        return text
    return (WINDOW_PADDING_FILLER + " ") * reps + text


def build_windowed_positive_records(templates: List[Dict], seed: int) -> List[Dict]:
    """This module's own `build_positive_records` (reskins at every
    EDIT_DEPTHS + the 5 hand paraphrases), PLUS a new VERBATIM class: each
    of the 25 real templates, completely unmodified. Verbatim-template
    positives are the exact shape the windowed tier exists to catch (a
    memorized template diluted by filler padding, not a reskin/paraphrase
    of one) and the registered bars explicitly need a "verbatim+padded
    detection" number distinct from the reskin/paraphrase groups."""
    records = build_positive_records(templates, seed)
    for t in templates:
        records.append({
            "template_id": t["id"], "kind": VERBATIM_KIND,
            "requested_depth": None, "actual_depth": None,
            "text": t["text"],
        })
    return records


def build_straddle_lengths(window_levels: List[Tuple[int, int, int]]) -> List[int]:
    """Target boundary-token lengths for the new multi-sentence negative
    class, straddling (below/at/above, per STRADDLE_DELTAS) EVERY
    window-ladder level's WIDTH -- computed from the REAL
    `SemanticNoveltyPenalty._window_levels` (never hardcoded), so this
    tracks the ladder automatically if the 25-template set or
    `window_growth` ever changes. The auditor's refinement this
    operationalizes: a negative shorter than every level's width never
    triggers windowing (SemanticNoveltyPenalty._window_texts's `n <=
    width: continue` skips that level entirely) and would silently never
    exercise the scoring path EXP-011 exists to calibrate."""
    lengths = set()
    for _cover, _stride, width in window_levels:
        for delta in STRADDLE_DELTAS:
            lengths.add(max(3, width + delta))
    return sorted(lengths)


def build_straddling_negatives(
    joke_pool: List[str], target_lengths: List[int], reps_per_length: int,
    seed: int,
) -> List[Dict]:
    """NEW negative class (the auditor's refinement to EXP-011's
    registered design): multi-sentence novel completions -- concatenations
    of 2-5 genuinely DIFFERENT corpus jokes (from `joke_pool`, already
    leak-excluded against the embedded corpus by the caller -- see
    `build_negatives`'s `exclude_texts`) plus non-joke prose (NONJOKE_PROSE)
    -- constructed at each length in `target_lengths` (boundary-tokens, per
    `_window_token_spans` -- the SAME tokenizer `SemanticNoveltyPenalty`'s
    own window ladder uses, so a length target here means exactly what it
    means to that ladder).

    Deterministic given `seed`: every (length, replicate-index) pair gets
    its own `random.Random` keyed on both, so re-running this script
    reproduces byte-identical negatives.

    Construction: sample k in [MIN_JOKES_PER_NEGATIVE,
    MAX_JOKES_PER_NEGATIVE] distinct jokes plus 1-3 NONJOKE_PROSE
    sentences, shuffle, join with spaces; if that draw doesn't already
    reach the target length, top up with more prose (cycled
    deterministically); finally TRUNCATE via `_window_token_spans`'s own
    spans to land at EXACTLY `L` boundary-tokens -- so the constructed
    negative's length precisely straddles the target, not just
    approximately (truncating at a span boundary, never mid-token)."""
    records = []
    for L in target_lengths:
        for i in range(reps_per_length):
            # A string seed, not a tuple -- Python 3.9+ deprecates hash-
            # based seeding of arbitrary objects (tuples included); str/
            # int/float/bytes stay fully supported and deterministic.
            rng = random.Random("%d:%d:%d" % (seed, L, i))
            k = min(rng.randint(MIN_JOKES_PER_NEGATIVE, MAX_JOKES_PER_NEGATIVE),
                    len(joke_pool))
            chosen_jokes = rng.sample(joke_pool, k)
            n_prose = rng.randint(1, min(3, len(NONJOKE_PROSE)))
            chosen_prose = rng.sample(NONJOKE_PROSE, n_prose)
            parts = chosen_jokes + chosen_prose
            rng.shuffle(parts)
            text = " ".join(parts)
            cycle = 0
            while len(_window_token_spans(text)) < L:
                text = text + " " + NONJOKE_PROSE[cycle % len(NONJOKE_PROSE)]
                cycle += 1
            spans = _window_token_spans(text)
            text = text[:spans[L - 1][1]]
            records.append({
                "target_length": L,
                "actual_length": len(_window_token_spans(text)),
                "n_jokes": k, "n_prose": n_prose, "text": text,
            })
    return records


def windowed_semantic_scores(texts: List[str], term: SemanticNoveltyPenalty,
                             embed_fn) -> List[float]:
    """Max-over-windows cosine-similarity score for each text in `texts`,
    against `term._template_embeddings` (templates-only reference set --
    apples-to-apples with the whole-text validation's own
    `template_only_scores`, and with `term.reference == "templates"`'s own
    default). Reuses `SemanticNoveltyPenalty._window_texts` directly
    (constructed on `term`) for EXACT scoring parity with the shipped
    windowed `__call__` path (env/semantic_novelty.py's windowed branch) --
    this function deliberately does NOT re-implement windowing; it only
    strips the threshold/ramp step, since a threshold sweep needs the raw
    score, not a pre-thresholded penalty. One embed_fn call per text (whole
    text first, then every window), matching that class's own performance
    contract exactly."""
    import numpy as np
    scores = []
    for text in texts:
        candidates = [text] + term._window_texts(text)
        embeddings = np.asarray(embed_fn(candidates))
        scores.append(float((embeddings @ term._template_embeddings.T).max()))
    return scores


def run_windowed_validation(args: argparse.Namespace) -> None:
    """EXP-011: re-sweep the semantic-novelty threshold for MAX-OVER-
    WINDOWS scoring (`SemanticNoveltyPenalty(windowed=True)`), per the
    registered design in EXPERIMENT_LOG.md's "Windowed novelty tiers"
    entry and its auditor's refinement (see this module's EXP-011
    constants block above for the full design rationale). Writes its own
    report.json to `args.out_dir` (separate from the whole-text
    validation's `experiment-runs/2026-07-17-semantic-novelty-validation/`)
    and prints the four registered bars (PASS/FAIL) plus the recommended
    threshold. Does NOT touch, re-run, or regress the whole-text
    validation path below (`main()` branches to this function BEFORE any
    of that code runs -- see `main`'s `if args.windowed` guard)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    padding_reps = _parse_padding_reps(args.padding_reps)
    corpus_dir = Path(args.corpus_dir).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = ("python3 -m env.validate_semantic_novelty --windowed "
              "--padding-reps %s --corpus-dir %s --out-dir %s "
              "--corpus-cap %d" %
              (args.padding_reps, args.corpus_dir, args.out_dir,
               args.corpus_cap))
    print("command: %s" % command)
    print("padding_reps: %r" % padding_reps)

    # ---- 1. templates + a manual corpus scan (for build_negatives's
    #         exclude_texts leak-prevention -- see EXP-009's own
    #         build_negatives docstring). SemanticNoveltyPenalty below
    #         re-scans internally at construction (same seed/cap -> the
    #         identical deterministic sample), which is a small amount of
    #         duplicated I/O traded for reusing the class's own
    #         construction path verbatim rather than hand-rolling a
    #         second, potentially-diverging one. ----
    templates = load_templates(corpus_dir)
    assert len(templates) == 25, (
        "expected the real 25-template file at %s, got %d templates" %
        (corpus_dir, len(templates)))

    remaining_cap = max(args.corpus_cap - len(templates), 0)
    sampled_texts, n_corpus_scanned = _reservoir_sample_jokes(
        corpus_dir, cap=remaining_cap, seed=20260717)
    corpus_texts_set = set(t["text"] for t in templates) | set(sampled_texts)

    # ---- 2. positives: reskins/paraphrases (reused generator) + verbatim ----
    positives = build_windowed_positive_records(templates, RESKIN_SEED)

    # ---- 3. negatives: baseline 100 (reused, unpadded) + straddling class ----
    baseline_negatives, base_n_scanned, base_n_leaks = build_negatives(
        corpus_dir, NEGATIVE_SAMPLE_N, NEGATIVE_SAMPLE_SEED,
        exclude_texts=corpus_texts_set)
    straddle_pool, pool_n_scanned, pool_n_leaks = build_negatives(
        corpus_dir, STRADDLE_POOL_N, STRADDLE_POOL_SEED,
        exclude_texts=corpus_texts_set)

    print("windowed positives: %d reskins (%d templates x %d depths) + %d "
         "paraphrases + %d verbatim = %d total" %
         (25 * len(EDIT_DEPTHS), 25, len(EDIT_DEPTHS), len(HAND_PARAPHRASES),
          len(templates), len(positives)))
    print("windowed negatives: %d baseline (reused from EXP-009) + %d "
         "straddle-pool candidates scanned" %
         (len(baseline_negatives), len(straddle_pool)))

    # ---- 4. real model, construct SemanticNoveltyPenalty(windowed=True)
    #         ONCE. This embeds the (templates + sampled) corpus with the
    #         real model and derives the window ladder from the REAL 25
    #         templates' boundary-tokenizer lengths -- every windowed
    #         score below reuses `term._window_texts` and
    #         `term._template_embeddings` directly (windowed_semantic_
    #         scores), never re-implemented here. ----
    t0 = time.perf_counter()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model_load_seconds = time.perf_counter() - t0

    def _embed_fn(texts):
        return model.encode(list(texts), batch_size=256,
                            normalize_embeddings=True, show_progress_bar=False)

    t1 = time.perf_counter()
    term = SemanticNoveltyPenalty(
        corpus_dir=corpus_dir, embed_fn=_embed_fn, windowed=True,
        reference="templates", corpus_cap=args.corpus_cap)
    construction_seconds = time.perf_counter() - t1
    print("window ladder (cover, stride, width): %r" % term._window_levels)

    straddle_lengths = build_straddle_lengths(term._window_levels)
    print("straddle target lengths (boundary-tokens): %r" % straddle_lengths)
    straddle_negative_records = build_straddling_negatives(
        straddle_pool, straddle_lengths, STRADDLE_REPS_PER_LENGTH,
        seed=STRADDLE_POOL_SEED)

    # ---- 5. score everything, reusing windowed_semantic_scores throughout ----
    t2 = time.perf_counter()

    pad_pairs = [(rec, reps) for rec in positives for reps in padding_reps]
    padded_texts = [pad_with_filler(rec["text"], reps) for rec, reps in pad_pairs]
    padded_scores_flat = windowed_semantic_scores(padded_texts, term, _embed_fn)
    for rec, reps in pad_pairs:
        rec.setdefault("padded_scores", {})
    for (rec, reps), score in zip(pad_pairs, padded_scores_flat):
        rec["padded_scores"][reps] = score

    baseline_negative_scores = windowed_semantic_scores(
        baseline_negatives, term, _embed_fn)
    straddle_texts = [r["text"] for r in straddle_negative_records]
    straddle_scores = windowed_semantic_scores(straddle_texts, term, _embed_fn)
    for rec, score in zip(straddle_negative_records, straddle_scores):
        rec["windowed_score"] = score
    expanded_negative_scores = baseline_negative_scores + straddle_scores

    scoring_seconds = time.perf_counter() - t2
    print("windowed scoring: %d positive-embeddings x %d pad levels + %d "
         "negatives (%d baseline + %d straddling) in %.3fs" %
         (len(positives), len(padding_reps), len(expanded_negative_scores),
          len(baseline_negative_scores), len(straddle_scores), scoring_seconds))

    # ---- 6. sweep thresholds. Groups are "kind@pad{reps}" (e.g.
    #         "verbatim@pad20") so the sweep table exposes detection at
    #         every (positive-kind, padding-level) cell, not just an
    #         aggregate. ----
    def group_key(rec):
        if rec["kind"] == "paraphrase":
            return "paraphrase"
        if rec["kind"] == VERBATIM_KIND:
            return VERBATIM_KIND
        return "depth_%d" % rec["actual_depth"]

    scores_by_group: Dict[str, List[float]] = {}
    for rec in positives:
        g = group_key(rec)
        for reps, score in rec["padded_scores"].items():
            scores_by_group.setdefault("%s@pad%d" % (g, reps), []).append(score)

    windowed_sweep = sweep("windowed_semantic", scores_by_group,
                          expanded_negative_scores, THRESHOLD_GRID)
    windowed_recommended = pick_threshold(windowed_sweep, TARGET_FPR)
    t = windowed_recommended["threshold"]

    # ---- 7. the four registered bars, at the recommended threshold ----
    bar1_fpr = windowed_recommended["fpr"]
    bar1_pass = bar1_fpr is not None and bar1_fpr <= TARGET_FPR

    verbatim_padded_scores = [
        score for rec in positives if rec["kind"] == VERBATIM_KIND
        for reps, score in rec["padded_scores"].items() if reps > 0]
    bar2_verbatim_padded_detection = (
        sum(1 for s in verbatim_padded_scores if s > t) / len(verbatim_padded_scores)
        if verbatim_padded_scores else None)
    bar2_pass = (bar2_verbatim_padded_detection is not None and
                bar2_verbatim_padded_detection >= BAR_VERBATIM_PADDED_DETECTION_MIN)

    if 5 in padding_reps and 50 in padding_reps:
        det_pad5 = (sum(1 for rec in positives if rec["padded_scores"][5] > t)
                    / len(positives))
        det_pad50 = (sum(1 for rec in positives if rec["padded_scores"][50] > t)
                     / len(positives))
        bar3_padding_invariance_delta = abs(det_pad5 - det_pad50)
        bar3_pass = bar3_padding_invariance_delta <= BAR_PADDING_INVARIANCE_MAX_DELTA
    else:
        det_pad5 = det_pad50 = bar3_padding_invariance_delta = None
        bar3_pass = None  # not evaluable with this --padding-reps set

    whole_text_paraphrase_detection = None
    bar4_windowed_unpadded_paraphrase_detection = None
    bar4_pass = None
    if WHOLE_TEXT_REPORT_PATH.exists() and 0 in padding_reps:
        with open(WHOLE_TEXT_REPORT_PATH) as f:
            whole_text_report = json.load(f)
        whole_text_paraphrase_detection = (
            whole_text_report["semantic"]["at_recommended_threshold"]
            ["detection"]["paraphrase"])
        paraphrase_pad0_scores = [rec["padded_scores"][0] for rec in positives
                                  if rec["kind"] == "paraphrase"]
        bar4_windowed_unpadded_paraphrase_detection = (
            sum(1 for s in paraphrase_pad0_scores if s > t) / len(paraphrase_pad0_scores)
            if paraphrase_pad0_scores else None)
        bar4_pass = (bar4_windowed_unpadded_paraphrase_detection is not None and
                    bar4_windowed_unpadded_paraphrase_detection >=
                    whole_text_paraphrase_detection)

    bars = {
        "fpr_le_0.05_on_expanded_negatives": {
            "pass": bar1_pass, "fpr": bar1_fpr, "target": TARGET_FPR},
        "verbatim_padded_detection_ge_0.95": {
            "pass": bar2_pass, "detection": bar2_verbatim_padded_detection,
            "target": BAR_VERBATIM_PADDED_DETECTION_MIN,
            "n": len(verbatim_padded_scores)},
        "padding_invariance_le_2pp_between_5_and_50_reps": {
            "pass": bar3_pass, "delta": bar3_padding_invariance_delta,
            "detection_pad5": det_pad5, "detection_pad50": det_pad50,
            "target_max_delta": BAR_PADDING_INVARIANCE_MAX_DELTA},
        "paraphrase_detection_not_below_exp009_whole_text_unpadded": {
            "pass": bar4_pass,
            "windowed_unpadded": bar4_windowed_unpadded_paraphrase_detection,
            "whole_text_operating_point": whole_text_paraphrase_detection},
    }
    all_bars_pass = all(b["pass"] for b in bars.values() if b["pass"] is not None) \
        and all(b["pass"] is not None for b in bars.values())
    decision = ("full_replacement_candidate" if all_bars_pass
               else "dilution_only_complement")

    # ---- 8. print + write report.json ----
    print()
    print("=== EXP-011 windowed semantic novelty -- RECOMMENDED "
         "threshold=%.2f (lowest clearing FPR<=%.2f on the expanded "
         "negative set) ===" % (t, TARGET_FPR))
    for name, b in bars.items():
        print("  [%s] %s -> %r" %
             ("PASS" if b["pass"] else ("FAIL" if b["pass"] is False else "N/A"),
              name, b))
    print("  decision: %s" % decision)

    fpr_curve = [r for r in windowed_sweep if abs(r["threshold"] - t) <= 0.05]

    report = {
        "experiment": "EXP-011-windowed-semantic-novelty-threshold",
        "command": command,
        "padding_reps": padding_reps,
        "window_levels": term._window_levels,
        "straddle_target_lengths": straddle_lengths,
        "test_set": {
            "n_positives": len(positives),
            "n_templates": len(templates),
            "edit_depths_requested": list(EDIT_DEPTHS),
            "n_hand_paraphrases": len(HAND_PARAPHRASES),
            "n_verbatim": len(templates),
            "n_baseline_negatives": len(baseline_negatives),
            "baseline_negative_sample_seed": NEGATIVE_SAMPLE_SEED,
            "baseline_negative_rows_scanned": base_n_scanned,
            "baseline_n_excluded_leaks": base_n_leaks,
            "n_straddling_negatives": len(straddle_negative_records),
            "straddle_pool_n": len(straddle_pool),
            "straddle_pool_seed": STRADDLE_POOL_SEED,
            "straddle_reps_per_length": STRADDLE_REPS_PER_LENGTH,
            "straddle_pool_n_scanned": pool_n_scanned,
            "straddle_pool_n_excluded_leaks": pool_n_leaks,
        },
        "wall_time": {
            "model_load_seconds": model_load_seconds,
            "term_construction_seconds": construction_seconds,
            "scoring_seconds": scoring_seconds,
            "corpus_cap_used": args.corpus_cap,
        },
        "recommended_threshold": t,
        "at_recommended_threshold": windowed_recommended,
        "fpr_curve_around_recommended": fpr_curve,
        "bars": bars,
        "decision": decision,
        "full_sweep": windowed_sweep,
        "positives_detail": positives,
        "negatives_detail": {
            "baseline": [{"text": txt, "windowed_score": s}
                        for txt, s in zip(baseline_negatives, baseline_negative_scores)],
            "straddling": straddle_negative_records,
        },
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % report_path)


# --------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=str(REAL_CORPUS_DIR))
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--corpus-cap", type=int, default=CORPUS_CAP_FOR_TIMING)
    ap.add_argument("--windowed", action="store_true",
                    help="EXP-011: validate max-over-sliding-windows "
                         "semantic-novelty scoring (SemanticNoveltyPenalty"
                         "(windowed=True)) instead of the shipped "
                         "whole-text mode -- see run_windowed_validation's "
                         "docstring. Writes a separate report.json and "
                         "never touches the whole-text path below.")
    ap.add_argument("--padding-reps", default=DEFAULT_PADDING_REPS,
                    help="Comma-separated filler-repetition levels "
                         "('0,5,20,50') each positive is embedded at, in "
                         "--windowed mode only. Ignored otherwise.")
    args = ap.parse_args()

    if args.windowed:
        run_windowed_validation(args)
        return

    corpus_dir = Path(args.corpus_dir).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = "python3 -m env.validate_semantic_novelty --corpus-dir %s " \
             "--out-dir %s --corpus-cap %d" % (args.corpus_dir, args.out_dir,
                                              args.corpus_cap)
    print("command: %s" % command)

    # ---- 1. load real templates + reservoir-sample the general corpus.
    #         This has to happen BEFORE build_negatives: negatives must be
    #         excluded from whatever the corpus-comparison sample will
    #         contain, or a "negative" can leak in as an exact duplicate
    #         of a corpus-embedded row (see build_negatives's
    #         exclude_texts docstring -- this is the exact bug the
    #         2026-07-17 corrected pass fixes). The scan itself only
    #         needs `_reservoir_sample_jokes` (pure stdlib), so it doesn't
    #         need numpy/sentence_transformers loaded yet. ----
    templates = load_templates(corpus_dir)
    assert len(templates) == 25, (
        "expected the real 25-template file at %s, got %d templates" %
        (corpus_dir, len(templates)))

    remaining_cap = max(args.corpus_cap - len(templates), 0)
    t3 = time.perf_counter()
    sampled_texts, n_corpus_scanned = _reservoir_sample_jokes(
        corpus_dir, cap=remaining_cap, seed=20260717)
    t4 = time.perf_counter()
    scan_seconds = t4 - t3
    corpus_texts = [t["text"] for t in templates] + sampled_texts
    corpus_texts_set = set(corpus_texts)

    positives = build_positive_records(templates, RESKIN_SEED)
    negatives, neg_n_scanned, n_excluded_leaks = build_negatives(
        corpus_dir, NEGATIVE_SAMPLE_N, NEGATIVE_SAMPLE_SEED,
        exclude_texts=corpus_texts_set)
    print("test set: %d positives (%d reskins across %d templates x %d "
          "depths + %d hand paraphrases), %d negatives (from %d "
          "commercial-safe rows scanned, %d exact-corpus-leak "
          "duplicate(s) excluded and redrawn)" %
          (len(positives), 25 * len(EDIT_DEPTHS), 25, len(EDIT_DEPTHS),
           len(HAND_PARAPHRASES), len(negatives), neg_n_scanned,
           n_excluded_leaks))

    # ---- 2. n-gram scores (the existing, shipped scorer; templates-only
    #         reference set -- see best_ngram_score) ----
    t0 = time.perf_counter()
    for rec in positives:
        rec["ngram_score"] = best_ngram_score(rec["text"], templates)
    negative_ngram_scores = [best_ngram_score(t, templates) for t in negatives]
    ngram_wall = time.perf_counter() - t0
    print("n-gram scoring: %d items in %.3fs" %
          (len(positives) + len(negatives), ngram_wall))

    # ---- 3. semantic scores (the new scorer), with phase-separated timing ----
    t0 = time.perf_counter()
    import numpy as np
    from sentence_transformers import SentenceTransformer
    t1 = time.perf_counter()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    t2 = time.perf_counter()
    import_seconds = t1 - t0
    model_load_seconds = t2 - t1

    t5 = time.perf_counter()
    corpus_embeddings = np.asarray(model.encode(
        corpus_texts, batch_size=256, normalize_embeddings=True,
        show_progress_bar=False))
    t6 = time.perf_counter()
    corpus_embed_seconds = t6 - t5
    embed_throughput = len(corpus_texts) / corpus_embed_seconds

    # Templates-only reference slice: corpus_texts is exactly
    # [template_texts...] + [sampled_texts...] in that order (see above),
    # so this is a FREE slice of the embeddings already computed -- no
    # second encode() call. THIS is the reference set threshold
    # calibration uses below, mirroring best_ngram_score's templates-only
    # reference exactly (apples-to-apples with the n-gram baseline -- see
    # module docstring for why the ORIGINAL full-corpus reference was
    # wrong).
    template_embeddings = corpus_embeddings[:len(templates)]

    print("semantic corpus construction: import=%.3fs model_load=%.3fs "
          "scan=%.3fs (%d rows) embed=%.3fs (%d items, %.1f items/sec)" %
          (import_seconds, model_load_seconds, scan_seconds, n_corpus_scanned,
           corpus_embed_seconds, len(corpus_texts), embed_throughput))

    full_corpus_rows = n_corpus_scanned  # every jokes.jsonl row scanned,
                                        # regardless of the cap used above
    estimated_full_embed_seconds = full_corpus_rows / embed_throughput
    print("extrapolated: embedding ALL %d scanned rows at this throughput "
          "would take ~%.1fs (~%.1f min) vs. %.1fs for the capped default" %
          (full_corpus_rows, estimated_full_embed_seconds,
           estimated_full_embed_seconds / 60.0, corpus_embed_seconds))

    def _embed_fn(texts):
        return model.encode(texts, batch_size=256, normalize_embeddings=True,
                            show_progress_bar=False)

    t7 = time.perf_counter()
    pos_texts = [r["text"] for r in positives]
    pos_semantic_scores, pos_general_corpus_scores = semantic_scores(
        pos_texts, _embed_fn, template_embeddings, corpus_embeddings)
    for rec, score, gscore in zip(positives, pos_semantic_scores,
                                  pos_general_corpus_scores):
        rec["semantic_score"] = score
        rec["general_corpus_max_sim"] = gscore
    negative_semantic_scores, negative_general_corpus_scores = semantic_scores(
        negatives, _embed_fn, template_embeddings, corpus_embeddings)
    query_embed_seconds = time.perf_counter() - t7
    print("semantic query scoring: %d items in %.3fs" %
          (len(positives) + len(negatives), query_embed_seconds))

    # ---- 4. group scores by edit depth / paraphrase, sweep thresholds ----
    def group_key(rec):
        if rec["kind"] == "paraphrase":
            return "paraphrase"
        return "depth_%d" % rec["actual_depth"]

    ngram_groups: Dict[str, List[float]] = {}
    semantic_groups: Dict[str, List[float]] = {}
    for rec in positives:
        g = group_key(rec)
        ngram_groups.setdefault(g, []).append(rec["ngram_score"])
        semantic_groups.setdefault(g, []).append(rec["semantic_score"])

    ngram_sweep = sweep("ngram", ngram_groups, negative_ngram_scores,
                       THRESHOLD_GRID)
    semantic_sweep = sweep("semantic", semantic_groups,
                          negative_semantic_scores, THRESHOLD_GRID)

    # n-gram's ACTUAL shipped operating point (CorpusNoveltyPenalty's
    # default threshold=0.35) -- what does the currently-deployed defense
    # already catch, not just the sweep's hypothetical best case.
    ngram_shipped_row = min(ngram_sweep,
                           key=lambda r: abs(r["threshold"] - NGRAM_SHIPPED_THRESHOLD))

    semantic_recommended = pick_threshold(semantic_sweep, TARGET_FPR)

    print()
    print("=== n-gram (trigram-Jaccard) at SHIPPED default threshold=%.2f ===" %
         ngram_shipped_row["threshold"])
    for g in sorted(ngram_groups):
        print("  %-12s detection=%.3f" % (g, ngram_shipped_row["detection"][g]))
    print("  %-12s FPR=%.3f" % ("negatives", ngram_shipped_row["fpr"]))

    print()
    print("=== semantic (cosine, all-MiniLM-L6-v2), TEMPLATES-ONLY "
         "reference set (apples-to-apples w/ n-gram above), RECOMMENDED "
         "threshold=%.2f (lowest clearing FPR<=%.2f) ===" %
         (semantic_recommended["threshold"], TARGET_FPR))
    for g in sorted(semantic_groups):
        print("  %-12s detection=%.3f" % (g, semantic_recommended["detection"][g]))
    print("  %-12s FPR=%.3f" % ("negatives", semantic_recommended["fpr"]))

    # Informational only -- general_corpus_max_sim is NEVER used for
    # threshold selection (see module docstring / semantic_scores'
    # docstring for why: it's confounded by the general corpus's own
    # pervasive near-duplicate content).
    neg_general_sorted = sorted(negative_general_corpus_scores)
    n_neg = len(neg_general_sorted)
    print()
    print("=== general_corpus_max_sim (INFORMATIONAL ONLY, templates + "
         "capped general-corpus sample -- NOT used for calibration) ===")
    print("  negatives    mean=%.3f median=%.3f p90=%.3f max=%.3f" % (
        statistics.mean(neg_general_sorted),
        statistics.median(neg_general_sorted),
        neg_general_sorted[int(0.9 * (n_neg - 1))] if n_neg else float("nan"),
        max(neg_general_sorted) if neg_general_sorted else float("nan")))

    # ---- 5. write report.json ----
    report = {
        "experiment": "EXP-009-semantic-novelty-validation",
        "command": command,
        "test_set": {
            "n_positives": len(positives),
            "n_templates": len(templates),
            "edit_depths_requested": list(EDIT_DEPTHS),
            "n_hand_paraphrases": len(HAND_PARAPHRASES),
            "n_negatives": len(negatives),
            "negative_sample_seed": NEGATIVE_SAMPLE_SEED,
            "negative_rows_scanned": neg_n_scanned,
            "reskin_seed": RESKIN_SEED,
            "n_excluded_leaks": n_excluded_leaks,
        },
        "wall_time": {
            "import_seconds": import_seconds,
            "model_load_seconds": model_load_seconds,
            "corpus_scan_seconds": scan_seconds,
            "corpus_scan_rows": n_corpus_scanned,
            "corpus_embed_seconds": corpus_embed_seconds,
            "corpus_embed_n_items": len(corpus_texts),
            "corpus_embed_throughput_items_per_sec": embed_throughput,
            "query_embed_seconds": query_embed_seconds,
            "query_embed_n_items": len(positives) + len(negatives),
            "ngram_scoring_seconds": ngram_wall,
            "estimated_full_corpus_embed_seconds": estimated_full_embed_seconds,
            "corpus_cap_used": args.corpus_cap,
        },
        "ngram": {
            "shipped_default_threshold": NGRAM_SHIPPED_THRESHOLD,
            "at_shipped_default": ngram_shipped_row,
            "full_sweep": ngram_sweep,
        },
        "semantic": {
            "reference_set": "templates_only",  # apples-to-apples with
                                                # ngram's own reference set
                                                # -- see module docstring
            "target_fpr": TARGET_FPR,
            "recommended_threshold": semantic_recommended["threshold"],
            "at_recommended_threshold": semantic_recommended,
            "full_sweep": semantic_sweep,
        },
        "positives_detail": positives,
        "negatives_scores": {
            "ngram": negative_ngram_scores,
            "semantic": negative_semantic_scores,
            "general_corpus_max_sim": negative_general_corpus_scores,
        },
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % report_path)


if __name__ == "__main__":
    main()

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
from env.semantic_novelty import _reservoir_sample_jokes  # noqa: E402

REAL_CORPUS_DIR = Path("~/Experiments/good-humored-data/corpus").expanduser()
OUTPUT_DIR = _REPO_ROOT / "experiment-runs" / "2026-07-17-semantic-novelty-validation"

EDIT_DEPTHS = (1, 2, 4)
RESKIN_SEED = 20260717          # see generate_reskin's docstring
NEGATIVE_SAMPLE_SEED = 20260717
NEGATIVE_SAMPLE_N = 100
NGRAM_SHIPPED_THRESHOLD = 0.35  # CorpusNoveltyPenalty's actual default
CORPUS_CAP_FOR_TIMING = 50_000  # env.semantic_novelty.DEFAULT_CORPUS_CAP
THRESHOLD_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05..0.95
TARGET_FPR = 0.05               # CLAUDE.md-style success criterion

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


# --------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=str(REAL_CORPUS_DIR))
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--corpus-cap", type=int, default=CORPUS_CAP_FOR_TIMING)
    args = ap.parse_args()

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

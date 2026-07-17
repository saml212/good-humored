"""Embedding-based, paraphrase-robust novelty penalty -- the theory-grounded
fix for `CorpusNoveltyPenalty`'s documented worst exploit.

MOTIVATION (see `references/theory-operationalization.md` RQ4 and
`docs/THEORY-MAP.md`'s "priority translation #1" -- both read in full before
this module was written): `env/rewards.py`'s `CorpusNoveltyPenalty` compares
completions to the memorized-joke corpus via trigram-Jaccard, which is pure
surface n-gram overlap. Its own docstring documents, and
`env/tests/test_rewards.py::test_two_word_template_reskin_fully_evades_and_is_locked_in`
pins down as a regression test, that substituting ~2 content words in a
memorized template routinely drops trigram-Jaccard below threshold and
evades the penalty ENTIRELY (reward 0.0, not even a ramped partial penalty).
n-gram overlap fundamentally cannot see past a handful of substituted words
-- catching that requires semantic similarity. This module supplies that as
an ADDITIONAL, OPTIONAL reward tier (see `env/rewards.py`'s
`RewardConfig.semantic_novelty_weight`, default 0.0 -- off, a new tier, not
a silent behavior change to the existing stack).

GUARDED IMPORT, same pattern as `benchmark/label_space.py`: `numpy` and
`sentence_transformers` are imported lazily, inside `__init__`/`__call__`,
never at module level -- so this file stays importable with zero ML deps,
and (critically) so `env/rewards.py`'s own "provably pure-stdlib" default
import path is untouched: `reward_stack()`/`combined_reward()` only import
THIS module lazily, and only when `RewardConfig.semantic_novelty_weight !=
0.0` -- see that module's wiring. Importing `env.semantic_novelty` itself
never requires torch/sentence-transformers; only CONSTRUCTING a
`SemanticNoveltyPenalty` without an injected `embed_fn` does.

DEGRADED MODE IS THE OPPOSITE OF `LabelSpace`'s, ON PURPOSE. `LabelSpace`
silently degrades to an identity mapping when `sentence_transformers` is
missing -- fine for a topic-label canonicalization helper (worst case:
benchmark analysis metrics revert to pre-EXP-002 behavior). It is NOT fine
here: CLAUDE.md's own hard rule is that "an inert novelty layer silently
pretending to work is this project's documented nightmare," and this
module's entire reason to exist is closing the exact evasion a full audit
night was spent documenting. Construction therefore FAILS LOUDLY by default
(`SemanticNoveltyUnavailable`, an `ImportError` subclass) if neither a real
embedding backend nor an injected `embed_fn` is available. `allow_degraded=
True` is an explicit, deliberate escape (e.g. a fast test sweep, or an
operator who has consciously decided to accept the risk) -- and even then
it warns loudly, ONCE, at construction (not silently, not per-call), and
every subsequent call returns 0.0.

WARNING (RESOLVED 2026-07-17, history kept so the reasoning isn't lost) --
TEMPLATES-VS-GENERAL-CORPUS CONFLATION: an earlier version of this file
computed `__call__`'s similarity against `self._corpus_embeddings`
(templates PLUS up to `corpus_cap` sampled general-corpus rows)
UNCONDITIONALLY, while `DEFAULT_THRESHOLD` was calibrated against
similarity to the 25 memorized TEMPLATES ONLY (see that constant's
comment) -- a scope mismatch between what was validated and what
actually ran. The corrected validation found the general corpus is
pervasively internally near-duplicated (recycled/reworded reposts
unrelated to the 25 templates -- `_dedup_stats.json` confirms only
exact-string dedup was ever done), so genuinely NOVEL completions
routinely score 0.6-0.97 similarity against *something* in a 50K
general-corpus sample (see
experiment-runs/2026-07-17-semantic-novelty-validation/report.json's
`negatives_scores.general_corpus_max_sim`: median ~0.60, p90 ~0.83) --
far above `DEFAULT_THRESHOLD=0.38`. Enabling the term at the calibrated
threshold against that old implementation would have penalized most
novel completions, not just template paraphrases.

FIX: `SemanticNoveltyPenalty` now takes a `reference` constructor arg,
`"templates"` (DEFAULT) or `"corpus"`. Default `__call__` scores ONLY
against the 25 template embeddings -- the exact construct this term
exists to operationalize ("is this a paraphrase of a KNOWN memorized
joke," matching `CorpusNoveltyPenalty`'s n-gram tier's own
templates-only reference set), and the exact reference set
`DEFAULT_THRESHOLD=0.38` was calibrated against
(`env/validate_semantic_novelty.py`'s `semantic_scores` function,
`template_only_scores`). `reference="corpus"` is a separate, explicitly
opt-in mode measuring a DIFFERENT construct ("does this resemble
anything in a large, redundant scraped corpus," i.e. general
corpus-redundancy, not memorized-template paraphrase) -- it has NO
default threshold; constructing with `reference="corpus"` and no
`threshold` raises `ValueError` rather than silently reusing
`DEFAULT_THRESHOLD`, since that threshold is invalid for this
construct's similarity distribution (see numbers above).

WARNING -- FIXED 50K RESERVOIR IS A THEORETICAL BUT REAL OVERFIT SURFACE:
`corpus_cap`'s reservoir sample is deterministic (`DEFAULT_SAMPLE_SEED`),
so it is the SAME ~50K/1.2M rows for the life of a training run. A policy
under enough RL pressure could in principle learn to avoid similarity
specifically to THAT fixed sample without avoiding the ~1.15M un-sampled
corpus rows -- gaming the sample, not the underlying "don't reproduce a
memorized joke" objective. Low priority relative to the conflation above
(the 25 templates -- the actual highest-value target -- are always fully
included, never subject to sampling), but real: consider periodically
re-sampling the seed across training, or expanding the cap, if this term
is ever relied on for a long run.

PADDING/DILUTION -- CLOSED FOR WINDOWED MODE (2026-07-17 fix; the
original audit finding kept on record below). The exploit, empirically
confirmed against the real model + real corpus: prepending generic
filler sentences to a VERBATIM memorized template dilutes the whole-text
mean-pooled embedding until similarity drops below threshold -- ~15
repetitions of one filler sentence brought the penalty to ~7% severity,
~20 brought it to exactly 0.0, and it stayed evaded at any longer
padding. The same trick zeroed out CorpusNoveltyPenalty's exact-hash AND
trigram tiers at just ~5 repetitions. It attacked the
mean-pooled-embedding design itself, not the threshold.

STATUS OF THE FIX, per tier:
  - n-gram tier (`env.rewards.CorpusNoveltyPenalty`): CLOSED, ON BY
    DEFAULT (`windowed=True`) -- max-over-sliding-windows with a stride-1
    exact-alignment guarantee; the verbatim-inside-padding case now
    scores severity 1.0. See that class's docstring for the full design,
    the default-ON justification, and its residual risks.
  - semantic tier (this module): CLOSED IN WINDOWED MODE, which is
    OPT-IN (`SemanticNoveltyPenalty(..., windowed=True)`, default OFF)
    PENDING EXP-011 -- max cosine over sliding windows against the 25
    templates, keeping the templates-only reference and
    DEFAULT_THRESHOLD=0.38. The calibration-transfer ASSUMPTION: a
    window that contains the paraphrase, with bounded surrounding
    filler, approximates the paraphrase-alone embedding closely enough
    that the templates-only threshold still separates (the window
    ladder guarantees every candidate span is fully contained in a
    window it occupies >=1/3 of -- see SemanticNoveltyPenalty's
    docstring -- and the audit measured threshold-crossing only below
    ~10% joke fraction, so 1/3 leaves margin). EMPIRICAL SPOT-CHECK
    (2026-07-17, real model + real templates, offline): the assumption
    HOLDS on the detection side -- the 20-rep dilution exploit scores
    severity ~0.96 windowed vs exactly 0.0 whole-text -- but FAILS on
    the false-positive side: a genuinely novel multi-sentence
    completion's best WINDOW similarity against the 25 templates came
    out ~0.55 (vs ~0.3 whole-text), i.e. short windows of arbitrary
    text sit closer to short joke sentences in MiniLM space than whole
    completions do, and at threshold 0.38 windowed mode DOES penalize
    novel long completions. EXP-011 (re-run
    env/validate_semantic_novelty.py's positives/negatives through
    windowed scoring, positives embedded in 0/5/20/50 reps of padding,
    negatives INCLUDING multi-sentence novel completions -- the
    negative class windowing newly exposes -- then re-sweep for the
    lowest threshold with FPR<=0.05; expect it to land well above
    0.38) is REQUIRED before any windowed threshold is trusted. Until
    then the shipped, validated whole-text behavior is preserved
    exactly by the default, and enabling windowed=True at
    DEFAULT_THRESHOLD is known-miscalibrated, not merely unvalidated.

RESIDUAL RISKS WINDOWING DOES NOT SOLVE (both tiers): (1) a memorized
joke or paraphrase interleaved word-by-word with filler has no
contiguous span for any window to isolate -- every window is diluted at
every scale and trigrams are broken by construction; still open, and
only accidentally mitigated by composition (word-by-word interleaving is
exactly what SelfRepetitionPenalty/comprehensibility/judge punish);
(2) padding past the window caps (`max_scan_tokens` for n-gram,
`max_windows`' coverage frontier here) evades the window tiers -- both
caps and their exact frontiers are documented on their constructors;
(3) the n-gram tier's 2-word-reskin gap is unchanged by windowing (a
padded reskin now scores exactly what the bare reskin scores -- uniform,
not fixed); the semantic tier remains the reskin/paraphrase defense.
"""

import json
import random
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from benchmark.joke_novelty import load_templates
from env.rewards import _contents, _normalize, _window_token_spans

_MODEL_NAME = "all-MiniLM-L6-v2"

# Corpus embedding cap and sampling seed -- see `_reservoir_sample_jokes`'s
# docstring for the exact deterministic sampling scheme, and
# `env/validate_semantic_novelty.py` for the empirical wall-time numbers
# this default cap was chosen against (embedding the FULL ~1.2M-joke corpus
# was measured too slow for a per-training-step-relevant construction cost;
# see that script's report.json / this project's EXP-009 write-up for the
# exact number).
DEFAULT_CORPUS_CAP = 50_000
DEFAULT_SAMPLE_SEED = 20260717

# Calibrated 2026-07-17 (CORRECTED pass) against
# experiment-runs/2026-07-17-semantic-novelty-validation/report.json:
# positives = the 25 real ChatGPT templates reskinned at edit depths 1/2/4
# content words (deterministic substitution, see validate_semantic_novelty.
# generate_reskin) plus 5 hand-written full paraphrases (the hard case);
# negatives = 100 deterministically-sampled, genuinely different jokes from
# the commercial-safe corpus, with exact-text duplicates of the embedded
# corpus sample excluded and redrawn (`build_negatives`'s `exclude_texts`
# -- see that function's docstring for the held-out-set leak this closes).
# Threshold chosen as the LOWEST value clearing false-positive-rate <=
# 0.05 on the negative set, scored ONLY against the 25 templates
# (`semantic_scores`' `template_only_scores` -- apples-to-apples with
# `best_ngram_score`'s own templates-only reference set), maximizing
# detection subject to that FPR ceiling -- see that report's full
# threshold sweep for every other operating point.
#
# SUPERSEDES TWO EARLIER, WRONG VALUES, both left here as a record of what
# went wrong rather than quietly overwritten:
#   - 0.94: the FIRST (buggy) validation run's recommendation, calibrated
#     against `corpus_embeddings` (templates + a 50K general-corpus
#     sample) instead of templates alone. That reference set is pervasively
#     internally near-duplicated (recycled/reworded jokes unrelated to the
#     25 templates -- confirmed via `_dedup_stats.json`, which shows only
#     exact-string dedup was ever done on the corpus), which inflated
#     "negative" similarity scores and forced the FPR<=0.05 threshold pick
#     above every genuine positive's score -- 0.0 detection at every
#     depth, an ARTIFACT of test-set construction, not a real result. 3/100
#     negatives were additionally found to be EXACT duplicates of rows
#     independently sampled into that same corpus set (a held-out-set
#     leak now closed by `build_negatives`'s `exclude_texts`).
#   - 0.60: a placeholder written into this file before the first
#     validation run ever completed (never actually calibrated against a
#     real report.json) -- an artifact of this module and the validation
#     script being authored in the same session that died mid-run.
#
# Re-run env/validate_semantic_novelty.py and update this comment (and the
# constant) if the corpus, the reskin wordlist, or the embedding model ever
# change.
#
# ONLY valid for `reference="templates"` (the default -- see
# SemanticNoveltyPenalty's `reference` param). It is NOT a valid threshold
# for `reference="corpus"`'s similarity distribution -- see the module
# docstring's WARNING for the numbers; `reference="corpus"` requires its
# own explicit `threshold` and has no default.
DEFAULT_THRESHOLD = 0.38


class SemanticNoveltyUnavailable(ImportError):
    """Raised at `SemanticNoveltyPenalty` construction when neither
    `numpy` + `sentence_transformers` NOR an injected `embed_fn` is
    available, and `allow_degraded` was not explicitly set. Subclasses
    `ImportError` so a caller doing broad `except ImportError:` handling
    for optional-ML-dependency setup code catches this too."""


def _reservoir_sample_jokes(
    corpus_dir: Path, cap: int, seed: int
) -> Tuple[List[str], int]:
    """Single-pass, deterministic reservoir sample (Algorithm R) of up to
    `cap` joke texts from every `jokes.jsonl` under `corpus_dir`, treated
    as ONE logical stream in sorted-file-path-then-line order -- so the
    same `seed` + corpus contents always yields the same sample,
    independent of filesystem directory-iteration order and independent of
    how many `jokes.jsonl` files exist. Returns `(sampled_texts,
    n_scanned)` so callers can report exact coverage (sampled / scanned),
    not just the cap.

    Deliberately single-pass (does not load the whole corpus into memory
    first, then sample) -- CLAUDE.md's own hardware note is a local dev
    box; a ~1.2M-line corpus loaded whole before sampling is needless
    memory pressure when reservoir sampling gets the same guarantee
    (uniform sample of size `cap` from a stream of unknown length) in
    O(cap) memory.

    `cap <= 0` scans (for an accurate `n_scanned` count) but samples
    nothing -- the caller (`SemanticNoveltyPenalty.__init__`) hits this
    when the always-kept template set already fills or exceeds the
    configured corpus cap.
    """
    rng = random.Random(seed)
    reservoir: List[str] = []
    n_scanned = 0
    for path in sorted(corpus_dir.rglob("jokes.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                n_scanned += 1
                if cap <= 0:
                    continue
                if len(reservoir) < cap:
                    reservoir.append(text)
                else:
                    j = rng.randint(0, n_scanned - 1)
                    if j < cap:
                        reservoir[j] = text
    return reservoir, n_scanned


class SemanticNoveltyPenalty:
    """"is it not a joke the internet already told, EVEN IF reworded" --
    embedding-based sibling of `env.rewards.CorpusNoveltyPenalty`. Weight
    -1.5 by default (mirrors that class's default, and the recommended
    weight documented on `RewardConfig.semantic_novelty_weight`).

    Score = max cosine similarity between the completion's embedding and
    every REFERENCE-SET embedding, computed once per `__call__` batch via a
    single `(n_completions, n_reference)` matrix multiply -- not a python
    loop over the reference set per completion. Penalty ramps linearly from
    `threshold` (0 penalty) to 1.0 similarity (full `weight`), identical in
    shape and rationale to `CorpusNoveltyPenalty`'s template tier: at or
    below `threshold` the contribution is exactly 0.0 (below-threshold
    similarity is treated as "not the same joke," not a small residual
    penalty); above it, severity ramps linearly so a near-verbatim reskin
    (similarity close to 1.0) costs close to the full `weight` while a
    borderline case just above threshold costs little.

    `reference` selects WHICH embedding set that similarity is computed
    against -- this is the one load-bearing knob this class exposes, and
    the two modes measure genuinely different constructs:

      "templates" (DEFAULT): score against the 25 memorized-template
        embeddings ONLY. This is what this term exists to operationalize
        -- "is this a paraphrase of a KNOWN memorized joke" -- matching
        `CorpusNoveltyPenalty`'s own n-gram tier's reference set exactly
        (apples-to-apples), and the exact reference set
        `DEFAULT_THRESHOLD=0.38` was calibrated against (see that
        constant's comment). Use this unless you have a specific,
        separately-calibrated reason not to.

      "corpus": score against templates PLUS up to `corpus_cap`
        reservoir-sampled general-corpus rows. This measures a DIFFERENT
        construct -- "does this resemble anything in a large, redundant
        scraped corpus" (general corpus-redundancy, not
        memorized-template paraphrase; see module docstring's WARNING for
        why the general corpus's own pervasive near-duplication makes
        this a much noisier, higher-baseline signal). Has NO default
        threshold: constructing with `reference="corpus"` and no
        `threshold` raises `ValueError` rather than silently reusing
        `DEFAULT_THRESHOLD`, which is invalid for this construct. Pass an
        explicit, separately-calibrated `threshold` if you deliberately
        want this mode.

    Regardless of `reference`, corpus embeddings are precomputed ONCE at
    construction from `corpus_dir` (same directory convention as
    `CorpusNoveltyPenalty`: every `jokes.jsonl` found anywhere under it,
    plus a root-level `chatgpt-25-templates.jsonl`) -- the general-corpus
    scan/embed always happens (even in default `"templates"` mode) so
    switching `reference` never requires reconstruction, and construction
    time is dominated by the one-time embedding cost either way. The 25
    templates are ALWAYS included in full, never subject to sampling --
    they are the highest-value target this module exists to defend, and
    dropping one via random sampling would be a silent, undocumented
    coverage gap in exactly the case that matters most. Everything else
    (the general joke corpus, ~1.2M rows across commercial-safe +
    research-only) is deterministically reservoir-sampled down to
    `corpus_cap` total (default `DEFAULT_CORPUS_CAP`, see that constant's
    comment for the wall-time reasoning) -- see `_reservoir_sample_jokes`.

    Construction prints a one-line report of exactly how many corpus items
    were scanned vs. embedded (loud, not silent) -- see module docstring's
    "degraded mode is the opposite of LabelSpace's" note for why silence is
    not acceptable here.

    Degenerate-completion convention, matching the rest of the stack (see
    `env.rewards.IntraGroupDiversityReward`'s docstring for the shared
    rationale): a completion whose `_normalize` token set is empty (empty
    string, whitespace-only, emoji-only) never earns a nonzero penalty
    here -- it is skipped before any embedding call, exactly mirroring
    `CorpusNoveltyPenalty`'s existing behavior (an empty completion's
    trigram set is empty, `trigram_jaccard` returns 0.0 against every
    template, so `best <= threshold` and the reward is 0.0 there too; this
    class makes the same outcome explicit and skips the wasted embedding
    call rather than relying on an accidental zero).

    WINDOWED MODE (`windowed=True`, DEFAULT OFF -- opt-in pending
    EXP-011, see module docstring): closes the padding/dilution exploit
    for this tier by scoring max cosine over sliding, BOUNDARY-tokenized
    windows of the completion (plus the whole text, so windowed scores
    are always >= shipped scores) against the SAME reference embeddings.
    Default OFF preserves the shipped, validated whole-text behavior
    bit-for-bit; the templates-only reference and
    `DEFAULT_THRESHOLD=0.38` are kept in windowed mode on the
    calibration-transfer assumption stated in the module docstring --
    which a real-model spot-check found holds for DETECTION but NOT for
    the false-positive baseline (window similarity for novel text runs
    ~0.55, above 0.38), so EXP-011's re-swept threshold is REQUIRED
    before enabling this flag in a real run; see module docstring.

    "Boundary-tokenized" (2026-07-17 audit BLOCKER fix; see
    `_window_texts`'s docstring for the full explanation): windows are
    derived via `env.rewards._window_token_spans` -- punctuation,
    whitespace, and zero-width/format characters (Unicode category
    'Cf') all act as token separators -- rather than plain
    `text.split()` (whitespace only), which used to let padding GLUED to
    a template by punctuation or a zero-width character (no whitespace
    at all) fuse the template's edge word into the padding's edge word,
    same class of bug as `env.rewards.CorpusNoveltyPenalty`'s own
    windowed tier (lower severity here: approximate cosine similarity
    degrades gracefully on a misaligned edge rather than fully evading,
    but the fix is the same). Window CONTENT is always sliced from the
    ORIGINAL completion text, never reconstructed from stripped tokens,
    so a template's own internal punctuation (e.g. "don't") reaches the
    embedder intact.

    Window ladder + containment guarantee: window sizes derive from the
    templates' BOUNDARY-tokenizer lengths (NOT raw whitespace-token
    counts -- the two can differ for a template with internal
    punctuation, e.g. this project's own fixture template has 11
    boundary-tokens vs. 10 whitespace-tokens, from "don't"'s
    apostrophe). Cover targets C_k start at the shortest template length
    and double up to (longest template length + `window_growth`), each
    level using stride S_k = max(1, C_k // 2) and
    window size W_k = C_k + S_k - 1. Window starts are 0, S_k, 2S_k, ...
    plus a final tail window ending exactly at the last token. Covering
    lemma: for any contiguous span of P <= C_k tokens, the valid-start
    interval [p + P - W_k, p] has length W_k - P + 1 >= S_k, so it
    contains a multiple of S_k (or, at the right edge, the appended tail
    start) -- i.e. EVERY span up to the longest template length +
    `window_growth` (paraphrase-runs-longer headroom, default 8) is
    FULLY CONTAINED in at least one window. Dilution bound: a span of
    length P in (C_{k-1}, C_k] occupies >= P / W_k > C_{k-1} / (1.5 *
    2 * C_{k-1}) = 1/3 of its containing window, so in-window filler
    never exceeds 2/3 -- far from the ~90% dilution the audit measured
    as needed to cross below threshold. Spans shorter than the shortest
    template are still fully contained (level 0) at a weaker fraction;
    they are sub-template fragments, not full paraphrases.

    Performance + cap: all of a completion's windows (whole text first)
    are embedded in ONE `embed_fn` call per completion -- never one call
    per window. (Non-windowed mode keeps the shipped one-call-per-BATCH
    path untouched.) Identical window texts are deduplicated before
    embedding (adversarial repetition produces many duplicate windows --
    dedup makes the attack cheaper for us, not the attacker). Windows
    are capped at `max_windows` per completion (default 256), split
    evenly across ladder levels, keeping each level's EARLIEST windows.
    SECURITY IMPLICATION, stated plainly: when the cap binds, each
    level's coverage is a PREFIX of the completion (level k covers
    roughly its first budget * S_k + W_k tokens; >= ~640 whitespace
    tokens per level at defaults with real-corpus template lengths) and
    a memorized span pushed entirely past every level's frontier evades
    this tier's windows. That needs several hundred tokens of padding --
    ~5x ComprehensibilityReward's max_tokens band -- and the whole-text
    score still applies; a one-time RuntimeWarning fires if the cap ever
    binds so a training run can't silently live past the frontier.

    Degenerate windows follow the degenerate-completion convention
    above: a window whose `_normalize` token set is empty (all-emoji /
    whitespace padding) is skipped before embedding, exactly like a
    degenerate completion. The whole-completion guard runs first either
    way -- a fully degenerate completion is never windowed at all.

    `embed_fn`, if supplied, is `Callable[[Sequence[str]], <array-like of
    shape (len(texts), dim)>]` and is used in place of a real
    `SentenceTransformer` -- the ONLY way to unit-test this class's actual
    scoring logic without a model download (see
    `env/tests/test_semantic_novelty.py`). Embeddings it returns are
    assumed L2-normalized (unit norm) by the caller, exactly like
    `LabelSpace`'s `model.encode(..., normalize_embeddings=True)`
    convention -- so cosine similarity reduces to a plain dot product.
    Supplying `embed_fn` bypasses the `sentence_transformers` guarded
    import entirely (it is never even attempted), but `numpy` is still
    required for the similarity matrix multiply -- treated as part of the
    same "backend unavailable" failure mode as missing
    `sentence_transformers`, since without it this class cannot do its job
    at any reasonable speed either.
    """

    __name__ = "semantic_novelty_penalty"

    def __init__(
        self,
        corpus_dir: Optional[Union[str, Path]],
        weight: float = -1.5,
        threshold: Optional[float] = None,
        reference: str = "templates",
        corpus_cap: int = DEFAULT_CORPUS_CAP,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        embed_fn: Optional[Callable[[Sequence[str]], object]] = None,
        allow_degraded: bool = False,
        batch_size: int = 256,
        windowed: bool = False,
        window_growth: int = 8,
        max_windows: int = 256,
    ):
        if corpus_dir is None:
            raise ValueError(
                "semantic_novelty_penalty: corpus_dir is required (got "
                "None), same contract as CorpusNoveltyPenalty -- this term "
                "needs a real memorized-joke corpus directory to compute "
                "similarity against. See env/tests/fixtures/corpus/ for "
                "the tiny fixture shape, or ~/Experiments/good-humored-data "
                "for the real corpus.")

        corpus_path = Path(corpus_dir).expanduser()
        if not corpus_path.exists():
            raise FileNotFoundError(
                "semantic_novelty_penalty: corpus_dir %s does not exist." %
                corpus_path)

        if reference not in ("templates", "corpus"):
            raise ValueError(
                "semantic_novelty_penalty: reference must be 'templates' "
                "(default -- score against the 25 known-memorized "
                "templates only, the reference set DEFAULT_THRESHOLD was "
                "calibrated against) or 'corpus' (score against templates "
                "+ the sampled general corpus -- a separate, explicitly "
                "opt-in mode measuring corpus-redundancy, not "
                "memorized-template paraphrase; see class docstring and "
                "module docstring's WARNING). Got %r." % (reference,))

        if threshold is None:
            if reference == "templates":
                threshold = DEFAULT_THRESHOLD
            else:
                raise ValueError(
                    "semantic_novelty_penalty: reference='corpus' has no "
                    "calibrated default threshold -- DEFAULT_THRESHOLD "
                    "(%.2f) was calibrated ONLY against the 25-template "
                    "reference set (see that constant's comment) and is "
                    "NOT valid for the general corpus's similarity "
                    "distribution, which runs far higher for genuinely "
                    "novel content (see module docstring's WARNING: "
                    "negatives median ~0.60 there vs. a templates-only "
                    "FPR<=0.05 threshold of 0.38). Pass an explicit "
                    "threshold, separately calibrated for reference="
                    "'corpus', if you deliberately want this mode." %
                    DEFAULT_THRESHOLD)

        self.weight = weight
        self.threshold = threshold
        self.reference = reference
        self.windowed = windowed
        self.window_growth = window_growth
        self.max_windows = max_windows
        self.degraded = False
        self._embed = embed_fn
        self._cap_warned = False
        self._window_levels: List[Tuple[int, int, int]] = []
        self.n_templates = 0
        self.n_corpus_scanned = 0
        self.n_corpus_embedded = 0

        # Cheap check (corpus_dir existence) done above BEFORE the
        # potentially-expensive backend load below -- fail fast on a typo'd
        # path without first paying to load a real embedding model, mirroring
        # CorpusNoveltyPenalty's own None -> exists -> content check order.
        if embed_fn is None:
            try:
                import numpy  # noqa: F401  -- availability check only
                from sentence_transformers import SentenceTransformer
            except Exception as e:  # ImportError normally; broad on
                # purpose -- a half-broken torch install must be caught
                # here too, same as LabelSpace.fit()'s own broad except.
                if not allow_degraded:
                    raise SemanticNoveltyUnavailable(
                        "semantic_novelty_penalty: neither "
                        "sentence_transformers+numpy nor an embed_fn is "
                        "available (%r). This term exists SPECIFICALLY to "
                        "close CorpusNoveltyPenalty's documented 2-word- "
                        "reskin evasion -- an inert novelty layer that "
                        "silently returns 0.0 while pretending to work is "
                        "this project's documented nightmare (CLAUDE.md), "
                        "so construction fails loudly by default. Install "
                        "with: pip3 install --user sentence-transformers "
                        "numpy -- or pass allow_degraded=True if you have "
                        "deliberately decided to run without this tier "
                        "(it will warn once and contribute 0.0 to every "
                        "reward)." % (e,)) from e
                warnings.warn(
                    "semantic_novelty_penalty: DEGRADED MODE -- "
                    "sentence_transformers/numpy unavailable (%r) and "
                    "allow_degraded=True was passed explicitly. This term "
                    "will contribute EXACTLY 0.0 to every reward for the "
                    "life of this instance -- CorpusNoveltyPenalty's "
                    "2-word-reskin evasion is NOT covered by anything else "
                    "in the stack. This is a deliberate, acknowledged gap, "
                    "not a silent one -- fix it before a real training "
                    "run by installing sentence-transformers." % (e,),
                    RuntimeWarning, stacklevel=2)
                self.degraded = True
                return  # skip template/joke loading + embedding below --
                        # directory existence was already checked above,
                        # so this only skips the EXPENSIVE work, not the
                        # cheap config-typo check

            model = SentenceTransformer(_MODEL_NAME)

            def _st_embed(texts):
                return model.encode(
                    list(texts), batch_size=batch_size,
                    normalize_embeddings=True, show_progress_bar=False)

            self._embed = _st_embed

        templates = load_templates(corpus_path)
        template_texts = [t["text"] for t in templates]
        self.n_templates = len(template_texts)

        if reference == "templates" and self.n_templates == 0:
            raise FileNotFoundError(
                "semantic_novelty_penalty: reference='templates' (the "
                "default) requires at least one entry in "
                "'chatgpt-25-templates.jsonl' at %s's root -- found none. "
                "This mode's whole job is comparing against KNOWN "
                "memorized templates; with zero templates there is "
                "nothing to compare against. Pass reference='corpus' "
                "(with an explicit threshold) if you specifically want to "
                "compare against the general joke corpus instead." %
                corpus_path)

        # Windowed-mode ladder (see class docstring): cover targets C
        # from the shortest template length, doubling up to the longest
        # + `window_growth`; per level, stride S = max(1, C // 2) and
        # window size W = C + S - 1 (the smallest W satisfying the
        # covering lemma W >= P + S - 1 for spans up to P = C).
        # Template lengths are the only length anchor windowing has, so
        # windowed mode requires at least one template regardless of
        # `reference` (reference="templates" already enforces that).
        #
        # BOUNDARY-tokenizer lengths (`_window_token_spans`, shared with
        # `env.rewards.CorpusNoveltyPenalty`'s own windowed fix), NOT raw
        # `t.split()` whitespace-token counts (2026-07-17 audit BLOCKER
        # fix, same class of bug as that module -- see `_window_texts`
        # below for the full explanation). This can change the ladder
        # for a template containing internal punctuation: this project's
        # own fixture template ("Why don't scientists...") has 11
        # boundary-tokens vs. 10 raw whitespace-tokens, from "don't"'s
        # apostrophe -- `_window_texts` slices window text from the
        # ORIGINAL completion, so this length only affects window
        # SIZING, never destroys any character.
        tmpl_word_lens = [len(_window_token_spans(t)) for t in template_texts
                          if _window_token_spans(t)]
        if windowed and not tmpl_word_lens:
            raise ValueError(
                "semantic_novelty_penalty: windowed=True derives its "
                "window sizes from the templates' token lengths, but %s "
                "has no non-empty 'chatgpt-25-templates.jsonl' entries. "
                "Windowed mode has no length anchor without templates." %
                corpus_path)
        if tmpl_word_lens:
            cover = max(min(tmpl_word_lens), 3)
            p_max = max(tmpl_word_lens) + window_growth
            while True:
                cover = min(cover, p_max)
                stride = max(1, cover // 2)
                self._window_levels.append(
                    (cover, stride, cover + stride - 1))
                if cover >= p_max:
                    break
                cover *= 2

        remaining_cap = max(corpus_cap - len(template_texts), 0)
        sampled_texts, n_scanned = _reservoir_sample_jokes(
            corpus_path, cap=remaining_cap, seed=sample_seed)
        self.n_corpus_scanned = n_scanned

        corpus_texts = template_texts + sampled_texts
        if not corpus_texts:
            raise FileNotFoundError(
                "semantic_novelty_penalty: %s contains no 'jokes.jsonl' "
                "file(s) anywhere under it and no 'chatgpt-25-templates."
                "jsonl' at its root -- refusing to construct a novelty "
                "term with nothing to compare against (same refusal as "
                "CorpusNoveltyPenalty)." % corpus_path)

        print(
            "semantic_novelty_penalty: embedding %d corpus items "
            "(%d templates, ALWAYS kept + %d jokes reservoir-sampled from "
            "%d scanned, cap=%d, seed=%d)..." %
            (len(corpus_texts), self.n_templates, len(sampled_texts),
             n_scanned, corpus_cap, sample_seed))

        import numpy as np
        self._corpus_embeddings = np.asarray(self._embed(corpus_texts))
        self.n_corpus_embedded = len(corpus_texts)
        print("semantic_novelty_penalty: done (%d embedded, coverage "
              "%.4f%% of scanned jokes + 100%% of templates)." %
              (self.n_corpus_embedded,
               100.0 * len(sampled_texts) / n_scanned if n_scanned else 0.0))

        # Templates are always the first `n_templates` rows of
        # corpus_texts (see above) -- a free slice of the embeddings
        # already computed, no second embed call. This is the ONLY
        # embedding set `__call__` scores against in the default
        # reference="templates" mode (see class + module docstrings for
        # why); `reference="corpus"` uses the full `_corpus_embeddings`
        # instead. Selected once here, not re-branched per __call__.
        self._template_embeddings = self._corpus_embeddings[:self.n_templates]
        self._reference_embeddings = (
            self._template_embeddings if reference == "templates"
            else self._corpus_embeddings)

    def _window_texts(self, text: str) -> List[str]:
        """Sliding-window texts for one completion, per the class
        docstring's ladder: for each (cover, stride, width) level,
        BOUNDARY-tokenizer-token windows starting at 0, stride,
        2*stride, ... plus a tail window ending at the last token (the
        covering lemma's right-edge case). Duplicate window texts
        (adversarial repetition) and degenerate windows (empty
        `_normalize` token set, e.g. all-emoji padding) are skipped
        WITHOUT consuming window budget. Budget: `max_windows` split
        evenly across levels, keeping each level's earliest windows --
        when it binds, a one-time RuntimeWarning names the
        coverage-frontier security implication (see class docstring).

        2026-07-17 audit BLOCKER fix, same class of bug as
        `env.rewards.CorpusNoveltyPenalty`'s windowed tier (lower
        severity here -- this tier is approximate cosine similarity, not
        exact matching, so a misaligned window edge degrades gracefully
        rather than fully evading -- but the SAME fix applies): window
        POSITIONS now come from `_window_token_spans` (shared with
        `env.rewards`), which treats punctuation, whitespace, AND
        zero-width/format characters as token SEPARATORS, so padding
        glued to a memorized template by "." or a zero-width character
        (with no whitespace at all) no longer fuses the template's edge
        word into the padding's edge word the way plain `text.split()`
        did. Window CONTENT is still SLICED FROM THE ORIGINAL `text`
        (`text[start:end]`), never reconstructed by joining stripped
        boundary-tokens -- exactly like `CorpusNoveltyPenalty`'s fix,
        and for the same reason: a template's own internal punctuation
        (e.g. "don't") is itself a boundary character under this
        tokenizer, so slicing preserves it intact for the embedder to
        see, while joining stripped tokens would not ("don" + " " + "t"
        != "don't")."""
        spans = _window_token_spans(text)
        n = len(spans)
        if not self._window_levels:
            return []
        budget = max(1, self.max_windows // len(self._window_levels))
        out: List[str] = []
        seen = set()
        capped = False
        for _cover, stride, width in self._window_levels:
            if n <= width:
                continue  # whole text (always embedded) covers this level
            starts = list(range(0, n - width + 1, stride))
            if starts[-1] != n - width:
                starts.append(n - width)  # tail window: covering lemma's
                                          # right-edge case
            emitted = 0
            for s in starts:
                if emitted >= budget:
                    capped = True
                    break
                win = text[spans[s][0]:spans[s + width - 1][1]]
                if win in seen or not _normalize(win):
                    continue
                seen.add(win)
                out.append(win)
                emitted += 1
        if capped and not self._cap_warned:
            warnings.warn(
                "semantic_novelty_penalty: windowed mode hit max_windows"
                "=%d on a %d-token completion -- window coverage is now a "
                "PREFIX of the completion per ladder level, and a "
                "memorized span pushed past every level's frontier evades "
                "the window tier (whole-text scoring still applies). "
                "Completions this long are degenerate under the rest of "
                "the stack, but do not ignore this in a real training "
                "run: raise max_windows or investigate what the policy "
                "is emitting. This warning fires once per instance." %
                (self.max_windows, n), RuntimeWarning, stacklevel=3)
            self._cap_warned = True
        return out

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        texts = _contents(completions)
        rewards = [0.0] * len(texts)
        if self.degraded:
            return rewards

        # Degenerate-completion guard -- see class docstring. Skips the
        # embedding call entirely for empty/whitespace/emoji-only text,
        # rather than embedding it and hoping the model happens to place
        # it far from the corpus.
        non_degenerate_idx = [i for i, t in enumerate(texts) if _normalize(t)]
        if not non_degenerate_idx:
            return rewards

        import numpy as np

        if self.windowed:
            # Windowed mode: ONE embed_fn call per completion -- the
            # whole text first (so windowed scores are always >= the
            # shipped whole-text score), then every window. Severity is
            # the max over all of them, ramped exactly like the shipped
            # path below.
            for orig_i in non_degenerate_idx:
                text = texts[orig_i]
                candidates = [text] + self._window_texts(text)
                embeddings = np.asarray(self._embed(candidates))
                score = float(
                    (embeddings @ self._reference_embeddings.T).max())
                if score <= self.threshold:
                    continue
                severity = (score - self.threshold) / (1.0 - self.threshold)
                rewards[orig_i] = self.weight * severity
            return rewards

        query_texts = [texts[i] for i in non_degenerate_idx]
        query_embeddings = np.asarray(self._embed(query_texts))
        # (n_query, dim) @ (dim, n_reference) -> (n_query, n_reference);
        # both sides are unit-normalized (see class docstring's embed_fn
        # contract), so this dot product IS cosine similarity, exactly
        # LabelSpace's `embeddings @ embeddings.T` convention.
        # `self._reference_embeddings` is templates-only by default (see
        # __init__'s `reference` handling) -- NOT the full
        # `self._corpus_embeddings` -- so this term operationalizes
        # "paraphrase of a known memorized template," matching the
        # reference set DEFAULT_THRESHOLD was calibrated against.
        sims = query_embeddings @ self._reference_embeddings.T
        best = sims.max(axis=1)

        for local_i, orig_i in enumerate(non_degenerate_idx):
            score = float(best[local_i])
            if score <= self.threshold:
                continue
            severity = (score - self.threshold) / (1.0 - self.threshold)
            rewards[orig_i] = self.weight * severity
        return rewards

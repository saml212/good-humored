"""Unified record types for humor preference / reward-model training data.

Pure stdlib, no exceptions -- this module (like `firewall.py`) must stay
importable with zero deps on the barest box, since it's the load-bearing
type layer every other module in this package (and, later, every trainer)
imports first.

Three record shapes:

  - `RankedGroup`  -- one prompt/context + N candidate responses, each
    with a score and (when the source reports it) a rater count. This is
    the actual shape of Oogiri and NYCC data, and what a Bradley-Terry or
    listwise reward-model trainer wants directly.
  - `PreferencePair` -- context, chosen, rejected: the DPO/pairwise shape.
    Produced FROM a `RankedGroup` via `to_preference_pairs()` -- there is
    no other constructor path, by design (see below).
  - `CorpusJoke` -- a single joke with no comparison signal at all (the
    shape of `local_corpus.py`'s data: 1.2M individual jokes, most with an
    upvote-style `score` but no sibling candidates rated on the same
    prompt). Used for the novelty-penalty corpus and SFT, never for
    reward-model training directly -- there is nothing to compare it to.

All three carry MANDATORY provenance: `source_dataset`, `license_class`,
`source_id`. There is no optional-provenance path anywhere in this
package -- a record with unknown provenance cannot be firewalled, and an
unfirewallable record is exactly the thing CLAUDE.md's Data section
exists to prevent.

---

CRITICAL DESIGN RULE, enforced structurally, not just by convention
(CLAUDE.md Hard Rules; `references/corpus-sources.md`;
`.claude/skills/humor-rl/SKILL.md` "Data" section; `docs/TRANSFER-PLAN.md`
§3.1's own warning about the SocialGrep `score` field): raw popularity /
upvote / "star" counts must NEVER be converted into a comparison ACROSS
prompts. A joke with 500 upvotes on a viral post is not "funnier" than one
with 5 upvotes on an obscure post -- the count mostly reflects who saw it
and when, not how funny it was. This is the single most-cited failure
mode in this project's own literature review (`negative-results.md`;
NYCC's own documented RLHF/DPO limitations trace partly to exactly this).

The enforcement is structural, not a rule someone has to remember:

  1. `Candidate.score` only has meaning relative to its OWN
     `RankedGroup.candidates` siblings -- nothing in this module ever
     puts two `Candidate` objects from different groups in the same
     comparison.
  2. `to_preference_pairs()` -- the ONLY function in this entire package
     allowed to compare two scores against each other -- takes exactly
     ONE `RankedGroup` as its argument, never a list of groups, never a
     flat pool of candidates. Its pairing loop reads only
     `group.candidates`. There is no code path by which a candidate from
     group A can be paired against a candidate from group B: the
     function's own signature makes that a `TypeError` before any
     comparison logic even runs.
  3. `CorpusJoke` (single jokes, no siblings) has NO pairing function at
     all in this module or `local_corpus.py`. `local_corpus.py` must
     never synthesize a `RankedGroup` out of two unrelated
     `CorpusJoke`s just because they both happen to carry a `score` --
     see that module's docstring for the explicit refusal to do this.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# The only two valid values for `license_class` anywhere in this package.
# Deliberately a plain tuple (not `typing.Literal`-only) so runtime code
# (not just a type checker) can validate against it -- see
# `_validate_license_class` below and `firewall.ALL_LICENSE_CLASSES`.
LICENSE_CLASSES: Tuple[str, str] = ("commercial_safe", "research_only")


def _validate_license_class(license_class: str, owner: str) -> None:
    if license_class not in LICENSE_CLASSES:
        raise ValueError(
            "%s: license_class must be one of %r, got %r. There is no "
            "third option and no silent default -- every record in this "
            "package must be explicitly classified so the license "
            "firewall (firewall.py) can do its job." %
            (owner, LICENSE_CLASSES, license_class))


@dataclass(frozen=True)
class Candidate:
    """One candidate response within a `RankedGroup`.

    `score` is on whatever native scale the source dataset uses (Oogiri's
    `star` like-count, NYCC's binary winner/loser label, a Bradley-Terry
    strength, ...) -- comparable ONLY to other `Candidate`s in the same
    `RankedGroup.candidates` tuple. Do not extract a bare list of
    `Candidate` objects from several groups and compare their `.score`
    fields directly; that is precisely the cross-prompt popularity-bias
    mistake this schema exists to make structurally awkward to commit.

    `rater_count` is `None` when the source doesn't report how many
    independent raters produced `score` (e.g. Oogiri-GO's `star` field is
    an aggregate like-count with no rater N attached). A `None` here is
    informational, not an error -- callers that need real rater counts
    for a Bradley-Terry fit should check for `None` themselves.
    """

    text: str
    score: float
    rater_count: Optional[int] = None
    candidate_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                "Candidate.text must be a non-empty string, got %r" %
                (self.text,))


@dataclass(frozen=True)
class RankedGroup:
    """One prompt/context + its candidate responses, all rated within
    that single context. This is the unit of comparison for this whole
    package -- see module docstring's "CRITICAL DESIGN RULE".

    `candidates` is stored as a `tuple` (converted in `__post_init__` even
    if a caller passes a list) specifically so a `RankedGroup` is
    immutable end to end -- nothing can quietly append a candidate from
    somewhere else into an existing group after construction.
    """

    context: str
    candidates: Tuple[Candidate, ...]
    source_dataset: str
    license_class: str
    source_id: str
    group_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_license_class(self.license_class, "RankedGroup")
        if not isinstance(self.context, str) or not self.context.strip():
            raise ValueError(
                "RankedGroup.context must be a non-empty string, got %r" %
                (self.context,))
        cands = tuple(self.candidates)
        if len(cands) < 1:
            raise ValueError(
                "RankedGroup must have at least 1 candidate (source_id=%r "
                "had 0). A group with zero candidates carries no signal "
                "and should not have been constructed by the loader." %
                self.source_id)
        object.__setattr__(self, "candidates", cands)


@dataclass(frozen=True)
class PreferencePair:
    """The DPO/pairwise shape: context + one chosen + one rejected
    response. Constructed ONLY by `to_preference_pairs()` below (or a
    caller reimplementing that exact same within-group discipline) --
    there is deliberately no convenience constructor that takes two bare
    strings and scores, because that shape invites exactly the
    cross-prompt comparison this package exists to prevent.
    """

    context: str
    chosen: str
    rejected: str
    source_dataset: str
    license_class: str
    source_id: str
    chosen_score: float
    rejected_score: float
    score_gap: float
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_license_class(self.license_class, "PreferencePair")
        if self.chosen_score <= self.rejected_score:
            raise ValueError(
                "PreferencePair: chosen_score (%r) must be strictly "
                "greater than rejected_score (%r) -- 'chosen' means "
                "'scored higher within this group', not an arbitrary "
                "label." % (self.chosen_score, self.rejected_score))


@dataclass(frozen=True)
class CorpusJoke:
    """A single joke with no rating group -- the shape of
    `local_corpus.py`'s ~1.2M-row corpus. `score` here (when present) is
    an ORIGIN-CORPUS field carried through for informational/provenance
    purposes only (e.g. a Reddit upvote count) -- it is NEVER given
    comparison meaning by anything in this package. There is no function
    anywhere in `data_adapters/` that turns a list of `CorpusJoke` into a
    `RankedGroup` or `PreferencePair`; see `local_corpus.py`'s docstring
    for why that door is deliberately not opened.
    """

    text: str
    source_dataset: str
    license_class: str
    source_id: str
    score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_license_class(self.license_class, "CorpusJoke")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                "CorpusJoke.text must be a non-empty string, got %r" %
                (self.text,))


def to_preference_pairs(
    group: RankedGroup,
    min_score_gap: float = 0.0,
    max_pairs: Optional[int] = None,
) -> List[PreferencePair]:
    """Convert ONE `RankedGroup`'s candidates into `PreferencePair`s.

    This is the only pairing function in the package (see module
    docstring). It takes a single `RankedGroup` -- not a list, not a
    corpus -- so there is no way to accidentally pair candidates from two
    different prompts: the pairing loop below only ever reads
    `group.candidates`.

    Tie / near-tie exclusion rule: a pair is emitted only when
    `hi.score - lo.score > min_score_gap` (strict). At the default
    `min_score_gap=0.0`, exact ties (gap == 0) are excluded but any
    positive gap, however small, is kept. Raise `min_score_gap` to also
    drop near-ties. Rationale: a tied or near-tied pair carries no
    preference signal, and manufacturing a forced direction out of a
    coin-flip-close pair would inject noise into the training set --
    funniness judgments are already documented as noisy on their own
    (`.claude/skills/humor-rl/SKILL.md` "Data": ExPUNations reports
    Cohen's kappa = 0.41 for funniness ratings), so this converter should
    not make that noise worse by asserting confidence gaps don't support.

    Degenerate-pair guard: a pair whose `chosen` and `rejected` TEXT are
    identical (can happen if a source has duplicate candidate text at
    different scores) is skipped -- there is no real preference between a
    string and itself no matter the score gap.

    All pairwise combinations within the group are considered by default
    (O(n^2) in group size) -- for a group of size 25 that's up to 300
    pairs. Pass `max_pairs` to cap the number of highest-score-gap pairs
    returned per group if a downstream trainer can't absorb the
    combinatorial blowup from large groups; `None` (default) returns every
    pair that survives the tie-exclusion rule.

    Every `PreferencePair` inherits `source_dataset`, `license_class`,
    `source_id`, AND a copy of `group.metadata` directly from `group` --
    provenance propagates unchanged from the group to every pair derived
    from it. The `metadata` copy matters concretely, not just for
    completeness: e.g. `nycc.py` stamps `attribution_required`/
    `attribution` (a real CC-BY-4.0 compliance obligation) onto every
    `RankedGroup` it produces, and a downstream trainer that only ever
    consumes `PreferencePair`s (never the source `RankedGroup`s) must
    still see that obligation -- losing it here would silently drop a
    real licensing requirement, not just cosmetic metadata.
    """
    if not isinstance(group, RankedGroup):
        raise TypeError(
            "to_preference_pairs: expected a single RankedGroup, got %r. "
            "This function deliberately does not accept a list of groups "
            "or a flat candidate pool -- see schema.py's module docstring "
            "on why pairing only ever happens within one group." %
            (type(group),))

    if len(group.candidates) < 2:
        return []

    ranked = sorted(group.candidates, key=lambda c: c.score, reverse=True)
    pairs: List[Tuple[float, PreferencePair]] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            hi, lo = ranked[i], ranked[j]
            gap = hi.score - lo.score
            if gap <= min_score_gap:
                continue
            if hi.text == lo.text:
                continue
            pair = PreferencePair(
                context=group.context,
                chosen=hi.text,
                rejected=lo.text,
                source_dataset=group.source_dataset,
                license_class=group.license_class,
                source_id=group.source_id,
                chosen_score=hi.score,
                rejected_score=lo.score,
                score_gap=gap,
                metadata=dict(group.metadata),
            )
            pairs.append((gap, pair))

    pairs.sort(key=lambda t: t[0], reverse=True)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return [p for _, p in pairs]


def to_preference_pairs_batch(
    groups: Sequence[RankedGroup],
    min_score_gap: float = 0.0,
    max_pairs: Optional[int] = None,
) -> List[PreferencePair]:
    """Convenience wrapper: apply `to_preference_pairs` independently to
    each group in `groups` and concatenate the results.

    This is NOT a relaxation of the within-group rule -- each group is
    still processed in complete isolation by the single-group function
    above; this wrapper only saves the caller a list comprehension. There
    is still no code path where a candidate from `groups[0]` is compared
    against a candidate from `groups[1]`.
    """
    out: List[PreferencePair] = []
    for g in groups:
        out.extend(to_preference_pairs(g, min_score_gap=min_score_gap,
                                       max_pairs=max_pairs))
    return out

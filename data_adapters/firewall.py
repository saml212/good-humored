"""The license firewall: commercial-safe and research-only humor data must
never silently mix. Pure stdlib, no deps.

Every record this package produces (`RankedGroup`, `PreferencePair`,
`CorpusJoke` from `schema.py`) carries a `license_class` field. This
module is the single place a caller should go to turn "a pile of tagged
records" into "the records I am actually allowed to use for X" -- never
write an ad hoc inline `if r.license_class == "commercial_safe"` filter
in training code, because that's exactly the kind of easy-to-typo,
easy-to-forget check that lets research-only data leak into a commercial
artifact.

WHY this matters (not just a style preference): per `CLAUDE.md` (Data
section) and `references/corpus-sources.md`, this project's local corpus
mixes data acquired under genuinely different terms --
`commercial-safe/jokes.jsonl` (887,639 rows, CC-BY-4.0, SocialGrep) can go
into a commercial training run with attribution; `research-only/
jokes.jsonl` (310,151 rows: Fraser's unspecified-license set plus
taivop's three explicitly-non-commercial sets) cannot. Oogiri-GO
(`oogiri.py`) is treated as research_only in this package (see that
module's docstring for a flagged, unresolved discrepancy with the
dataset's own HF license tag). Getting this wrong doesn't fail loudly on
its own later -- a model trained on the wrong mix just quietly becomes an
artifact nobody can legally ship. That is why every public loader entry
point in this package requires the caller to pass `allowed_licenses`
explicitly: there is no default anywhere that includes `'research_only'`.
"""

from typing import Dict, Iterable, List, Sequence, TypeVar

from data_adapters.schema import LICENSE_CLASSES

T = TypeVar("T")  # RankedGroup | PreferencePair | CorpusJoke, or any
                   # object with a `.license_class` attribute.


def assert_license_class(records: Iterable[T], allowed: Sequence[str]) -> List[T]:
    """Hard gate: every record's `license_class` must be in `allowed`, or
    this raises immediately, naming the offending record's provenance.

    `allowed` has NO default -- it must be passed explicitly and must be
    non-empty. This is deliberate friction: the entire point of the
    firewall is that a caller cannot get research-only data by doing
    nothing, only by asking for it by name.

    Returns the records as a `list` (consuming any iterator/generator
    passed in) so the caller has a concrete, already-validated collection
    rather than a one-shot generator that might silently be re-iterated
    as empty later.

    Use this (rather than `split_by_license`) when you want a firewall
    that FAILS the whole batch on any violation -- e.g. right before
    handing a supposedly-single-license dataset to a trainer. Use
    `split_by_license` instead when you expect a mixed batch and want to
    partition it, not reject it.
    """
    if not allowed:
        raise ValueError(
            "assert_license_class: `allowed` must be a non-empty, "
            "explicit list of license classes (e.g. ['commercial_safe']). "
            "There is no default anywhere in data_adapters that includes "
            "'research_only' -- WHY: research-only data (Oogiri-GO under "
            "this package's own research_only assignment; Fraser/taivop "
            "per DATA.md's MANIFEST.md) must never silently end up in a "
            "commercial training run or shipped artifact. State your "
            "intent by naming the license class(es) you actually want.")

    unknown = sorted(set(allowed) - set(LICENSE_CLASSES))
    if unknown:
        raise ValueError(
            "assert_license_class: unknown license class(es) %r in "
            "`allowed`. Valid values are exactly %r -- no others exist "
            "in this package." % (unknown, LICENSE_CLASSES))

    records = list(records)
    for r in records:
        lc = getattr(r, "license_class", None)
        if lc not in allowed:
            raise ValueError(
                "assert_license_class: record from source_dataset=%r "
                "source_id=%r has license_class=%r, which is not in "
                "allowed=%r. This is the license firewall CLAUDE.md / "
                "DATA.md / references/corpus-sources.md require: "
                "commercial-safe and research-only humor data must never "
                "silently mix (acquisition terms differ -- CC-BY-4.0 vs. "
                "explicit non-commercial-use sources -- and a model "
                "trained on the wrong mix is not a legally shippable "
                "commercial artifact). If this record legitimately "
                "belongs in your training set, add its license_class to "
                "`allowed` explicitly -- do not widen a default to make "
                "this error go away." %
                (getattr(r, "source_dataset", "?"),
                 getattr(r, "source_id", "?"), lc, allowed))
    return records


def split_by_license(records: Iterable[T]) -> Dict[str, List[T]]:
    """Partition `records` into `{license_class: [records...]}`.

    Unlike `assert_license_class`, this never raises on an unexpected
    `license_class` value -- an unrecognized value becomes its own bucket
    key (e.g. `None` if a malformed record slipped through), so a caller
    iterating `.items()` will notice it as an oddly-named bucket rather
    than having it silently dropped or crashing the whole batch. Prefer
    `assert_license_class` when you want a hard failure instead of a
    visible-but-separate bucket.
    """
    out: Dict[str, List[T]] = {}
    for r in records:
        lc = getattr(r, "license_class", None)
        out.setdefault(lc, []).append(r)
    return out

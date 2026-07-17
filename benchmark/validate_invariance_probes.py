"""Repeated-item invariance probes for LABEL_PROMPT_V4 (adversarial audit
findings M1/M2/bicycle-splitting, EXP-008 addendum's two-tier redesign +
alias table, EXPERIMENT_LOG.md).

Three joke families were traced across the FULL wild corpus (not just
their v2 label histogram) and found to scatter under repeated exposure
to the SAME underlying joke, worded slightly differently each time:

  M1 (farming_scarecrow): the "outstanding in his field" scarecrow joke,
     36 verbatim occurrences, split v3's `other` (16) / `work` (10) /
     `farmer` (9) / `farm animal` (1) -- a 4-way split on one joke.
  M2 (skeleton_guts): the "they don't have the guts" skeleton joke, 28
     verbatim occurrences, split free-v2 `death` (11) / `bone` (9) /
     `skeleton` (8), then landed in v3's `other` 27/28 times.
  bicycle_two_tired: the "two-tired" bicycle joke, 23 verbatim
     occurrences, split v3's `other` (15) / `driving` (5) / `car` (2) /
     `exercise` (1) and free-v2 `bike` (9) / `bicycle` (7) / `vehicle`
     (2) / `exercise` (2) / `transportation` (2) / `cycling` (1) --
     resolved via the new `bike` -> `bicycle` alias (re-audit finding).

benchmark/fixtures/label_invariance_probes_v4.jsonl holds 6 verbatim
wild-text variants of each family (18 probes total; real paraphrases
actually observed in the corpus -- punctuation/wording/person changes,
not synthetic rewordings), each tagged with the
`expected_label`/`expected_tier` v4 SHOULD converge on now that
LABEL_PROMPT_V4 carries explicit disambiguation guidance and the alias
table resolves documented synonyms before the vocabulary check.

This script is the REAL-LABELER counterpart to
benchmark/tests/test_rejector_v4.py's TestInvarianceProbeFixtureParsing
(which only checks the parsing pipeline against a scripted-correct fake,
no network). This script actually calls a live provider and reports,
per family: whether all variants agree (invariant), whether there's a
strict majority, what the majority label is, and whether it matches the
target. This is a VALIDATION script, not a build artifact that runs
itself -- per this repo's build/validate split, it is meant to be run by
EXP-010 after a separate audit, not executed wholesale during the build
that wrote it (a `--family`/`--limit` pair exists specifically so a
smoke run can stay cheap: as few as 1 call instead of 18 -- the family
count is read from the fixture at load time, never hardcoded here, so
a future fourth family is a fixture edit, not a script change).

Usage:
  # full run, all three families, 6 variants each (18 calls):
  python3 -m benchmark.validate_invariance_probes --provider claude:haiku \\
      --out experiment-runs-scratch/v4-invariance-probes/report.json

  # cheap smoke: one family, one variant (1 call):
  python3 -m benchmark.validate_invariance_probes --provider claude:haiku \\
      --family skeleton_guts --limit 1
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List

from .providers import get_provider
from .rejector import label_topic_v4

PROBES_PATH = (Path(__file__).parent / "fixtures"
              / "label_invariance_probes_v4.jsonl")


def load_probes(path: Path = PROBES_PATH) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def majority(labels: List[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def has_strict_majority(labels: List[str]) -> bool:
    """False when repeats split with no true winner (1-1-1, 2-2, ...) --
    same discipline as validate_rejector.py's identically-named helper
    (audit W3 there): a fabricated "majority" must never leak into a
    pass/fail read of this report."""
    top = Counter(labels).most_common(1)[0][1]
    return top * 2 > len(labels)


def run_probes(probes: List[Dict], complete: Callable[[str], str]) -> Dict:
    by_family: Dict[str, List[Dict]] = defaultdict(list)
    for p in probes:
        by_family[p["family"]].append(p)

    families = {}
    for family, items in by_family.items():
        labels, tiers = [], []
        for item in items:
            label, tier = label_topic_v4(item["joke"], complete)
            labels.append(label)
            tiers.append(tier)
        expected = items[0]["expected_label"]
        maj = majority(labels)
        families[family] = {
            "n": len(items),
            "jokes": [it["joke"] for it in items],
            "labels": labels,
            "tiers": tiers,
            "invariant": len(set(labels)) == 1,
            "has_strict_majority": has_strict_majority(labels),
            "majority_label": maj,
            "expected_label": expected,
            "majority_matches_expected": maj == expected,
            "label_histogram": dict(Counter(labels).most_common()),
            "tier_histogram": dict(Counter(tiers).most_common()),
        }
    return {"families": families}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="claude:haiku",
                    help="resolved ONLY through get_provider() (never "
                         "invoked directly), so the neutral-cwd guard "
                         "applies.")
    ap.add_argument("--probes", default=str(PROBES_PATH))
    ap.add_argument("--family", default=None,
                    help="restrict to one family (farming_scarecrow, "
                         "skeleton_guts, or bicycle_two_tired) -- for a "
                         "cheap smoke run")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of variants PER FAMILY -- for "
                         "a cheap smoke run (e.g. --limit 1 costs at "
                         "most one call per family present in the "
                         "fixture -- 3 today -- before --family narrows "
                         "it further)")
    ap.add_argument("--out", default=None,
                    help="write the JSON report here in addition to "
                         "printing it")
    args = ap.parse_args()

    probes = load_probes(Path(args.probes))
    if args.family is not None:
        probes = [p for p in probes if p["family"] == args.family]
        if not probes:
            raise SystemExit("no probes matched --family %r" % args.family)
    if args.limit is not None:
        capped, seen = [], Counter()
        for p in probes:
            if seen[p["family"]] < args.limit:
                capped.append(p)
                seen[p["family"]] += 1
        probes = capped

    complete = get_provider(args.provider)
    report = run_probes(probes, complete)
    report["provider"] = args.provider

    print(json.dumps(report, indent=2))
    for family, r in report["families"].items():
        print("%-20s n=%-2d invariant=%-5s strict_majority=%-5s "
              "majority=%-14r expected=%-14r match=%s" %
              (family, r["n"], r["invariant"], r["has_strict_majority"],
               r["majority_label"], r["expected_label"],
               r["majority_matches_expected"]))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print("report written to %s" % out_path)


if __name__ == "__main__":
    main()

"""Dataset adapter layer: external humor datasets -> unified reward-model
/ preference training format, with a license firewall between
commercial-safe and research-only data.

    schema.py        -- RankedGroup, PreferencePair, CorpusJoke record
                         types, and to_preference_pairs()/
                         to_preference_pairs_batch(). Pure stdlib.
    firewall.py       -- assert_license_class() / split_by_license(): the
                         only sanctioned way to filter records by license
                         before handing them to a trainer. Pure stdlib.
    oogiri.py         -- Oogiri-GO (zhongshsh/CLoT-Oogiri-GO, HF) loader,
                         text-only (T2T), license_class='research_only'
                         (see that module's docstring for a flagged,
                         unresolved license discrepancy).
    nycc.py           -- New Yorker Caption Contest
                         (jmhessel/newyorker_caption_contest, HF) loader,
                         'ranking' config, text-only,
                         license_class='commercial_safe' (CC-BY-4.0).
    local_corpus.py   -- adapter over the already-downloaded
                         ~/Experiments/good-humored-data/corpus/ (887,639
                         commercial-safe + 310,151 research-only single
                         jokes, no rating groups) -> CorpusJoke.

Every public loader entry point (`oogiri.load_ranked_groups`,
`nycc.load_ranked_groups`, `local_corpus.load_corpus_jokes`,
`local_corpus.load_memorized_templates`) requires the caller to pass
`allowed_licenses` explicitly -- there is no default anywhere in this
package that includes `'research_only'`. See `firewall.py`'s module
docstring for why.

Nothing in this package makes a network call at import time. `oogiri.py`
and `nycc.py` only touch the network inside their `ensure_sample`/
`fetch_rows` functions, and only when a caller doesn't supply an
already-fetched local path/rows (which is exactly the path this package's
own test suite uses, via tiny fixture files, to stay network-free).
"""

from data_adapters.firewall import assert_license_class, split_by_license
from data_adapters.schema import (LICENSE_CLASSES, Candidate, CorpusJoke,
                                  PreferencePair, RankedGroup,
                                  to_preference_pairs,
                                  to_preference_pairs_batch)

__all__ = [
    "LICENSE_CLASSES", "Candidate", "RankedGroup", "PreferencePair",
    "CorpusJoke", "to_preference_pairs", "to_preference_pairs_batch",
    "assert_license_class", "split_by_license",
]

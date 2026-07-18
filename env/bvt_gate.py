"""BVT multiplicative gate -- `docs/THEORY-MAP.md` §12.1's registration-grade
spec for the next reward tier, operationalizing McGraw & Warren (2010,
*Psychological Science* 21(8), `mcgraw2010benign`)'s Benign Violation
Theory: three conditions are jointly necessary and sufficient for humor --
a situation appraised as a VIOLATION, appraised as BENIGN, and both
appraisals occurring simultaneously. Read §12.1 in full before touching
this file -- it is the spec, not a summary of it, and carries the theory
citation trail, the product-vs-min()-vs-soft-AND argument (product chosen
for a GRPO-specific gradient reason, not just "the theory says multiply"),
the full gaming analysis, and the pre-registered validation design
(fixture: `env/tests/fixtures/bvt_gate_fixture.jsonl`) this class's
behavior must clear before EXP-013 trusts it in a live run.

reward = weight * violation_score * benign_score

Product, not min() or geometric mean -- §12.1's own argument, restated
briefly here because it is the one design choice this file's correctness
depends on: `min(v, b)`'s gradient is exactly zero w.r.t. whichever term is
NOT currently the minimum, which under GRPO's within-group advantage
computation can collapse the whole term's learning signal to "whichever
single detector currently has within-group variance" -- exactly the
additive-stack failure this term exists to close, one level down. The
product's gradient (`d(vb)/dv = b`, `d(vb)/db = v`) never fully zeroes on
either side as long as the OTHER term is nonzero.

Two SEPARATE judge calls (`violation_judge`, `benign_judge`), not one
combined rubric -- avoids a halo effect where one overall "I like/dislike
this" impression contaminates both appraisals, defeating the point of
measuring two theoretically-independent constructs (§12.1's "Measuring
violation and benign SEPARATELY"). Cost: 2 judge calls per completion
instead of `judge_preference`'s 1.

Inert by default (`weight=0.0`), same discipline as
`env.semantic_novelty.SemanticNoveltyPenalty` / `RewardConfig.judge_weight`:
neither `violation_judge` nor `benign_judge` has a network-calling default.
Missing either one degrades to ONE loud `RuntimeWarning` (not per-call) and
a 0.0 contribution to every reward -- never a silent stub API call, same
convention as `env.rewards.JudgePreferenceReward`.

`violation_judge`/`benign_judge` are `Callable[[Any, str], float]`,
IDENTICAL in shape to `RewardConfig.judge` -- an existing judge callable's
provider plumbing (`benchmark/providers.py`'s `get_provider`) is directly
reusable here, just pointed at `VIOLATION_PROMPT`/`BENIGN_PROMPT` (below,
verbatim from §12.1) instead of a preference rubric. Both prompts ask for
an integer 0-10 on the first line, parsed the same
`benchmark/banter.py._parse_score`-style way and divided by 10 by whatever
real judge callable is wired up in EXP-013 -- THIS class only enforces the
resulting [0, 1] contract (see `JudgePreferenceReward`'s identical
enforcement and its rationale: an unnormalized 1-10 score would silently
dominate the rest of the additive reward stack this gate sits alongside).

Degenerate/refusal handling: deliberately NONE beyond what the injected
judges themselves decide -- this mirrors `JudgePreferenceReward`, which
also does not special-case empty/refusal completions before handing them
to `judge`. A refusal or empty completion is still a real text the
violation/benign judges can rate (most likely low-violation, high-benign,
i.e. a low product) rather than a case this class silently intercepts;
unlike `ComprehensibilityReward`'s explicit refusal override, §12.1's
pseudocode does not call for one here, and no deviation from that
pseudocode is taken.

NOT a fix for judge-hacking in general (§12.1's "what this loses" + gaming
analysis, restated briefly): the benign-washing guard is the
disclaimer-discounting instruction baked into `BENIGN_PROMPT` below,
validated (not just hoped for) via the fixture's
`disclaimer_washed_violation` class. Correlated-judge-template
exploitation (a policy converging on one structural template that trips
both judges for unrelated reasons) is explicitly NOT closeable by this
term alone -- §12.1 calls for a standing fakeability audit using
`SelfRepetitionPenalty`/`IntraGroupDiversityReward` alongside this term in
any real run, not a static pre-registration bar.
"""

import warnings
from typing import Any, Callable, List, Optional

from env.rewards import _contents

# Verbatim from docs/THEORY-MAP.md §12.1. Two independent rubrics, scored
# by two SEPARATE judge calls (see module docstring for why one combined
# call is rejected) -- deliberately silent on whether the text is funny,
# safe, or appropriate; each prompt scores ONLY its own appraisal axis.
VIOLATION_PROMPT = """You are assessing whether a piece of text VIOLATES a
norm, expectation, or how something 'should' be -- logically, socially, or
morally -- regardless of whether it is funny, safe, or appropriate. A text
can present a strong violation while being completely FUNNY, completely
OFFENSIVE, or completely NEUTRAL -- score ONLY the presence/strength of the
violation, never whether it's OK.

Text: {completion}

Output ONLY an integer 0 (no violation of any kind -- an unremarkable,
expected statement) to 10 (a severe, unambiguous violation) on the first line.
"""

BENIGN_PROMPT = """You are assessing whether a piece of text is BENIGN --
i.e. is there ALSO a competing appraisal that neutralizes any threat (it is
clearly fictional/hypothetical, the target is distant from the reader, a
norm holds that makes it acceptable, or nothing in it is severe enough to
threaten anyone)? Do NOT let the mere presence of a disclaimer ('just
kidding', 'jk', a fictional-frame label) drive your score by itself --
assess whether the underlying situation is genuinely safe once you set the
disclaimer aside, not whether the text CLAIMS to be joking. A text
describing a genuinely harmful act that merely appends 'jk' should score
LOW, not high.

Text: {completion}

Output ONLY an integer 0 (a genuine, unmitigated threat -- no safe reading
available) to 10 (completely safe; no unneutralized threat present) on the
first line.
"""


class BVTGateReward:
    """reward = weight * violation_score * benign_score. Inert by default
    (weight=0.0) -- same discipline as SemanticNoveltyPenalty. No network
    call unless both violation_judge and benign_judge are injected. See
    module docstring and docs/THEORY-MAP.md §12.1 for the full design
    rationale, gaming analysis, and pre-registered validation bars.
    """

    __name__ = "bvt_gate_reward"

    def __init__(self,
                violation_judge: Optional[Callable[[Any, str], float]] = None,
                benign_judge: Optional[Callable[[Any, str], float]] = None,
                weight: float = 0.0):
        self.violation_judge = violation_judge
        self.benign_judge = benign_judge
        self.weight = weight
        self._warned = False

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        texts = _contents(completions)
        if self.violation_judge is None or self.benign_judge is None:
            if not self._warned:
                warnings.warn(
                    "bvt_gate_reward: violation_judge/benign_judge not "
                    "both configured -- this term contributes 0.0 to "
                    "every reward. Pass RewardConfig(violation_judge=..., "
                    "benign_judge=...) before training -- both must be "
                    "set, since the product formula needs both appraisals "
                    "to mean anything (a missing judge is not a partial "
                    "signal, it is no signal). This warning fires once "
                    "per BVTGateReward instance, not once per call.",
                    RuntimeWarning, stacklevel=2)
                self._warned = True
            return [0.0] * len(texts)

        prompts_iter = prompts if prompts is not None else [None] * len(texts)
        rewards = []
        for p, c in zip(prompts_iter, texts):
            v = float(self.violation_judge(p, c))
            b = float(self.benign_judge(p, c))
            for name, score in (("violation_judge", v), ("benign_judge", b)):
                if not (0.0 <= score <= 1.0):
                    raise ValueError(
                        "bvt_gate_reward: %s must return [0, 1], got %r. "
                        "Normalize your judge's native scale (e.g. the "
                        "0-10 scale VIOLATION_PROMPT/BENIGN_PROMPT ask for "
                        "-> divide by 10) before passing it in, same "
                        "contract as JudgePreferenceReward's judge." %
                        (name, score))
            rewards.append(self.weight * v * b)
        return rewards

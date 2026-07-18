"""Two-stage incongruity-resolution gate -- `docs/THEORY-MAP.md` §12.2's
registration-grade spec. Read §12.2 in full before touching this file --
it is the spec, not a summary of it, and carries the theory citation trail
(Suls 1972's two-stage model; Kao/Trott's cross-validated null result that
surprisal magnitude does NOT predict funniness among things that are
already jokes -- the load-bearing empirical constraint this whole design
is built around), the two-tier Tier A/Tier B infrastructure argument (Tier
A -- real logprob-based surprisal -- is blocked on an unconfirmed
`api:glm` logprob pre-probe and explicitly NOT built here; this file is
Tier B, the judge-based "predict-then-diverge" proxy, registered as
buildable now), the full gaming analysis, and the pre-registered
validation design (fixture:
`env/tests/fixtures/incongruity_gate_fixture.jsonl`) this class's behavior
must clear before EXP-014 trusts it in a live run.

Stage 1 (incongruity/surprise): a predictor guesses, COLD (no hint this is
a joke), the most unsurprising continuation of a completion's setup. Large
distance from the real punchline = surprising.

Stage 2 (resolution): the SAME predictor, now PRIMED ("expect a twist"),
guesses again. If the primed guess lands measurably closer to the real
punchline than the cold guess did, that is this design's proxy for "a
reconciling rule exists" -- Wyer & Collins' "resolution" stage, minus the
explicitly-out-of-scope third stage (elaboration).

reward = weight if (gate_1_surprise AND gate_2_resolution) else 0.0

A STRICT AND, not a graded sum or a scalar reward proportional to
`distance_cold` -- this is the single most load-bearing design constraint
in §12.2, restated here because getting it wrong silently re-creates the
exact failure two independent papers (Kao et al. 2016; Trott et al. 2025)
falsify a decade apart: surprisal magnitude has NO established
relationship to funniness among things that are already jokes (ambiguity
r=.03, p=.697 within-puns in Kao et al.). A policy rewarded for HOW MUCH
`distance_cold` exceeds `surprise_threshold` would be maximizing exactly
the quantity this term's own citation trail says does not track
funniness. This is a GATE: zero or `weight`, never a magnitude in between.

Inert by default (`weight=0.0`). `predictor` matches
`benchmark/providers.py`'s `Callable[[str], str]` contract (NOT `judge`'s
`(prompt, completion) -> float` shape -- this class needs raw text
generations to compare by embedding distance, not a score) -- no
network-calling default, same discipline as every other injectable
callable in this stack. `embed_fn` matches
`env.semantic_novelty.SemanticNoveltyPenalty`'s injection contract
exactly: `Callable[[Sequence[str]], <array-like (n, dim)>]`, assumed
L2-normalized so cosine similarity reduces to a plain dot product; a
missing real backend (`sentence_transformers`/`numpy`) and no injected
`embed_fn` raises `SemanticNoveltyUnavailable` (imported from
`env.semantic_novelty`, NOT a new exception type -- §12.2's pseudocode
explicitly calls for reusing that type) unless `allow_degraded=True` is
passed explicitly, matching that class's "fail loudly by default, quiet
degradation only on explicit opt-in" discipline exactly.

DEVIATION FROM §12.2's LITERAL WIRING TEXT, flagged loudly per this
build's instructions (see also env/rewards.py's reward_stack() for the
matching note): §12.2's own prose says BVTGateReward needs "no lazy
import" while implying (by silence) this class follows
SemanticNoveltyPenalty's LAZY pattern -- both are in fact lazily imported
by reward_stack() here, INCLUDING BVTGateReward, which the prose describes
as importable unconditionally at env/rewards.py's top level. That literal
reading was not taken: this module (and env/bvt_gate.py) import `_contents`
FROM `env.rewards` at their own top level, so an unconditional top-level
`from env.bvt_gate import BVTGateReward` inside env/rewards.py would be a
genuine circular import (env.rewards -> env.bvt_gate -> env.rewards).
Lazily importing both new gates inside reward_stack(), gated on their own
nonzero weight exactly like SemanticNoveltyPenalty already is, avoids that
hazard with zero behavioral or performance cost (reward_stack() runs once
per training-run setup, not per completion) and is what this build's own
task instructions explicitly asked for ("lazy imports", plural). See
env/rewards.py's reward_stack() docstring for the mirrored note.
"""

import warnings
from typing import Callable, List, Optional, Sequence

from env.rewards import _contents
from env.semantic_novelty import SemanticNoveltyUnavailable, _MODEL_NAME

# Verbatim from docs/THEORY-MAP.md §12.2.
SPLIT_PROMPT = """Below is a joke. Identify its SETUP (everything before the
reveal) and its PUNCHLINE (the reveal itself). If the joke has no
identifiable setup/punchline structure (e.g. it is a single-clause pun with
no buildup), output exactly "NO_SPLIT" and nothing else.

Joke: {completion}

Output exactly two lines:
SETUP: <the setup text>
PUNCHLINE: <the punchline text>
"""

PREDICT_COLD_PROMPT = """Setup: {setup}

Predict, in ONE short sentence, the single most natural, UNSURPRISING way
you would expect this to continue -- not a joke, just the ordinary
continuation a typical person would expect.

Output ONLY your one-sentence predicted continuation.
"""

PREDICT_PRIMED_PROMPT = """Setup: {setup}

This is the setup of a joke that ends in a clever twist, pun, or reframe.
Predict, in ONE short sentence, what you think the twist/punchline is.

Output ONLY your one-sentence predicted punchline.
"""


class TwoStageIncongruityGate:
    """reward = weight if (surprising cold AND resolves when primed) else 0.
    Inert by default. `predictor` matches providers.py's Callable[[str],
    str] contract (NOT judge's (prompt, completion)->float shape -- this
    class needs raw text generations, not a score). `embed_fn` matches
    SemanticNoveltyPenalty's injection contract exactly. See module
    docstring and docs/THEORY-MAP.md §12.2 for the full design rationale,
    gaming analysis, and pre-registered validation bars.

    A completion the predictor marks NO_SPLIT (or that fails to parse both
    SETUP/PUNCHLINE lines) returns 0.0 -- a documented "can't apply" case,
    matching this codebase's existing None-sentinel convention
    (`banter.py`'s `_judge_once`, `rejector.py`'s `UNPARSEABLE`) rather
    than pretending every completion has an identifiable setup/punchline
    structure.
    """

    __name__ = "two_stage_incongruity_gate"

    def __init__(self, predictor: Optional[Callable[[str], str]] = None,
                embed_fn: Optional[Callable[[Sequence[str]], object]] = None,
                weight: float = 0.0, surprise_threshold: float = 0.5,
                drop_threshold: float = 0.15, allow_degraded: bool = False):
        self.predictor = predictor
        self.weight = weight
        self.surprise_threshold = surprise_threshold
        self.drop_threshold = drop_threshold
        self.degraded = False
        self._embed = embed_fn

        # Guarded sentence_transformers/numpy import + loud
        # SemanticNoveltyUnavailable-style failure, exactly mirroring
        # SemanticNoveltyPenalty.__init__ (see module docstring): only
        # attempted when no embed_fn was injected, so unit tests never
        # need a real embedding backend (env/tests/test_incongruity_gate.py
        # injects a fake embed_fn exclusively).
        if embed_fn is None:
            try:
                import numpy  # noqa: F401  -- availability check only
                from sentence_transformers import SentenceTransformer
            except Exception as e:  # ImportError normally; broad on
                # purpose -- a half-broken torch install must be caught
                # here too, same as SemanticNoveltyPenalty's own broad
                # except.
                if not allow_degraded:
                    raise SemanticNoveltyUnavailable(
                        "two_stage_incongruity_gate: neither "
                        "sentence_transformers+numpy nor an embed_fn is "
                        "available (%r). This term exists SPECIFICALLY to "
                        "operationalize Suls (1972)'s two-stage "
                        "incongruity-resolution gate -- an inert gate "
                        "that silently returns 0.0 while pretending to "
                        "work is this project's documented nightmare "
                        "(CLAUDE.md), so construction fails loudly by "
                        "default. Install with: pip3 install --user "
                        "sentence-transformers numpy -- or pass "
                        "allow_degraded=True if you have deliberately "
                        "decided to run without this gate (it will warn "
                        "once and contribute 0.0 to every reward)." %
                        (e,)) from e
                warnings.warn(
                    "two_stage_incongruity_gate: DEGRADED MODE -- "
                    "sentence_transformers/numpy unavailable (%r) and "
                    "allow_degraded=True was passed explicitly. This gate "
                    "will contribute EXACTLY 0.0 to every reward for the "
                    "life of this instance. This is a deliberate, "
                    "acknowledged gap, not a silent one -- fix it before "
                    "a real training run by installing "
                    "sentence-transformers." % (e,),
                    RuntimeWarning, stacklevel=2)
                self.degraded = True
                return  # skip backend construction below

            model = SentenceTransformer(_MODEL_NAME)

            def _st_embed(texts):
                return model.encode(
                    list(texts), normalize_embeddings=True,
                    show_progress_bar=False)

            self._embed = _st_embed

    def _split(self, text: str):
        """Ask `predictor` to identify SETUP/PUNCHLINE per SPLIT_PROMPT.
        Returns (None, None) on an explicit "NO_SPLIT" response or any
        parse failure (missing either line) -- the documented "can't
        apply" sentinel, never a guessed/default split."""
        raw = self.predictor(SPLIT_PROMPT.format(completion=text))
        if not raw or not raw.strip():
            return None, None
        stripped = raw.strip()
        if stripped == "NO_SPLIT":
            return None, None
        setup = None
        punchline = None
        for line in stripped.splitlines():
            line = line.strip()
            if line.startswith("SETUP:"):
                setup = line[len("SETUP:"):].strip()
            elif line.startswith("PUNCHLINE:"):
                punchline = line[len("PUNCHLINE:"):].strip()
        if not setup or not punchline:
            return None, None
        return setup, punchline

    def _distance(self, a: str, b: str) -> float:
        """1 - cosine_similarity(embed(a), embed(b)). embed_fn's output is
        assumed unit-normalized (SemanticNoveltyPenalty's own embed_fn
        contract), so the dot product IS cosine similarity."""
        import numpy as np
        emb = np.asarray(self._embed([a, b]))
        return float(1.0 - np.dot(emb[0], emb[1]))

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        texts = _contents(completions)
        if self.degraded or self.predictor is None:
            return [0.0] * len(texts)
        rewards = []
        for text in texts:
            setup, punchline = self._split(text)
            if setup is None:
                rewards.append(0.0)
                continue
            cold = self.predictor(PREDICT_COLD_PROMPT.format(setup=setup))
            primed = self.predictor(PREDICT_PRIMED_PROMPT.format(setup=setup))
            d_cold = self._distance(cold, punchline)
            d_primed = self._distance(primed, punchline)
            passes = (d_cold >= self.surprise_threshold
                      and d_primed < d_cold
                      and (d_cold - d_primed) >= self.drop_threshold)
            rewards.append(self.weight if passes else 0.0)
        return rewards

"""The trainable RL environment package for good-humored.

`benchmark/` measures; `env/` trains. This package turns the same
underlying machinery (rejector labeling, banter scenarios, novelty
checks) that `benchmark/` uses for offline evaluation into the pieces a
GRPO training run needs at the moment GPUs arrive:

  rewards.py     -- the decomposed reward stack (README.md "The RL
                     environment" table), as TRL-GRPO-compatible
                     reward_funcs.
  cascade_env.py -- Track 1 (the rejection cascade) as a step-wise
                     reset()/step() training environment.
  banter_env.py  -- Track 2 (contextual banter) as a step-wise
                     reset()/step() training environment.

Nothing in this package makes a network call or invokes a CLI. Every
judge/labeler/partner dependency is an explicitly injected callable —
see each module's docstring for its pluggable-callable contract (there
are two distinct ones in this package: rewards.py's judge returns a
normalized float in [0, 1]; banter_env.py's judge is a raw text
completion callable, reusing benchmark.banter's rubric-based parser
unmodified. Do not assume they're interchangeable).
"""

from .banter_env import BanterEnv
from .cascade_env import CascadeEnv
from .rewards import (ComprehensibilityReward, CorpusNoveltyPenalty,
                      IntraGroupDiversityReward, JudgePreferenceReward,
                      RewardConfig, SelfRepetitionPenalty, combined_reward,
                      reward_stack)

__all__ = [
    "RewardConfig", "combined_reward", "reward_stack",
    "JudgePreferenceReward", "CorpusNoveltyPenalty", "SelfRepetitionPenalty",
    "IntraGroupDiversityReward", "ComprehensibilityReward",
    "CascadeEnv", "BanterEnv",
]

#!/usr/bin/env python3
"""End-to-end smoke test: a TINY causal LM trained for a handful of real
TRL `GRPOTrainer` steps, consuming `env/rewards.py`'s reward stack exactly
as a full training run would (CLAUDE.md hard rule: smoke test every
model/environment before spending real GPU time on it -- this is that
rule applied to the environment package itself, on a laptop, today).

Unlike `env/smoke.py` (pure stdlib, calls reward functions directly with
hand-built completions), this script proves the reward stack survives
contact with the REAL trainer: real tokenization, real batched generation,
real advantage computation, a real backward pass and optimizer step. The
fake judge is the only stand-in (deterministic, local, wordlist-based --
NO network/API call, per the task's constraint); every reward TERM is the
genuine `env.rewards` code.

Run: python3 -m env.integration.trl_grpo_smoke
(from a venv with torch/transformers/trl/accelerate/datasets installed --
see the venv at ~/Experiments/good-humored-data/venvs/trl-smoke/)

--------------------------------------------------------------------------
FINDING -- a real crash this smoke test exists to catch, captured precisely
--------------------------------------------------------------------------
On this machine (macOS 15.7.7, Apple Silicon, torch==2.13.0,
transformers==5.14.1, trl==1.8.0), constructing `GRPOTrainer` with
`GRPOConfig(beta=<nonzero>, use_cpu=True)` reliably SEGFAULTS (SIGSEGV,
"possible pointer authentication failure") the moment TRL builds the
frozen reference-model copy that beta>0 requires. The crash is NOT a
Python exception (no traceback -- the process dies silently under a
background pthread); macOS's own crash reporter
(~/Library/Logs/DiagnosticReports/Python-*.ips) pins it exactly to
`at::native::mps::mps_copy_` / `copy_cast_kernel_mps` inside
libtorch_cpu.dylib, invoked from `torch::autograd::THPVariable_to` -- i.e.
some internal tensor `.to(device)` call is being routed through the MPS
backend's copy kernel EVEN THOUGH `GRPOConfig(use_cpu=True)` was set,
and that kernel crashes hard. Reproduced twice identically before
isolating it (see the two crash reports from this session's PIDs).

Workaround applied below, since this is a torch/accelerate/macOS
interaction bug, not a bug in this repo's code: monkeypatch
`torch.backends.mps.is_available` to return `False` *before* importing
transformers/datasets/trl, so nothing downstream ever decides an
auxiliary tensor belongs on `mps`. This is a workaround, not a fix --
report it, don't paper over it in `env/rewards.py` (out of scope; that
file owns none of this).

--------------------------------------------------------------------------
FINDING -- skill-doc divergence
--------------------------------------------------------------------------
`.claude/skills/grpo-rl-training/SKILL.md` and
`.claude/skills/trl-fine-tuning/SKILL.md` both use `max_prompt_length` as
a standard `GRPOConfig`/generation-config field. In trl==1.8.0,
`max_prompt_length` has been REMOVED from `GRPOConfig` entirely (it only
still exists on newer *experimental* trainers under `trl.experimental.*`,
e.g. SSD/SDPO/SDFT). This script has no need for it (prompts are written
short by construction below), but anyone following either skill's copy-
pasted config against a current `trl` install will hit a `TypeError:
unexpected keyword argument 'max_prompt_length'` -- worth a skill-doc fix.
"""

import os

os.environ.setdefault("HF_HOME", os.path.expanduser(
    "~/Experiments/good-humored-data/hf-cache"))  # hard rule: never let the
                                                    # HF cache default elsewhere
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

# --- crash workaround, see module docstring "FINDING -- a real crash" ---
torch.backends.mps.is_available = lambda: False

from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from env.rewards import RewardConfig, reward_stack  # noqa: E402

FIXTURE_CORPUS_DIR = _REPO_ROOT / "env" / "tests" / "fixtures" / "corpus"

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"  # 135M, Apache-2.0, base (non-chat) --
                                            # smallest model this task's
                                            # candidate list offers; CPU-only
                                            # training of it is fast enough for
                                            # a laptop smoke test
NUM_GENERATIONS = 4
PER_DEVICE_TRAIN_BATCH_SIZE = 8  # == 2 x NUM_GENERATIONS -> two distinct
                                 # prompt-groups per step, so
                                 # IntraGroupDiversityReward's multi-group
                                 # chunking loop actually gets exercised
                                 # (a single-group batch would leave that
                                 # code path untested)
MAX_STEPS = 3
MAX_COMPLETION_LENGTH = 48
LEARNING_RATE = 1e-5
BETA = 0.02  # matches docs/TRANSFER-PLAN.md's recommended 0.01-0.02 KL
             # anchor, deliberately higher than grpo-rl-training's generic
             # default -- also the exact setting that triggers the crash
             # above without the mps workaround, so exercising it here is
             # the point, not incidental

# A tiny, deterministic, local "funniness" judge: NO network/API call, no
# model -- just presence of words from a fixed comedy-adjacent wordlist.
# Base score 0.2 (every completion gets *something*, matching how a real
# judge would rarely output a hard zero), +0.2 per distinct hit, capped at
# 1.0 to respect JudgePreferenceReward's enforced [0, 1] contract.
FUNNY_WORDS = ("cat", "banana", "joke", "laugh", "pun", "robot", "chicken",
              "walks into a bar", "why did", "knock knock")


def fake_judge(prompt, completion: str) -> float:
    text = completion.lower()
    hits = sum(1 for w in FUNNY_WORDS if w in text)
    return min(1.0, 0.2 + 0.2 * hits)


# 12 inline joke-request prompts (task calls for 8-16). Deliberately short
# (no max_prompt_length available in this trl version to truncate for us --
# see module docstring) and deliberately plain strings, not chat-format,
# since MODEL_NAME is a base model with no chat template.
PROMPTS = [
    "Tell me a joke about cats.",
    "Tell me a joke about computers.",
    "Tell me a joke about coffee.",
    "Tell me a joke about robots.",
    "Tell me a joke about the ocean.",
    "Tell me a joke about mountains.",
    "Tell me a joke about pizza.",
    "Tell me a joke about spreadsheets.",
    "Tell me a joke about weekends.",
    "Tell me a joke about weather.",
    "Tell me a joke about bicycles.",
    "Tell me a joke about libraries.",
]


def documented_bounds(cfg: RewardConfig) -> dict:
    """Every term in reward_stack() is `weight * component` where
    `component` is documented to sit in [0, 1] (a judge score, a ramped
    novelty/repetition severity, a mean pairwise trigram distance, or a
    0/0.25/0.5/0.75/1.0-banded structural score) -- so each term's own
    valid range is exactly [min(0, weight), max(0, weight)], regardless of
    the weight's sign. Computed from the actual RewardConfig in use, not
    hardcoded, so this stays correct if weights are ever overridden."""
    weight_by_name = {
        "judge_preference": cfg.judge_weight,
        "corpus_novelty_penalty": cfg.corpus_novelty_weight,
        "self_repetition_penalty": cfg.self_repetition_weight,
        "intra_group_diversity": cfg.intra_group_diversity_weight,
        "comprehensibility": cfg.comprehensibility_weight,
    }
    return {name: tuple(sorted((0.0, w))) for name, w in weight_by_name.items()}


class RewardTracker:
    """Wraps every reward_stack() term in a counting/validating shim so the
    acceptance checks below have real evidence, not an assumption:

      - counts[name] -- proves every term actually got called by the real
        trainer (not silently skipped/short-circuited).
      - violations -- accumulates any finite/length/documented-range
        violation, checked on EVERY call, not just spot-checked.
      - step_records / stdout JSON lines -- one line per completed "round"
        (reward_stack()'s 5 terms are always invoked in the same fixed
        order every time TRL computes rewards for a batch -- see
        `_calculate_rewards` in trl/trainer/grpo_trainer.py -- so "the
        first term was just called" reliably marks the start of a new
        round and "the last term was just called" reliably marks its
        end).
    """

    def __init__(self, term_names, bounds):
        self.term_names = term_names
        self.bounds = bounds
        self.counts = {n: 0 for n in term_names}
        self.violations = []
        self.step_records = []
        self._round_values = {}
        self._round_index = 0

    def wrap(self, funcs):
        return [self._wrap_one(f) for f in funcs]

    def _wrap_one(self, func):
        name = func.__name__
        tracker = self

        def wrapped(prompts, completions, **kwargs):
            values = func(prompts=prompts, completions=completions, **kwargs)
            tracker._record(name, values, len(completions))
            return values

        wrapped.__name__ = name  # TRL uses __name__ for per-term logging
        return wrapped

    def _record(self, name, values, expected_len):
        self.counts[name] += 1

        if len(values) != expected_len:
            self.violations.append(
                "%s: returned %d rewards for a batch of %d completions"
                % (name, len(values), expected_len))

        lo, hi = self.bounds[name]
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v):
                self.violations.append("%s: non-finite/non-numeric reward %r"
                                       % (name, v))
                continue
            if not (lo - 1e-6 <= v <= hi + 1e-6):
                self.violations.append(
                    "%s: reward %.6f outside documented range [%.4f, %.4f]"
                    % (name, v, lo, hi))

        if name == self.term_names[0]:
            self._round_index += 1
            self._round_values = {}
        self._round_values[name] = list(values)

        if name == self.term_names[-1]:
            means = {n: (sum(vals) / len(vals) if vals else float("nan"))
                     for n, vals in self._round_values.items()}
            record = {
                "event": "step_reward_means",
                "step": self._round_index,
                "n_completions": expected_len,
                "reward_means": means,
            }
            self.step_records.append(record)
            print(json.dumps(record), flush=True)


def register_grad_finite_hooks(model, log):
    """Register a backward hook on 2 parameters (first + a middle one, by
    name) that records (name, is_finite) the INSTANT autograd computes
    each gradient -- this is the bulletproof way to "spot-check a couple
    of parameters post-step" for NaN/Inf, since it doesn't depend on
    guessing whether `.grad` is still populated by the time some later
    callback fires (it might already have been zeroed for the next step).
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable) < 2:
        raise RuntimeError(
            "model has fewer than 2 trainable parameters -- cannot spot-check "
            "gradients as required by the smoke test's acceptance checks")
    picks = [trainable[0], trainable[len(trainable) // 2]]
    for name, p in picks:
        def _hook(grad, name=name):
            log.append((name, bool(torch.isfinite(grad).all().item())))
        p.register_hook(_hook)
    return [name for name, _ in picks]


def main() -> None:
    violations = []

    cfg = RewardConfig(
        judge=fake_judge,
        joke_corpus_dir=FIXTURE_CORPUS_DIR,  # tiny fixture corpus, NOT the
                                              # real ~/Experiments/good-humored-data
        group_size=NUM_GENERATIONS,
    )
    raw_funcs = reward_stack(cfg)
    term_names = [f.__name__ for f in raw_funcs]
    bounds = documented_bounds(cfg)
    tracker = RewardTracker(term_names, bounds)
    wrapped_funcs = tracker.wrap(raw_funcs)

    print("Loading %s ..." % MODEL_NAME, flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model_load_s = time.time() - t0

    grad_log = []
    hooked_params = register_grad_finite_hooks(model, grad_log)

    dataset = Dataset.from_dict({"prompt": PROMPTS})

    out_dir = tempfile.mkdtemp(prefix="trl-grpo-smoke-")
    try:
        args = GRPOConfig(
            output_dir=out_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            max_steps=MAX_STEPS,
            logging_steps=1,
            report_to=[],       # no wandb / no external reporting
            use_cpu=True,       # CPU-only, deterministic (see crash finding)
            beta=BETA,
            learning_rate=LEARNING_RATE,
            dataloader_num_workers=0,
            save_strategy="no",  # smoke test has no use for a checkpoint
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=wrapped_funcs,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        print("Training %d step(s) ..." % MAX_STEPS, flush=True)
        t0 = time.time()
        trainer.train()
        train_s = time.time() - t0

        steps_run = trainer.state.global_step

        # ------------------------------------------------ acceptance checks

        # 1. every reward term actually got called, once per step run
        for name in term_names:
            if tracker.counts[name] != steps_run:
                violations.append(
                    "%s: called %d time(s), expected exactly %d (== training "
                    "steps run)" % (name, tracker.counts[name], steps_run))

        # 2/3. finite + correct-length + documented-range, gathered live
        violations.extend(tracker.violations)

        # 4. loss finite for every logged step
        loss_entries = [e for e in trainer.state.log_history if "loss" in e]
        if not loss_entries:
            violations.append("no 'loss' entries found in trainer.state.log_history")
        for e in loss_entries:
            if not math.isfinite(e["loss"]):
                violations.append("step %s: non-finite loss %r"
                                  % (e.get("step"), e["loss"]))

        # 5. no NaN/Inf in grads, spot-checked on 2 params every step
        expected_grad_events = steps_run * len(hooked_params)
        if len(grad_log) < expected_grad_events:
            violations.append(
                "gradient hooks fired %d time(s), expected >= %d (%d "
                "parameters x %d steps) -- backward pass may not have "
                "reached the checked parameters every step"
                % (len(grad_log), expected_grad_events, len(hooked_params),
                   steps_run))
        for name, is_finite in grad_log:
            if not is_finite:
                violations.append(
                    "parameter %r: non-finite gradient observed post-step" % name)

    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    summary = {
        "event": "smoke_result",
        "passed": len(violations) == 0,
        "model": MODEL_NAME,
        "steps_run": steps_run,
        "model_load_s": round(model_load_s, 2),
        "train_s": round(train_s, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "reward_term_call_counts": tracker.counts,
        "hooked_grad_params": hooked_params,
        "violations": violations,
    }
    print(json.dumps(summary, indent=2), flush=True)

    if violations:
        print("\nSMOKE TEST FAILED -- %d violation(s):" % len(violations),
              file=sys.stderr)
        for v in violations:
            print("  - %s" % v, file=sys.stderr)
        sys.exit(1)

    print("\nSMOKE TEST PASSED -- TRL GRPOTrainer trained %d real step(s) "
          "against env/rewards.py's actual reward stack, no violations."
          % steps_run)
    sys.exit(0)


if __name__ == "__main__":
    main()

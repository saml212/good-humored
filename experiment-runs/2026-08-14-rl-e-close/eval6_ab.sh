#!/bin/bash
# RL-D A/B eval (eval6): base vs GRPO-smooth-trained, identical seeds, identical fleet.
# PRE-REGISTERED (EXPERIMENT_LOG RL-C pins): 500 paired sessions at seed-base
# 10,000,000 (reserved eval space, disjoint from 9M train), judged on the
# CERTIFIED objective. Success: delta >= +0.03 at t >= 2 + diversity guards.
# PREREQ: merge ckpt via verl.model_merger; serve base :8002 + trained :8005
# (max-model-len 16384); partner :8004 and audience :8003 alive.
set -u
GH=/data/good-humored
cd $GH/repo
export HF_HOME=$GH/hf-cache
V=$GH/venv/bin/python
$V -m env.banter_rollout --base-url http://127.0.0.1:8002/v1 --model qwen3-30b-base \
  --partner-base-url http://127.0.0.1:8004/v1 --partner-model qwen3-235b-a22b \
  --n-sessions 1000 --session-offset 12000000 --workers 96 --temperature 1.0 \
  --provocation-rate 0.5 --out $GH/runs/eval6_base.jsonl >> $GH/logs/eval6_ab.log 2>&1
$V -m env.banter_rollout --base-url http://127.0.0.1:8005/v1 --model qwen3-30b-rle \
  --partner-base-url http://127.0.0.1:8004/v1 --partner-model qwen3-235b-a22b \
  --n-sessions 1000 --session-offset 12000000 --workers 96 --temperature 1.0 \
  --provocation-rate 0.5 --out $GH/runs/eval6_rle.jsonl >> $GH/logs/eval6_ab.log 2>&1
$V -m env.score_banter --transcripts $GH/runs/eval6_base.jsonl --limit 1000 --workers 24 \
  --audience-url http://127.0.0.1:8003/v1 --audience-model glm-4.5-air \
  --out $GH/runs/eval6_base.scored.json >> $GH/logs/eval6_ab.log 2>&1
$V -m env.score_banter --transcripts $GH/runs/eval6_rle.jsonl --limit 1000 --workers 24 \
  --audience-url http://127.0.0.1:8003/v1 --audience-model glm-4.5-air \
  --out $GH/runs/eval6_rle.scored.json >> $GH/logs/eval6_ab.log 2>&1
echo "EVAL4_AB_DONE" >> $GH/logs/eval6_ab.log

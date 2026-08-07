#!/bin/bash
# Lane keeper v3 -- perpetual banter generation, BOTH policy lanes
# concurrent per iteration (30B on GPU 1, GLM-Air on GPUs 2-3, shared
# 235B partner on GPUs 4-7) so no policy GPU idles while the other
# lane runs; (temperature, provocation_rate) rotate per iteration and
# session offsets advance so seeded schedules differ batch to batch.
# GLM also serves score-keeper audience calls concurrently (batched).
# Stop: touch /data/good-humored/STOP_LANES
set -u
GH=/data/good-humored
cd $GH/repo || exit 1
i=$(ls $GH/runs/banter_stream_*.jsonl 2>/dev/null | sed 's/.*_0*\([0-9]*\)\.jsonl/\1/' | sort -n | tail -1)
i=${i:-0}
TEMPS=(1.0 0.9 1.1)
PROVS=(0.35 0.25 0.50)
while [ ! -f $GH/STOP_LANES ]; do
  i=$((i+1))
  T="${TEMPS[$((i % 3))]}"
  R="${PROVS[$(((i / 3) % 3))]}"
  N=$(printf "%03d" $i)
  echo "=== iter $i temp=$T prov=$R (30B + GLM lanes concurrent) ===" \
    >> $GH/logs/lane_keeper.log
  $GH/venv/bin/python -m env.banter_rollout \
    --base-url http://127.0.0.1:8002/v1 --model qwen3-30b-a3b \
    --partner-base-url http://127.0.0.1:8004/v1 --partner-model qwen3-235b-a22b \
    --n-sessions 1000 --session-offset $((i*1000)) --workers 96 \
    --temperature "$T" --provocation-rate "$R" \
    --out $GH/runs/banter_stream_$N.jsonl \
    >> $GH/logs/lane_keeper.log 2>&1 &
  PID_30B=$!
  $GH/venv/bin/python -m env.banter_rollout \
    --base-url http://127.0.0.1:8003/v1 --model glm-4.5-air \
    --partner-base-url http://127.0.0.1:8004/v1 --partner-model qwen3-235b-a22b \
    --n-sessions 1000 --session-offset $((250000 + i*1000)) --workers 96 \
    --temperature "$T" --provocation-rate "$R" \
    --out $GH/runs/glm_stream_$N.jsonl \
    >> $GH/logs/lane_keeper_glm.log 2>&1 &
  PID_GLM=$!
  wait $PID_30B $PID_GLM
done
echo "lane keeper v3 stopped after iter $i" >> $GH/logs/lane_keeper.log

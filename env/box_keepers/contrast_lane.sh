#!/bin/bash
# Contrast lane -- 8B self-play on GPU 0 (otherwise idle). Weak-model
# banter is deliberate NEGATIVE-contrast data for emulator/curation
# training: the scorer runs over it too, so the curation distribution
# spans genuinely-bad to genuinely-good instead of only strong-model
# output. Stop: touch /data/good-humored/STOP_LANES
set -u
GH=/data/good-humored
cd $GH/repo || exit 1
i=$(ls $GH/runs/contrast_stream_*.jsonl 2>/dev/null | sed 's/.*_0*\([0-9]*\)\.jsonl/\1/' | sort -n | tail -1)
i=${i:-0}
TEMPS=(1.0 0.9 1.1)
while [ ! -f $GH/STOP_LANES ]; do
  i=$((i+1))
  T="${TEMPS[$((i % 3))]}"
  echo "=== contrast batch $i temp=$T offset=$((500000 + i*1000)) ===" \
    >> $GH/logs/contrast_lane.log
  $GH/venv/bin/python -m env.banter_rollout \
    --base-url http://127.0.0.1:8001/v1 --model qwen3-8b \
    --n-sessions 1000 --session-offset $((500000 + i*1000)) --workers 64 \
    --temperature "$T" \
    --out "$GH/runs/contrast_stream_$(printf "%03d" $i).jsonl" \
    >> $GH/logs/contrast_lane.log 2>&1
done
echo "contrast lane stopped after batch $i" >> $GH/logs/contrast_lane.log

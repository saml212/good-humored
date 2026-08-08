#!/bin/bash
# Parameterized perpetual policy lane (replaces the two-lane barrier
# loop in lane_keeper_v2.sh -- the wait on both lanes idled the faster
# lane's GPU every iteration; independent loops close that hole).
# Usage: policy_lane.sh MODEL BASE_URL FILE_PREFIX OFFSET_BASE LOGFILE \
#          [TEMPS_CSV] [PARTNER_MODEL] [PARTNER_URL]
# TEMPS_CSV optional (default "1.0,0.9,1.1") -- temperature response is
# model-specific (GLM improves to 1.1 on all four metrics; 30B is flat).
# PARTNER defaults to the shared 235B; pass both partner args for a
# self-play lane (soaks a policy server's idle duty cycle -- sessions
# bunch-synchronize on the shared partner bottleneck otherwise).
# Stop: touch /data/good-humored/STOP_LANES
set -u
MODEL=$1; URL=$2; PREFIX=$3; OFFBASE=$4; LOG=$5
IFS=',' read -ra TEMPS <<< "${6:-1.0,0.9,1.1}"
PARTNER_MODEL="${7:-qwen3-235b-a22b}"
PARTNER_URL="${8:-http://127.0.0.1:8004/v1}"
GH=/data/good-humored
cd $GH/repo || exit 1
i=$(ls $GH/runs/${PREFIX}_*.jsonl 2>/dev/null | sed 's/.*_0*\([0-9]*\)\.jsonl/\1/' | sort -n | tail -1)
i=${i:-0}
# v0.3 rotation (2026-08-07 cycle 2): P=0.35 retired -- monotone
# confirmed twice (P=0.65 beat 0.50 beat 0.35 on curation AND reaction
# in both lanes); 0.65 incumbent, 0.50 control, 0.80 probe
PROVS=(0.65 0.50 0.80)
while [ ! -f $GH/STOP_LANES ]; do
  i=$((i+1))
  # modulo by ACTUAL array length -- indexing by a hardcoded 3 killed
  # two 2-temp lanes instantly under set -u (TEMPS[2] unbound), one of
  # them silently on its first post-restart iteration
  T="${TEMPS[$((i % ${#TEMPS[@]}))]}"
  R="${PROVS[$(((i / ${#TEMPS[@]}) % ${#PROVS[@]}))]}"
  echo "=== $PREFIX batch $i temp=$T prov=$R ===" >> "$LOG"
  $GH/venv/bin/python -m env.banter_rollout \
    --base-url "$URL" --model "$MODEL" \
    --partner-base-url "$PARTNER_URL" --partner-model "$PARTNER_MODEL" \
    --n-sessions 1000 --session-offset $((OFFBASE + i*1000)) --workers 96 \
    --temperature "$T" --provocation-rate "$R" \
    --out "$GH/runs/${PREFIX}_$(printf "%03d" $i).jsonl" \
    >> "$LOG" 2>&1
done
echo "$PREFIX lane stopped after batch $i" >> "$LOG"

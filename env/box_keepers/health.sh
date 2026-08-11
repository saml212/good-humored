#!/bin/bash
# Standing health check -- session EXISTENCE + output FRESHNESS per
# lane, because log-tail reading proved worthless: a keeper that
# crashes on its first iteration leaves a plausible-looking tail
# forever. Run from the iterate cycle; exits nonzero if anything is
# down so the caller cannot miss it.
GH=/data/good-humored
FAIL=0
check_session() {
  tmux has-session -t "$1" 2>/dev/null || { echo "DOWN: tmux $1"; FAIL=1; }
}
check_fresh() {  # file, max-age-minutes, label
  if [ -z "$(find "$1" -mmin -"$2" 2>/dev/null)" ]; then
    echo "STALE: $3 ($1 older than $2 min)"; FAIL=1
  fi
}
# TRAINING ERA (2026-08-10): sampling lanes + scorers retired by
# STOP_LANES at 2,947 batches; expected topology = partner + audience
# serving for the GRPO env. Trainer sessions are transient and
# watched by their own stall-aware watchers, not this script.
for s in gh_serve_glm gh_serve_235b; do
  check_session "$s"
done
echo "bank: gen=$(ls $GH/runs/*_stream_*.jsonl 2>/dev/null | wc -l) scored=$(ls $GH/runs/*.scored.json 2>/dev/null | wc -l) (frozen)"
tmux has-session -t gh_grpo_v2 2>/dev/null && echo "grpo_v2: session up" || echo "grpo_v2: no session"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr "\n" " "; echo
[ "$FAIL" = 0 ] && echo "HEALTH OK" || echo "HEALTH FAIL"
exit $FAIL

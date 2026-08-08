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
for s in gh_serve_8b gh_serve_30b gh_serve_glm gh_serve_235b \
         gh_lane_30b gh_lane_glm gh_lane_glmself gh_contrast_lane \
         gh_score_keeper gh_score_keeper2; do
  check_session "$s"
done
# newest output file per lane must be recent (batches take ~4-9 min)
for pref in banter_stream glm_stream glmself_stream contrast_stream; do
  newest=$(ls -t $GH/runs/${pref}_*.jsonl 2>/dev/null | head -1)
  [ -n "$newest" ] && check_fresh "$newest" 30 "$pref lane output"
done
newest_scored=$(ls -t $GH/runs/*.scored.json 2>/dev/null | head -1)
[ -n "$newest_scored" ] && check_fresh "$newest_scored" 45 "scoring output"
echo "gen=$(ls $GH/runs/*_stream_*.jsonl 2>/dev/null | wc -l) scored=$(ls $GH/runs/*.scored.json 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr "\n" " "; echo
[ "$FAIL" = 0 ] && echo "HEALTH OK" || echo "HEALTH FAIL"
exit $FAIL

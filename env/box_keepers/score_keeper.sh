#!/bin/bash
# Score keeper -- perpetual scoring pass. Sweeps runs/ for COMPLETED
# stream batches (run_summary line present) that lack a .scored.json,
# scores each (certified gates + GLM-Air audience reaction diagnostic,
# 500-session subsample so scoring keeps pace with generation), then
# rebuilds the rolling curation master + top-5 transcript extract for
# the human read. Stop: touch /data/good-humored/STOP_LANES
# Optional arg 1: "desc" scans newest-first so a SECOND keeper
# instance can drain the backlog from the other end (generation is 3
# lanes wide; one scorer falls behind). mkdir-lock claims prevent
# double-scoring between instances.
set -u
ORDER="${1:-asc}"
GH=/data/good-humored
export HF_HOME=$GH/hf-cache
cd $GH/repo || exit 1
while [ ! -f $GH/STOP_LANES ]; do
  found=0
  # a keeper killed mid-batch leaves its lock behind; scoring takes
  # <10 min, so >90-min locks are stale and would silently orphan
  # their batch forever
  find $GH/runs -maxdepth 1 -name "*.scored.json.lock" -mmin +90 -exec rmdir {} + 2>/dev/null
  FILES=$(ls $GH/runs/banter_stream_*.jsonl $GH/runs/glm_stream_*.jsonl $GH/runs/glmself_stream_*.jsonl $GH/runs/contrast_stream_*.jsonl 2>/dev/null)
  # desc instance: POLICY lanes newest-first, contrast last -- the
  # contrast lane generates 3-4x faster and otherwise monopolizes the
  # fresh end while the batches that feed the read/card wait
  [ "$ORDER" = "desc" ] && FILES=$( { ls $GH/runs/banter_stream_*.jsonl $GH/runs/glm_stream_*.jsonl $GH/runs/glmself_stream_*.jsonl 2>/dev/null | sort -r; ls $GH/runs/contrast_stream_*.jsonl 2>/dev/null | sort -r; } )
  for f in $FILES; do
    [ -e "$f" ] || continue
    out="${f%.jsonl}.scored.json"
    # .failed marker prevents an infinite retry loop on a bad batch --
    # failures stay LOUD in the log but don't wedge the keeper
    { [ -e "$out" ] || [ -e "$out.failed" ]; } && continue
    grep -q run_summary "$f" || continue   # still generating
    mkdir "$out.lock" 2>/dev/null || continue   # claimed by other instance
    found=1
    # contrast batches: half subsample -- they feed distribution
    # stats and emulator negatives, not the demo channel, and their
    # 3-4x generation rate is the entire scoring backlog (gap 57 vs
    # 11-16 on policy lanes, cycle 19)
    LIMIT=500
    case "$f" in *contrast*) LIMIT=250;; esac
    echo "=== scoring $(basename $f) (limit $LIMIT) ===" >> $GH/logs/score_keeper.log
    $GH/venv/bin/python -m env.score_banter \
      --transcripts "$f" --limit $LIMIT --workers 24 \
      --audience-url http://127.0.0.1:8003/v1 --audience-model glm-4.5-air \
      --out "$out" >> $GH/logs/score_keeper.log 2>&1 \
      || { echo "SCORE FAILED: $f" >> $GH/logs/score_keeper.log; rm -f "$out"; touch "$out.failed"; }
    rmdir "$out.lock" 2>/dev/null
    # rebuild rolling curation after every batch so the freshest view
    # is always on disk for the human read
    $GH/venv/bin/python -m env.curate_banter \
      --scored-glob "$GH/runs/*_stream_*.scored.json" \
      --transcripts-dir $GH/runs \
      --out-master $GH/runs/curation_master.json \
      --out-top $GH/runs/curation_top5.txt \
      >> $GH/logs/score_keeper.log 2>&1 \
      || echo "CURATE FAILED" >> $GH/logs/score_keeper.log
  done
  [ "$found" = 0 ] && sleep 60
done
echo "score keeper stopped" >> $GH/logs/score_keeper.log

#!/bin/bash
# Score keeper -- perpetual scoring pass. Sweeps runs/ for COMPLETED
# stream batches (run_summary line present) that lack a .scored.json,
# scores each (certified gates + GLM-Air audience reaction diagnostic,
# 500-session subsample so scoring keeps pace with generation), then
# rebuilds the rolling curation master + top-5 transcript extract for
# the human read. Stop: touch /data/good-humored/STOP_LANES
set -u
GH=/data/good-humored
export HF_HOME=$GH/hf-cache
cd $GH/repo || exit 1
while [ ! -f $GH/STOP_LANES ]; do
  found=0
  for f in $GH/runs/banter_stream_*.jsonl $GH/runs/glm_stream_*.jsonl $GH/runs/contrast_stream_*.jsonl; do
    [ -e "$f" ] || continue
    out="${f%.jsonl}.scored.json"
    # .failed marker prevents an infinite retry loop on a bad batch --
    # failures stay LOUD in the log but don't wedge the keeper
    { [ -e "$out" ] || [ -e "$out.failed" ]; } && continue
    grep -q run_summary "$f" || continue   # still generating
    found=1
    echo "=== scoring $(basename $f) ===" >> $GH/logs/score_keeper.log
    $GH/venv/bin/python -m env.score_banter \
      --transcripts "$f" --limit 500 --workers 24 \
      --audience-url http://127.0.0.1:8003/v1 --audience-model glm-4.5-air \
      --out "$out" >> $GH/logs/score_keeper.log 2>&1 \
      || { echo "SCORE FAILED: $f" >> $GH/logs/score_keeper.log; rm -f "$out"; touch "$out.failed"; }
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

#!/usr/bin/env bash
# Serial bench queue on the laptop: each line of runs/bench_queue.txt is "<tag> <comma-separated program files>"; lines are consumed
# in order once no bench process is running. Append lines at any time. Stop with: touch runs/bench_queue.stop
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate; touch runs/bench_queue.txt; : > runs/bench_queue.done
while [ ! -f runs/bench_queue.stop ]; do
  if pgrep -f "bench_methods.py --tag" >/dev/null; then sleep 60; continue; fi
  line=$(grep -vxFf runs/bench_queue.done runs/bench_queue.txt | head -1)
  if [ -z "$line" ]; then sleep 60; continue; fi
  echo "$line" >> runs/bench_queue.done; tag=${line%% *}; progs=${line#* }
  echo "[$(date '+%H:%M')] starting $tag: $progs" >> runs/bench_queue.log
  JAX_PLATFORMS=cpu caffeinate -i python scripts/bench_methods.py --tag "$tag" --programs "$progs" > "runs/bench_$tag.log" 2>&1
  echo "[$(date '+%H:%M')] finished $tag" >> runs/bench_queue.log
done

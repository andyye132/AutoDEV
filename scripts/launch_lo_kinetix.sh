#!/usr/bin/env bash
# Run 14: racing multi-engine×regret from run 13 best; rung-1 gate on 14-target race split.

cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:?}
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); launching run 14" > runs/evolve_lo_kinetix.log
sleep 90   # let the server import JAX before the first program arrives
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20 AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq_lo_kinetix AUTODEV_TARGET_NORM=runs/norms_run13.json
export AUTODEV_INIT=runs/evolve_run7_multi/candidates/run7_best_gen035_0.5888.py
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_lo_kinetix --eval_jobs 3 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_lo_kinetix.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_lo_kinetix.log

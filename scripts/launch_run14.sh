#!/usr/bin/env bash
# Run 14: racing multi-engine×regret from run 13 best; rung-1 gate on 14-target race split.

cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:?first server}
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); launching run 14" > runs/evolve_run14.log
sleep 90   # let the server import JAX before the first program arrives
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20 AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq14
export AUTODEV_INIT=runs/evolve_run13_multi_regret/candidates/run13_best_gen038_0.7450.py
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run14_racing --eval_jobs 6 --proposal_jobs 4 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run14.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run14.log

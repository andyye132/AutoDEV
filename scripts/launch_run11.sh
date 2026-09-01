#!/usr/bin/env bash
# Run 11: search with VERIFIED fitness on the 42-target run-11 split (24 reachable-hard exploration-tier search targets + the 24 easy-bench search targets; runs/hyak/split_run11.json), seeded from split-and-pump — the hard-bench leader (0.440/50% vs action-repeat 0.373/42%). Fitness server = A40 job 39347133 on queue evalq11, 3 seeds, 128x1024.

cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:-39347133}
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); launching run 11" > runs/evolve_run11.log
sleep 90   # let the server import JAX before the first program arrives
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20 AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq11
export AUTODEV_INIT=evolve/methods/split_pump.py
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run11_hard --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run11.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run11.log

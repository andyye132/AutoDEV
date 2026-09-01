#!/usr/bin/env bash
# Run 4: GPU fitness. The LLM loop runs on the laptop; each candidate's fitness is computed by the Hyak fitness server
# (scripts/eval_server.py, job 39301031) on the size-l search split, 3 seeds, 128x1024 steps. Seed program = run-1 best.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:-39301031}
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); launching run 4" > runs/evolve_run4.log
sleep 90   # let the server import JAX before the first program arrives
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20
export AUTODEV_INIT=runs/evolve_overnight/candidates/cand1_gen037_0.6994.py
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run4_gpu --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run4.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run4.log

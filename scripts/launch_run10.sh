#!/usr/bin/env bash
# Run 10: search on the fixed harness with VERIFIED fitness (a hit counts only if its chain replays), seeded from split-and-pump (the survey hybrid; top-2 on the easy bench, top on the sealed-class subset). The LLM loop runs on the laptop; each candidate's fitness is computed by the Hyak fitness server
# (scripts/eval_server.py, A40 job 39301031, successor 39316382 on the same queue) on the size-l search split, 3 seeds, 128x1024 steps. Seed program = pair-ladder: seed family matters more than weighting (run 2), so a new family may lift the ceiling.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:-39345904}
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); launching run 10" > runs/evolve_run10.log
sleep 90   # let the server import JAX before the first program arrives
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20 AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq10
export AUTODEV_INIT=evolve/methods/split_pump.py
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run10_splitpump --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run10.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run10.log

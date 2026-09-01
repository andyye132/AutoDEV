#!/usr/bin/env bash
# Run 5 (to start after run 4): GPU fitness on the *reachable* search subset (runs/hyak/split_reachable.json: 103 of 127
# instances — the 24 no tester/probe ever reaches are constant zeros that only shrink the fitness range). Held-out splits are
# untouched. Seed = best program of run 4 (export first) or the run-1 best. Needs its own fitness server(s):
#   ssh hyak 'cd /gscratch/raivn/andy132/AutoDev && sbatch --export=ALL,SPLIT=runs/hyak/split_reachable.json,QUEUE=runs/hyak/evalq5,MAX_HOURS=10 scripts/slurm/eval_server.sbatch'
# then run:  SRV=<jobid> AUTODEV_INIT=<seed program> scripts/launch_run5.sh
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:?server job id}; INIT=${AUTODEV_INIT:-runs/evolve_overnight/candidates/cand1_gen037_0.6994.py}
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); launching run 5 from $INIT" > runs/evolve_run5.log
sleep 90
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20 AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq5
export AUTODEV_INIT=$INIT
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run5_reach --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run5.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run5.log

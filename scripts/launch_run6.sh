#!/usr/bin/env bash
# Run 6: GPU fitness on the REACHABLE-ONLY search split (runs/hyak/split_reachable.json), its own server/queue (evalq6),
# seeded from the run-4 best. Waits for the server job to be running.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:-39311725}; INIT=${AUTODEV_INIT:-runs/evolve_run4_gpu/candidates/cand1_gen030_0.6460.py}
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 120; done
echo "server $SRV running at $(date); launching run 6 from $INIT" > runs/evolve_run6.log
sleep 120
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=7200 AUTODEV_REMOTE_POLL=20 AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq6
export AUTODEV_INIT=$INIT
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run6_reach --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run6.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run6.log

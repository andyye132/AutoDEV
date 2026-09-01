#!/usr/bin/env bash
# Run 7: MULTI-ENGINE fitness (Kinetix reachable split + Craftax + Gymnax MountainCar/Acrobot/DeepSea, weighted), its own
# server/queue (evalq7), seeded from the run-4 best. Held out from the fitness: Kinetix held-out levels/class, Craftax held-out
# worlds/achievements, Gymnax held-out seeds — and any engine not in the fitness (XLand-MiniGrid, planned).
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:?server job id}; INIT=${AUTODEV_INIT:-evolve/initial.py}   # Go-Explore seed: the run-4 program was selected by the pre-fix (racy) fitness
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 120; done
echo "server $SRV running at $(date); launching run 7 (multi-engine fitness) from $INIT" > runs/evolve_run7.log
sleep 180
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=9000 AUTODEV_REMOTE_POLL=30 AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq7
export AUTODEV_INIT=$INIT
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run7_multi --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run7.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run7.log

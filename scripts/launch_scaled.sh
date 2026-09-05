#!/usr/bin/env bash
# Scaled multi-engine evolution, seeded from run7, chained-server longevity (the $-cap was NOT what stopped run7 --
# the server's 11.5h wall-clock was; only $20/$40 spent). Faithful run7 objective: AUTODEV_SUB_WEIGHTS / TARGET_NORM
# deliberately UNSET (the client re-weights engines equally; see the launch playbook). Run under nohup/tmux.
#   SRV=<server jobid> [INIT=<seed.py>] [GENS=100] [MAXCOST=100] [RESULTS=runs/evolve_scaled] bash scripts/launch_scaled.sh
set -u
cd /Users/andy132/projects/research/ACT/AutoDev
. .venv/bin/activate
SRV=${SRV:?server job id required}
INIT=${INIT:-runs/evolve_run7_multi/candidates/run7_best_gen035_0.5888.py}
GENS=${GENS:-100}; MAXCOST=${MAXCOST:-100}; RESULTS=${RESULTS:-runs/evolve_scaled}
echo "waiting for server $SRV to be Running..."
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running; warming 180s for JAX import + queue-dir creation"
sleep 180
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=9000 AUTODEV_REMOTE_POLL=30 \
       AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq_scaled
export AUTODEV_INIT="$INIT"
echo "launching evolution: gens=$GENS max_cost=$MAXCOST seed=$INIT -> $RESULTS"
python evolve/run_evolution.py --generations "$GENS" --results_dir "$RESULTS" \
  --eval_jobs 3 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost "$MAXCOST" \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high"
echo "EVOLUTION EXITED $?"

#!/usr/bin/env bash
# Run 13: multi-engine × regret synthesis. Fitness = weighted per-substrate regret (kinetix reachable + craftax + gymnax classics;
# MiniGrid AND Breakout held out as transfer tests), paired-delta feedback vs a frozen reference, text feedback, novelty rejection.
# THREE servers share evalq13 (atomic inbox claims) and eval_jobs=4 keeps them fed -> ~3x wall-clock speedup.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:?first server job id}
SEED_PROG=${SEED_PROG:?seed program path}
Q=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq13
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); caching the reference evaluation" > runs/evolve_run13.log
REF=ref_run13_$(date +%s)
scp -q "$SEED_PROG" "hyak:$Q/inbox/$REF.py"
until scp -q "hyak:$Q/outbox/$REF.json" runs/ref_run13.json 2>/dev/null; do sleep 30; done
echo "reference cached at $(date)" >> runs/evolve_run13.log
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=15 AUTODEV_REMOTE_QUEUE=$Q
export AUTODEV_INIT=$SEED_PROG
export AUTODEV_TARGET_NORM=runs/norms_run13.json AUTODEV_REF_RESULT=runs/ref_run13.json AUTODEV_TEXT_FEEDBACK=1 AUTODEV_NOVELTY=1
export AUTODEV_SUB_WEIGHTS='{"kinetix:l":1.0,"craftax":1.0,"gymnax:MountainCar-v0":0.34,"gymnax:Acrobot-v1":0.33,"gymnax:DeepSea-bsuite":0.33}'
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run13_multi_regret --eval_jobs 4 --proposal_jobs 3 --islands 2 --archive_size 30 --max_cost 45 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run13.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run13.log

#!/usr/bin/env bash
# Run 12: regret-normalized paired-delta fitness on the 42-target run-11 split, seeded from run 11's best.
# New infrastructure vs run 11: per-target regret vs best-known (AUTODEV_TARGET_NORM), paired delta vs a frozen reference on
# identical tasks+seeds (AUTODEV_REF_RESULT, cached below before the loop starts), Eureka-style text feedback to the proposer
# (AUTODEV_TEXT_FEEDBACK=1), LLM novelty rejection (AUTODEV_NOVELTY=1), and a closed-loop design hint in the task prompt.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
SRV=${SRV:?server job id}
SEED_PROG=${SEED_PROG:-runs/evolve_run11_hard/candidates/run11_best_gen004_0.7917.py}
Q=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq12
until [ "$(ssh -o ConnectTimeout=20 -o BatchMode=yes hyak "squeue -j $SRV -h -o %t" 2>/dev/null)" = "R" ]; do sleep 60; done
echo "server $SRV running at $(date); caching the reference evaluation" > runs/evolve_run12.log
REF=ref_run12_$(date +%s)
scp -q "$SEED_PROG" "hyak:$Q/inbox/$REF.py"
until scp -q "hyak:$Q/outbox/$REF.json" runs/ref_run12.json 2>/dev/null; do sleep 30; done
echo "reference cached at $(date): $(python -c "import json;d=json.load(open('runs/ref_run12.json'));print(round(d['score'],4), d['detected'])")" >> runs/evolve_run12.log
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20 AUTODEV_REMOTE_QUEUE=$Q
export AUTODEV_INIT=$SEED_PROG
export AUTODEV_TARGET_NORM=runs/norms_run12.json AUTODEV_REF_RESULT=runs/ref_run12.json AUTODEV_TEXT_FEEDBACK=1 AUTODEV_NOVELTY=1
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run12_regret --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run12.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run12.log

#!/usr/bin/env bash
# Run 3: rotated held-out class (ANOMALY held out; OOB in search). Launch after run 2 finishes.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
export AUTODEV_STEPS=1024 AUTODEV_WORKERS=64 AUTODEV_SEEDS=0,1 AUTODEV_SPLIT=runs/split_rot_anomaly.json
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run3_rot --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" > runs/evolve_run3.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run3.log

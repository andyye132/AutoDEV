#!/usr/bin/env bash
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
export JAX_PLATFORMS=cpu SHINKA_HEADLESS_COMMAND="npx -y @roberttlange/headless"
export AUTODEV_SUBSTRATE=xminigrid:MiniGrid-DoorKey-8x8
export AUTODEV_SPLIT=runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json
export AUTODEV_WORKERS=64 AUTODEV_STEPS=2048 AUTODEV_SEEDS=0,1
export AUTODEV_TASK_MSG_FILE=evolve/task_doorkey.txt
export AUTODEV_INIT=evolve/initial.py
python evolve/run_evolution.py --generations 18 --results_dir runs/evolve_fresh_doorkey \
  --models "headless/claude@claude-sonnet-5?effort=medium" --eval_jobs 1 --proposal_jobs 1 --islands 1 --archive_size 20 --max_cost 12 \
  >> runs/evolve_fresh_doorkey.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_fresh_doorkey.log

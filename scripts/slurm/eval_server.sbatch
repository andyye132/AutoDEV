#!/usr/bin/env bash
#SBATCH --job-name=autodev-evalsrv
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# Fitness server for the laptop-side evolution loop (scripts/eval_server.py). One GPU; exits after MAX_HOURS or when killed.
#   env: SEEDS=0,1,2 WORKERS=128 STEPS=1024 SIZE=l SPLIT=runs/hyak/split.json WHICH=search QUEUE=runs/hyak/evalq MAX_HOURS=11.5 CHUNK=16 AUTODEV_HARD_WEIGHT=0
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
export AUTODEV_HARD_WEIGHT="${AUTODEV_HARD_WEIGHT:-0}"
export JAX_TRACEBACK_FILTERING=off
SEEDS_="${SEEDS:-0,1,2}"; SEEDS_="${SEEDS_//:/,}"   # colon lists survive --export (commas do not)
python scripts/eval_server.py --queue "${QUEUE:-runs/hyak/evalq}" --seeds "${SEEDS_}" --workers "${WORKERS:-128}" --steps "${STEPS:-1024}" \
  --size "${SIZE:-l}" --split "${SPLIT:-runs/hyak/split.json}" --which "${WHICH:-search}" --max_hours "${MAX_HOURS:-11.5}" --chunk "${CHUNK:-16}"
echo EVALSRV_DONE

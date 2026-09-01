#!/usr/bin/env bash
#SBATCH --job-name=autodev-eval-ppo
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
export AUTODEV_SPLIT="${SPLIT:-runs/hyak/split.json}"
SPL="${SPLITS:-search,heldout_levels_instances,heldout_class_instances}"; SPL="${SPL//:/,}"
# 128 envs x 64 rollout x 32 updates = 262,144 env steps per instance (= the program testers' 128x2048 budget)
python scripts/eval_ppo_baselines.py --size l --seeds 0,1,2 --num_envs 128 --rollout 64 --num_updates "${NUM_UPDATES:-32}" --chunk 8 --out runs/hyak/eval_ppo${TAG:-}.json --splits "$SPL"
echo EVAL_PPO_DONE

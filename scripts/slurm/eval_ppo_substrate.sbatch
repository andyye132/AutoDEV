#!/usr/bin/env bash
#SBATCH --job-name=autodev-pposub
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# PPO+intrinsic RL testers on a substrate at matched budget.
#   env: SUBSTRATE=..., SPLIT=..., WHICH=search,heldout_levels_instances, OBJ=count,rnd,icm,curiosity_proximity,zero,
#        SEEDS=100,101,102, NUP=32 (1x; 128=4x; 512=16x), NENV=128, ROLL=64, CHUNK=16, OUT=runs/ppo_<task>.json
cd /gscratch/raivn/andy132/AutoDev
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/autodev_multi
export JAX_PLATFORMS=cuda
SEEDS_="${SEEDS:-100,101,102}"; SEEDS_="${SEEDS_//:/,}"
python scripts/eval_ppo_substrate.py --substrate "$SUBSTRATE" --split "$SPLIT" --which "${WHICH:-search,heldout_levels_instances}" \
  --objectives "${OBJ:-count,rnd,icm,curiosity_proximity,zero}" --seeds "$SEEDS_" \
  --num_envs "${NENV:-128}" --rollout "${ROLL:-64}" --num_updates "${NUP:-32}" --chunk "${CHUNK:-16}" --out "${OUT}"

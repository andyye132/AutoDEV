#!/usr/bin/env bash
#SBATCH --job-name=autodev-levels
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
python scripts/gen_levels.py --n 64 --size l --out runs/hyak/fresh_levels
python scripts/design_checks.py --tiers s,m,l --size l --out runs/hyak/design_checks.json
echo LEVELS_DONE

#!/usr/bin/env bash
#SBATCH --job-name=autodev-tune
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
export AUTODEV_SPLIT=runs/hyak/split.json
python scripts/tune_constants.py --program "${PROGRAM:-runs/evolved/candidates/cand1_gen037_0.6994.py}" --trials "${TRIALS:-40}" --seeds 0,1 --workers 128 --steps 1024 --size l --split_path runs/hyak/split.json --out runs/hyak/tuning_$(basename "${PROGRAM:-cand1}" .py).json
echo TUNE_DONE

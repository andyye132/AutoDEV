#!/usr/bin/env bash
#SBATCH --job-name=autodev-guided
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
python scripts/eval_guided.py --splits search,heldout_levels_instances,heldout_class_instances --seeds 0,1,2 --workers 128 --steps 2048 --size l --split_path runs/hyak/split.json --chunk 16 --out runs/hyak/guided_ceiling.json
echo GUIDED_DONE

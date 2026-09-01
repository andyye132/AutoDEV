#!/usr/bin/env bash
#SBATCH --job-name=autodev-pool
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
python scripts/build_pool.py --tiers s,m,l --size l --seeds 2 --workers 128 --steps 2048 --slots 512 --chunk 16 --out runs/hyak
python scripts/make_split.py --pool runs/hyak/pool.json --out runs/hyak/split.json --heldout_class OOB --heldout_frac 0.3 --trivial 512 --keep_unfound
echo POOL_DONE

#!/usr/bin/env bash
#SBATCH --job-name=autodev-prep-fresh
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# K-Fresh suite: planted defects on the 64 generated levels (never seen by any search). All instances go to "search" of this
# pool (the whole suite is held-out relative to the discovery runs); one level is held out by the split builder's minimum.
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
mkdir -p runs/hyak/fresh_defects
python scripts/build_pool.py --levels_index runs/hyak/fresh_levels/index.json --size l --seeds 2 --programs random,repeat_random,goexplore --workers 128 --steps 2048 --slots 512 --chunk 16 --out runs/hyak/fresh_defects
python scripts/make_split.py --pool runs/hyak/fresh_defects/pool.json --out runs/hyak/fresh_defects/split.json --heldout_class none --heldout_frac 0.02 --trivial 512 --keep_unfound
echo FRESH_POOL_DONE

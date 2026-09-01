#!/usr/bin/env bash
#SBATCH --job-name=autodev-prep-goal
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# K-Goal suite: unmodified shipped levels, target = the level's own win condition. Pool (probes by the hand-designed testers),
# then a level split (30% held out by hash order, no held-out class).
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
mkdir -p runs/hyak/goal
python scripts/build_pool.py --tiers s,m,l --size l --seeds 1 --classes GOAL --programs random,repeat_random,goexplore --workers 128 --steps 2048 --slots 512 --chunk 16 --out runs/hyak/goal
python scripts/make_split.py --pool runs/hyak/goal/pool.json --out runs/hyak/goal/split.json --heldout_class none --heldout_frac 0.3 --trivial 512 --keep_unfound
echo GOAL_POOL_DONE

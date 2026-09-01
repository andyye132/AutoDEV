#!/usr/bin/env bash
#SBATCH --job-name=autodev-eval
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# Evaluate hand-designed + evolved programs on search / held-out-levels / held-out-class splits at scale.
#   env: PROGRAM_FILES=comma-separated .py files, TAG=name, SEEDS=0,1,2,3,4, STEPS=2048, WORKERS=128, EXTRA_ARGS="--restarts 4"
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
export AUTODEV_SPLIT="${SPLIT:-runs/hyak/split.json}"
# lists may be passed colon-separated (sbatch --export splits on commas)
PROGS="${PROGRAMS-random,repeat_random,goexplore}"; PROGS="${PROGS//:/,}"; PFILES="${PROGRAM_FILES:-}"; PFILES="${PFILES//:/,}"
# evaluation seeds are disjoint from the search fitness seeds 0-2 (Opus audit H5)
SEEDS_="${SEEDS:-100,101,102,103,104}"; SEEDS_="${SEEDS_//:/,}"; SPL="${SPLITS:-search,heldout_levels_instances,heldout_class_instances}"; SPL="${SPL//:/,}"
python scripts/eval_baselines.py --programs "$PROGS" --program_files "$PFILES" \
  --splits "$SPL" --seeds "$SEEDS_" --workers "${WORKERS:-128}" --steps "${STEPS:-2048}" --size l \
  --out runs/hyak/eval_${TAG:-programs}.json ${EXTRA_ARGS:-}
echo EVAL_DONE

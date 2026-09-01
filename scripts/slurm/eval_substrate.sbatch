#!/usr/bin/env bash
#SBATCH --job-name=autodev-eval-sub
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# Evaluate frozen programs on a non-Kinetix substrate (Gymnax/Craftax) with the same protocol. Colon-separated lists.
#   env: SUBSTRATE=gymnax:MountainCar-v0 SPLIT=runs/substrates/gymnax_MountainCar-v0_split.json PROGRAMS PROGRAM_FILES SPLITS SEEDS WORKERS STEPS SLOTS CHUNK TAG
export AUTODEV_ROOT=/gscratch/raivn/andy132/AutoDev
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/autodev_multi
cd "$AUTODEV_ROOT"; mkdir -p logs runs/hyak
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export AUTODEV_SPLIT="${SPLIT:?split path}"
echo "job $SLURM_JOB_ID on $(hostname) started $(date); gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); substrate ${SUBSTRATE:?}"
PROGS="${PROGRAMS-random,repeat_random,goexplore}"; PROGS="${PROGS//:/,}"; PFILES="${PROGRAM_FILES:-}"; PFILES="${PFILES//:/,}"
SEEDS_="${SEEDS:-0,1,2,3,4}"; SEEDS_="${SEEDS_//:/,}"; SPL="${SPLITS:-search,heldout_levels_instances}"; SPL="${SPL//:/,}"
python scripts/eval_baselines.py --substrate "$SUBSTRATE" --programs "$PROGS" --program_files "$PFILES" --splits "$SPL" --seeds "$SEEDS_" \
  --workers "${WORKERS:-64}" --steps "${STEPS:-2048}" --slots "${SLOTS:-128}" --chunk "${CHUNK:-8}" --out runs/hyak/eval_${TAG:-substrate}.json
echo EVAL_DONE

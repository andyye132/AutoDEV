#!/usr/bin/env bash
#SBATCH --job-name=autodev-craftax
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# C-Ach suite (Craftax achievements) with the SAME frozen programs; uses the separate craftax conda env.
#   env: PROGRAMS, PROGRAM_FILES, WORLDS=4, ACHS=0,1,5,9,11,20,29,61,28,49, WORKERS=32, STEPS=2048, SLOTS=64, SEEDS=0,1,2, CHUNK=4, TAG
export AUTODEV_ROOT=/gscratch/raivn/andy132/AutoDev
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/autodev_multi
cd "$AUTODEV_ROOT"; mkdir -p logs runs/hyak
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
echo "job $SLURM_JOB_ID on $(hostname) started $(date); gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
PROGS="${PROGRAMS-random,repeat_random,goexplore}"; PROGS="${PROGS//:/,}"; PFILES="${PROGRAM_FILES:-}"; PFILES="${PFILES//:/,}"
ACHS_="${ACHS:-0,1,5,9,11,20,29,61,28,49}"; ACHS_="${ACHS_//:/,}"; SEEDS_="${SEEDS:-0,1,2}"; SEEDS_="${SEEDS_//:/,}"
python scripts/eval_craftax.py --programs "$PROGS" --program_files "$PFILES" --worlds "${WORLDS:-4}" \
  --achievements "$ACHS_" --workers "${WORKERS:-32}" --steps "${STEPS:-2048}" --slots "${SLOTS:-64}" --seeds "$SEEDS_" \
  --chunk "${CHUNK:-4}" --out runs/hyak/eval_craftax${TAG:-}.json
echo CRAFTAX_DONE

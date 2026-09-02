#!/usr/bin/env bash
#SBATCH --job-name=autodev-chimera
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# Re-verify the exploit: run PROGRAM under fixed harness and legacy race (same 16 instances, GPU), count chimera slots.
#   env: PROGRAM=path, N=16, WORKERS=128, STEPS=2048, OUT=results/chimera_reverify.json
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
JAX_PLATFORMS=cuda python scripts/chimera_audit.py --program "${PROGRAM:-runs/evolve_overnight/candidates/cand1_gen037_0.6994.py}" \
  --n "${N:-16}" --workers "${WORKERS:-128}" --steps "${STEPS:-2048}" --out "${OUT:-results/chimera_reverify.json}"

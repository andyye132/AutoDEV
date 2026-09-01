#!/usr/bin/env bash
#SBATCH --job-name=autodev-evolve
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
# The evolution driver runs on this node and evaluates candidates locally on its GPU (job_type=local).
# Requires an LLM API key in the environment (ANTHROPIC_API_KEY or OPENAI_API_KEY) or a headless CLI agent.
python evolve/run_evolution.py --generations ${GENERATIONS:-150} --results_dir runs/evolve_$(date +%Y%m%d_%H%M)

#!/usr/bin/env bash
#SBATCH --job-name=autodev-evalsrv-multi
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# Multi-engine fitness server: weighted mean over substrates' search splits (Kinetix reachable split + Craftax + Gymnax).
#   env: QUEUE=runs/hyak/evalq7 SEEDS=0,1,2 MAX_HOURS=11.5 SUBSTRATES="kind|split|weight|workers|steps|slots|chunk;..."
export AUTODEV_ROOT=/gscratch/raivn/andy132/AutoDev
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/autodev_multi
cd "$AUTODEV_ROOT"; mkdir -p logs runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu JAX_TRACEBACK_FILTERING=off
echo "job $SLURM_JOB_ID on $(hostname) started $(date); gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
DEFAULT_SUBS="kinetix:l|runs/hyak/split_reachable.json|1.0|128|1024|256|16;craftax|runs/substrates/craftax_split.json|1.0|32|2048|64|4;gymnax:MountainCar-v0|runs/substrates/gymnax_MountainCar-v0_split.json|0.34|64|512|128|6;gymnax:Acrobot-v1|runs/substrates/gymnax_Acrobot-v1_split.json|0.33|64|512|128|6;gymnax:DeepSea-bsuite|runs/substrates/gymnax_DeepSea-bsuite_split.json|0.33|64|512|128|6"
python scripts/eval_server.py --queue "${QUEUE:-runs/hyak/evalq7}" --seeds "${SEEDS:-0,1,2}" --max_hours "${MAX_HOURS:-11.5}" --substrates "${SUBSTRATES:-$DEFAULT_SUBS}"
echo EVALSRV_DONE

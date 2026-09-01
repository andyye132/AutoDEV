#!/usr/bin/env bash
#SBATCH --job-name=autodev-bench
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# Method bench on a GPU: same stratified subset and protocol as the Mac bench (fixed harness, verified hits) at a bigger budget.
#   env: PROGRAMS (colon-separated program files), TAG, SEEDS (colon list), WORKERS, STEPS, SPLIT
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh; cd /gscratch/raivn/andy132/AutoDev
echo "job $SLURM_JOB_ID on $(hostname) started $(date); gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
P="${PROGRAMS//:/,}"; S="${SEEDS:-100:101:102}"; S="${S//:/,}"
python scripts/bench_methods.py --tag "${TAG:-gpu}" --programs "$P" --split "${SPLIT:-runs/hyak/split_bench.json}" --seeds "$S" --workers "${WORKERS:-128}" --steps "${STEPS:-1024}" --chunk 8
echo BENCH_DONE

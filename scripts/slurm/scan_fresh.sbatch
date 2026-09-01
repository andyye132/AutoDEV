#!/usr/bin/env bash
#SBATCH --job-name=autodev-scan-fresh
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
PROGRAM_FILES=${PROGRAM_FILES:-}
PFILES="${PROGRAM_FILES//:/,}"; PROGS="${PROGRAMS-random,repeat_random,goexplore}"; PROGS="${PROGS//:/,}"
# waits for the fresh levels index written by prep_levels
while [ ! -f runs/hyak/fresh_levels/index.json ]; do sleep 60; done
python scripts/scan_clean.py --levels_index runs/hyak/fresh_levels/index.json --size l --programs "$PROGS" ${PFILES:+--program_files $PFILES} --workers "${WORKERS:-256}" --steps "${STEPS:-4096}" --slots "${SLOTS:-1024}" --seeds "${SEEDS:-3}" --chain_log "${CHAIN_LOG:-262144}" --out runs/hyak/scan_fresh${TAG:-}.json
echo SCAN_FRESH_DONE

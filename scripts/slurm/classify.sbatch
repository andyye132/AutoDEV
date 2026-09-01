#!/usr/bin/env bash
#SBATCH --job-name=autodev-classify
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# GPU-side replay classification of a scan file (to compare with the laptop CPU replay). env: SCAN=runs/hyak/scan_fresh.json INDEX=1
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
python scripts/classify_hits.py --scan "${SCAN:-runs/hyak/scan_fresh.json}" --size l ${INDEX:+--index}
echo CLASSIFY_DONE

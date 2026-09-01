#!/usr/bin/env bash
#SBATCH --job-name=autodev-pipeline
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --output=/gscratch/raivn/andy132/AutoDev/logs/%x-%j.out
# Full pipeline on one GPU node: pool -> split -> evolution (needs LLM access: ANTHROPIC_API_KEY, or the claude CLI for headless)
#   -> evaluate seed/baselines/evolved on all splits -> scan clean shipped + fresh levels with every tester -> bug report.
source /gscratch/raivn/andy132/AutoDev/scripts/slurm/_common.sh
set -x
SIZE=${SIZE:-l}; TIERS=${TIERS:-s,m,l}; GENS=${GENERATIONS:-120}; STAMP=$(date +%Y%m%d_%H%M)
export AUTODEV_STEPS=${AUTODEV_STEPS:-2048} AUTODEV_WORKERS=${AUTODEV_WORKERS:-128} AUTODEV_SEEDS=${AUTODEV_SEEDS:-0,1}
python scripts/build_pool.py --tiers $TIERS --size $SIZE --seeds 2 --workers 128 --steps 2048 --slots 512 --chunk 16
python scripts/make_split.py --heldout_class OOB --heldout_frac 0.3 --trivial 512 --keep_unfound
python scripts/gen_levels.py --n 64 --size $SIZE
python evolve/run_evolution.py --generations $GENS --results_dir runs/evolve_$STAMP --eval_jobs 2 --proposal_jobs 3 --models "${MODELS:-headless/claude@claude-sonnet-5?effort=medium}"
BEST=runs/evolve_$STAMP/best/main.py
python scripts/eval_baselines.py --programs random,repeat_random,goexplore --program_files evolve/initial.py,$BEST --seeds 0,1,2,3,4 --workers 128 --steps 2048 --size $SIZE --out runs/eval_all_$STAMP.json
python scripts/scan_clean.py --tiers $TIERS --size $SIZE --programs random,repeat_random,goexplore --program_files $BEST --workers 256 --steps 4096 --slots 1024 --seeds 3 --out runs/scan_shipped_$STAMP.json
python scripts/scan_clean.py --levels_index runs/fresh_levels/index.json --size $SIZE --programs random,repeat_random,goexplore --program_files $BEST --workers 256 --steps 4096 --slots 1024 --seeds 3 --out runs/scan_fresh_$STAMP.json
python scripts/design_checks.py --tiers $TIERS --size $SIZE --out runs/design_checks_$STAMP.json
python scripts/report_bugs.py --scans runs/scan_shipped_$STAMP.json,runs/scan_fresh_$STAMP.json --design runs/design_checks_$STAMP.json --out runs/bug_report_$STAMP.md --render --size $SIZE

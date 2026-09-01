#!/usr/bin/env bash
# After run 3 (rotated held-out class = ANOMALY) ends: export its top-k candidates, sync, and queue two Hyak evaluations on the
# rotated split (candidates; baselines + run-1 best for comparison), chained behind AFTER=<jobid> to respect the A40 cap.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
AFTER=${AFTER:?job id to chain behind}
python scripts/select_candidates.py --run runs/evolve_run3_rot --k 2 --out runs/evolve_run3_rot/candidates
ls runs/evolve_run3_rot/candidates/*.py
rsync -az --timeout=120 runs/evolve_run3_rot/candidates hyak:/gscratch/raivn/andy132/AutoDev/runs/evolve_run3_rot/
FILES=$(ls runs/evolve_run3_rot/candidates/*.py | tr '\n' ',' | sed 's/,$//')
ssh -o ConnectTimeout=20 hyak "cd /gscratch/raivn/andy132/AutoDev && J1=\$(sbatch --parsable --dependency=afterany:$AFTER --export=ALL,PROGRAMS=,PROGRAM_FILES=$FILES,TAG=run3_rot,SPLIT=runs/hyak/split_rot_anomaly.json scripts/slurm/eval_programs.sbatch) && echo run3 candidates eval: \$J1 && J2=\$(sbatch --parsable --dependency=afterany:\$J1 --export=ALL,PROGRAMS=random,repeat_random,goexplore,PROGRAM_FILES=runs/evolved/candidates/cand1_gen037_0.6994.py,TAG=baselines_rot,SPLIT=runs/hyak/split_rot_anomaly.json scripts/slurm/eval_programs.sbatch) && echo baselines on rotated split: \$J2"

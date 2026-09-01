#!/usr/bin/env bash
# Rival discovery scaffolds against the SAME Hyak fitness server/queue as the evolutionary runs (AUTODEV_REMOTE_EVAL=1).
#   BUDGET=6  scripts/launch_scaffolds.sh   -> pilot (test T5: plumbing + at least one program above the seed)
#   BUDGET=40 scripts/launch_scaffolds.sh   -> full comparison at the evolutionary runs' evaluation budget
# Seed = the run-1 best (same seed as run 4) so the three scaffolds start from the same program. Runs loop and tree concurrently;
# each evaluation is scored by whichever server is free (~11 min per evaluation, two servers).
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
BUDGET=${BUDGET:-6}; TAG=${TAG:-pilot}
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=7200 AUTODEV_REMOTE_POLL=20
SEED=${AUTODEV_INIT:-runs/evolve_overnight/candidates/cand1_gen037_0.6994.py}
for S in loop tree; do
  (nohup caffeinate -i python evolve/scaffolds.py --strategy $S --budget $BUDGET --seed_program $SEED --results_dir runs/scaffold_${S}_${TAG} \
     --model "headless/claude@claude-sonnet-5?effort=medium" > runs/scaffold_${S}_${TAG}.log 2>&1 < /dev/null &)
  echo "started scaffold=$S budget=$BUDGET -> runs/scaffold_${S}_${TAG}"
done

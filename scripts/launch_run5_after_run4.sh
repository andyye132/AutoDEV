#!/usr/bin/env bash
# Waits for run 4 to exit, then starts run 5 on the SAME fitness servers/queue (split.json, 3 seeds, 128x1024), seeded from the
# strongest independent design at smoke budget (movertarget) — tests whether the search improves a structurally different program.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
until grep -q "EVOLVE EXIT" runs/evolve_run4.log 2>/dev/null; do sleep 120; done
# make sure at least one fitness server is still running
until ssh -o ConnectTimeout=20 -o BatchMode=yes hyak 'squeue -u andy132 -h -n autodev-evalsrv -t R -o %i' 2>/dev/null | grep -q .; do echo "no server running at $(date)" >> runs/evolve_run5.log; sleep 300; done
echo "run 4 exited; launching run 5 (seed movertarget) at $(date)" > runs/evolve_run5.log
export AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_TIMEOUT=5400 AUTODEV_REMOTE_POLL=20
export AUTODEV_INIT=evolve/proposals3/movertarget.py
python evolve/run_evolution.py --generations 40 --results_dir runs/evolve_run5_mover --eval_jobs 2 --proposal_jobs 2 --islands 2 --archive_size 30 --max_cost 40 \
  --models "headless/claude@claude-sonnet-5?effort=medium,headless/claude@claude-opus-5?effort=high" >> runs/evolve_run5.log 2>&1
echo "EVOLVE EXIT $?" >> runs/evolve_run5.log

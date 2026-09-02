#!/usr/bin/env bash
# Evaluate all methods on the graded bug-finding benchmark at matched budget (128x2048). Testers via eval_baselines,
# PPO (default + high-entropy, best-of) via eval_ppo_baselines. Split via AUTODEV_SPLIT / --split_path.
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu AUTODEV_SPLIT=runs/graded/split.json
R7=runs/evolve_run7_multi/candidates/run7_best_gen035_0.5888.py
echo "=== testers (LLM-evolved + baselines) ==="
.venv/bin/python scripts/eval_baselines.py --substrate kinetix:l \
  --programs random,repeat_random,goexplore \
  --program_files "$R7,evolve/methods/saturate.py,evolve/methods/split_pump.py,evolve/proposals/macrosweep.py" \
  --splits search --seeds 100,101,102 --workers 128 --steps 2048 --size l --out runs/graded/eval_testers.json 2>&1 | grep -E "search|ANOMALY" | sed 's/^/  /'
echo "=== PPO default (ent=0.01) ==="
.venv/bin/python scripts/eval_ppo_baselines.py --split_path runs/graded/split.json --splits search --seeds 100,101,102 \
  --size l --num_envs 128 --rollout 64 --num_updates 32 --ent_coef 0.01 --out runs/graded/eval_ppo_default.json 2>&1 | grep -E "search" | sed 's/^/  /'
echo "=== PPO high-entropy (ent=0.1, tuned) ==="
.venv/bin/python scripts/eval_ppo_baselines.py --split_path runs/graded/split.json --splits search --seeds 100,101,102 \
  --size l --num_envs 128 --rollout 64 --num_updates 32 --ent_coef 0.1 --out runs/graded/eval_ppo_tuned.json 2>&1 | grep -E "search" | sed 's/^/  /'
echo "GRADED EVAL DONE"

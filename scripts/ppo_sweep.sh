#!/usr/bin/env bash
# PPO hyperparameter sweep on the tasks where PPO fails hardest, at 1x budget. Closes the "untuned PPO" fairness question.
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu
sweep_task () { local name=$1 sub=$2 split=$3
  for lr in 0.0001 0.0003 0.001; do for ent in 0.0 0.01 0.05; do
    echo "=== $name lr=$lr ent=$ent ==="
    .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
      --which search --objectives count,curiosity_proximity --seeds 100,101,102 \
      --num_envs 64 --rollout 64 --num_updates 32 --chunk 16 --lr $lr --ent_coef $ent \
      --out "runs/ppo_sweep_${name}_lr${lr}_ent${ent}.json" 2>&1 | grep -E "score.*detected" | sed "s/^/  /"
  done; done
}
sweep_task doorkey "xminigrid:MiniGrid-DoorKey-8x8" runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json
sweep_task breakout "gymnax:Breakout-MinAtar" runs/substrates/gymnax_Breakout-MinAtar_split.json
echo "SWEEP DONE"

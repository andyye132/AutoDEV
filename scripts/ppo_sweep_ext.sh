#!/usr/bin/env bash
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu
sub="xminigrid:MiniGrid-DoorKey-8x8"; split=runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json
for lr in 0.00003 0.0001; do for ent in 0.1 0.2; do
  echo "=== doorkey-ext lr=$lr ent=$ent ==="
  .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
    --which search --objectives count,curiosity_proximity --seeds 100,101,102 \
    --num_envs 64 --rollout 64 --num_updates 32 --chunk 16 --lr $lr --ent_coef $ent \
    --out "runs/ppo_sweep_doorkey_lr${lr}_ent${ent}.json" 2>&1 | grep -E "score.*detected" | sed "s/^/  /"
done; done
echo "EXT SWEEP DONE"

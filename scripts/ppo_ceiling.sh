#!/usr/bin/env bash
cd /Users/andy132/projects/research/ACT/AutoDev; export JAX_PLATFORMS=cpu
sub="xminigrid:MiniGrid-DoorKey-8x8"; split=runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json
for ent in 0.3 0.5; do
  echo "=== doorkey-ceil lr=0.0001 ent=$ent ==="
  .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
    --which search --objectives count,curiosity_proximity --seeds 100,101,102 \
    --num_envs 64 --rollout 64 --num_updates 32 --chunk 16 --lr 0.0001 --ent_coef $ent \
    --out "runs/ppo_sweep_doorkey_lr0.0001_ent${ent}.json" 2>&1 | grep -E "score.*detected" | sed "s/^/  /"
done
echo "CEILING DONE"

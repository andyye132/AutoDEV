#!/usr/bin/env bash
# PPO budget-scaling curves (4x and 16x) across the diverse substrates, to show where/if PPO catches up to the
# zero-training LLM tester — the honest anti-"you-under-trained-PPO" evidence. 4x=num_updates 128, 16x=512.
# The LLM tester's budget is fixed at 1x (131072); these curves scale ONLY PPO's budget.
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu
OBJ="count,icm,curiosity_proximity"   # the 3 that did best at 1x (keep it cheap at high budget)
SEEDS="100,101,102"; NENV=64; ROLL=64
run () { local name=$1 sub=$2 split=$3 nup=$4 tag=$5
  echo "=== $name ${tag} ($sub) ==="
  .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
    --which search --objectives "$OBJ" --seeds "$SEEDS" \
    --num_envs $NENV --rollout $ROLL --num_updates $nup --chunk 16 --out "runs/ppo_${name}_${tag}.json" 2>&1 | grep -vE "WARNING|Grain"
}
for tag_nup in "4x:128" "16x:512"; do
  tag=${tag_nup%%:*}; nup=${tag_nup##*:}
  run doorkey   "xminigrid:MiniGrid-DoorKey-8x8" runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json $nup $tag
  run fourrooms "xminigrid:MiniGrid-FourRooms"    runs/substrates/xminigrid_MiniGrid-FourRooms_split.json $nup $tag
  run breakout  "gymnax:Breakout-MinAtar"         runs/substrates/gymnax_Breakout-MinAtar_split.json $nup $tag
  run sokoban   "jumanji:Sokoban-v0"              runs/substrates/jumanji_Sokoban_split.json $nup $tag
done
echo "ALL PPO BUDGET CURVES DONE"

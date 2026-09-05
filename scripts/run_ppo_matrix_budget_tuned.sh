#!/usr/bin/env bash
# TUNED PPO budget-scaling curves (4x, 16x) — the HONEST Fig-2. Uses the same tuned config (lr=1e-4, ent=0.2)
# that produced the 1x tuned table, so 1x/4x/16x are one consistent tuned curve. LLM tester stays fixed at 1x.
# Prior budget curves used DEFAULT (under-tuned) PPO and understated where RL lands with more budget.
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu
LR=${TUNED_LR:-0.0001}; ENT=${TUNED_ENT:-0.2}
OBJ="count,icm,curiosity_proximity"     # best-of at 1x; keep cheap at high budget
SEEDS="100,101,102"; NENV=64; ROLL=64
echo "tuned budget curves: lr=$LR ent=$ENT"
run () { local name=$1 sub=$2 split=$3 nup=$4 tag=$5
  echo "=== $name ${tag} (tuned) ==="
  .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
    --which search --objectives "$OBJ" --seeds "$SEEDS" \
    --num_envs $NENV --rollout $ROLL --num_updates $nup --chunk 16 --lr $LR --ent_coef $ENT \
    --out "runs/ppo_${name}_${tag}_tuned.json" 2>&1 | grep -E "score.*detected|substrate=" | sed 's/^/  /'
}
for tag_nup in "4x:128" "16x:512"; do
  tag=${tag_nup%%:*}; nup=${tag_nup##*:}
  run doorkey   "xminigrid:MiniGrid-DoorKey-8x8" runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json $nup $tag
  run fourrooms "xminigrid:MiniGrid-FourRooms"    runs/substrates/xminigrid_MiniGrid-FourRooms_split.json $nup $tag
  run breakout  "gymnax:Breakout-MinAtar"         runs/substrates/gymnax_Breakout-MinAtar_split.json $nup $tag
  run sokoban   "jumanji:Sokoban-v0"              runs/substrates/jumanji_Sokoban_split.json $nup $tag
done
echo "TUNED BUDGET CURVES DONE"

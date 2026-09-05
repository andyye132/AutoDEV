#!/usr/bin/env bash
# 16x-only tuned budget runs (relaunch after the 4x/16x run was killed mid-DoorKey-16x). Same tuned config as the table.
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu
LR=0.0001; ENT=0.2; OBJ="count,icm,curiosity_proximity"; SEEDS="100,101,102"; NENV=64; ROLL=64; NUP=512
run () { local name=$1 sub=$2 split=$3
  echo "=== $name 16x (tuned) ==="
  .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
    --which search --objectives "$OBJ" --seeds "$SEEDS" \
    --num_envs $NENV --rollout $ROLL --num_updates $NUP --chunk 16 --lr $LR --ent_coef $ENT \
    --out "runs/ppo_${name}_16x_tuned.json" 2>&1 | grep -E "score.*detected|substrate=" | sed 's/^/  /'
}
run doorkey   "xminigrid:MiniGrid-DoorKey-8x8" runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json
run fourrooms "xminigrid:MiniGrid-FourRooms"    runs/substrates/xminigrid_MiniGrid-FourRooms_split.json
run breakout  "gymnax:Breakout-MinAtar"         runs/substrates/gymnax_Breakout-MinAtar_split.json
run sokoban   "jumanji:Sokoban-v0"              runs/substrates/jumanji_Sokoban_split.json
echo "16X TUNED DONE"

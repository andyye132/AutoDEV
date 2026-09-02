#!/usr/bin/env bash
# Full PPO-tester matrix on CPU across diverse substrates, matched budget (64x64x32 = 131072 env steps = the substrate-suite tester budget).
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu
OBJ="count,rnd,icm,curiosity_proximity,zero"; SEEDS="100,101,102"; NENV=64; ROLL=64; NUP=32
run () { local name=$1 sub=$2 split=$3
  echo "=== $name ($sub) ==="
  .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
    --which search,heldout_levels_instances --objectives "$OBJ" --seeds "$SEEDS" \
    --num_envs $NENV --rollout $ROLL --num_updates $NUP --chunk 16 --out "runs/ppo_${name}.json" 2>&1 | grep -vE "WARNING|Grain"
}
run doorkey   "xminigrid:MiniGrid-DoorKey-8x8" runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json
run fourrooms "xminigrid:MiniGrid-FourRooms"    runs/substrates/xminigrid_MiniGrid-FourRooms_split.json
run breakout  "gymnax:Breakout-MinAtar"         runs/substrates/gymnax_Breakout-MinAtar_split.json
run sokoban   "jumanji:Sokoban-v0"              runs/substrates/jumanji_Sokoban_split.json
run mountaincar "gymnax:MountainCar-v0"         runs/substrates/gymnax_MountainCar-v0_split.json
echo "ALL PPO MATRIX DONE"

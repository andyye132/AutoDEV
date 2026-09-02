#!/usr/bin/env bash
# Re-run PPO on ALL tasks with the FROZEN tuned hyperparameters (found by the sweep on DoorKey/Breakout), 5 seeds.
# The fair protocol: sweep on a couple tasks, freeze the best config, apply everywhere. Closes the "untuned PPO" question.
#   env: TUNED_LR (default 0.0001), TUNED_ENT (default 0.05)
cd /Users/andy132/projects/research/ACT/AutoDev
export JAX_PLATFORMS=cpu
LR=${TUNED_LR:-0.0001}; ENT=${TUNED_ENT:-0.05}
echo "tuned PPO: lr=$LR ent=$ENT, 5 seeds"
run () { local name=$1 sub=$2 split=$3 nenv=${4:-64} roll=${5:-64} nup=${6:-32}
  echo "=== $name (tuned) ==="
  .venv/bin/python scripts/eval_ppo_substrate.py --substrate "$sub" --split "$split" \
    --which search,heldout_levels_instances --objectives count,rnd,icm,curiosity_proximity,zero \
    --seeds 100,101,102,103,104 --num_envs $nenv --rollout $roll --num_updates $nup --chunk 16 \
    --lr $LR --ent_coef $ENT --out "runs/ppo_${name}_tuned.json" 2>&1 | grep -E "score.*detected|substrate=" | sed "s/^/  /"
}
run doorkey     "xminigrid:MiniGrid-DoorKey-8x8" runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json
run fourrooms   "xminigrid:MiniGrid-FourRooms"    runs/substrates/xminigrid_MiniGrid-FourRooms_split.json
run breakout    "gymnax:Breakout-MinAtar"         runs/substrates/gymnax_Breakout-MinAtar_split.json
run sokoban     "jumanji:Sokoban-v0"              runs/substrates/jumanji_Sokoban_split.json
run mountaincar "gymnax:MountainCar-v0"           runs/substrates/gymnax_MountainCar-v0_split.json
run deepsea24   "gymnax:DeepSea-bsuite:24"        runs/substrates/gymnax_DeepSea-bsuite_split.json
run craftax     "craftax"                          runs/substrates/craftax_split.json 32 64 32
echo "TUNED PPO RERUN DONE"

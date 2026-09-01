#!/usr/bin/env bash
# One-time env build on the Hyak login node (pip only, no compute). Env lives in gscratch per CLUSTER_GUIDE.md.
set -euo pipefail
ENV_PREFIX=/mmfs1/gscratch/raivn/andy132/conda_envs/autodev
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda create -y --prefix "$ENV_PREFIX" python=3.11
conda activate "$ENV_PREFIX"
pip install --upgrade pip
pip install "jax[cuda12]" "kinetix-env==3.0.0" optax flax shinka-evolve numpy pillow
python - <<'PY'
import jax; print("jax", jax.__version__, "devices:", jax.devices())
import kinetix, shinka; print("kinetix ok, shinka ok")
PY
echo "Setup done."

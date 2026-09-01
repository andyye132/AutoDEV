# Sourced by the sbatch scripts. Conventions from cluster_scripts_backup/CLUSTER_GUIDE.md (bristle).
export AUTODEV_ROOT=/gscratch/raivn/andy132/AutoDev
export CONDA_ENV=/mmfs1/gscratch/raivn/andy132/conda_envs/autodev
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$AUTODEV_ROOT"
mkdir -p logs runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
echo "job $SLURM_JOB_ID on $(hostname) started $(date); gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

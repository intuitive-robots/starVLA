#!/bin/bash
# Evaluate one DROID checkpoint with four deterministic diffusion seeds.

#SBATCH --job-name=droid_open_loop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:59:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/droid_open_loop_%j.out
#SBATCH --error=slurm_logs/droid_open_loop_%j.err

set -eo pipefail

STARVLA_REPO="${STARVLA_REPO:-/e/project1/m3/blank4/code/starVLA}"
cd "$STARVLA_REPO"

ml load CUDA
export SLURM_MPI_TYPE=none
source /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/etc/profile.d/conda.sh
conda activate starVLA
set -u

export LD_LIBRARY_PATH="/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${STARVLA_REPO}:/e/home/jusers/blank4/jupiter/blank4/code/marigold_data:/e/home/jusers/blank4/jupiter/blank4/code/marigold_train:${PYTHONPATH:-}"
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR="/e/scratch/m3/blank4/cache/triton/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/e/scratch/m3/blank4/cache/inductor/${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export USE_TF=0 USE_TORCH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-64}"
export LEROBOT_PARQUET_CACHE_SIZE="${LEROBOT_PARQUET_CACHE_SIZE:-4}"
export LEROBOT_PREFETCH_MP4=0
export LEROBOT_SKIP_FILE_CHECK=1

BASE_SEED="${BASE_SEED:-0}"
export STARVLA_REPO BASE_SEED

# Booster nodes have four GPUs.  Run one independent deterministic diffusion
# seed per GPU; --seed_subdir prevents the shards from writing the same files.
srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --mpi=none --kill-on-bad-exit=1 \
    bash -c '
        set -euo pipefail
        cd "$STARVLA_REPO"
        seed=$((BASE_SEED + SLURM_PROCID))
        device="$SLURM_LOCALID"
        echo "open-loop seed=$seed rank=$SLURM_PROCID device=$device host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        python scripts/eval_droid_marigold_open_loop.py "$@" \
            --seed "$seed" --device "$device" --seed_subdir
    ' _ "$@"

#!/bin/bash
#SBATCH --job-name=starvla_libero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=07:30:00
#SBATCH --output=slurm_logs/starvla_%j.out
#SBATCH --error=slurm_logs/starvla_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
# Uncomment and set partition/account as needed:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

# Usage:
#   sbatch train_libero_slurm.sh
#   sbatch train_libero_slurm.sh --config path/to/my_config.yaml
#   sbatch train_libero_slurm.sh --config path/to/my_config.yaml --trainer.max_train_steps 50000 --run_id my_run

set -eo pipefail

# ── Parse --config; collect remaining args as CLI overrides ──────────────
#CONFIG_YAML="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
CONFIG_YAML="./examples/LIBERO/train_files/starvla_real_robot_qwen_08_base.yaml"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_YAML="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Environment modules ───────────────────────────────────────────────────────
ml load CUDA

# ── Conda ─────────────────────────────────────────────────────────────────────
source ~/blank4/envs/miniforge3/etc/profile.d/conda.sh
conda activate starVLA
set -u

# ── Library paths ─────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export TORCH_USE_CUDA_DSA=1

# ── HuggingFace / cache dirs ──────────────────────────────────────────────────
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR=/e/home/jusers/blank4/jupiter/blank4/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/e/home/jusers/blank4/jupiter/blank4/cache/inductor

# ── Offline / version flags ───────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export DISABLE_VERSION_CHECK=1

# ── NCCL / GLOO ───────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# ── Multi-node accelerate settings (single-node defaults) ─────────────────────
MASTER_ADDR="${SLURM_NODELIST:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}"

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p slurm_logs

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $NUM_PROCESSES"
echo "Config:        $CONFIG_YAML"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"

# ── Launch ────────────────────────────────────────────────────────────────────
# accelerate launch \
#     --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#     --num_processes "${NUM_PROCESSES}" \
#     starVLA/training/train_starvla.py \
#     --config_yaml "${CONFIG_YAML}" \
#     "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"



accelerate launch \
    --config_file starVLA/config/deepseeds/accelerate_multigpu_4gpu.yaml \
    --num_processes "${NUM_PROCESSES}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --use_deepspeed false \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

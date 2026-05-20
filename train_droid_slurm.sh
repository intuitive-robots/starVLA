#!/bin/bash
#SBATCH --job-name=starvla_droid
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/starvla_%j.out
#SBATCH --error=slurm_logs/starvla_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
# Uncomment and set partition/account as needed:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

# Usage:
#   sbatch train_droid_slurm.sh
#   sbatch train_droid_slurm.sh --config path/to/my_config.yaml
#   sbatch train_droid_slurm.sh --config path/to/my_config.yaml --trainer.max_train_steps 50000 --run_id my_run

set -eo pipefail

# ── Parse --config; collect remaining args as CLI overrides ──────────────
CONFIG_YAML="./examples/DROID/train_files/train_droid.yaml"

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

# ── Multi-node accelerate settings ─────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
TOTAL_GPUS="${TOTAL_GPUS:-$((SLURM_NNODES * GPUS_PER_NODE))}"

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p slurm_logs

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Nodes:         $SLURM_NNODES"
echo "Master addr:   $MASTER_ADDR"
echo "Master port:   $MASTER_PORT"
echo "GPUs / node:   $GPUS_PER_NODE"
echo "Total GPUs:    $TOTAL_GPUS"
echo "Config:        $CONFIG_YAML"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"

# ── Launch ────────────────────────────────────────────────────────────────────
LAUNCH_CMD=(
    accelerate launch
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
    --machine_rank "$SLURM_PROCID"
    --num_machines "$SLURM_NNODES"
    --num_processes "$TOTAL_GPUS"
    starVLA/training/train_starvla.py
    --config_yaml "$CONFIG_YAML"
)

if ((${#EXTRA_ARGS[@]})); then
    LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Launch command: ${LAUNCH_CMD[*]}"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 "${LAUNCH_CMD[@]}"

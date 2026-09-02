#!/bin/bash
#SBATCH --job-name=starvla_libero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=09:30:00
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

# `sbatch` executes a spool COPY of this script, so BASH_SOURCE[0] points under Slurm's
# spool directory rather than this checkout. Anchor explicitly to the shared repository.
STARVLA_REPO=/e/project1/m3/blank4/code/starVLA
cd "$STARVLA_REPO"

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
# Large, variable-length CoT batches can fragment the CUDA allocator between
# forwards.  Callers may override this, but expandable segments are the safer
# default for long-running VLA jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Offline / version flags ───────────────────────────────────────────────────
# USE_TF=0: transformers otherwise probes for TensorFlow, and that scan over
# site-packages is wide enough to hit a stale NFS handle and take the process out
# with a SIGBUS before any traceback -- diagnosed as the cause of t3_maskft's five
# consecutive exit-7 failures on five different nodes.
export USE_TF=0 USE_TORCH=1
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
ACCELERATE_CONFIG_FILE="${STARVLA_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

# ── Pre-flight: is this node's view of the filesystems intact? ────────────────
# Nodes intermittently come up with STALE NFS handles for /e/project1 and $HOME. A process
# that starts anyway dies mid-import with `OSError: [Errno 116] Stale file handle` followed
# by SIGBUS when an mmap'd page vanishes -- under accelerate that surfaces only as a bare
# exit 7 with no traceback. It cost five consecutive t3_maskft failures on five nodes,
# misdiagnosed twice before instrumenting a single process directly.
# This only REPORTS and exits: it turns a silent 2-minute exit-7 with no traceback into a
# one-line diagnosis. It deliberately does not resubmit -- relaunching is a human decision.
preflight_ok=1
for d in "/e/project1/m3/blank4/code/starVLA" "$HOME/blank4/envs/miniforge3" "/e/scratch/m3/blank4"; do
    ls "$d" >/dev/null 2>&1 || { echo "PREFLIGHT FAIL: cannot stat $d on $(hostname)"; preflight_ok=0; }
done
python -c "import torch, transformers" >/dev/null 2>&1 || { echo "PREFLIGHT FAIL: import torch/transformers on $(hostname)"; preflight_ok=0; }
if [ "$preflight_ok" -ne 1 ]; then
    echo "PREFLIGHT FAILED on $(hostname): this node's mounts are stale. Not resubmitting."
    exit 1
fi
echo "preflight OK on $(hostname)"

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $NUM_PROCESSES"
echo "Config:        $CONFIG_YAML"
echo "Accelerate:    $ACCELERATE_CONFIG_FILE"
echo "CUDA allocator:${PYTORCH_CUDA_ALLOC_CONF}"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
    --config_file "${ACCELERATE_CONFIG_FILE}" \
    --num_processes "${NUM_PROCESSES}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --use_deepspeed true \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"



# accelerate launch \
#     --config_file starVLA/config/deepseeds/accelerate_multigpu_4gpu.yaml \
#     --num_processes "${NUM_PROCESSES}" \
#     starVLA/training/train_starvla.py \
#     --config_yaml "${CONFIG_YAML}" \
#     --use_deepspeed false \
#     "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

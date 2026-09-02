#!/bin/bash
# Two-node LIBERO training: one launcher task per node, four Accelerate workers per node.
# Global batch for the cam3d retry is 8 examples/GPU * 8 GPUs * accumulation 1 = 64.
#SBATCH --job-name=starvla_libero_2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/starvla_%j.out
#SBATCH --error=slurm_logs/starvla_%j.err
#SBATCH --partition=booster
#SBATCH -A m3

set -euo pipefail

STARVLA_REPO=/e/project1/m3/blank4/code/starVLA
cd "$STARVLA_REPO"

CONFIG_YAML=./examples/LIBERO/train_files/starvla_real_robot_qwen_08_base.yaml
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

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
NUM_MACHINES="$SLURM_NNODES"
NUM_PROCESSES="$((SLURM_NNODES * 4))"
ACCELERATE_CONFIG_FILE="${STARVLA_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_memory.yaml}"

export MASTER_ADDR MASTER_PORT NUM_MACHINES NUM_PROCESSES ACCELERATE_CONFIG_FILE

echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "GPUs:          $NUM_PROCESSES (4 per node)"
echo "Config:        $CONFIG_YAML"
echo "Accelerate:    $ACCELERATE_CONFIG_FILE"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=4 \
    --kill-on-bad-exit=1 bash -c '
        # The Conda CUDA activation hook probes optional variables before defining them,
        # so match the proven single-node launcher and enable nounset only afterwards.
        set -eo pipefail
        config_yaml="$1"
        shift

        cd /e/project1/m3/blank4/code/starVLA
        ml load CUDA
        source /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/etc/profile.d/conda.sh
        conda activate starVLA
        set -u

        export LD_LIBRARY_PATH="/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        export TORCH_USE_CUDA_DSA=1
        export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
        export TRANSFORMERS_CACHE="$HF_HOME/transformers"
        export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
        export HF_DATASETS_CACHE="$HF_HOME/datasets"
        export TRITON_CACHE_DIR="/e/scratch/m3/blank4/cache/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
        export TORCHINDUCTOR_CACHE_DIR="/e/scratch/m3/blank4/cache/inductor/${SLURM_JOB_ID}/node${SLURM_NODEID}"
        mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

        export USE_TF=0 USE_TORCH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export WANDB_MODE=offline DISABLE_VERSION_CHECK=1
        export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1
        export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=10000 NCCL_SOCKET_TIMEOUT_MS=360000

        preflight_ok=1
        for d in /e/project1/m3/blank4/code/starVLA /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3 /e/scratch/m3/blank4; do
            ls "$d" >/dev/null 2>&1 || { echo "PREFLIGHT FAIL: cannot stat $d on $(hostname)"; preflight_ok=0; }
        done
        python -c "import torch, transformers" >/dev/null 2>&1 || {
            echo "PREFLIGHT FAIL: import torch/transformers on $(hostname)"
            preflight_ok=0
        }
        gpu_count="$(nvidia-smi -L | wc -l)"
        if [[ "$gpu_count" -ne 4 ]]; then
            echo "PREFLIGHT FAIL: expected 4 visible GPUs on $(hostname), got $gpu_count"
            preflight_ok=0
        fi
        [[ "$preflight_ok" -eq 1 ]] || exit 1
        echo "preflight OK on $(hostname): machine_rank=$SLURM_NODEID GPUs=$gpu_count"

        accelerate launch \
            --config_file "$ACCELERATE_CONFIG_FILE" \
            --num_processes "$NUM_PROCESSES" \
            --num_machines "$NUM_MACHINES" \
            --machine_rank "$SLURM_NODEID" \
            --main_process_ip "$MASTER_ADDR" \
            --main_process_port "$MASTER_PORT" \
            starVLA/training/train_starvla.py \
            --config_yaml "$config_yaml" \
            --use_deepspeed true \
            "$@"
    ' _ "$CONFIG_YAML" "${EXTRA_ARGS[@]}"

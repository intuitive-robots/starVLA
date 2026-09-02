#!/bin/bash
# Generic multi-node DROID training launcher.
# Examples:
#   sbatch train_droid_new.sh
#   sbatch --nodes=4 train_droid_new.sh
#   sbatch --nodes=4 --export=ALL,GPUS_PER_NODE=4 train_droid_new.sh --trainer.max_train_steps 50000

#SBATCH --job-name=starvla_droid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:45:00
# Ask only the batch shell for an early warning. It converts this into a shared
# flag that every training rank observes at a safe optimizer-step boundary.
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm_logs/starvla_%j.out
#SBATCH --error=slurm_logs/starvla_%j.err
#SBATCH --partition=booster
#SBATCH -A m3

set -euo pipefail

#CONFIG_YAML=./examples/DROID/train_files/train_droid.yaml
CONFIG_YAML=./examples/DROID/train_files/train_droid_delta_eef.yaml
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

# Use the directory from which sbatch was invoked unless explicitly overridden.
STARVLA_REPO="${STARVLA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_MACHINES="$SLURM_NNODES"
NUM_PROCESSES="$((NUM_MACHINES * GPUS_PER_NODE))"

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
if [[ "${SYSTEMNAME:-}" =~ ^(juwelsbooster|juwels|jurecadc|jusuf)$ ]]; then
    MASTER_ADDR="${MASTER_ADDR}i"
fi
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

# This Accelerate config selects DeepSpeed and points at the ZeRO-2 JSON config.
ACCELERATE_CONFIG_FILE="${STARVLA_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_memory.yaml}"

export STARVLA_REPO GPUS_PER_NODE NUM_MACHINES NUM_PROCESSES
export MASTER_ADDR MASTER_PORT ACCELERATE_CONFIG_FILE

STARVLA_PREEMPT_FLAG="/e/scratch/m3/blank4/starvla_preempt/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$STARVLA_PREEMPT_FLAG")"
unlink "$STARVLA_PREEMPT_FLAG" 2>/dev/null || true
export STARVLA_PREEMPT_FLAG

request_preempt_checkpoint() {
    echo "$(date --iso-8601=seconds) Slurm time-limit signal received; requesting a safe checkpoint"
    touch "$STARVLA_PREEMPT_FLAG"
}
trap request_preempt_checkpoint USR1 TERM

mkdir -p slurm_logs

echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "GPUs / node:   $GPUS_PER_NODE"
echo "Total GPUs:    $NUM_PROCESSES"
echo "Master addr:   $MASTER_ADDR"
echo "Master port:   $MASTER_PORT"
echo "Repository:    $STARVLA_REPO"
echo "Config:        $CONFIG_YAML"
echo "Accelerate:    $ACCELERATE_CONFIG_FILE"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"

srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task="$GPUS_PER_NODE" \
    --kill-on-bad-exit=1 bash -c '
        set -eo pipefail
        config_yaml="$1"
        shift

        cd "$STARVLA_REPO"
        ml load CUDA
        source /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/etc/profile.d/conda.sh
        conda activate starVLA
        set -u

        export LD_LIBRARY_PATH="/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        export TORCH_USE_CUDA_DSA=1
        export PYTHONPATH="${STARVLA_REPO}:/e/home/jusers/blank4/jupiter/blank4/code/marigold_data:/e/home/jusers/blank4/jupiter/blank4/code/marigold_train:${PYTHONPATH:-}"

        export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
        export TRANSFORMERS_CACHE="$HF_HOME/transformers"
        export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
        export HF_DATASETS_CACHE="$HF_HOME/datasets"
        export TRITON_CACHE_DIR="/e/scratch/m3/blank4/cache/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
        export TORCHINDUCTOR_CACHE_DIR="/e/scratch/m3/blank4/cache/inductor/${SLURM_JOB_ID}/node${SLURM_NODEID}"
        mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

        export USE_TF=0 USE_TORCH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export WANDB_MODE=offline DISABLE_VERSION_CHECK=1 NO_ALBUMENTATIONS_UPDATE=1
        export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1
        export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=10000 NCCL_SOCKET_TIMEOUT_MS=360000

        # Keep caches bounded per worker.  These values are multiplied by
        # (nodes x ranks/node x DataLoader workers/rank).  Full-file MP4
        # prefetch is harmful for the ~195 MiB original DROID videos under
        # random episode sampling and is disabled by default upstream.
        export LEROBOT_PARQUET_CACHE_SIZE="${LEROBOT_PARQUET_CACHE_SIZE:-4}"
        export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-64}"
        export LEROBOT_PREFETCH_MP4="${LEROBOT_PREFETCH_MP4:-0}"
        export LEROBOT_SKIP_FILE_CHECK="${LEROBOT_SKIP_FILE_CHECK:-1}"

        visible_gpus="$(nvidia-smi -L | wc -l)"
        if [[ "$visible_gpus" -ne "$GPUS_PER_NODE" ]]; then
            echo "PREFLIGHT FAIL: expected $GPUS_PER_NODE visible GPUs on $(hostname), got $visible_gpus"
            exit 1
        fi
        python -c "import accelerate, torch, transformers" >/dev/null
        echo "preflight OK on $(hostname): machine_rank=$SLURM_NODEID GPUs=$visible_gpus"

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
    ' _ "$CONFIG_YAML" "${EXTRA_ARGS[@]}" &
srun_pid=$!

# A signal interrupts bash's wait builtin. Keep waiting while the srun process
# is alive so the distributed checkpoint can finish before the allocation ends.
srun_status=0
while true; do
    set +e
    wait "$srun_pid"
    srun_status=$?
    set -e
    if kill -0 "$srun_pid" 2>/dev/null; then
        continue
    fi
    break
done
exit "$srun_status"

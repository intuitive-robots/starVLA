#!/bin/bash
# Probe one per-GPU micro-batch size using the exact DROID/GR00T training stack.
# Usage: sbatch --export=ALL,BATCH_SIZE=16 probe_droid_batch_slurm.sh

#SBATCH --job-name=droid_bsz_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/droid_bsz_probe_%j.out
#SBATCH --error=slurm_logs/droid_bsz_probe_%j.err
#SBATCH --partition=booster
#SBATCH -A m3

set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:?submit with --export=ALL,BATCH_SIZE=<n>}"
PROBE_RUN_ID="${PROBE_RUN_ID:-droid_bsz_probe_${BATCH_SIZE}_${SLURM_JOB_ID}}"
STARVLA_REPO="${STARVLA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [[ "${SYSTEMNAME:-}" =~ ^(juwelsbooster|juwels|jurecadc|jusuf)$ ]]; then
    MASTER_ADDR="${MASTER_ADDR}i"
fi
MASTER_PORT="$((24000 + SLURM_JOB_ID % 16000))"

mkdir -p slurm_logs

srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=4 \
    --kill-on-bad-exit=1 bash -c '
        set -eo pipefail
        cd "$1"

        ml load CUDA
        source /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/etc/profile.d/conda.sh
        conda activate starVLA
        set -u

        export LD_LIBRARY_PATH="/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        export PYTHONPATH="${1}:/e/home/jusers/blank4/jupiter/blank4/code/marigold_data:/e/home/jusers/blank4/jupiter/blank4/code/marigold_train:${PYTHONPATH:-}"
        export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
        export TRANSFORMERS_CACHE="$HF_HOME/transformers"
        export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
        export HF_DATASETS_CACHE="$HF_HOME/datasets"
        export TRITON_CACHE_DIR="/e/scratch/m3/blank4/cache/triton/${SLURM_JOB_ID}"
        export TORCHINDUCTOR_CACHE_DIR="/e/scratch/m3/blank4/cache/inductor/${SLURM_JOB_ID}"
        mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export USE_TF=0 USE_TORCH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export WANDB_MODE=disabled DISABLE_VERSION_CHECK=1 NO_ALBUMENTATIONS_UPDATE=1
        export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1
        # Match the production launcher: random access to long DROID videos
        # must not copy complete MP4 files into every worker cache.
        export LEROBOT_PARQUET_CACHE_SIZE=4
        export LEROBOT_VIDEO_DECODER_CACHE_SIZE=64
        export LEROBOT_PREFETCH_MP4=0
        export LEROBOT_SKIP_FILE_CHECK=1

        echo "PROBE_START batch_size=$2 host=$(hostname) visible_gpus=$(nvidia-smi -L | wc -l)"

        accelerate launch \
            --config_file starVLA/config/deepseeds/deepspeed_zero2_memory.yaml \
            --num_processes 4 \
            --num_machines 1 \
            --machine_rank 0 \
            --main_process_ip "$3" \
            --main_process_port "$4" \
            starVLA/training/train_starvla.py \
            --config_yaml examples/DROID/train_files/train_droid_delta_eef.yaml \
            --use_deepspeed true \
            --run_id "$5" \
            --run_root_dir /e/scratch/m3/blank4/starvla_batch_probes \
            --datasets.vla_data.per_device_batch_size "$2" \
            --trainer.max_train_steps 2 \
            --trainer.num_warmup_steps 0 \
            --trainer.eval_interval 1 \
            --trainer.save_interval 1000000 \
            --trainer.logging_frequency 1 \
            --trainer.save_final_checkpoint false \
            --trainer.open_loop_eval false

        echo "PROBE_PASS batch_size=$2"
    ' _ "$STARVLA_REPO" "$BATCH_SIZE" "$MASTER_ADDR" "$MASTER_PORT" "$PROBE_RUN_ID"

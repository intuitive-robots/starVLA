#!/bin/bash
# One-node, four-rank comparison of DROID loader batching designs.
# Usage: sbatch benchmark_droid_loader_slurm.sh

#SBATCH --job-name=droid_loader_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/droid_loader_bench_%j.out
#SBATCH --error=slurm_logs/droid_loader_bench_%j.err
#SBATCH --partition=booster
#SBATCH -A m3

set -euo pipefail

STARVLA_REPO="${STARVLA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RESULT_DIR="${RESULT_DIR:-${STARVLA_REPO}/playground/dataloader_benchmarks/${SLURM_JOB_ID}}"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
mkdir -p "$RESULT_DIR" slurm_logs

srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=4 \
    --kill-on-bad-exit=1 bash -c '
        set -eo pipefail
        cd "$1"
        result_dir="$2"
        master_port="$3"

        ml load CUDA
        source /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/etc/profile.d/conda.sh
        conda activate starVLA
        set -u

        export PYTHONPATH="${PWD}:/e/home/jusers/blank4/jupiter/blank4/code/marigold_data:/e/home/jusers/blank4/jupiter/blank4/code/marigold_train:${PYTHONPATH:-}"
        export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
        export DISABLE_VERSION_CHECK=1 NO_ALBUMENTATIONS_UPDATE=1
        export LEROBOT_PARQUET_CACHE_SIZE=4
        export LEROBOT_VIDEO_DECODER_CACHE_SIZE=64
        export LEROBOT_PREFETCH_MP4=0
        export LEROBOT_SKIP_FILE_CHECK=1
        export NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0

        if [ "${BENCHMARK_PROFILE:-comparison}" = "tuning" ]; then
            specs=(
                "original_rr_w8_pf1 droid_lerobot_delta_eef round_robin 8 1 12"
                "original_rr_w16_pf2 droid_lerobot_delta_eef round_robin 16 2 12"
                "original_rr_w24_pf4 droid_lerobot_delta_eef round_robin 24 4 12"
                "original_rr_w24_pf8 droid_lerobot_delta_eef round_robin 24 8 12"
            )
        else
            specs=(
                "original_worker droid_lerobot_delta_eef worker_batched 8 1 20"
                "original_round_robin droid_lerobot_delta_eef round_robin 8 1 20"
                "resized_worker droid_lerobot_resized_success_delta_eef worker_batched 8 1 20"
                "resized_round_robin droid_lerobot_resized_success_delta_eef round_robin 8 1 20"
            )
        fi

        for spec in "${specs[@]}"
        do
            read -r name data_mix mode workers prefetch steps <<< "$spec"
            echo "===== $name ====="
            accelerate launch \
                --multi_gpu \
                --num_processes 4 \
                --num_machines 1 \
                --main_process_port "$master_port" \
                scripts/benchmark_droid_loader.py \
                --mode "$mode" \
                --data-mix "$data_mix" \
                --batch-size 32 \
                --num-workers "$workers" \
                --prefetch-factor "$prefetch" \
                --steps "$steps" \
                --consumer-delay 0.9 \
                --output "$result_dir/${name}.json"
            master_port=$((master_port + 1))
        done
    ' _ "$STARVLA_REPO" "$RESULT_DIR" "$MASTER_PORT"

echo "Results: $RESULT_DIR"

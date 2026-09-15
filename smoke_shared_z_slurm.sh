#!/bin/bash
#SBATCH --job-name=smoke_shared_z
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:30:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/sharedz_smoke_%j.out
#SBATCH --error=slurm_logs/sharedz_smoke_%j.err

set -uo pipefail

STARVLA=/e/project1/m3/blank4/code/starVLA
CFG="$STARVLA/examples/simBenchmarks/LIBERO/train_files"
TRAIN="$STARVLA/train_libero_slurm.sh"
TAG="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$STARVLA/slurm_logs/sharedz_smoke_${TAG}"
RUN_DIR="/e/scratch/m3/blank4/sharedz_smoke_runs/${TAG}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

configs=(
  ervla_zbase_pi_sharedz_control
  ervla_zunsup_pi_sharedz_dist
  ervla_zsup_pi_sharedz_ground_temporal
  ervla_zshuf_pi_sharedz_ground_temporal
  ervla_znodrop_pi_sharedz_ground_temporal
  ervla_zonly_pi_sharedz_ground_temporal
  ervla_zsupdec_pi_sharedz_ground_temporal
)

failed=0
for wave_start in 0 4; do
  remaining=$((${#configs[@]} - wave_start))
  wave_size=$((remaining < 4 ? remaining : 4))
  wave=("${configs[@]:wave_start:wave_size}")
  wave_csv=$(IFS=,; echo "${wave[*]}")
  gpu_map=$(seq -s, 0 $((wave_size - 1)))

  SMOKE_WAVE="$wave_csv" SMOKE_LOG_DIR="$LOG_DIR" SMOKE_RUN_DIR="$RUN_DIR" \
  SMOKE_CFG="$CFG" SMOKE_TRAIN="$TRAIN" \
  srun --exact --nodes=1 --ntasks="$wave_size" --ntasks-per-node="$wave_size" \
      --cpus-per-task=48 --gpus-per-task=1 --gpu-bind="map_gpu:$gpu_map" \
      --kill-on-bad-exit=0 bash -c '
        IFS=, read -r -a configs <<< "$SMOKE_WAVE"
        slot="$SLURM_LOCALID"
        config="${configs[$slot]}"
        log="$SMOKE_LOG_DIR/${config}.log"
        export CUDA_VISIBLE_DEVICES="$slot"
        export NUM_PROCESSES=1
        export MASTER_PORT="$((29700 + slot))"
        export STARVLA_ACCELERATE_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_memory.yaml
        if bash "$SMOKE_TRAIN" \
          --config "$SMOKE_CFG/${config}.yaml" \
          --run_root_dir "$SMOKE_RUN_DIR" \
          --run_id "$config" \
          --trainer.max_train_steps 1 \
          --trainer.num_warmup_steps 0 \
          --trainer.eval_interval 1 \
          --trainer.open_loop_eval false \
          --trainer.save_interval 999999 \
          --trainer.save_final_checkpoint false \
          --trainer.logging_frequency 1 \
          --datasets.vla_data.per_device_batch_size 1 \
          --datasets.vla_data.num_workers 1 \
          --datasets.vla_data.persistent_workers false \
          >>"$log" 2>&1; then
          echo __SHARED_Z_SMOKE_PASS__ >>"$log"
        else
          status="$?"
          echo "__SHARED_Z_SMOKE_FAIL__ status=$status" >>"$log"
          exit "$status"
        fi
      '

  for ((offset=0; offset<wave_size; offset++)); do
    config="${configs[$((wave_start + offset))]}"
    log="$LOG_DIR/${config}.log"
    if rg -q '^__SHARED_Z_SMOKE_PASS__$' "$log"; then
      echo "PASS $config"
    else
      echo "FAIL $config ($log)" >&2
      failed=1
    fi
  done
done

echo "logs=$LOG_DIR"
echo "runs=$RUN_DIR"
exit "$failed"

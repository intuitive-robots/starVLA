#!/bin/bash
#SBATCH --job-name=smoke_enc_mlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/encoder_mlm_smoke_%j.out
#SBATCH --error=slurm_logs/encoder_mlm_smoke_%j.err

set -uo pipefail

STARVLA=/e/project1/m3/blank4/code/starVLA
CFG="$STARVLA/examples/simBenchmarks/LIBERO/train_files"
TRAIN="$STARVLA/train_libero_slurm.sh"
TAG="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$STARVLA/slurm_logs/encoder_mlm_smoke_${TAG}"
RUN_DIR="/e/scratch/m3/blank4/encoder_mlm_smoke_runs/${TAG}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

configs=(
  ervla_mlm_pi_encoder_cam3d
  ervla_mlm_pi_encoder_cam3d_rand
  ervla_mlm_pi_encoder_control
)
config_csv=$(IFS=,; echo "${configs[*]}")

SMOKE_CONFIGS="$config_csv" SMOKE_LOG_DIR="$LOG_DIR" SMOKE_RUN_DIR="$RUN_DIR" \
SMOKE_CFG="$CFG" SMOKE_TRAIN="$TRAIN" \
srun --exact --nodes=1 --ntasks=3 --ntasks-per-node=3 \
    --cpus-per-task=80 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2 \
    --kill-on-bad-exit=0 bash -c '
      IFS=, read -r -a configs <<< "$SMOKE_CONFIGS"
      slot="$SLURM_LOCALID"
      config="${configs[$slot]}"
      log="$SMOKE_LOG_DIR/${config}.log"
      export CUDA_VISIBLE_DEVICES="$slot"
      export NUM_PROCESSES=1
      export MASTER_PORT="$((29800 + slot))"
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
        echo __ENCODER_MLM_SMOKE_PASS__ >>"$log"
      else
        status="$?"
        echo "__ENCODER_MLM_SMOKE_FAIL__ status=$status" >>"$log"
        exit "$status"
      fi
    '

failed=0
for config in "${configs[@]}"; do
  log="$LOG_DIR/${config}.log"
  if rg -q '^__ENCODER_MLM_SMOKE_PASS__$' "$log"; then
    echo "PASS $config"
  else
    echo "FAIL $config ($log)" >&2
    failed=1
  fi
done
echo "logs=$LOG_DIR"
echo "runs=$RUN_DIR"
exit "$failed"

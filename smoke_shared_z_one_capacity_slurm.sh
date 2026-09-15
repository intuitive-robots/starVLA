#!/bin/bash
#SBATCH --job-name=cap_shared_z_one
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:59:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/sharedz_capacity_one_%j.out
#SBATCH --error=slurm_logs/sharedz_capacity_one_%j.err

set -euo pipefail
cd /e/project1/m3/blank4/code/starVLA

arm="${1:?usage: smoke_shared_z_one_capacity_slurm.sh <config-stem>}"
case "$arm" in
  ervla_zsup_pi_sharedz_ground_temporal|ervla_zsupdec_pi_sharedz_ground_temporal) ;;
  *) echo "unsupported capacity arm: $arm" >&2; exit 2 ;;
esac

export NUM_PROCESSES=4
export STARVLA_ACCELERATE_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_memory.yaml
run_root="/e/scratch/m3/blank4/sharedz_capacity_runs/${SLURM_JOB_ID}"
mkdir -p "$run_root"

bash train_libero_slurm.sh \
  --config "examples/LIBERO/train_files/${arm}.yaml" \
  --run_root_dir "$run_root" \
  --run_id "$arm" \
  --trainer.max_train_steps 1 \
  --trainer.num_warmup_steps 0 \
  --trainer.eval_interval 1 \
  --trainer.open_loop_eval false \
  --trainer.save_interval 999999 \
  --trainer.save_final_checkpoint false \
  --trainer.logging_frequency 1

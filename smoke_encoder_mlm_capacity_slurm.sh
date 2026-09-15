#!/bin/bash
#SBATCH --job-name=cap_enc_mlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/encoder_mlm_capacity_%j.out
#SBATCH --error=slurm_logs/encoder_mlm_capacity_%j.err

set -eo pipefail
export STARVLA_ACCELERATE_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_memory.yaml
export NUM_PROCESSES=4
export MASTER_PORT=29850

RUN_ROOT="/e/scratch/m3/blank4/encoder_mlm_capacity_${SLURM_JOB_ID}"
bash /e/project1/m3/blank4/code/starVLA/train_libero_slurm.sh \
  --config /e/project1/m3/blank4/code/starVLA/examples/LIBERO/train_files/ervla_mlm_pi_encoder_cam3d.yaml \
  --run_root_dir "$RUN_ROOT" \
  --run_id encoder_mlm_capacity \
  --trainer.max_train_steps 1 \
  --trainer.num_warmup_steps 0 \
  --trainer.eval_interval 1 \
  --trainer.open_loop_eval false \
  --trainer.save_interval 999999 \
  --trainer.save_final_checkpoint false \
  --trainer.logging_frequency 1

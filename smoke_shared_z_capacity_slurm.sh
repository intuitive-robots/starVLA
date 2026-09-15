#!/bin/bash
#SBATCH --job-name=cap_shared_z
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:59:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/sharedz_capacity_%j.out
#SBATCH --error=slurm_logs/sharedz_capacity_%j.err

set -uo pipefail
cd /e/project1/m3/blank4/code/starVLA
export NUM_PROCESSES=4
export STARVLA_ACCELERATE_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_memory.yaml

tag="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
run_root="/e/scratch/m3/blank4/sharedz_capacity_runs/$tag"
mkdir -p "$run_root"

failed=0
for arm in ervla_zsup_pi_sharedz_ground_temporal ervla_zsupdec_pi_sharedz_ground_temporal; do
  echo "CAPACITY_START $arm $(date -Is)"
  if bash train_libero_slurm.sh \
      --config "examples/simBenchmarks/LIBERO/train_files/${arm}.yaml" \
      --run_root_dir "$run_root" \
      --run_id "$arm" \
      --trainer.max_train_steps 1 \
      --trainer.num_warmup_steps 0 \
      --trainer.eval_interval 1 \
      --trainer.open_loop_eval false \
      --trainer.save_interval 999999 \
      --trainer.save_final_checkpoint false \
      --trainer.logging_frequency 1; then
    echo "CAPACITY_PASS $arm $(date -Is)"
  else
    status="$?"
    echo "CAPACITY_FAIL $arm status=$status $(date -Is)"
    failed=1
  fi
done
exit "$failed"

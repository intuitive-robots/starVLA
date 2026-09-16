#!/bin/bash
# Capture action/state sequences only from clean episodes completed by the
# real-alternating PI encoder control.
# Four policy replicas use all four GPUs; simulator workers are CPU/EGL clients.
#SBATCH --job-name=dfr_success_capture
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_success_capture_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_success_capture_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
OUT=${DFR_COUNTERFACTUAL_ROOT:-${ROOT}/outputs/paired_success_pi_alternating_v1}
CKPT=${DFR_CAPTURE_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/checkpoints/steps_20000_pytorch_model.pt}
TRIALS=${DFR_CAPTURE_TRIALS_PER_TASK:-8}
mkdir -p "${OUT}/source_trajectories" "${OUT}/clean_eval"

cd "${STAR}"
for suite in libero_spatial libero_object libero_goal libero_10; do
  echo "Capturing clean-successful trajectories for ${suite}"
  # One task per invocation ensures every simulator process creates exactly
  # one EGL context. This avoids both same-GPU construction races and the
  # driver abort observed when one process closes a task and creates another.
  for task_partition in {0..9}; do
    your_ckpt="${CKPT}" \
    output_dir="${OUT}/clean_eval/${suite}" \
    capture_successful_dir="${OUT}/source_trajectories" \
    workers_per_gpu=1 servers_per_gpu=1 max_batch_size=4 max_wait_time=1.0 \
    save_video=false SKIP_AGGREGATE=true \
      bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
        "${suite}" 4 1 "${TRIALS}" "" "${task_partition}" 10
  done
  /e/project1/m3/blank4/envs/miniforge3/envs/libero/bin/python \
    examples/LIBERO/eval_files/parallel_eval/aggregate_results.py \
    --root_path "${OUT}/clean_eval/${suite}" --task_suite_name "${suite}"
done

count=$(find "${OUT}/source_trajectories" -type f -name '*.json' | wc -l)
echo "Captured ${count} successful trajectories under ${OUT}/source_trajectories"
test "${count}" -ge 120

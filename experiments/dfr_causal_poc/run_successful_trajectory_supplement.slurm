#!/bin/bash
# Add clean-success captures for one task using all four GPUs on one Booster node.
# Intended only for a task that has <3 successes in the primary capture.
#SBATCH --job-name=dfr_success_supplement
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_success_supplement_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_success_supplement_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
OUT=${DFR_COUNTERFACTUAL_ROOT:?set DFR_COUNTERFACTUAL_ROOT}
CKPT=${DFR_CAPTURE_CHECKPOINT:?set DFR_CAPTURE_CHECKPOINT}
SUITE=${DFR_CAPTURE_SUITE:?set DFR_CAPTURE_SUITE}
TASK=${DFR_CAPTURE_TASK:?set DFR_CAPTURE_TASK (0..9)}
TRIALS=${DFR_CAPTURE_TRIALS_PER_TASK:-50}

case "${SUITE}" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) exit 2 ;; esac
if [ "${TASK}" -lt 0 ] || [ "${TASK}" -gt 9 ]; then exit 2; fi

mkdir -p "${OUT}/source_trajectories" "${OUT}/clean_eval/${SUITE}"
cd "${STAR}"
your_ckpt="${CKPT}" \
output_dir="${OUT}/clean_eval/${SUITE}" \
capture_successful_dir="${OUT}/source_trajectories" \
workers_per_gpu=1 servers_per_gpu=1 max_batch_size=4 max_wait_time=1.0 \
save_video=false SKIP_AGGREGATE=true \
  bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
    "${SUITE}" 4 1 "${TRIALS}" "" "${TASK}" 10

/e/project1/m3/blank4/envs/miniforge3/envs/libero/bin/python \
  examples/LIBERO/eval_files/parallel_eval/aggregate_results.py \
  --root_path "${OUT}/clean_eval/${SUITE}" --task_suite_name "${SUITE}"

successes=$(find "${OUT}/source_trajectories" -maxdepth 1 -type f \
  -name "${SUITE}_task$(printf '%02d' "${TASK}")_episode*.json" | wc -l)
echo "${SUITE} task ${TASK}: ${successes} successful captures after supplement"
test "${successes}" -ge 3

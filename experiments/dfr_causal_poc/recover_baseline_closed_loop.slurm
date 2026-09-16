#!/bin/bash
#SBATCH --job-name=dfr_base_recover
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=288
#SBATCH --time=03:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_base_recover_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_base_recover_%j.err

# Recovery for a partially completed baseline screen. The original run used
# four simulator workers per GPU; if only the first four of its sixteen raw
# ranges are missing, this reruns exactly those ranges with one worker per GPU.
# Existing ranges 4--15 are retained and the normal aggregator combines all
# sixteen files into the original exact-140 result.

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
REFIT_ROOT=${DFR_REFIT_OUT:-${ROOT}/outputs/action_refit_v1}
CKPT=${DFR_REFIT_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_p_causal_pi_masked/checkpoints/steps_20000_pytorch_model.pt}
PLUS_EXACT=${DFR_REFIT_PLUS_EXACT:-140}
CLEAN_TRIALS=${DFR_REFIT_CLEAN_TRIALS:-3}
RESULT_ROOT=${REFIT_ROOT}/baseline/closed_loop
PLUS_OUT=${RESULT_ROOT}/libero_plus_exact${PLUS_EXACT}
CLEAN_OUT=${RESULT_ROOT}/libero_clean_${CLEAN_TRIALS}x10

test -s "${CKPT}"
mkdir -p "${PLUS_OUT}" "${CLEAN_OUT}"
cd "${STAR}"

# LIBERO+-10 has 2,519 raw configurations. In the failed 16-way launch, the
# four missing ranges cover [0,632). Treat this node as partition 0 of 4 so the
# auto wrapper creates one range per GPU: [0,158), ..., [474,632).
export your_ckpt="${CKPT}"
export output_dir="${PLUS_OUT}"
export exact_tasks_per_suite="${PLUS_EXACT}"
export max_batch_size=16
export max_wait_time=1.0
export LIBERO_PLUS_RUNTIME=apptainer
bash examples/LIBERO-plus/eval_files/parallel_eval/auto_eval_libero_plus.sh \
  libero_10 4 158 1 0 1 "0,1,2,3" 1 4

test -s "${PLUS_OUT}/libero_10/overall_results.json"
test "$(jq -r '.overall.total_count' "${PLUS_OUT}/libero_10/overall_results.json")" -eq "${PLUS_EXACT}"

# The failed parent job never reached clean evaluation. One worker per GPU is
# sufficient for 30 episodes and avoids the simulator concurrency failure.
export output_dir="${CLEAN_OUT}"
workers_per_gpu=1 servers_per_gpu=1 max_batch_size=4 max_wait_time=1.0 \
  your_ckpt="${CKPT}" \
  bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
    libero_10 4 1 "${CLEAN_TRIALS}"

test -s "${CLEAN_OUT}/overall_results.json"
test "$(jq -r '.libero_10.total_count' "${CLEAN_OUT}/overall_results.json")" -eq 30
echo "Recovered baseline closed-loop evaluation: ${RESULT_ROOT}"

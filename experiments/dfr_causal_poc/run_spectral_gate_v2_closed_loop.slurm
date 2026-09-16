#!/bin/bash
#SBATCH --job-name=dfr_gate_v2_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=288
#SBATCH --time=03:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_gate_v2_eval_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_gate_v2_eval_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
GATE_ROOT=${DFR_GATE_OUT:-${ROOT}/outputs/spectral_gate_v2}
ARM=${DFR_GATE_ARM:?set DFR_GATE_ARM to a spectral-gate v2 arm}
PLUS_EXACT=${DFR_GATE_PLUS_EXACT:-140}
CLEAN_TRIALS=${DFR_GATE_CLEAN_TRIALS:-3}

CKPT=${GATE_ROOT}/${ARM}/checkpoints/aligned_pytorch_model.pt
case "${ARM}" in
  warm_final_balanced_v2)
    unset STARVLA_PI_ALIGNMENT_PATH || true
    ;;
  static_orthogonal_gate_warm_final|conditional_orthogonal_gate_warm_final|static_random_gate_warm_final_v2)
    GATE=${GATE_ROOT}/${ARM}/alignment_gate.pt
    test -s "${GATE}"
    export STARVLA_PI_ALIGNMENT_PATH=${GATE}
    ;;
  *)
    echo "Unknown spectral-gate v2 arm: ${ARM}" >&2
    exit 2
    ;;
esac
test -s "${CKPT}"

RESULT_ROOT=${GATE_ROOT}/${ARM}/closed_loop
PLUS_OUT=${RESULT_ROOT}/libero_plus_exact${PLUS_EXACT}
CLEAN_OUT=${RESULT_ROOT}/libero_clean_${CLEAN_TRIALS}x10
mkdir -p "${PLUS_OUT}" "${CLEAN_OUT}"

cd "${STAR}"
export your_ckpt="${CKPT}"
export output_dir="${PLUS_OUT}"
bash eval_libero_plus_slurm.sh \
  --ckpt "${CKPT}" \
  --suite libero_10 \
  --num_gpus 4 \
  --workers_per_gpu 4 \
  --servers_per_gpu 1 \
  --num_trials 1 \
  --exact_tasks_per_suite "${PLUS_EXACT}" \
  --max_batch_size 16 \
  --max_wait_time 1.0 \
  --sim-runtime apptainer

export output_dir="${CLEAN_OUT}"
your_ckpt="${CKPT}" output_dir="${CLEAN_OUT}" \
  workers_per_gpu=4 servers_per_gpu=1 max_batch_size=16 max_wait_time=1.0 \
  bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
    libero_10 4 4 "${CLEAN_TRIALS}"

test -s "${PLUS_OUT}/libero_10/overall_results.json"
test -s "${CLEAN_OUT}/overall_results.json"
echo "Closed-loop spectral-gate v2 evaluation complete: arm=${ARM} root=${RESULT_ROOT}"

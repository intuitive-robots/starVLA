#!/bin/bash
#SBATCH --job-name=pi_endpoint_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=288
#SBATCH --time=08:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_endpoint_eval_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_endpoint_eval_%j.err

set -euo pipefail
STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
STUDY=${PI_ENDPOINT_ROOT:-${ROOT}/outputs/pi_unrolled_endpoint_adapters_alternating_v1}
CKPT=${PI_ENDPOINT_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/checkpoints/steps_20000_pytorch_model.pt}
ADAPTER=${PI_ENDPOINT_ADAPTER:-${STUDY}/endpoint_correct/adapter.pt}
OUT=${PI_ENDPOINT_EVAL_OUT:-${STUDY}/closed_loop}
PLUS_EXACT=${PI_ENDPOINT_PLUS_EXACT:-140}
CLEAN_TRIALS=${PI_ENDPOINT_CLEAN_TRIALS:-3}
ARMS=${PI_ENDPOINT_EVAL_ARMS:-baseline endpoint_correct}
WORKERS_PER_GPU=${PI_ENDPOINT_WORKERS_PER_GPU:-1}
# LIBERO+ can abort natively while switching between environments.  Run each
# selected task in a fresh process, with one sequential stream per GPU, so a
# retry never discards already completed trajectories.
export STARVLA_ISOLATE_EXACT_TASKS=${PI_ENDPOINT_ISOLATE_EXACT_TASKS:-1}
test -s "${CKPT}"
test -s "${ADAPTER}"
cd "${STAR}"

for arm in ${ARMS}; do
  case "${arm}" in
    baseline) unset STARVLA_PI_PROJECTION_LORA_PATH || true ;;
    endpoint_correct) export STARVLA_PI_PROJECTION_LORA_PATH="${ADAPTER}" ;;
    *) echo "unknown arm ${arm}" >&2; exit 2 ;;
  esac
  result=${OUT}/${arm}
  plus=${result}/libero_plus_exact${PLUS_EXACT}
  clean=${result}/libero_clean_${CLEAN_TRIALS}x10
  mkdir -p "${plus}" "${clean}"
  export your_ckpt="${CKPT}" output_dir="${plus}"
  bash eval_libero_plus_slurm.sh \
    --ckpt "${CKPT}" --suite libero_10 --num_gpus 4 \
    --workers_per_gpu "${WORKERS_PER_GPU}" --servers_per_gpu 1 --num_trials 1 \
    --exact_tasks_per_suite "${PLUS_EXACT}" --max_batch_size 16 \
    --max_wait_time 1.0 --sim-runtime apptainer
  test -s "${plus}/libero_10/overall_results.json"

  export output_dir="${clean}"
  your_ckpt="${CKPT}" output_dir="${clean}" \
    workers_per_gpu="${WORKERS_PER_GPU}" servers_per_gpu=1 max_batch_size=16 max_wait_time=1.0 \
    bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
      libero_10 4 "${WORKERS_PER_GPU}" "${CLEAN_TRIALS}"
  test -s "${clean}/overall_results.json"
done
echo "PI endpoint closed-loop evaluation complete: ${OUT}"

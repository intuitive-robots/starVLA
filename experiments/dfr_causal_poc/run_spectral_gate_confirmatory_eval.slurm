#!/bin/bash
#SBATCH --job-name=dfr_gate_confirm_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=288
#SBATCH --time=03:00:00
#SBATCH --array=0-12%2
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_gate_confirm_eval_%A_%a.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_gate_confirm_eval_%A_%a.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
OUT=${DFR_CONFIRM_OUT:-${ROOT}/outputs/spectral_gate_confirmatory}
SOURCE_CKPT=${DFR_GATE_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_p_causal_pi_masked/checkpoints/steps_20000_pytorch_model.pt}
SEED_BASE=${DFR_CONFIRM_SEED_BASE:-20260907}
PLUS_EXACT=${DFR_CONFIRM_PLUS_EXACT:-420}
# LIBERO-Plus perturbation instances do not all expose multiple stored initial
# states. Broaden coverage with distinct instances rather than indexing a
# nonexistent second/third state within an instance.
PLUS_TRIALS=${DFR_CONFIRM_PLUS_TRIALS:-1}
CLEAN_TRIALS=${DFR_CONFIRM_CLEAN_TRIALS:-10}

task_id=${SLURM_ARRAY_TASK_ID:?this launcher must run as a Slurm array}
arms=(warm_final_balanced_v2 static_orthogonal_gate_warm_final static_random_gate_warm_final_v2)

if [ "${task_id}" -eq 0 ]; then
  label=source
  ckpt=${SOURCE_CKPT}
  result_root=${OUT}/source/closed_loop
  unset STARVLA_PI_ALIGNMENT_PATH || true
else
  flat=$((task_id - 1))
  seed=$((SEED_BASE + flat / 3))
  arm=${arms[$((flat % 3))]}
  label=seed_${seed}/${arm}
  arm_root=${OUT}/seed_${seed}/${arm}
  ckpt=${arm_root}/checkpoints/aligned_pytorch_model.pt
  result_root=${arm_root}/closed_loop
  case "${arm}" in
    warm_final_balanced_v2)
      unset STARVLA_PI_ALIGNMENT_PATH || true
      ;;
    static_orthogonal_gate_warm_final|static_random_gate_warm_final_v2)
      gate=${arm_root}/alignment_gate.pt
      test -s "${gate}"
      export STARVLA_PI_ALIGNMENT_PATH=${gate}
      ;;
  esac
fi
test -s "${ckpt}"

plus_out=${result_root}/libero_plus_exact${PLUS_EXACT}_trials${PLUS_TRIALS}
clean_out=${result_root}/libero_clean_${CLEAN_TRIALS}x10
mkdir -p "${plus_out}" "${clean_out}"

cd "${STAR}"
export your_ckpt=${ckpt}
export output_dir=${plus_out}
bash eval_libero_plus_slurm.sh \
  --ckpt "${ckpt}" \
  --suite libero_10 \
  --num_gpus 4 \
  --workers_per_gpu 4 \
  --servers_per_gpu 1 \
  --num_trials "${PLUS_TRIALS}" \
  --exact_tasks_per_suite "${PLUS_EXACT}" \
  --max_batch_size 16 \
  --max_wait_time 1.0 \
  --sim-runtime apptainer

export output_dir=${clean_out}
your_ckpt="${ckpt}" output_dir="${clean_out}" \
  workers_per_gpu=4 servers_per_gpu=1 max_batch_size=16 max_wait_time=1.0 \
  bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
    libero_10 4 4 "${CLEAN_TRIALS}"

test -s "${plus_out}/libero_10/overall_results.json"
test -s "${clean_out}/overall_results.json"
echo "Confirmatory evaluation complete: ${label}"

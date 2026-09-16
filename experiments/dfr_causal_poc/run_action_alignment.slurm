#!/bin/bash
#SBATCH --job-name=dfr_align
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_align_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_align_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
CKPT=${DFR_ALIGN_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_p_causal_pi_masked/checkpoints/steps_20000_pytorch_model.pt}
CONFIG=${DFR_ALIGN_CONFIG:-${STAR}/examples/LIBERO/train_files/ervla_p_causal_pi.yaml}
OUT=${DFR_ALIGN_OUT:-${ROOT}/outputs/action_alignment_v1}
STEPS=${DFR_ALIGN_STEPS:-1000}
BASIS_PAIRS=${DFR_ALIGN_BASIS_PAIRS:-256}
BASIS_BATCH_SIZE=${DFR_ALIGN_BASIS_BATCH_SIZE:-8}
VALIDATION_EXAMPLES=${DFR_ALIGN_VALIDATION_EXAMPLES:-48}
LOG_EVERY=${DFR_ALIGN_LOG_EVERY:-50}
SKIP_CHECKPOINT=${DFR_ALIGN_SKIP_CHECKPOINT:-0}
mkdir -p "${OUT}"

export PYTHONPATH=${STAR}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export STAR ROOT PY CKPT CONFIG OUT STEPS BASIS_PAIRS BASIS_BATCH_SIZE
export VALIDATION_EXAMPLES LOG_EVERY SKIP_CHECKPOINT

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  arms=(warm_final_balanced learned_gate_all learned_gate_all_warm_final random_gate_all_warm_final)
  arm=${arms[$SLURM_LOCALID]}
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  skip_checkpoint=()
  if [ "${SKIP_CHECKPOINT}" = 1 ]; then skip_checkpoint=(--skip-checkpoint); fi
  "${PY}" "${ROOT}/align_action_head.py" \
    --checkpoint "${CKPT}" \
    --config "${CONFIG}" \
    --output-dir "${OUT}/${arm}" \
    --arm "${arm}" \
    --steps "${STEPS}" \
    --basis-pairs "${BASIS_PAIRS}" \
    --basis-batch-size "${BASIS_BATCH_SIZE}" \
    --validation-examples "${VALIDATION_EXAMPLES}" \
    --log-every "${LOG_EVERY}" \
    "${skip_checkpoint[@]}"
'

for arm in warm_final_balanced learned_gate_all learned_gate_all_warm_final random_gate_all_warm_final; do
  test -s "${OUT}/${arm}/report.json"
  if [ "${SKIP_CHECKPOINT}" != 1 ] && [ "${arm}" != learned_gate_all ]; then
    test -s "${OUT}/${arm}/checkpoints/aligned_pytorch_model.pt"
  fi
done
echo "All zero-initialized alignment arms complete: ${OUT}"

#!/bin/bash
#SBATCH --job-name=dfr_action_refit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_action_refit_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_action_refit_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
CKPT=${DFR_REFIT_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_p_causal_pi_masked/checkpoints/steps_20000_pytorch_model.pt}
CONFIG=${DFR_REFIT_CONFIG:-${STAR}/examples/LIBERO/train_files/ervla_p_causal_pi.yaml}
OUT=${DFR_REFIT_OUT:-${ROOT}/outputs/action_refit_v1}
STEPS=${DFR_REFIT_STEPS:-600}
BATCH_SIZE=${DFR_REFIT_BATCH_SIZE:-4}
VALIDATION_EXAMPLES=${DFR_REFIT_VALIDATION_EXAMPLES:-48}
SKIP_CHECKPOINT=${DFR_REFIT_SKIP_CHECKPOINT:-0}
mkdir -p "${OUT}"

export PYTHONPATH=${STAR}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export STAR ROOT PY CKPT CONFIG OUT STEPS BATCH_SIZE VALIDATION_EXAMPLES SKIP_CHECKPOINT

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  arms=(fresh_final_balanced fresh_decoder_balanced warm_decoder_balanced fresh_decoder_clean)
  arm=${arms[$SLURM_LOCALID]}
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  skip_checkpoint=()
  if [ "${SKIP_CHECKPOINT}" = 1 ]; then skip_checkpoint=(--skip-checkpoint); fi
  "${PY}" "${ROOT}/refit_action_decoder.py" \
    --checkpoint "${CKPT}" \
    --config "${CONFIG}" \
    --output-dir "${OUT}/${arm}" \
    --arm "${arm}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-examples "${VALIDATION_EXAMPLES}" \
    "${skip_checkpoint[@]}"
'

for arm in fresh_final_balanced fresh_decoder_balanced warm_decoder_balanced fresh_decoder_clean; do
  test -s "${OUT}/${arm}/report.json"
  if [ "${SKIP_CHECKPOINT}" != 1 ]; then
    test -s "${OUT}/${arm}/checkpoints/dfr_refit_pytorch_model.pt"
  fi
done
echo "All DFR action-refit arms complete: ${OUT}"

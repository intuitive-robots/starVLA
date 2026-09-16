#!/bin/bash
#SBATCH --job-name=dfr_gate_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_gate_v2_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_gate_v2_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
CKPT=${DFR_GATE_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_p_causal_pi_masked/checkpoints/steps_20000_pytorch_model.pt}
CONFIG=${DFR_GATE_CONFIG:-${STAR}/examples/LIBERO/train_files/ervla_p_causal_pi.yaml}
OUT=${DFR_GATE_OUT:-${ROOT}/outputs/spectral_gate_v2}
STEPS=${DFR_GATE_STEPS:-2000}
MICROBATCHES=${DFR_GATE_MICROBATCHES:-4}
BASIS_PAIRS=${DFR_GATE_BASIS_PAIRS:-1024}
BASIS_BATCH_SIZE=${DFR_GATE_BASIS_BATCH_SIZE:-8}
VALIDATION_EXAMPLES=${DFR_GATE_VALIDATION_EXAMPLES:-256}
LOG_EVERY=${DFR_GATE_LOG_EVERY:-100}
SKIP_CHECKPOINT=${DFR_GATE_SKIP_CHECKPOINT:-0}
mkdir -p "${OUT}"

export PYTHONPATH=${STAR}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export STAR ROOT PY CKPT CONFIG OUT STEPS MICROBATCHES BASIS_PAIRS BASIS_BATCH_SIZE
export VALIDATION_EXAMPLES LOG_EVERY SKIP_CHECKPOINT

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  arms=(warm_final_balanced_v2 static_orthogonal_gate_warm_final conditional_orthogonal_gate_warm_final static_random_gate_warm_final_v2)
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
    --batch-size 4 \
    --microbatches-per-step "${MICROBATCHES}" \
    --nuisances background lighting sensor_noise camera_jitter \
    --basis-pairs "${BASIS_PAIRS}" \
    --basis-batch-size "${BASIS_BATCH_SIZE}" \
    --validation-examples "${VALIDATION_EXAMPLES}" \
    --log-every "${LOG_EVERY}" \
    "${skip_checkpoint[@]}"
'

for arm in warm_final_balanced_v2 static_orthogonal_gate_warm_final conditional_orthogonal_gate_warm_final static_random_gate_warm_final_v2; do
  test -s "${OUT}/${arm}/report.json"
  if [ "${SKIP_CHECKPOINT}" != 1 ]; then
    test -s "${OUT}/${arm}/checkpoints/aligned_pytorch_model.pt"
  fi
done
echo "All spectral-gate v2 arms complete: ${OUT}"

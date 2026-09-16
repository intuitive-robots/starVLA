#!/bin/bash
#SBATCH --job-name=pi_endpoint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_endpoint_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_endpoint_%j.err

set -euo pipefail
STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
CKPT=${PI_ENDPOINT_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/checkpoints/steps_20000_pytorch_model.pt}
MANIFEST=${PI_ENDPOINT_MANIFEST:-${ROOT}/outputs/paired_success_pi_alternating_v2/counterfactuals/manifest.jsonl}
STATS=${PI_ENDPOINT_STATS:-${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/dataset_statistics.json}
OUT=${PI_ENDPOINT_OUT:-${ROOT}/outputs/pi_unrolled_endpoint_adapters_alternating_v1}
TRAIN_PAIRS=${PI_ENDPOINT_TRAIN_PAIRS:-2048}
VALIDATION_PAIRS=${PI_ENDPOINT_VALIDATION_PAIRS:-256}
SMOKE=${PI_ENDPOINT_SMOKE:-0}
mkdir -p "${OUT}"
export PYTHONPATH=${STAR}:${ROOT}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export STAR ROOT PY CKPT MANIFEST STATS OUT TRAIN_PAIRS VALIDATION_PAIRS SMOKE

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  arms=(endpoint_correct endpoint_no_pair endpoint_shuffled local_flow)
  arm=${arms[$SLURM_LOCALID]}
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  smoke_args=(); if [ "${SMOKE}" = 1 ]; then smoke_args=(--smoke); fi
  "${PY}" "${ROOT}/fit_pi_unrolled_endpoint_adapters.py" \
    --manifest "${MANIFEST}" --checkpoint "${CKPT}" --stats "${STATS}" \
    --output-dir "${OUT}/${arm}" --arm "${arm}" \
    --train-pairs "${TRAIN_PAIRS}" --validation-pairs "${VALIDATION_PAIRS}" \
    "${smoke_args[@]}"
'
for arm in endpoint_correct endpoint_no_pair endpoint_shuffled local_flow; do
  test -s "${OUT}/${arm}/report.json"
  test -s "${OUT}/${arm}/adapter.pt"
done
"${PY}" "${ROOT}/summarize_pi_unrolled_endpoint_adapters.py" --root "${OUT}"
echo "PI unrolled endpoint ablation complete: ${OUT}"

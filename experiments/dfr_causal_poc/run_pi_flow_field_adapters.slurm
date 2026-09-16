#!/bin/bash
#SBATCH --job-name=pi_flow_adapt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_flow_adapt_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_flow_adapt_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
CKPT=${PI_FLOW_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/checkpoints/steps_20000_pytorch_model.pt}
MANIFEST=${PI_FLOW_MANIFEST:-${ROOT}/outputs/paired_success_pi_alternating_v2/counterfactuals/manifest.jsonl}
STATS=${PI_FLOW_STATS:-${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/dataset_statistics.json}
OUT=${PI_FLOW_OUT:-${ROOT}/outputs/pi_flow_field_adapters_alternating_v1}
TRAIN_PAIRS=${PI_FLOW_TRAIN_PAIRS:-2048}
VALIDATION_PAIRS=${PI_FLOW_VALIDATION_PAIRS:-256}
RANK=${PI_FLOW_RANK:-4}
LR=${PI_FLOW_LR:-3e-4}
SMOKE=${PI_FLOW_SMOKE:-0}
mkdir -p "${OUT}"

export PYTHONPATH=${STAR}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export STAR ROOT PY CKPT MANIFEST STATS OUT TRAIN_PAIRS VALIDATION_PAIRS RANK LR SMOKE

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  arms=(projection_lora xattn_qk_lora xattn_vout_lora xattn_head_gate)
  arm=${arms[$SLURM_LOCALID]}
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  smoke_args=()
  if [ "${SMOKE}" = 1 ]; then smoke_args=(--smoke); fi
  "${PY}" "${ROOT}/fit_pi_flow_field_adapters.py" \
    --manifest "${MANIFEST}" \
    --checkpoint "${CKPT}" \
    --stats "${STATS}" \
    --output-dir "${OUT}/${arm}" \
    --arm "${arm}" \
    --rank "${RANK}" \
    --train-pairs "${TRAIN_PAIRS}" \
    --validation-pairs "${VALIDATION_PAIRS}" \
    --learning-rate "${LR}" \
    "${smoke_args[@]}"
'

for arm in projection_lora xattn_qk_lora xattn_vout_lora xattn_head_gate; do
  test -s "${OUT}/${arm}/report.json"
  test -s "${OUT}/${arm}/adapter.pt"
done
"${PY}" "${ROOT}/summarize_pi_flow_field_adapters.py" --root "${OUT}"
echo "PI flow-field adapter screen complete: ${OUT}"

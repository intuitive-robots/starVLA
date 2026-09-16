#!/bin/bash
#SBATCH --job-name=pi_endpoint_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_endpoint_test_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/pi_endpoint_test_%j.err

set -euo pipefail
STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
STUDY=${ROOT}/outputs/pi_unrolled_endpoint_adapters_alternating_v1
PY=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python
CKPT=${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/checkpoints/steps_20000_pytorch_model.pt
MANIFEST=${ROOT}/outputs/paired_success_pi_alternating_v2/counterfactuals/manifest.jsonl
STATS=${STAR}/playground/Checkpoints/ervla_w2_pi_encoder_actiononly_pifix_nolatent/dataset_statistics.json
ADAPTER=${STUDY}/endpoint_correct/adapter.pt
OUT=${STUDY}/sealed_test
mkdir -p "${OUT}"
export PYTHONPATH=${STAR}:${ROOT} TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export STAR ROOT STUDY PY CKPT MANIFEST STATS ADAPTER OUT
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  "${PY}" "${ROOT}/evaluate_pi_endpoint_adapter.py" \
    --manifest "${MANIFEST}" --checkpoint "${CKPT}" --stats "${STATS}" \
    --adapter "${ADAPTER}" --output "${OUT}/shard_${SLURM_LOCALID}.json" \
    --shard-id "${SLURM_LOCALID}" --num-shards 4
'
for shard in 0 1 2 3; do test -s "${OUT}/shard_${shard}.json"; done
"${PY}" "${ROOT}/summarize_pi_endpoint_test.py" --root "${OUT}"

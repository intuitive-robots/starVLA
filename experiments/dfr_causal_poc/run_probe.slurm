#!/bin/bash
#SBATCH --job-name=dfr_p_causal_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_p_causal_probe_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_p_causal_probe_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
CAPTURE=${DFR_POC_CAPTURE_OUT:-${ROOT}/outputs/paired_libero10_v1}
OUT=${DFR_POC_PROBE_OUT:-${ROOT}/outputs/p_causal_probe_v1}
CKPT=${DFR_POC_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_p_causal_pi_masked/checkpoints/steps_20000_pytorch_model.pt}
BATCH_SIZE=${DFR_POC_BATCH_SIZE:-3}
mkdir -p "${OUT}"

export PYTHONPATH=${STAR}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STAR ROOT PY CAPTURE OUT CKPT BATCH_SIZE

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  ranks=(1 4 8 16)
  rank=${ranks[$SLURM_LOCALID]}
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  shard_out=${OUT}/shards/rank_${rank}
  mkdir -p "${shard_out}"
  "${PY}" "${ROOT}/probe_paired_plus.py" \
    --manifest "${CAPTURE}/manifest.jsonl" \
    --checkpoint "${CKPT}" \
    --output-dir "${shard_out}" \
    --batch-size "${BATCH_SIZE}" \
    --test-tasks 3 \
    --max-rank 16 \
    --ranks "${rank}"
'

"${PY}" "${ROOT}/merge_probe_reports.py" \
  --root "${OUT}" \
  --ranks 1,4,8,16

test -s "${OUT}/report.json"
echo "P-causal diagnostic complete: ${OUT}"

#!/bin/bash
# Frozen paired readout screen for one VLA. Capture and fitting both use 4 GPUs.
#SBATCH --job-name=dfr_pair_readout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_pair_readout_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_pair_readout_%j.err

set -euo pipefail
STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
PAIR_ROOT=${DFR_COUNTERFACTUAL_ROOT:-${ROOT}/outputs/paired_success_v1}
MODEL_NAME=${DFR_PAIR_MODEL_NAME:?set DFR_PAIR_MODEL_NAME}
CKPT=${DFR_PAIR_CHECKPOINT:?set DFR_PAIR_CHECKPOINT}
OUT=${DFR_PAIR_OUTPUT:-${PAIR_ROOT}/readouts/${MODEL_NAME}}
BATCH=${DFR_PAIR_BATCH_SIZE:-3}
STATS=$(dirname "$(dirname "${CKPT}")")/dataset_statistics.json
test -s "${PAIR_ROOT}/counterfactuals/manifest.jsonl"
test -s "${CKPT}"
test -s "${STATS}"
mkdir -p "${OUT}"
export STAR ROOT PY PAIR_ROOT MODEL_NAME CKPT OUT BATCH STATS
export PYTHONPATH=${STAR}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  "${PY}" "${ROOT}/capture_successful_pair_features.py" \
    --manifest "${PAIR_ROOT}/counterfactuals/manifest.jsonl" \
    --checkpoint "${CKPT}" --stats "${STATS}" --output-dir "${OUT}" \
    --rank "$SLURM_LOCALID" --world-size 4 --batch-size "${BATCH}"
'

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  case "$SLURM_LOCALID" in
    0) loci=(raw_preprojector projected_memory) ;;
    1) loci=(pi_input pi_early) ;;
    2) loci=(pi_middle) ;;
    3) loci=(pi_late) ;;
  esac
  "${PY}" "${ROOT}/fit_successful_pair_readouts.py" \
    --root "${OUT}" --loci "${loci[@]}" --rank "$SLURM_LOCALID" --world-size 4
'

"${PY}" "${ROOT}/summarize_successful_pair_readouts.py" --root "${OUT}"
test -s "${OUT}/summary.json"

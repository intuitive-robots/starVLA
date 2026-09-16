#!/bin/bash
#SBATCH --job-name=dfr_pair_capture
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=01:30:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_pair_capture_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_pair_capture_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
OUT=${DFR_POC_CAPTURE_OUT:-${ROOT}/outputs/paired_libero10_v1}
SIF=${LIBERO_PLUS_SIF:-${STAR}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}
VARIANTS=${DFR_POC_VARIANTS_PER_TASK:-3}
mkdir -p "${OUT}"
export STAR ROOT OUT SIF VARIANTS

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  shard=$SLURM_LOCALID
  gpu=$SLURM_LOCALID
  shard_out=${OUT}/shards/${shard}
  cache=${TMPDIR:-/tmp}/dfr_pair_numba_${SLURM_JOB_ID}_${shard}
  mkdir -p "${shard_out}" "${cache}"
  CUDA_VISIBLE_DEVICES=${gpu} apptainer exec --cleanenv --nv \
    --bind "${STAR}:${STAR}:rw" \
    --pwd "${STAR}" \
    --env LIBERO_HOME=/app/LIBERO-plus \
    --env MUJOCO_GL=egl \
    --env PYOPENGL_PLATFORM=egl \
    --env CUDA_VISIBLE_DEVICES=${gpu} \
    --env MUJOCO_EGL_DEVICE_ID=${gpu} \
    --env NUMBA_CACHE_DIR=${cache} \
    "${SIF}" \
    /opt/conda/envs/libero/bin/python \
    "${ROOT}/capture_paired_plus.py" \
    --output-root "${shard_out}" \
    --variants-per-task "${VARIANTS}" \
    --num-shards 4 \
    --shard-id "${shard}"
'

expected_pairs=$((10 * 3 * VARIANTS))
/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python \
  "${ROOT}/merge_capture.py" \
  --root "${OUT}" \
  --expected-pairs "${expected_pairs}" \
  --num-shards 4

test -s "${OUT}/manifest.jsonl"
echo "paired capture complete: ${OUT}"

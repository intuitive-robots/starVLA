#!/bin/bash
# Render successful clean trajectories with exact state-matched Plus appearance changes.
#SBATCH --job-name=dfr_cf_render
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_render_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_render_%j.err

set -euo pipefail
STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
OUT=${DFR_COUNTERFACTUAL_ROOT:-${ROOT}/outputs/paired_success_v1}
TRAJECTORY_ROOT=${DFR_COUNTERFACTUAL_TRAJECTORY_ROOT:-${OUT}/source_trajectories}
SIF=${LIBERO_PLUS_SIF:-${STAR}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}
ANCHORS=${DFR_COUNTERFACTUAL_ANCHORS:-12}
EXPECTED_TASKS=${DFR_COUNTERFACTUAL_EXPECTED_TASKS:-40}
MINIMUM_TASKS=${DFR_COUNTERFACTUAL_MINIMUM_TASKS:-20}
MINIMUM_TASKS_PER_SUITE=${DFR_COUNTERFACTUAL_MINIMUM_TASKS_PER_SUITE:-3}
mkdir -p "${OUT}/counterfactuals"
export STAR ROOT OUT TRAJECTORY_ROOT SIF ANCHORS EXPECTED_TASKS MINIMUM_TASKS MINIMUM_TASKS_PER_SUITE

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  shard=$SLURM_LOCALID
  gpu=$SLURM_LOCALID
  shard_out=${OUT}/counterfactuals/shards/${shard}
  cache=${TMPDIR:-/tmp}/dfr_cf_numba_${SLURM_JOB_ID}_${shard}
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
    "${SIF}" /opt/conda/envs/libero/bin/python \
    "${ROOT}/render_successful_counterfactuals.py" \
      --trajectory-root "${TRAJECTORY_ROOT}" \
      --output-root "${shard_out}" \
      --anchors-per-trajectory "${ANCHORS}" \
      --expected-tasks "${EXPECTED_TASKS}" \
      --minimum-tasks "${MINIMUM_TASKS}" \
      --minimum-tasks-per-suite "${MINIMUM_TASKS_PER_SUITE}" \
      --num-shards 4 --shard-id "${shard}"
'

/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python \
  "${ROOT}/merge_successful_counterfactuals.py" \
  --root "${OUT}/counterfactuals" --num-shards 4 \
  --expected-tasks "${EXPECTED_TASKS}" \
  --minimum-tasks "${MINIMUM_TASKS}" \
  --minimum-tasks-per-suite "${MINIMUM_TASKS_PER_SUITE}"
test -s "${OUT}/counterfactuals/manifest.jsonl"

#!/bin/bash
#SBATCH --job-name=dfr_cf_replay
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_replay_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_replay_%j.err

set -euo pipefail
STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PAIR_ROOT=${DFR_CF_PAIR_ROOT:?set DFR_CF_PAIR_ROOT}
MANIFEST=${PAIR_ROOT}/counterfactuals/manifest.jsonl
OUT=${PAIR_ROOT}/closed_loop_pi_late_v1/source_replay
SIF=${LIBERO_PLUS_SIF:-${STAR}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}
EXPECTED_PERTURBED=${DFR_CF_EXPECTED_PERTURBED:?set DFR_CF_EXPECTED_PERTURBED}
EXPECTED_CLEAN=${DFR_CF_EXPECTED_CLEAN:?set DFR_CF_EXPECTED_CLEAN}
mkdir -p "${OUT}"

export STAR ROOT PAIR_ROOT MANIFEST OUT SIF
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  shard=$SLURM_LOCALID
  cache=${TMPDIR:-/tmp}/dfr_cf_replay_${SLURM_JOB_ID}_${shard}
  mkdir -p "${cache}"
  CUDA_VISIBLE_DEVICES=${shard} apptainer exec --cleanenv --nv \
    --bind "${STAR}:${STAR}:rw" --pwd "${STAR}" \
    --env LIBERO_HOME=/app/LIBERO-plus --env PYTHONPATH="${STAR}" \
    --env MUJOCO_GL=egl --env PYOPENGL_PLATFORM=egl \
    --env CUDA_VISIBLE_DEVICES=${shard} --env MUJOCO_EGL_DEVICE_ID=${shard} \
    --env NUMBA_CACHE_DIR="${cache}" \
    "${SIF}" /opt/conda/envs/libero/bin/python \
      "${ROOT}/eval_successful_pair_closed_loop.py" \
        --manifest "${MANIFEST}" --output "${OUT}/shard_${shard}.jsonl" \
        --arm source_replay --source-replay --shard-id "${shard}" --num-shards 4
'

count=$(find "${OUT}" -name 'shard_*.jsonl' -exec cat {} + | wc -l)
success=$(find "${OUT}" -name 'shard_*.jsonl' -exec cat {} + | \
  /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python -c \
  'import json,sys; rows=[json.loads(x) for x in sys.stdin if x.strip()]; print(sum(r["success"] for r in rows))')
expected=$((EXPECTED_PERTURBED + EXPECTED_CLEAN))
test "${count}" -eq "${expected}"
echo "Source replay: ${success}/${count} successful (expected cases=${expected})"

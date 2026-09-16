#!/bin/bash
# Validation-only trust-region sweep. Each of four independent settings owns one GPU.
#SBATCH --job-name=dfr_cf_trust
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=05:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_trust_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_trust_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PAIR_ROOT=${DFR_CF_PAIR_ROOT:?set DFR_CF_PAIR_ROOT}
MANIFEST=${PAIR_ROOT}/counterfactuals/manifest.jsonl
CKPT=${DFR_CF_CHECKPOINT:?set DFR_CF_CHECKPOINT}
READOUT=${DFR_CF_READOUT:?set DFR_CF_READOUT}
OUT=${DFR_CF_TRUST_OUT:-${PAIR_ROOT}/trust_sweep_v1}
SIF=${LIBERO_PLUS_SIF:-${STAR}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}
EXPECTED_PERTURBED=${DFR_CF_EXPECTED_PERTURBED:?set DFR_CF_EXPECTED_PERTURBED}
EXPECTED_CLEAN=${DFR_CF_EXPECTED_CLEAN:?set DFR_CF_EXPECTED_CLEAN}
PYTHON=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python
PAIRED_ROLLOUT=${DFR_CF_PAIRED_ROLLOUT:-0}

test -s "${MANIFEST}"
test -s "${CKPT}"
test -s "${READOUT}"
test -s "${SIF}"
mkdir -p "${OUT}/servers"

CONFIG=${OUT}/config.json
if [ ! -s "${CONFIG}" ]; then
  "${PYTHON}" -c 'import json,sys; json.dump([
    {"name":"strength_0","strength":0.0,"norm_cap_ratio":None},
    {"name":"strength_0p1","strength":0.1,"norm_cap_ratio":None},
    {"name":"strength_0p25","strength":0.25,"norm_cap_ratio":None},
    {"name":"cap_0p05","strength":1.0,"norm_cap_ratio":0.05}], open(sys.argv[1],"w"), indent=2)' "${CONFIG}"
fi

export STAR ROOT PAIR_ROOT MANIFEST CKPT READOUT OUT SIF PYTHON CONFIG PAIRED_ROLLOUT
export EXPECTED_PERTURBED EXPECTED_CLEAN
cd "${STAR}"

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  rank=${SLURM_LOCALID}
  name=$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])][\"name\"])" "${CONFIG}" "${rank}")
  strength=$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])][\"strength\"])" "${CONFIG}" "${rank}")
  cap=$("${PYTHON}" -c "import json,sys; x=json.load(open(sys.argv[1]))[int(sys.argv[2])][\"norm_cap_ratio\"]; print(\"\" if x is None else x)" "${CONFIG}" "${rank}")
  port=$((18600 + rank))
  visible_gpu=${CUDA_VISIBLE_DEVICES:-${rank}}
  mkdir -p "${OUT}/${name}"
  server_pid=""
  cleanup() {
    if [ -n "${server_pid}" ]; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  CUDA_VISIBLE_DEVICES=${visible_gpu} \
    STARVLA_FLOW_RESIDUAL_READOUT="${READOUT}" \
    STARVLA_FLOW_RESIDUAL_STRENGTH="${strength}" \
    STARVLA_FLOW_RESIDUAL_NORM_CAP_RATIO="${cap}" \
    STARVLA_PAIRED_ROLLOUT="${PAIRED_ROLLOUT}" \
    "${PYTHON}" deployment/model_server/server_policy.py \
      --ckpt_path "${CKPT}" --port "${port}" --use_bf16 --idle_timeout -1 \
      > "${OUT}/servers/${name}.log" 2>&1 &
  server_pid=$!

  ready=0
  for _ in $(seq 1 180); do
    if "${PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(.2); raise SystemExit(s.connect_ex((\"127.0.0.1\", ${port})))"; then
      ready=1
      break
    fi
    sleep 2
  done
  if [ "${ready}" -ne 1 ]; then
    echo "policy server for ${name} did not become ready" >&2
    exit 1
  fi

  cache=${TMPDIR:-/tmp}/dfr_cf_trust_${SLURM_JOB_ID}_${rank}
  mkdir -p "${cache}"
  CUDA_VISIBLE_DEVICES=${visible_gpu} apptainer exec --cleanenv --nv \
    --bind "${STAR}:${STAR}:rw" --pwd "${STAR}" \
    --env LIBERO_HOME=/app/LIBERO-plus --env PYTHONPATH="${STAR}" \
    --env STARVLA_FAST_GLASS_BLUR=1 \
    --env MUJOCO_GL=egl --env PYOPENGL_PLATFORM=egl \
    --env CUDA_VISIBLE_DEVICES=${visible_gpu} --env MUJOCO_EGL_DEVICE_ID=${visible_gpu} \
    --env NUMBA_CACHE_DIR="${cache}" \
    "${SIF}" /opt/conda/envs/libero/bin/python \
      "${ROOT}/eval_successful_pair_closed_loop.py" \
        --manifest "${MANIFEST}" --output "${OUT}/${name}/shard_0.jsonl" \
        --arm paired_pi_late --host 127.0.0.1 --port "${port}" --checkpoint "${CKPT}" \
        --split validation --shard-id 0 --num-shards 1
'

"${PYTHON}" "${ROOT}/summarize_successful_pair_trust_sweep.py" \
  --root "${OUT}" --output "${OUT}/summary.json" --config "${CONFIG}" \
  --expected-perturbed "${EXPECTED_PERTURBED}" --expected-clean "${EXPECTED_CLEAN}"

echo "Successful-pair trust sweep complete: ${OUT}"

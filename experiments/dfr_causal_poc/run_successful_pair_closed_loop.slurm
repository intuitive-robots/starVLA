#!/bin/bash
# Four independent one-GPU workers, matching the topology used by the
# successful trust-region sweeps.
#SBATCH --job-name=dfr_cf_rollout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_rollout_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_cf_rollout_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PAIR_ROOT=${DFR_CF_PAIR_ROOT:-${ROOT}/outputs/paired_success_groot_encoder_v3}
MANIFEST=${DFR_CF_MANIFEST:-${PAIR_ROOT}/counterfactuals/manifest.jsonl}
CKPT=${DFR_CF_CHECKPOINT:-${STAR}/playground/Checkpoints/libero_gr00t_arm_plain/final_model/pytorch_model.pt}
READOUT=${DFR_CF_READOUT:-${PAIR_ROOT}/readouts/groot_encoder/probe_shards/rank_3_pi_late_paired_balanced.pt}
OUT=${DFR_CF_CLOSED_LOOP_OUT:-${PAIR_ROOT}/closed_loop_pi_late_v1}
SIF=${LIBERO_PLUS_SIF:-${STAR}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}
MAX_CASES=${DFR_CF_MAX_CASES_PER_DOMAIN:-0}
EXPECTED_PERTURBED=${DFR_CF_EXPECTED_PERTURBED:-104}
EXPECTED_CLEAN=${DFR_CF_EXPECTED_CLEAN:-26}
SPLIT=${DFR_CF_SPLIT:-test}
STRENGTH=${DFR_CF_RESIDUAL_STRENGTH:-1.0}
NORM_CAP_RATIO=${DFR_CF_RESIDUAL_NORM_CAP_RATIO:-}
RUN_SOURCE_REPLAY=${DFR_CF_RUN_SOURCE_REPLAY:-0}
PAIRED_ROLLOUT=${DFR_CF_PAIRED_ROLLOUT:-0}
PYTHON=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python

test -s "${MANIFEST}"
test -s "${CKPT}"
test -s "${READOUT}"
test -s "${SIF}"
mkdir -p "${OUT}/servers"

export STAR ROOT PAIR_ROOT MANIFEST CKPT READOUT OUT SIF MAX_CASES
export EXPECTED_PERTURBED EXPECTED_CLEAN SPLIT STRENGTH NORM_CAP_RATIO
export RUN_SOURCE_REPLAY PAIRED_ROLLOUT PYTHON
cd "${STAR}"

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  rank=${SLURM_LOCALID}
  visible_gpu=${CUDA_VISIBLE_DEVICES:-${rank}}
  port=$((18400 + rank))
  server_pid=""

  stop_server() {
    if [ -n "${server_pid}" ]; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
      server_pid=""
    fi
  }
  trap stop_server EXIT INT TERM

  run_client_with_retries() {
    local arm=$1
    local output_dir=$2
    local source_replay=$3
    local cache=${TMPDIR:-/tmp}/dfr_cf_rollout_${SLURM_JOB_ID}_${arm}_${rank}
    mkdir -p "${output_dir}" "${cache}"
    for attempt in 1 2 3; do
      local attempt_cache=${cache}_${attempt}
      mkdir -p "${attempt_cache}"
      extra_args=()
      if [ "${source_replay}" -eq 1 ]; then
        extra_args+=(--source-replay)
      else
        extra_args+=(--host 127.0.0.1 --port "${port}" --checkpoint "${CKPT}")
      fi
      if CUDA_VISIBLE_DEVICES=${visible_gpu} apptainer exec --cleanenv --nv \
        --bind "${STAR}:${STAR}:rw" --pwd "${STAR}" \
        --env LIBERO_HOME=/app/LIBERO-plus --env PYTHONPATH="${STAR}" \
        --env STARVLA_FAST_GLASS_BLUR=1 \
        --env MUJOCO_GL=egl --env PYOPENGL_PLATFORM=egl \
        --env CUDA_VISIBLE_DEVICES=${visible_gpu} --env MUJOCO_EGL_DEVICE_ID=${visible_gpu} \
        --env NUMBA_CACHE_DIR="${attempt_cache}" \
        "${SIF}" /opt/conda/envs/libero/bin/python \
          "${ROOT}/eval_successful_pair_closed_loop.py" \
            --manifest "${MANIFEST}" --output "${output_dir}/shard_${rank}.jsonl" \
            --arm "${arm}" --shard-id "${rank}" --num-shards 4 \
            --max-cases-per-domain "${MAX_CASES}" --split "${SPLIT}" \
            "${extra_args[@]}" \
            >> "${output_dir}/shard_${rank}.log" 2>&1; then
        return 0
      fi
      echo "retrying ${arm} shard ${rank} after attempt ${attempt}" \
        >> "${output_dir}/shard_${rank}.log"
    done
    return 1
  }

  run_policy_arm() {
    local arm=$1
    local adapter=""
    if [ "${arm}" = paired_pi_late ]; then
      adapter=${READOUT}
    fi
    mkdir -p "${OUT}/${arm}"
    CUDA_VISIBLE_DEVICES=${visible_gpu} \
      STARVLA_FLOW_RESIDUAL_READOUT="${adapter}" \
      STARVLA_FLOW_RESIDUAL_STRENGTH="${STRENGTH}" \
      STARVLA_FLOW_RESIDUAL_NORM_CAP_RATIO="${NORM_CAP_RATIO}" \
      STARVLA_PAIRED_ROLLOUT="${PAIRED_ROLLOUT}" \
      "${PYTHON}" deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT}" --port "${port}" --use_bf16 --idle_timeout -1 \
        > "${OUT}/servers/${arm}_gpu${rank}.log" 2>&1 &
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
      echo "policy server for ${arm} shard ${rank} did not become ready" >&2
      return 1
    fi

    run_client_with_retries "${arm}" "${OUT}/${arm}" 0
    stop_server
  }

  run_policy_arm baseline
  run_policy_arm paired_pi_late
  if [ "${RUN_SOURCE_REPLAY}" -eq 1 ]; then
    run_client_with_retries source_replay "${OUT}/source_replay" 1
  fi
'

if [ "${MAX_CASES}" -eq 0 ]; then
  "${PYTHON}" "${ROOT}/summarize_successful_pair_closed_loop.py" \
    --root "${OUT}" --output "${OUT}/summary.json" \
    --expected-perturbed "${EXPECTED_PERTURBED}" --expected-clean "${EXPECTED_CLEAN}"
fi

echo "Successful-pair closed-loop evaluation complete: ${OUT}"

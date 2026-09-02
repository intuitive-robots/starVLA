#!/bin/bash
set -euo pipefail
cd /e/home/jusers/blank4/jupiter/blank4/code/starVLA

ROOT=$(pwd)
VANILLA_LIBERO_HOME=${ROOT}/playground/sims/LIBERO
STARVLA_PYTHON=/e/project1/m3/blank4/envs/miniforge3/envs/starVLA/bin/python
LIBERO_PYTHON=/e/project1/m3/blank4/envs/miniforge3/envs/libero/bin/python
PORT=10103
OUT=vis/libero_object_4cm_traces_corrected
mkdir -p "${OUT}"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date --iso-8601=seconds)] rendering shared observations"
mkdir -p "${OUT}/inputs"
for task_id in $(seq 0 9); do
  rendered=false
  for attempt in $(seq 1 10); do
    echo "render task=${task_id} attempt=${attempt}"
    if MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
      CUDA_VISIBLE_DEVICES=0 LIBERO_HOME="${VANILLA_LIBERO_HOME}" \
      LIBERO_CONFIG_PATH="${VANILLA_LIBERO_HOME}/libero" \
      PYTHONPATH="${ROOT}:${VANILLA_LIBERO_HOME}" \
      "${LIBERO_PYTHON}" scripts/render_libero_object_perturbed_initials.py \
        --task-id "${task_id}" --output-dir "${OUT}/inputs"; then
      rendered=true
      break
    fi
  done
  if [ "${rendered}" != true ]; then
    echo "Failed to render a valid frame for task ${task_id} after 10 attempts" >&2
    exit 1
  fi
done

run_model() {
  local label=$1
  local checkpoint=$2
  local model_out=${OUT}/${label}
  mkdir -p "${model_out}"
  echo "[$(date --iso-8601=seconds)] starting ${label}"
  DEBUG=0 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="${ROOT}:${VANILLA_LIBERO_HOME}" \
    "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
      --ckpt_path "${checkpoint}" --port "${PORT}" --use_bf16 \
      --idle_timeout -1 --max_batch_size 16 --max_wait_time 0.1 \
      --cot_max_new_tokens 72 > "${model_out}/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    if "${STARVLA_PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(.2); s.connect(('127.0.0.1',${PORT})); s.close()" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server failed for ${label}" >&2
    tail -80 "${model_out}/server.log" >&2
    return 1
  fi
  PYTHONPATH="${ROOT}:${VANILLA_LIBERO_HOME}" "${LIBERO_PYTHON}" \
    scripts/infer_libero_trace_overlays_from_images.py \
      --input-dir "${OUT}/inputs" --output-dir "${model_out}" \
      --model-label "${label}" --port "${PORT}"
  cleanup
  SERVER_PID=
  echo "[$(date --iso-8601=seconds)] finished ${label}"
}

run_model ours playground/Checkpoints/libero_qwen08b_oft_cot_trace_ours_ft/final_model/pytorch_model.pt
run_model detector playground/Checkpoints/libero_qwen08b_oft_cot_trace_det/final_model/pytorch_model.pt
"${LIBERO_PYTHON}" scripts/make_libero_trace_comparison_grid.py \
  --ours "${OUT}/ours" --det "${OUT}/detector" \
  --output "${OUT}/ours_vs_detector_ep0.png"
echo "[$(date --iso-8601=seconds)] visualization complete"

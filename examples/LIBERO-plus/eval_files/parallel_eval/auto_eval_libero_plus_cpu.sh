#!/bin/bash
set -euo pipefail

SCRIPT_PATH="./examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_in_one_cpu.sh"
AGGREGATE_SCRIPT="./examples/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME="${LIBERO_HOME:-/e/project1/m3/blank4/code/starVLA/playground/sims/LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
# Use Mesa software EGL to avoid NVIDIA EGL 590.x cross-process global-state crash
_MESA_GL_DIR=/e/software/default/stages/2026/software/OpenGL/2025.09-GCCcore-14.3.0
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __EGL_VENDOR_LIBRARY_DIRS="${_MESA_GL_DIR}/share/glvnd/egl_vendor.d"
export LD_LIBRARY_PATH="${_MESA_GL_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBGL_DRIVERS_PATH="${_MESA_GL_DIR}/lib/dri"
# Use llvmpipe (LLVM JIT multi-threaded) instead of swrast (single-threaded)
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
export LP_NUM_THREADS="${LP_NUM_THREADS:-8}"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# See auto_eval_libero_plus.sh: keep numba's JIT cache off shared NFS.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/starvla_numba_cache}"

LIBERO_PLUS_CONDA_ENV="${LIBERO_PLUS_CONDA_ENV:-libero-plus}"
POLICY_SERVER_CONDA_ENV="${POLICY_SERVER_CONDA_ENV:-starVLA}"
your_ckpt="${your_ckpt:-path_to_checkpoint}"
export output_dir="${output_dir:-}"
server_host="${server_host:-127.0.0.1}"
base_port="${base_port:-10093}"
server_idle_timeout="${server_idle_timeout:--1}"
# === End of environment variable configuration ===
###########################################################################################

task_suite_name="${1:-libero_goal}"
num_gpus="${2:-4}"
tasks_per_gpu="${3:-10}"
workers_per_gpu="${4:-1}"
partition_idx="${5:-0}"
num_trials_per_task="${6:-1}"
gpu_ids_csv="${7:-}"


workers_per_gpu=1


if [ "${your_ckpt}" = "path_to_checkpoint" ]; then
    echo "[ERROR] Please set your_ckpt to a real checkpoint path."
    exit 1
fi

if [ -z "${output_dir}" ]; then
    ckpt_parent_dir=$(dirname "$(dirname "${your_ckpt}")")
    output_dir="${ckpt_parent_dir}/results/libero-plus"
fi

mkdir -p "${output_dir}"

server_pids=()
server_logs=()

cleanup() {
    if [ ${#server_pids[@]} -gt 0 ]; then
        echo "Stopping policy servers: ${server_pids[*]}"
        for pid in "${server_pids[@]}"; do
            kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        done
        wait "${server_pids[@]}" 2>/dev/null || true
    fi
}

wait_for_port() {
    local _host="$1"
    local port="$2"
    # See auto_eval_libero_plus.sh: server load is slow (registry scan + ckpt),
    # and multiple servers/GPU load concurrently, so 60s isn't enough.
    local attempts="${WAIT_FOR_PORT_ATTEMPTS:-600}"

    for ((attempt=1; attempt<=attempts; attempt++)); do
        if ss -ltnH "( sport = :${port} )" 2>/dev/null | grep -q ":${port}"; then
            return 0
        fi
        sleep 1
    done

    return 1
}

trap cleanup EXIT

if [ -z "${gpu_ids_csv}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpu_ids_csv="${CUDA_VISIBLE_DEVICES}"
fi

if [ -z "${gpu_ids_csv}" ]; then
    gpu_ids=()
    for ((gpu_idx=0; gpu_idx<num_gpus; gpu_idx++)); do
        gpu_ids+=("${gpu_idx}")
    done
    gpu_ids_csv=$(IFS=,; echo "${gpu_ids[*]}")
else
    IFS=',' read -r -a gpu_ids <<< "${gpu_ids_csv}"
    # Respect the num_gpus argument: truncate if CUDA_VISIBLE_DEVICES has more
    if [ "${#gpu_ids[@]}" -gt "${num_gpus}" ]; then
        gpu_ids=("${gpu_ids[@]:0:${num_gpus}}")
    else
        num_gpus=${#gpu_ids[@]}
    fi
    gpu_ids_csv=$(IFS=,; echo "${gpu_ids[*]}")
fi

case "${task_suite_name}" in
    libero_10)
        suite_size=2519
        ;;
    libero_goal)
        suite_size=2591
        ;;
    libero_object)
        suite_size=2518
        ;;
    libero_spatial)
        suite_size=2402
        ;;
    *)
        echo "[ERROR] Unknown LIBERO-plus task suite: ${task_suite_name}"
        exit 1
        ;;
esac

tasks_per_partition=$((num_gpus * tasks_per_gpu))
start_idx=$((partition_idx * tasks_per_partition))
end_idx=$((start_idx + tasks_per_partition))
if [ "${end_idx}" -gt "${suite_size}" ]; then
    end_idx=${suite_size}
fi

if [ "${start_idx}" -ge "${suite_size}" ]; then
    echo "[ERROR] Partition ${partition_idx} starts at ${start_idx}, beyond suite size ${suite_size}."
    exit 1
fi

echo "=========================================="
echo " LIBERO-plus Auto Eval"
echo "=========================================="
echo " Suite            : ${task_suite_name}"
echo " Checkpoint       : ${your_ckpt}"
echo " Output dir       : ${output_dir}"
echo " Conda env        : ${LIBERO_PLUS_CONDA_ENV}"
echo " Server env       : ${POLICY_SERVER_CONDA_ENV}"
echo " GPUs             : ${gpu_ids_csv}"
echo " Tasks / GPU      : ${tasks_per_gpu}"
echo " Workers / GPU    : ${workers_per_gpu}"
echo " Partition index  : ${partition_idx}"
echo " Trials / task    : ${num_trials_per_task}"
echo " Task range       : [${start_idx}, ${end_idx}) of ${suite_size}"
echo " Server host      : ${server_host}"
echo " Base port        : ${base_port}"
echo "=========================================="

server_log_dir="${output_dir}/server_logs"
mkdir -p "${server_log_dir}"

for ((i=0; i<num_gpus; i++)); do
    gpu_id=${gpu_ids[$i]}
    port=$((base_port + i))
    # partition_idx prefix: see auto_eval_libero_plus.sh -- avoids cross-node
    # filename collisions in the shared output_dir under multi-node runs.
    log_file="${server_log_dir}/partition${partition_idx}_gpu_${gpu_id}_port_${port}.log"
    server_logs+=("${log_file}")
    echo "Launching policy server on gpu=${gpu_id}, port=${port}, log=${log_file}"
    setsid bash -lc "CUDA_VISIBLE_DEVICES=${gpu_id} CUDA_LAUNCH_BLOCKING=1 conda run --no-capture-output -n \"${POLICY_SERVER_CONDA_ENV}\" python deployment/model_server/server_policy.py --ckpt_path \"${your_ckpt}\" --port \"${port}\" --use_bf16 --idle_timeout \"${server_idle_timeout}\"" > "${log_file}" 2>&1 &
    server_pids+=($!)
done

for ((i=0; i<num_gpus; i++)); do
    port=$((base_port + i))
    if ! wait_for_port "${server_host}" "${port}"; then
        echo "[ERROR] Policy server on port ${port} did not become ready."
        if [ -f "${server_logs[$i]}" ]; then
            echo "----- tail ${server_logs[$i]} -----"
            tail -n 80 "${server_logs[$i]}" || true
            echo "-----------------------------------"
        fi
        exit 1
    fi
done

bash "${SCRIPT_PATH}" \
    "${task_suite_name}" \
    "${start_idx}" \
    "${end_idx}" \
    "${num_gpus}" \
    "${tasks_per_gpu}" \
    "${workers_per_gpu}" \
    "${num_trials_per_task}" \
    "${gpu_ids_csv}" \
    "${base_port}" \
    "${server_host}" \
    "true"

python "${AGGREGATE_SCRIPT}" --root_path "${output_dir}"
echo "Aggregated results written to ${output_dir}/overall_results.json"

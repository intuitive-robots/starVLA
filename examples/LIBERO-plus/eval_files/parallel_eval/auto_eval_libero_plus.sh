#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Compute nodes do not necessarily expose git on PATH.  This script lives at
# examples/LIBERO-plus/eval_files/parallel_eval, exactly four levels below the
# StarVLA root, so resolve the checkout without an external VCS command.
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
SCRIPT_PATH="${SCRIPT_DIR}/eval_libero_in_one.sh"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_results.py"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME="${LIBERO_HOME:-/e/project1/m3/blank4/code/starVLA/playground/sims/LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# numba JIT-compiles on first use and writes its cache INTO the installed package
# dir by default. With many server/worker processes (servers_per_gpu * workers_per_gpu
# * num_gpus, times num_nodes) all importing it near-simultaneously against a
# shared NFS-mounted conda env, that turns into a write race NFS handles badly:
# "Stale file handle" errors on numba/* and Bus errors (mmap'd cache pages going
# invalid mid-access) under load. Redirect to node-local /tmp so it's never a
# shared write target across processes/nodes.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/starvla_numba_cache}"

LIBERO_PLUS_CONDA_ENV="${LIBERO_PLUS_CONDA_ENV:-libero-plus}"
POLICY_SERVER_CONDA_ENV="${POLICY_SERVER_CONDA_ENV:-starVLA}"
LIBERO_PLUS_SIF="${LIBERO_PLUS_SIF:-${REPO_ROOT}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}"
LIBERO_PLUS_RUNTIME="${LIBERO_PLUS_RUNTIME:-auto}"
your_ckpt="${your_ckpt:-path_to_checkpoint}"
export output_dir="${output_dir:-}"
server_host="${server_host:-127.0.0.1}"
base_port="${base_port:-10093}"
server_idle_timeout="${server_idle_timeout:--1}"
# Batches concurrent requests from multiple workers into one predict_action()
# call instead of serving one at a time -- see batch_dispatcher.py. Measured
# on this model: ~30x throughput at batch 32, ~112x at batch 128, for a ~15%
# latency cost per batch. Set to (at least) workers_per_gpu; 1 = old behavior.
max_batch_size="${max_batch_size:-32}"
max_wait_time="${max_wait_time:-1.0}"
# 0 (default) = full suite (~2400-2600 task instances). Set to e.g. 500 to
# only evaluate ~500 task instances of this suite. IMPORTANT: this suite's task
# indices are grouped into large contiguous blocks by perturbation category
# (e.g. libero_10 index 0-499 is entirely "table" perturbations, 500-999 is
# "view", ...) -- taking a plain prefix would only ever test ONE category. So
# this instead sets a stride = suite_size / max_tasks_per_suite and evaluates
# every stride'th task index across the FULL suite, which spreads the sample
# proportionally across every category the suite contains.
max_tasks_per_suite="${max_tasks_per_suite:-0}"
# Exact evenly spaced global sample count. Unlike max_tasks_per_suite, this
# does not restart an integer stride in each worker shard.
exact_tasks_per_suite="${exact_tasks_per_suite:-0}"
# Draw the model's generated 2D trace on saved rollout videos (explicit-CoT
# checkpoints only). Exported so eval_libero_in_one.sh's workers inherit it.
export overlay_trace="${overlay_trace:-False}"
# Quantitative evaluations need only per-shard JSON. Saving every rollout can
# exhaust project inode quotas, so video is explicitly opt-in.
export save_video="${save_video:-False}"
export object_perturb_m="${object_perturb_m:-0.0}"
export object_perturb_roles="${object_perturb_roles:-source,target}"
export object_perturb_seed="${object_perturb_seed:-20260812}"
# === End of environment variable configuration ===
###########################################################################################

task_suite_name="${1:-libero_goal}"
num_gpus="${2:-4}"
# 0 (default) = auto: size tasks_per_gpu from the suite's true length so this
# partition's [start_idx,end_idx) covers exactly its 1/num_partitions share
# (see suite_size lookup below). Pass a positive value to override.
tasks_per_gpu="${3:-0}"
workers_per_gpu="${4:-1}"
partition_idx="${5:-0}"
num_trials_per_task="${6:-1}"
gpu_ids_csv="${7:-}"
# Policy servers per GPU. Each costs one full copy of the model in that GPU's
# memory but gives real inference concurrency to workers sharing that GPU
# (the server runs inference synchronously per request). Best throughput when
# workers_per_gpu is a multiple of servers_per_gpu.
servers_per_gpu="${8:-1}"
# Total independent partitions this suite is split into -- i.e. total nodes
# when driven multi-node by eval_libero_plus_slurm.sh (partition_idx is then
# that node's rank). 1 = single partition covering the whole suite, as before.
num_partitions="${9:-1}"
# Set by the multi-node wrapper: each node would otherwise redundantly
# aggregate a partial (racy) snapshot of the shared output_dir; the wrapper
# does one aggregation itself after every partition has finished instead.
skip_aggregate="${SKIP_AGGREGATE:-false}"

case "${LIBERO_PLUS_RUNTIME}" in
    auto)
        if command -v apptainer >/dev/null 2>&1 && [ -f "${LIBERO_PLUS_SIF}" ]; then
            LIBERO_PLUS_RUNTIME=apptainer
        else
            LIBERO_PLUS_RUNTIME=conda
        fi
        ;;
    apptainer)
        command -v apptainer >/dev/null 2>&1 || {
            echo "[ERROR] LIBERO_PLUS_RUNTIME=apptainer but apptainer is unavailable."
            exit 1
        }
        [ -f "${LIBERO_PLUS_SIF}" ] || {
            echo "[ERROR] LIBERO-Plus SIF not found: ${LIBERO_PLUS_SIF}"
            exit 1
        }
        ;;
    conda) ;;
    *)
        echo "[ERROR] LIBERO_PLUS_RUNTIME must be auto, apptainer, or conda; got ${LIBERO_PLUS_RUNTIME}."
        exit 1
        ;;
esac
export LIBERO_PLUS_RUNTIME LIBERO_PLUS_SIF

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
    # Server load is slow: full dataset-registry scan + ~1.7GB ckpt, and with
    # servers_per_gpu > 1 several server processes do this concurrently on the
    # same node (I/O + GPU-init contention). 60s (the old default) isn't enough
    # once more than ~1 server/GPU is loading at once.
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

if [ "${exact_tasks_per_suite}" -lt 0 ] || [ "${exact_tasks_per_suite}" -gt "${suite_size}" ]; then
    echo "[ERROR] exact_tasks_per_suite must be in [0, ${suite_size}], got ${exact_tasks_per_suite}."
    exit 1
fi

# stride evaluates every stride'th task index instead of shrinking the range
# to a prefix -- keeps suite_size (and therefore the GPU/partition split
# below) spanning the FULL suite, so every category is still represented.
stride=1
if [ "${max_tasks_per_suite}" -gt 0 ] && [ "${max_tasks_per_suite}" -lt "${suite_size}" ]; then
    stride=$((suite_size / max_tasks_per_suite))
fi

# Auto-size tasks_per_gpu from the true suite length so num_partitions
# partitions (nodes) cover the suite exactly once between them, each getting
# ceil(suite_size / (num_partitions * num_gpus)) tasks per GPU.
if [ "${tasks_per_gpu}" -le 0 ]; then
    tasks_per_gpu=$(( (suite_size + num_partitions * num_gpus - 1) / (num_partitions * num_gpus) ))
fi

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

num_servers=$((num_gpus * servers_per_gpu))

echo "=========================================="
echo " LIBERO-plus Auto Eval"
echo "=========================================="
echo " Suite            : ${task_suite_name}"
echo " Checkpoint       : ${your_ckpt}"
echo " Output dir       : ${output_dir}"
echo " Conda env        : ${LIBERO_PLUS_CONDA_ENV}"
echo " Server env       : ${POLICY_SERVER_CONDA_ENV}"
echo " Simulator runtime: ${LIBERO_PLUS_RUNTIME}"
if [ "${LIBERO_PLUS_RUNTIME}" = apptainer ]; then
    echo " Simulator image  : ${LIBERO_PLUS_SIF}"
fi
echo " GPUs             : ${gpu_ids_csv}"
echo " Tasks / GPU      : ${tasks_per_gpu}"
echo " Workers / GPU    : ${workers_per_gpu}"
echo " Servers / GPU    : ${servers_per_gpu}  (total servers: ${num_servers}, ports ${base_port}..$((base_port + num_servers - 1)))"
echo " Max batch size   : ${max_batch_size}  (max_wait_time=${max_wait_time}s)"
echo " Object perturb   : ${object_perturb_m}m roles=${object_perturb_roles} seed=${object_perturb_seed}"
echo " Partition index  : ${partition_idx} / ${num_partitions}"
echo " Trials / task    : ${num_trials_per_task}"
if [ "${exact_tasks_per_suite}" -gt 0 ]; then
    echo " Task range       : [${start_idx}, ${end_idx}) of ${suite_size}  (raw partition range)"
    echo " Exact global sample: ${exact_tasks_per_suite} evenly spaced tasks/suite"
else
    echo " Task range       : [${start_idx}, ${end_idx}) of ${suite_size}  (stride=${stride}, ~$(( (end_idx - start_idx + stride - 1) / stride )) tasks evaluated)"
fi
echo " Server host      : ${server_host}"
echo " Base port        : ${base_port}"
echo "=========================================="

if [ "${workers_per_gpu}" -lt "${servers_per_gpu}" ]; then
    echo "[WARN] workers_per_gpu=${workers_per_gpu} < servers_per_gpu=${servers_per_gpu}: some servers will idle."
fi

server_log_dir="${output_dir}/server_logs"
mkdir -p "${server_log_dir}"

for ((g=0; g<num_gpus; g++)); do
    gpu_id=${gpu_ids[$g]}
    for ((s=0; s<servers_per_gpu; s++)); do
        port=$((base_port + g * servers_per_gpu + s))
        # Prefixed with partition_idx: gpu_id/port are LOCAL to each node, so under
        # the multi-node wrapper every node reuses the same gpu_id/port values and
        # would otherwise collide on this filename in the shared (NFS) output_dir --
        # multiple nodes' processes writing the "same" log file concurrently produces
        # interleaved/corrupted content, not an error, so it silently misleads
        # debugging. partition_idx (this node's rank) makes it unique per node.
        log_file="${server_log_dir}/partition${partition_idx}_gpu_${gpu_id}_port_${port}.log"
        server_logs+=("${log_file}")
        echo "Launching policy server on gpu=${gpu_id}, port=${port}, log=${log_file}"
        setsid bash -lc "CUDA_VISIBLE_DEVICES=${gpu_id} conda run --no-capture-output -n \"${POLICY_SERVER_CONDA_ENV}\" python deployment/model_server/server_policy.py --ckpt_path \"${your_ckpt}\" --port \"${port}\" --use_bf16 --idle_timeout \"${server_idle_timeout}\" --max_batch_size \"${max_batch_size}\" --max_wait_time \"${max_wait_time}\"" > "${log_file}" 2>&1 &
        server_pids+=($!)
    done
done

for ((i=0; i<num_servers; i++)); do
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
    "true" \
    "${servers_per_gpu}" \
    "${stride}" \
    "${exact_tasks_per_suite}"

if [ "${skip_aggregate}" != "true" ]; then
    # Suite-scoped (writes <output_dir>/<suite>/overall_results.json), not the
    # shared combined file -- so aggregating one suite can never race with or
    # overwrite another suite's results. Combine explicitly across suites with:
    #   python examples/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py --root_path <output_dir>
    python "${AGGREGATE_SCRIPT}" --root_path "${output_dir}" --task_suite_name "${task_suite_name}"
fi

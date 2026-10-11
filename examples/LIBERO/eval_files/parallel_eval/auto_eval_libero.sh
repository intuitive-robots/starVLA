#!/bin/bash
# Parallel vanilla-LIBERO evaluation: one policy server per GPU, N sim workers per GPU.
#
#   bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
#       <suite> <num_gpus> <workers_per_gpu> <num_trials> [gpu_ids_csv] [partition_idx] [num_partitions]
#
# Example (full libero_goal suite, 4 GPUs x 3 workers, standard 50 trials/task):
#   your_ckpt=/path/to/pytorch_model.pt \
#   bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh libero_goal 4 3 50
#
# Multi-node: partition_idx/num_partitions slice the suite's episode range
# BEFORE the per-worker split, so node `partition_idx` of `num_partitions`
# only ever touches its own [start,end) share -- see eval_libero_slurm.sh for
# the srun-driven wrapper that sets these automatically per node.
set -euo pipefail

# Keep simulator imports scalar.  Native BLAS/OpenMP pools otherwise multiply
# across workers and can exhaust process/thread resources on a full eval node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Native MuJoCo/EGL aborts must remain visible through exit status, but writing
# multi-gigabyte core files into the shared checkout only adds NFS pressure.
ulimit -c 0

SHARD_SCRIPT="./examples/LIBERO/eval_files/parallel_eval/eval_libero_shard.py"
AGGREGATE_SCRIPT="./examples/LIBERO/eval_files/parallel_eval/aggregate_results.py"

###########################################################################################
# === Please modify the following paths according to your environment ===
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# NOTE: deliberately NOT "${LIBERO_HOME:-...}". A LIBERO_HOME exported for a
# LIBERO-plus run would silently win and this script would evaluate the plus
# fork instead of vanilla LIBERO. Override with VANILLA_LIBERO_HOME if needed.
export LIBERO_HOME="${VANILLA_LIBERO_HOME:-${REPO_ROOT}/playground/sims/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# Drop any inherited LIBERO-plus entries so the vanilla checkout resolves first.
# NOTE: `|| true` is required — with an empty PYTHONPATH the greps match nothing,
# exit 1, and `set -eo pipefail` would kill the script before any output.
_clean_pythonpath=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v 'LIBERO-plus' | grep -v '^$' | paste -sd: - || true)
if [ -n "${_clean_pythonpath}" ]; then
    export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${_clean_pythonpath}"
else
    export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}"
fi

# Absolute interpreters — do NOT rely on `python` resolving correctly (a stray
# venv on PATH will silently win over the activated conda env).
# The conda envs were replaced by packed Apptainer images on 2026-09-14, so these
# now point at wrappers that exec `python` inside the matching image. Still
# absolute paths, and still overridable, so the reasoning above is unchanged.
LIBERO_PYTHON="${LIBERO_PYTHON:-${REPO_ROOT}/scripts/env/libero_python}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${REPO_ROOT}/scripts/env/starvla_python}"

your_ckpt="${your_ckpt:-path_to_checkpoint}"
output_dir="${output_dir:-}"
server_host="${server_host:-127.0.0.1}"
base_port="${base_port:-10093}"
server_idle_timeout="${server_idle_timeout:--1}"
save_video="${save_video:-false}"   # 500 episodes/suite is a lot of mp4s
# Policy servers per GPU. The server runs inference synchronously inside its
# async handler, so workers sharing one server serialize their forward passes.
# Running several servers on a GPU gives real inference concurrency; each costs
# one full copy of the model in that GPU's memory (~1.7GB for a 0.8B bf16 ckpt).
# Best throughput when workers_per_gpu is a multiple of servers_per_gpu.
servers_per_gpu="${servers_per_gpu:-1}"
# Batches concurrent requests from multiple workers into one predict_action()
# call instead of serving one at a time -- see deployment/model_server/tools/
# batch_dispatcher.py. Measured on the 0.8B Qwen3.5 OFT-CoT model: ~30x
# throughput at batch 32, ~112x at batch 128, for a ~15% latency cost per
# batch. Set to (at least) workers_per_gpu; 1 = old unbatched behavior.
max_batch_size="${max_batch_size:-32}"
max_wait_time="${max_wait_time:-1.0}"
cot_max_new_tokens="${cot_max_new_tokens:-0}"
object_perturb_m="${object_perturb_m:-0.0}"
object_perturb_roles="${object_perturb_roles:-source,target}"
object_perturb_seed="${object_perturb_seed:-20260812}"
capture_successful_dir="${capture_successful_dir:-}"
# EGL initialization on the Booster driver can fail transiently when several
# contexts start together. Retry the complete shard so a successful job always
# has complete episode coverage. Override to 1 to disable retries.
worker_max_attempts="${worker_max_attempts:-3}"
# === End of environment variable configuration ===
###########################################################################################

# One suite, or several as a comma-separated list ("libero_spatial,libero_10").
# A list is evaluated by ONE set of policy servers: the callers used to invoke
# this script once per suite, which tore down and reloaded every server between
# suites at ~1.1 min each -- ~3.4 min per arm wasted on a 4-suite sweep, and a
# full cold reload if the page cache had turned over. Passing a single suite
# behaves exactly as before.
task_suite_list="${1:-libero_goal}"
num_gpus="${2:-4}"
workers_per_gpu="${3:-2}"
max_concurrent_workers_per_gpu="${max_concurrent_workers_per_gpu:-${workers_per_gpu}}"
num_trials_per_task="${4:-50}"
gpu_ids_csv="${5:-}"
# Total independent partitions this suite's episodes are split into -- i.e.
# total nodes when driven multi-node (partition_idx is then that node's
# rank). 1 = single partition covering the whole suite, as before.
partition_idx="${6:-0}"
num_partitions="${7:-1}"
# Set by the multi-node wrapper: each node would otherwise redundantly
# aggregate a partial (racy) snapshot of the shared output_dir; the wrapper
# does one aggregation itself after every partition has finished instead.
skip_aggregate="${SKIP_AGGREGATE:-false}"
# chain_suites (default true): every worker slot runs its slice of ALL suites back to back. The launch stagger
# is paid once and there is no barrier between suites, instead of relaunching every worker per suite and waiting
# for the slowest episode each time (~30 s stagger + tail per suite per node). 2026-10-07 benchmark, 2,000
# episodes, 4 suites: 15-27% faster on 2-16 GPUs (16 GPUs 4.1 -> 3.0 min), same success, no failures.
# Shard files and aggregation are unchanged. chain_suites=false restores the per-suite loop.
chain_suites="${chain_suites:-true}"

IFS=',' read -r -a TASK_SUITES <<< "${task_suite_list}"
for _s in "${TASK_SUITES[@]}"; do
    case "${_s}" in
        libero_spatial|libero_object|libero_goal|libero_10|libero_90) ;;
        *) echo "[ERROR] Unknown LIBERO task suite: ${_s}"; exit 1 ;;
    esac
done

if [ "${your_ckpt}" = "path_to_checkpoint" ]; then
    echo "[ERROR] Please set your_ckpt to a real checkpoint path."
    exit 1
fi
if [ ! -f "${your_ckpt}" ]; then
    echo "[ERROR] Checkpoint not found: ${your_ckpt}"; exit 1
fi
for p in "${LIBERO_PYTHON}" "${STARVLA_PYTHON}"; do
    [ -x "$p" ] || { echo "[ERROR] Interpreter not executable: $p"; exit 1; }
done

# ── Gripper normalization mode for the SERVER ─────────────────────────────
# The policy server rebuilds the training-time transform from CODE, but the
# checkpoint doesn't record which gripper normalization it was trained with.
# Derive it from the checkpoint's own stats: action.min[6] == 0 means the data
# was already {0,1} and no gripper normalization was applied ("none"); == -1
# means raw LIBERO actions normalized with "binary_invert".
if [ -z "${STARVLA_LIBERO_GRIPPER_NORM:-}" ]; then
    _stats=$(dirname "$(dirname "${your_ckpt}")")/dataset_statistics.json
    if [ -f "${_stats}" ]; then
        STARVLA_LIBERO_GRIPPER_NORM=$("${LIBERO_PYTHON}" -c "
import json,sys
d=json.load(open('${_stats}')); k=next(iter(d))
print('none' if float(d[k]['action']['min'][6]) > -0.5 else 'binary_invert')
" 2>/dev/null || echo "binary_invert")
    else
        STARVLA_LIBERO_GRIPPER_NORM=binary_invert
    fi
fi
export STARVLA_LIBERO_GRIPPER_NORM
echo " Gripper norm     : ${STARVLA_LIBERO_GRIPPER_NORM}  (server-side un-normalization)"

if [ -z "${output_dir}" ]; then
    ckpt_parent_dir=$(dirname "$(dirname "${your_ckpt}")")
    output_dir="${ckpt_parent_dir}/results/libero"
fi
mkdir -p "${output_dir}"

# ── Resolve GPU ids ───────────────────────────────────────────────────────
if [ -z "${gpu_ids_csv}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpu_ids_csv="${CUDA_VISIBLE_DEVICES}"
fi
if [ -z "${gpu_ids_csv}" ]; then
    gpu_ids=(); for ((i=0; i<num_gpus; i++)); do gpu_ids+=("${i}"); done
else
    IFS=',' read -r -a gpu_ids <<< "${gpu_ids_csv}"
    if [ "${#gpu_ids[@]}" -gt "${num_gpus}" ]; then
        gpu_ids=("${gpu_ids[@]:0:${num_gpus}}")
    else
        num_gpus=${#gpu_ids[@]}
    fi
fi
gpu_ids_csv=$(IFS=,; echo "${gpu_ids[*]}")

num_workers=$((num_gpus * workers_per_gpu))
num_servers=$((num_gpus * servers_per_gpu))

# Per-suite episode accounting. Called once per suite inside the loop below,
# because every suite has its own task count and therefore its own partition
# slice; the policy servers are suite-agnostic and are started once, before it.
compute_suite_ranges() {
    local suite="$1"
    case "${suite}" in
        libero_90) n_tasks=90 ;;
        *)         n_tasks=10 ;;
    esac
    total_episodes=$((n_tasks * num_trials_per_task))

    # ── Slice this node's share of the suite (multi-node) ─────────────────
    # Same contiguous chunk+remainder split as the per-worker split below, just
    # one level up: [0, total_episodes) -> num_partitions pieces, this node takes
    # piece partition_idx. Everything after this point (num_workers etc.) then
    # operates on partition_episodes / current_start/current_end exactly as the
    # single-node case already did, just over a sub-range instead of the whole suite.
    partition_chunk=$((total_episodes / num_partitions))
    partition_remainder=$((total_episodes % num_partitions))
    if [ "${partition_idx}" -lt "${partition_remainder}" ]; then
        partition_start=$((partition_idx * (partition_chunk + 1)))
        partition_end=$((partition_start + partition_chunk + 1))
    else
        partition_start=$((partition_remainder * (partition_chunk + 1) + (partition_idx - partition_remainder) * partition_chunk))
        partition_end=$((partition_start + partition_chunk))
    fi
    if [ "${partition_start}" -ge "${total_episodes}" ]; then
        echo "[ERROR] Partition ${partition_idx} starts at ${partition_start}, beyond ${total_episodes} total episodes."
        return 1
    fi
    partition_episodes=$((partition_end - partition_start))
}

if [ "${workers_per_gpu}" -lt "${servers_per_gpu}" ]; then
    echo "[WARN] workers_per_gpu=${workers_per_gpu} < servers_per_gpu=${servers_per_gpu}: some servers will idle."
fi
if [ "${max_concurrent_workers_per_gpu}" -lt 1 ] || [ "${max_concurrent_workers_per_gpu}" -gt "${workers_per_gpu}" ]; then
    echo "[ERROR] max_concurrent_workers_per_gpu must be in [1, ${workers_per_gpu}], got ${max_concurrent_workers_per_gpu}."
    exit 1
fi

echo "=========================================="
echo " LIBERO Parallel Eval"
echo "=========================================="
echo " Suites           : ${TASK_SUITES[*]}  (${num_trials_per_task} trials/task)"
echo " Partition        : ${partition_idx} / ${num_partitions}"
echo " Checkpoint       : ${your_ckpt}"
echo " Output dir       : ${output_dir}"
echo " LIBERO_HOME      : ${LIBERO_HOME}"
echo " GPUs             : ${gpu_ids_csv}"
echo " Workers / GPU    : ${workers_per_gpu}  (total workers: ${num_workers})"
echo " Live sims / GPU  : ${max_concurrent_workers_per_gpu}"
echo " Servers / GPU    : ${servers_per_gpu}  (total servers: ${num_servers}, ports ${base_port}..$((base_port + num_servers - 1)))"
echo " Max batch size   : ${max_batch_size}  (max_wait_time=${max_wait_time}s)"
echo " CoT max tokens   : ${cot_max_new_tokens}  (0 = checkpoint default)"
echo " Save video       : ${save_video}"
echo " Object perturb   : ${object_perturb_m}m roles=${object_perturb_roles} seed=${object_perturb_seed}"
echo " Success capture  : ${capture_successful_dir:-disabled}"
echo "=========================================="

server_pids=(); server_logs=()

cleanup() {
    if [ ${#server_pids[@]} -gt 0 ]; then
        echo "Stopping policy servers: ${server_pids[*]}"
        for pid in "${server_pids[@]}"; do
            kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        done
        wait "${server_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_port() {
    local port="$1" attempts=600   # server load is slow: registry scan + 1.7GB ckpt
    for ((attempt=1; attempt<=attempts; attempt++)); do
        if ss -ltnH "( sport = :${port} )" 2>/dev/null | grep -q ":${port}"; then return 0; fi
        sleep 1
    done
    return 1
}

# ── Launch one policy server per GPU ──────────────────────────────────────
server_log_dir="${output_dir}/server_logs"; mkdir -p "${server_log_dir}"
for ((g=0; g<num_gpus; g++)); do
    gpu_id=${gpu_ids[$g]}
    for ((s=0; s<servers_per_gpu; s++)); do
        port=$((base_port + g * servers_per_gpu + s))
        # Prefixed with partition_idx: gpu_id/port are LOCAL to each node, so under
        # multi-node every node reuses the same gpu_id/port values and would
        # otherwise collide on this filename in the shared (NFS) output_dir --
        # multiple nodes' processes writing the "same" log file concurrently
        # produces interleaved/corrupted content, not an error, so it silently
        # misleads debugging. partition_idx (this node's rank) makes it unique.
        log_file="${server_log_dir}/partition${partition_idx}_gpu_${gpu_id}_port_${port}.log"
        server_logs+=("${log_file}")
        echo "Launching policy server on gpu=${gpu_id}, port=${port}, log=${log_file}"
        setsid env CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${REPO_ROOT}" \
            STARVLA_LIBERO_GRIPPER_NORM="${STARVLA_LIBERO_GRIPPER_NORM}" \
            "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
                --ckpt_path "${your_ckpt}" --port "${port}" --use_bf16 \
                --idle_timeout "${server_idle_timeout}" \
                --max_batch_size "${max_batch_size}" --max_wait_time "${max_wait_time}" \
                $( [ "${cot_max_new_tokens}" -gt 0 ] && echo "--cot_max_new_tokens ${cot_max_new_tokens}" ) \
            > "${log_file}" 2>&1 &
        server_pids+=($!)
    done
done

for ((i=0; i<num_servers; i++)); do
    port=$((base_port + i))
    echo "Waiting for policy server on port ${port} ..."
    if ! wait_for_port "${port}"; then
        echo "[ERROR] Policy server on port ${port} did not become ready."
        [ -f "${server_logs[$i]}" ] && { echo "----- tail ${server_logs[$i]} -----"; tail -n 80 "${server_logs[$i]}"; }
        exit 1
    fi
done
echo "All ${num_servers} policy servers ready."

# One shard with retries: <suite> <start> <end> <gpu_id> <port> <worker>. Body shared by both launch modes.
run_shard() {
    local task_suite_name="$1" current_start="$2" current_end="$3" gpu_id="$4" port="$5" w="$6" attempt rc
    shard_result="${output_dir}/logs/${task_suite_name}/${current_start}_to_${current_end}.json"
    if [ "${STARVLA_RESUME_EVAL:-0}" = "1" ] && [ -s "${shard_result}" ]; then
        echo "Worker ${w}: already complete, skipping (${shard_result})"
        return 0
    fi
    for ((attempt=1; attempt<=worker_max_attempts; attempt++)); do
        # robosuite 1.4 validates this value against the physical
        # IDs listed in CUDA_VISIBLE_DEVICES before CUDA remapping.
        if CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
            "${LIBERO_PYTHON}" "${SHARD_SCRIPT}" \
                --args.task-suite-name "${task_suite_name}" \
                --args.num-trials-per-task "${num_trials_per_task}" \
                --args.start-idx "${current_start}" --args.end-idx "${current_end}" \
                --args.output-dir "${output_dir}" \
                --args.host "${server_host}" --args.port "${port}" \
                --args.object-perturb-m "${object_perturb_m}" \
                --args.object-perturb-roles "${object_perturb_roles}" \
                --args.object-perturb-seed "${object_perturb_seed}" \
                $( [ -n "${capture_successful_dir}" ] && echo "--args.capture-successful-dir ${capture_successful_dir}" ) \
                $( [ "${save_video}" = "true" ] && echo "--args.save-video" || echo "--args.no-save-video" ); then
            return 0
        else
            rc=$?
        fi
        echo "[WARN] Worker ${w} shard [${current_start},${current_end}) failed (rc=${rc}, attempt ${attempt}/${worker_max_attempts})."
        [ "${attempt}" -lt "${worker_max_attempts}" ] && sleep $((attempt * 10))
    done
    echo "[ERROR] Worker ${w} shard [${current_start},${current_end}) exhausted ${worker_max_attempts} attempts."
    return "${rc}"
}

if [ "${chain_suites}" = "true" ]; then
    # Precompute every (suite, worker) range, then launch each worker slot once.
    declare -A chain_start chain_end
    for task_suite_name in "${TASK_SUITES[@]}"; do
        compute_suite_ranges "${task_suite_name}" || exit 1
        echo "##### chained suite ${task_suite_name}: ${n_tasks} tasks x ${num_trials_per_task} trials = ${total_episodes} episodes;" \
             "partition ${partition_idx}/${num_partitions} -> [${partition_start}, ${partition_end}) = ${partition_episodes}"
        chunk=$((partition_episodes / num_workers))
        remainder=$((partition_episodes % num_workers))
        current_start=${partition_start}
        for ((w=0; w<num_workers; w++)); do
            current_end=$((current_start + chunk + (w < remainder ? 1 : 0)))
            chain_start["${task_suite_name},${w}"]=${current_start}
            chain_end["${task_suite_name},${w}"]=${current_end}
            current_start=$current_end
        done
    done
    gpu_launcher_pids=()
    for ((g=0; g<num_gpus; g++)); do
        gpu_id=${gpu_ids[$g]}
        (
            pids=()
            launcher_rc=0
            for ((local_worker=0; local_worker<workers_per_gpu; local_worker++)); do
                w=$((g * workers_per_gpu + local_worker))
                port=$((base_port + g * servers_per_gpu + local_worker % servers_per_gpu))
                sleep 2   # EGL start-up stagger, once per worker slot instead of once per suite
                (
                    slot_rc=0
                    for task_suite_name in "${TASK_SUITES[@]}"; do
                        current_start=${chain_start["${task_suite_name},${w}"]}
                        current_end=${chain_end["${task_suite_name},${w}"]}
                        [ "$current_start" -ge "$current_end" ] && continue
                        echo "Worker ${w}: gpu=${gpu_id} port=${port} ${task_suite_name} [${current_start}, ${current_end})"
                        run_shard "${task_suite_name}" "${current_start}" "${current_end}" "${gpu_id}" "${port}" "${w}" || slot_rc=1
                    done
                    exit "${slot_rc}"
                ) &
                pids+=($!)
            done
            for pid in "${pids[@]}"; do
                wait "${pid}" || launcher_rc=1
            done
            exit "${launcher_rc}"
        ) &
        gpu_launcher_pids+=($!)
    done
    workers_rc=0
    for pid in "${gpu_launcher_pids[@]}"; do
        wait "${pid}" || workers_rc=1
    done
    if [ "${workers_rc}" -ne 0 ]; then
        echo "[ERROR] One or more simulator shards failed; refusing to aggregate partial results."
        exit 1
    fi
    echo "All workers finished (chained suites: ${TASK_SUITES[*]})."
    if [ "${skip_aggregate}" != "true" ]; then
        for task_suite_name in "${TASK_SUITES[@]}"; do
            echo "Aggregating ${task_suite_name}..."
            "${LIBERO_PYTHON}" "${AGGREGATE_SCRIPT}" --root_path "${output_dir}" --task_suite_name "${task_suite_name}"
        done
    fi
    TASK_SUITES=()   # the per-suite loop below then has nothing to do
fi

# ── Every suite below reuses the servers started above ────────────────────
for task_suite_name in "${TASK_SUITES[@]}"; do
compute_suite_ranges "${task_suite_name}" || exit 1
echo ""
echo "##### suite ${task_suite_name}: ${n_tasks} tasks x ${num_trials_per_task} trials = ${total_episodes} episodes;" \
     "partition ${partition_idx}/${num_partitions} -> [${partition_start}, ${partition_end}) = ${partition_episodes}"

# ── Precompute every worker's episode range up front ──────────────────────
# Pure arithmetic (cheap), so it can stay a single sequential pass -- the
# result is identical to the old inline computation, just split out so the
# (costly, EGL-staggered) launch loop below can be parallelized per GPU.
chunk=$((partition_episodes / num_workers))
remainder=$((partition_episodes % num_workers))
current_start=${partition_start}
worker_start=(); worker_end=()
for ((w=0; w<num_workers; w++)); do
    if [ "$w" -lt "$remainder" ]; then
        current_end=$((current_start + chunk + 1))
    else
        current_end=$((current_start + chunk))
    fi
    worker_start+=("${current_start}")
    worker_end+=("${current_end}")
    current_start=$current_end
done

# ── Launch workers: one background stream per GPU, staggered 2s only WITHIN
# that stream ──────────────────────────────────────────────────────────────
# The EGL race (libnvidia-eglcore.so global state on eglGetPlatformDisplayEXT,
# driver 590.x) is per-device, so different GPUs' worker streams don't need to
# be staggered against each other. Previously all workers across every GPU
# shared one global serial 2s stagger (num_workers * 2s wall time just to
# finish launching, e.g. ~200s for 100 workers); now it's num_gpus parallel
# streams of workers_per_gpu each (~workers_per_gpu * 2s, e.g. ~50s for the
# same 100 workers over 4 GPUs).
gpu_launcher_pids=()
for ((g=0; g<num_gpus; g++)); do
    gpu_id=${gpu_ids[$g]}
    (
        pids=()
        launcher_rc=0
        for ((local_worker=0; local_worker<workers_per_gpu; local_worker++)); do
            w=$((g * workers_per_gpu + local_worker))
            current_start=${worker_start[$w]}
            current_end=${worker_end[$w]}
            [ "$current_start" -ge "$current_end" ] && continue

            # Spread this GPU's workers round-robin over that GPU's servers.
            server_slot=$((local_worker % servers_per_gpu))
            port=$((base_port + g * servers_per_gpu + server_slot))

            echo "Worker ${w}: gpu=${gpu_id} port=${port} episodes [${current_start}, ${current_end})"

            sleep 2

            ( run_shard "${task_suite_name}" "${current_start}" "${current_end}" "${gpu_id}" "${port}" "${w}" ) &
            pids+=($!)
            if [ "${#pids[@]}" -ge "${max_concurrent_workers_per_gpu}" ]; then
                for pid in "${pids[@]}"; do
                    wait "${pid}" || launcher_rc=1
                done
                pids=()
            fi
        done
        for pid in "${pids[@]}"; do
            wait "${pid}" || launcher_rc=1
        done
        exit "${launcher_rc}"
    ) &
    gpu_launcher_pids+=($!)
done

# Wait on the per-GPU launcher subshells only (each already waited on its own
# workers). A bare `wait` would also wait on the policy servers, which run
# with --idle_timeout -1 and never exit, hanging the script after everything
# else has finished.
workers_rc=0
for pid in "${gpu_launcher_pids[@]}"; do
    wait "${pid}" || workers_rc=1
done
if [ "${workers_rc}" -ne 0 ]; then
    echo "[ERROR] One or more simulator shards failed; refusing to aggregate partial results."
    exit 1
fi
echo "All workers finished."
if [ "${skip_aggregate}" != "true" ]; then
    echo "Aggregating..."
    "${LIBERO_PYTHON}" "${AGGREGATE_SCRIPT}" --root_path "${output_dir}" --task_suite_name "${task_suite_name}"
fi

done   # for task_suite_name in "${TASK_SUITES[@]}"

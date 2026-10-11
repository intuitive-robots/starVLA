#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Keep simulator startup independent of git: Booster compute-node PATHs do not
# always include it.  This directory is four levels below the StarVLA root.
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
EVAL_SCRIPT="${REPO_ROOT}/examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py"

# A worker imports NumPy/OpenCV/PyTorch even when inference is served remotely.
# Without explicit caps, their native BLAS/OpenMP pools can create ~130 threads
# per process.  At 16 workers/GPU that becomes >8k threads per node and has
# caused native SIGABRTs on the booster nodes.  Keep simulator workers scalar;
# model-side batching/concurrency lives in the separate policy-server process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Native simulator failures otherwise emit 5-7 GB cores per worker into the
# shared repository.  Logs and shard exit status remain available for diagnosis.
ulimit -c 0
###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME="${LIBERO_HOME:-path_to_LIBERO-plus_code}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}" # let eval_libero find the LIBERO tools
export PYTHONPATH="$(pwd):${PYTHONPATH}" # let LIBERO find the websocket tools from main repo

# See auto_eval_libero_plus.sh: keep numba's JIT cache off shared NFS -- many
# concurrent sim workers importing it at once otherwise races on the shared
# conda env path (observed as "Stale file handle" / Bus error under load).
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/starvla_numba_cache}"

LIBERO_PLUS_CONDA_ENV="${LIBERO_PLUS_CONDA_ENV:-libero-plus}"
LIBERO_PLUS_SIF="${LIBERO_PLUS_SIF:-${REPO_ROOT}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}"
LIBERO_PLUS_RUNTIME="${LIBERO_PLUS_RUNTIME:-auto}"
your_ckpt="${your_ckpt:-path_to_checkpoint}"
output_dir="${output_dir:-path_to_output_dir}"
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=$1
start_idx=$2
end_idx=$3
num_gpus=${4:-1}
tasks_per_gpu=${5:-0}
workers_per_gpu=${6:-1}
# Keep shard geometry separate from live simulator concurrency.  Resume keys
# include the shard id/count, so lowering workers_per_gpu would make completed
# files unusable and could duplicate episodes.  This cap runs the same shards
# in waves instead.
max_concurrent_workers_per_gpu="${max_concurrent_workers_per_gpu:-${workers_per_gpu}}"
num_trials_per_task=${7:-1}
gpu_ids_csv=${8:-}
base_port=${9:-10093}
server_host=${10:-127.0.0.1}
use_server=${11:-true}
# Policy servers per GPU (see auto_eval_libero_plus.sh). Workers on a GPU
# round-robin over that GPU's servers instead of all sharing a single one.
servers_per_gpu=${12:-1}
# Evaluate every stride'th task index instead of every one -- see
# auto_eval_libero_plus.sh's max_tasks_per_suite for why (perturbation
# categories are large contiguous index blocks, so a plain sub-range is a
# biased, not representative, sample). 1 (default) = every task, unchanged.
stride=${13:-1}
# Exactly this many evenly spaced indices from the complete suite. Each raw
# worker range filters the same deterministic global index set, so their union
# has this exact size without overlaps.
exact_sample_count=${14:-0}
# Deal this partition's task list out to the workers round-robin (worker i gets
# tasks i, i+N, i+2N, ...) instead of giving each a contiguous block. LIBERO-plus
# groups perturbation categories into large contiguous index blocks whose episodes
# differ by >5x in cost, so contiguous blocks hand different workers wildly
# different total workloads: measured on libero_10 with 32 workers, shards
# finished spread over 52 minutes (median at +8 min) and the last five ran alone
# on an otherwise idle node for 45 of the suite's 69 minutes. Round-robin gives
# every worker the same category mix. The union of evaluated tasks is identical,
# so results stay comparable to contiguously sharded runs. 0 = old behavior.
interleave_shards="${STARVLA_INTERLEAVE_SHARDS:-1}"
# This node's rank and the total node count, when driven multi-node by
# auto_eval_libero_plus.sh. Under round-robin every node is handed the WHOLE
# suite and takes global shards partition_idx*num_workers + i, so the stripe is
# dealt across nodes as well as within one -- otherwise the node holding an
# expensive perturbation band gates the whole job (1.80x measured on libero_10
# over 4 nodes). Shard filenames carry the global index, so nodes writing into
# the same shared output_dir never collide.
partition_idx=${15:-0}
num_partitions=${16:-1}
# The NVIDIA EGL stack occasionally fails context initialization when several
# simulator processes start together. A shard is safe to repeat because its
# result filenames are deterministic and are overwritten on success.
worker_max_attempts="${worker_max_attempts:-3}"
# Exact-sample evaluations can isolate every selected task in its own simulator
# process.  This avoids a native EGL/robosuite failure during environment
# transitions from discarding all earlier results in a large raw-index shard.
# Four long-lived launcher streams still keep exactly one simulator active on
# each GPU; only the failed task is retried.
isolate_exact_tasks="${STARVLA_ISOLATE_EXACT_TASKS:-0}"
skip_existing_episodes="${STARVLA_SKIP_EXISTING_EPISODES:-0}"

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

if [ "${LIBERO_PLUS_RUNTIME}" = conda ]; then
    eval "$(conda shell.bash hook)"
    conda activate "${LIBERO_PLUS_CONDA_ENV}"
fi

mkdir -p "${output_dir}"
output_dir=$(realpath -m "${output_dir}")
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/starvla_mpl_${SLURM_JOB_ID:-local}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

apptainer_bind_args=()
if [ "${LIBERO_PLUS_RUNTIME}" = apptainer ]; then
    # The model checkpoint stays host-side.  A read-only bind of its run root
    # is only needed so the simulator client can find dataset_statistics.json
    # when gripper_encoding=auto.  Results get a separate writable bind.
    apptainer_bind_args+=(--bind "${REPO_ROOT}:${REPO_ROOT}:ro")
    if [ -e "${your_ckpt}" ]; then
        ckpt_path=$(realpath "${your_ckpt}")
        ckpt_scope=$(dirname "$(dirname "${ckpt_path}")")
        apptainer_bind_args+=(--bind "${ckpt_scope}:${ckpt_scope}:ro")
    fi
    apptainer_bind_args+=(--bind "${output_dir}:${output_dir}:rw")
fi

# NOTE on STARVLA_EVAL_INCLUDE_STATE below: it is forwarded EMPTY by default, not
# as "0". `apptainer exec --cleanenv` drops the host environment, so it has to be
# passed explicitly -- but passing a hard "0" would override the checkpoint's own
# answer. Empty means "ask the checkpoint": eval_libero_model.py then follows the
# server's `expects_state` handshake. A literal "1"/"0" still forces the decision,
# for deliberately evaluating a state-trained model blind.
run_eval_worker() {
    local gpu_id=$1
    shift

    if [ "${LIBERO_PLUS_RUNTIME}" = apptainer ]; then
        # robosuite 1.4 validates MUJOCO_EGL_DEVICE_ID against the physical IDs
        # listed in CUDA_VISIBLE_DEVICES before CUDA remapping.
        CUDA_VISIBLE_DEVICES="${gpu_id}" apptainer exec --cleanenv --nv \
            "${apptainer_bind_args[@]}" \
            --pwd "${REPO_ROOT}" \
            --env "PYTHONPATH=${REPO_ROOT}" \
            --env "LIBERO_HOME=/app/LIBERO-plus" \
            --env "LIBERO_CONFIG_PATH=/opt/libero-config" \
            --env "MUJOCO_GL=${MUJOCO_GL}" \
            --env "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}" \
            --env "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}" \
            --env "CUDA_VISIBLE_DEVICES=${gpu_id}" \
            --env "MUJOCO_EGL_DEVICE_ID=${gpu_id}" \
            --env "NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}" \
            --env "MPLCONFIGDIR=${MPLCONFIGDIR}" \
            --env "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
            --env "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}" \
            --env "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
            --env "STARVLA_FAST_GLASS_BLUR=${STARVLA_FAST_GLASS_BLUR:-0}" \
            --env "STARVLA_PAIRED_ROLLOUT=${STARVLA_PAIRED_ROLLOUT:-0}" \
            --env "STARVLA_EVAL_INCLUDE_STATE=${STARVLA_EVAL_INCLUDE_STATE:-}" \
            --env "STARVLA_EGL_INIT_LOCK=${STARVLA_EGL_INIT_LOCK:-}" \
            "${LIBERO_PLUS_SIF}" \
            /opt/conda/envs/libero/bin/python "${EVAL_SCRIPT}" "$@"
    else
        CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
            python "${EVAL_SCRIPT}" "$@"
    fi
}

echo "Simulator runtime: ${LIBERO_PLUS_RUNTIME}$([ "${LIBERO_PLUS_RUNTIME}" = apptainer ] && printf ' (%s)' "${LIBERO_PLUS_SIF}")"

if [ -n "${gpu_ids_csv}" ]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_ids_csv}"
else
    gpu_ids=()
    for ((gpu_idx=0; gpu_idx<num_gpus; gpu_idx++)); do
        gpu_ids+=("${gpu_idx}")
    done
fi

num_gpu_slots=${#gpu_ids[@]}
if [ "${num_gpu_slots}" -eq 0 ]; then
    echo "[ERROR] No GPUs were provided."
    exit 1
fi
if [ "${max_concurrent_workers_per_gpu}" -lt 1 ] || [ "${max_concurrent_workers_per_gpu}" -gt "${workers_per_gpu}" ]; then
    echo "[ERROR] max_concurrent_workers_per_gpu must be in [1, ${workers_per_gpu}], got ${max_concurrent_workers_per_gpu}."
    exit 1
fi

if [ "${isolate_exact_tasks}" = "1" ] && [ "${exact_sample_count}" -gt 0 ]; then
    case "${task_suite_name}" in
        libero_10) suite_size=2519 ;;
        libero_goal) suite_size=2591 ;;
        libero_object) suite_size=2518 ;;
        libero_spatial) suite_size=2402 ;;
        *)
            echo "[ERROR] Unknown LIBERO-plus task suite for exact-task isolation: ${task_suite_name}."
            exit 1
            ;;
    esac
    if [ "${exact_sample_count}" -gt "${suite_size}" ]; then
        echo "[ERROR] exact_sample_count=${exact_sample_count} exceeds suite size ${suite_size}."
        exit 1
    fi

    exact_task_ids=()
    for ((sample_idx=0; sample_idx<exact_sample_count; sample_idx++)); do
        task_id=$((sample_idx * suite_size / exact_sample_count))
        if [ "${task_id}" -ge "${start_idx}" ] && [ "${task_id}" -lt "${end_idx}" ]; then
            exact_task_ids+=("${task_id}")
        fi
    done
    echo "Exact-task isolation: ${#exact_task_ids[@]} selected tasks, ${num_gpu_slots} one-process-per-GPU streams."

    # A failed task must not discard the tasks queued behind it. Each stream now
    # records the failure and carries on; failed ids land in this manifest so a
    # re-run can target only them. STARVLA_RESUME_EVAL=1 additionally skips tasks
    # whose result file already exists, which makes that re-run cheap.
    failed_manifest="${output_dir}/failed_tasks_${task_suite_name}.txt"
    : > "${failed_manifest}"
    resume_eval="${STARVLA_RESUME_EVAL:-0}"

    isolated_launcher_pids=()
    for ((gpu_slot=0; gpu_slot<num_gpu_slots; gpu_slot++)); do
        gpu_id=${gpu_ids[$gpu_slot]}
        (
            stream_rc=0
            for ((task_offset=gpu_slot; task_offset<${#exact_task_ids[@]}; task_offset+=num_gpu_slots)); do
                task_id=${exact_task_ids[$task_offset]}
                task_end=$((task_id + 1))
                server_slot=$((task_offset % servers_per_gpu))
                worker_port=$((base_port + gpu_slot * servers_per_gpu + server_slot))
                task_result="${output_dir}/logs/${task_suite_name}/${task_id}_to_${task_end}.json"
                if [ "${resume_eval}" = "1" ] && [ -s "${task_result}" ]; then
                    echo "Exact task ${task_id}: already complete, skipping (${task_result})"
                    continue
                fi
                echo "Exact task ${task_offset}/${#exact_task_ids[@]}: gpu=${gpu_id}, task=${task_id}, host=${server_host}, port=${worker_port}"
                task_rc=1
                for ((attempt=1; attempt<=worker_max_attempts; attempt++)); do
                    if run_eval_worker "${gpu_id}" \
                        --pretrained_path "$your_ckpt" \
                        --task_suite_name "$task_suite_name" \
                        --num_trials_per_task "$num_trials_per_task" \
                        --output_dir "$output_dir" \
                        --host "$server_host" \
                        --port "$worker_port" \
                        --use_server "$use_server" \
                        --start_idx "$task_id" \
                        --end_idx "$task_end" \
                        --stride "$stride" \
                        --exact_sample_count "$exact_sample_count" \
                        --skip_existing_episodes "$skip_existing_episodes" \
                        --gripper_encoding "${gripper_encoding:-auto}" \
                        --object_perturb_m "${object_perturb_m:-0.0}" \
                        --object_perturb_roles "${object_perturb_roles:-source,target}" \
                        --object_perturb_seed "${object_perturb_seed:-20260812}" \
                        --save_video "${save_video:-False}" \
                        --overlay_trace "${overlay_trace:-False}"; then
                        task_rc=0
                        break
                    else
                        task_rc=$?
                    fi
                    echo "[WARN] Exact task ${task_id} failed (rc=${task_rc}, attempt ${attempt}/${worker_max_attempts})."
                    [ "${attempt}" -lt "${worker_max_attempts}" ] && sleep $((attempt * 10))
                done
                if [ "${task_rc}" -ne 0 ]; then
                    echo "[ERROR] Exact task ${task_id} exhausted ${worker_max_attempts} attempts; continuing with the rest."
                    echo "${task_id}" >> "${failed_manifest}"
                    stream_rc=1
                fi
            done
            exit "${stream_rc}"
        ) &
        isolated_launcher_pids+=($!)
    done

    isolated_rc=0
    for pid in "${isolated_launcher_pids[@]}"; do
        wait "${pid}" || isolated_rc=1
    done
    if [ "${isolated_rc}" -ne 0 ]; then
        n_failed=$(wc -l < "${failed_manifest}" 2>/dev/null || echo 0)
        echo "[ERROR] ${n_failed} of ${#exact_task_ids[@]} exact tasks failed; NOT aggregating, because a"
        echo "        success rate over a subset would silently understate the real one."
        echo "        Completed tasks are on disk and are not lost. Failed task ids:"
        echo "        ${failed_manifest}"
        echo "        Re-run the same command with STARVLA_RESUME_EVAL=1 to retry only these."
        exit 1
    fi
    exit 0
fi

# One shard with retries. Args: <suite> <stride> <result_tag> <failed_manifest> <start> <end> <shard> <num_shards>
# <part index> <gpu_id> <port>. Shared by the per-suite launch below and the chained launch (chain specs).
run_part() {
    local task_suite_name="$1" stride="$2" result_tag="$3" failed_manifest="$4" current_start="$5" current_end="$6"
    local current_shard="$7" current_num_shards="$8" i="$9" gpu_id="${10}" worker_port="${11}" shard_suffix shard_result attempt rc
    # eval_libero_model.py appends "_shard<i>of<M>" to its own sample_tag when
    # num_shards > 1 (see its sample_tag construction). The shell never did, so
    # the --resume probe looked for 0_to_2519_exact1000.json while disk held
    # 0_to_2519_exact1000_shard43of48.json: the test could never match and
    # --resume silently re-ran every shard, with zero "skipping" lines.
    shard_suffix=""
    if [ "${current_num_shards}" -gt 1 ]; then
        shard_suffix="_shard${current_shard}of${current_num_shards}"
    fi
    shard_result="${output_dir}/logs/${task_suite_name}/${current_start}_to_${current_end}${result_tag}${shard_suffix}.json"
    if [ "${resume_eval}" = "1" ] && [ -s "${shard_result}" ]; then
        echo "Part ${i} [${current_start},${current_end}): already complete, skipping"
        return 0
    fi
    for ((attempt=1; attempt<=worker_max_attempts; attempt++)); do
        if run_eval_worker "${gpu_id}" \
            --pretrained_path "$your_ckpt" \
            --task_suite_name "$task_suite_name" \
            --num_trials_per_task "$num_trials_per_task" \
            --output_dir "$output_dir" \
            --host "$server_host" \
            --port "$worker_port" \
            --use_server "$use_server" \
            --start_idx "$current_start" \
            --end_idx "$current_end" \
            --stride "$stride" \
            --exact_sample_count "$exact_sample_count" \
            --skip_existing_episodes "$skip_existing_episodes" \
            --shard_idx "$current_shard" \
            --num_shards "$current_num_shards" \
            --gripper_encoding "${gripper_encoding:-auto}" \
            --object_perturb_m "${object_perturb_m:-0.0}" \
            --object_perturb_roles "${object_perturb_roles:-source,target}" \
            --object_perturb_seed "${object_perturb_seed:-20260812}" \
            --save_video "${save_video:-False}" \
            --overlay_trace "${overlay_trace:-False}"; then
            return 0
        else
            rc=$?
        fi
        echo "[WARN] Part ${i} [${current_start},${current_end}) shard ${current_shard}/${current_num_shards} failed (rc=${rc}, attempt ${attempt}/${worker_max_attempts})."
        [ "${attempt}" -lt "${worker_max_attempts}" ] && sleep $((attempt * 10))
    done
    echo "[ERROR] Part ${i} [${current_start},${current_end}) shard ${current_shard}/${current_num_shards} exhausted ${worker_max_attempts} attempts."
    echo "${current_start} ${current_end} ${current_shard} ${current_num_shards}" >> "${failed_manifest}"
    return "${rc}"
}

num_workers=$((num_gpu_slots * workers_per_gpu))

# ── Chained suites (STARVLA_CHAIN_SPECS, set by auto_eval_libero_plus.sh when chain_suites=true) ──
# "suite:start:end:stride:interleave;suite:...": every worker slot is launched once (2 s stagger once) and runs
# its share of each suite back to back, instead of relaunching all workers per suite and waiting for the slowest
# shard every time. Per-suite partitioning, shard ids, result names, failed manifests and resume are unchanged.
if [ -n "${STARVLA_CHAIN_SPECS:-}" ]; then
    declare -A c_start c_end c_shard c_nshards
    IFS=';' read -r -a chain_specs <<< "${STARVLA_CHAIN_SPECS}"
    chain_suites=(); declare -A c_stride c_tag c_manifest
    for spec in "${chain_specs[@]}"; do
        IFS=':' read -r c_suite c_s c_e c_str c_il <<< "${spec}"
        chain_suites+=("${c_suite}")
        c_stride[$c_suite]=${c_str}
        if [ "${exact_sample_count}" -gt 0 ]; then c_tag[$c_suite]="_exact${exact_sample_count}"
        elif [ "${c_str}" != "1" ]; then c_tag[$c_suite]="_stride${c_str}"
        else c_tag[$c_suite]=""; fi
        c_manifest[$c_suite]="${output_dir}/failed_shards_${c_suite}.txt"
        : > "${c_manifest[$c_suite]}"
        if [ "${c_il}" = "1" ]; then
            for ((w=0; w<num_workers; w++)); do
                c_start["$c_suite,$w"]=${c_s}; c_end["$c_suite,$w"]=${c_e}
                c_shard["$c_suite,$w"]=$((partition_idx * num_workers + w)); c_nshards["$c_suite,$w"]=$((num_partitions * num_workers))
            done
        else
            c_total=$((c_e - c_s)); c_chunk=$((c_total / num_workers)); c_rem=$((c_total % num_workers)); c_cur=${c_s}
            for ((w=0; w<num_workers; w++)); do
                c_next=$((c_cur + c_chunk + (w < c_rem ? 1 : 0)))
                [ "${c_next}" -gt "${c_e}" ] && c_next=${c_e}
                c_start["$c_suite,$w"]=${c_cur}; c_end["$c_suite,$w"]=${c_next}
                c_shard["$c_suite,$w"]=0; c_nshards["$c_suite,$w"]=1
                c_cur=${c_next}
            done
        fi
        echo "Chained suite ${c_suite}: [${c_s}, ${c_e}) stride=${c_str} interleave=${c_il} over ${num_workers} workers"
    done
    resume_eval="${STARVLA_RESUME_EVAL:-0}"
    gpu_launcher_pids=()
    for ((gpu_slot=0; gpu_slot<num_gpu_slots; gpu_slot++)); do
        gpu_id=${gpu_ids[$gpu_slot]}
        (
            pids=()
            launcher_rc=0
            for ((local_worker=0; local_worker<workers_per_gpu; local_worker++)); do
                i=$((gpu_slot * workers_per_gpu + local_worker))
                worker_port=$((base_port + gpu_slot * servers_per_gpu + local_worker % servers_per_gpu))
                sleep 2   # EGL start-up stagger, once per worker slot instead of once per suite
                (
                    slot_rc=0
                    for c_suite in "${chain_suites[@]}"; do
                        [ "${c_start[$c_suite,$i]}" -ge "${c_end[$c_suite,$i]}" ] && continue
                        echo "Part ${i}: gpu=${gpu_id} ${c_suite} [${c_start[$c_suite,$i]}, ${c_end[$c_suite,$i]}) shard=${c_shard[$c_suite,$i]}/${c_nshards[$c_suite,$i]} port=${worker_port}"
                        run_part "${c_suite}" "${c_stride[$c_suite]}" "${c_tag[$c_suite]}" "${c_manifest[$c_suite]}" \
                            "${c_start[$c_suite,$i]}" "${c_end[$c_suite,$i]}" "${c_shard[$c_suite,$i]}" "${c_nshards[$c_suite,$i]}" \
                            "${i}" "${gpu_id}" "${worker_port}" || slot_rc=1
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
        echo "[ERROR] Shard(s) failed in chained suites (${chain_suites[*]}). NOT aggregating."
        echo "        Failed ranges per suite: ${output_dir}/failed_shards_<suite>.txt"
        echo "        Re-run the SAME command with STARVLA_RESUME_EVAL=1 to retry only those."
        exit 1
    fi
    echo "All chained suites finished: ${chain_suites[*]}"
    exit 0
fi

total=$((end_idx - start_idx))
chunk_size=$((total / num_workers))
remainder=$((total % num_workers))
current_start=$start_idx

if [ "${tasks_per_gpu}" -gt 0 ] && [ "${interleave_shards}" != "1" ]; then
    expected_total=$((tasks_per_gpu * num_gpu_slots))
    if [ "${expected_total}" -ne "${total}" ]; then
        echo "[WARN] Range size ${total} does not match visible_gpus * tasks_per_gpu = ${expected_total}. Using the explicit start/end range."
    fi
fi

# ── Precompute every worker's task range up front (cheap, pure arithmetic) ──
# so the actual (costly, EGL-staggered) launch loop below can run one
# independent stream per GPU in parallel instead of one big serial queue.
worker_start=(); worker_end=(); worker_shard=(); worker_num_shards=()
if [ "${interleave_shards}" = "1" ]; then
    # Every worker sees the whole range and filters it to its own round-robin
    # slice (eval_libero_model.py --shard_idx/--num_shards). Every node uses the
    # same num_workers, so global shard ids partition the suite exactly once.
    total_shards=$((num_partitions * num_workers))
    shard_base=$((partition_idx * num_workers))
    for ((i=0; i<num_workers; i++)); do
        worker_start+=("$start_idx")
        worker_end+=("$end_idx")
        worker_shard+=("$((shard_base + i))")
        worker_num_shards+=("${total_shards}")
    done
    echo "Shard layout: round-robin, global shards ${shard_base}..$((shard_base + num_workers - 1)) of ${total_shards} (node ${partition_idx}/${num_partitions}) over [${start_idx}, ${end_idx})"
fi
for ((i=0; i<num_workers; i++)); do
    if [ "${interleave_shards}" = "1" ]; then break; fi
    if [ $i -lt $remainder ]; then
        current_end=$((current_start + chunk_size + 1))
    else
        current_end=$((current_start + chunk_size))
    fi
    if [ $current_end -gt $end_idx ]; then
        current_end=$end_idx
    fi
    if [ $current_start -ge $current_end ]; then
        break
    fi
    worker_start+=("$current_start")
    worker_end+=("$current_end")
    worker_shard+=("0")
    worker_num_shards+=("1")
    current_start=$current_end
    if [ $current_start -ge $end_idx ]; then
        break
    fi
done
num_active_workers=${#worker_start[@]}

# ── Launch workers: one background stream per GPU, staggered 2s only WITHIN
# that stream ──────────────────────────────────────────────────────────────
# The EGL race (libnvidia-eglcore.so global state on eglGetPlatformDisplayEXT,
# driver 590.x) is per-device, so different GPUs' worker streams don't need to
# be staggered against each other. Previously every worker on the node shared
# one global stagger of `sleep $((i*2))` executed SEQUENTIALLY in the main
# loop before each worker launched -- since that sleep duration itself grew
# with i and blocked the loop, cumulative wait time before launching worker i
# was i*(i+1) seconds (quadratic, not linear): worker 99 wouldn't start until
# ~9900s (~2.75h) in. Now it's num_gpu_slots parallel streams of
# workers_per_gpu each, flat 2s within each stream.
# A shard that exhausts its retries should cost only that shard. Failed ranges
# are recorded here so a re-run can target them, and STARVLA_RESUME_EVAL=1 skips
# shards whose result JSON already exists. Neither changes which episodes are
# rolled out -- the partitioning above is untouched, so results stay directly
# comparable with earlier runs.
failed_manifest="${output_dir}/failed_shards_${task_suite_name}.txt"
: > "${failed_manifest}"
resume_eval="${STARVLA_RESUME_EVAL:-0}"
if [ "${exact_sample_count}" -gt 0 ]; then
    result_tag="_exact${exact_sample_count}"
elif [ "${stride}" != "1" ]; then
    result_tag="_stride${stride}"
else
    result_tag=""
fi

gpu_launcher_pids=()
for ((gpu_slot=0; gpu_slot<num_gpu_slots; gpu_slot++)); do
    gpu_id=${gpu_ids[$gpu_slot]}
    (
        pids=()
        launcher_rc=0
        for ((local_worker=0; local_worker<workers_per_gpu; local_worker++)); do
            i=$((gpu_slot * workers_per_gpu + local_worker))
            [ "$i" -ge "$num_active_workers" ] && break

            current_start=${worker_start[$i]}
            current_end=${worker_end[$i]}
            current_shard=${worker_shard[$i]}
            current_num_shards=${worker_num_shards[$i]}
            server_slot=$((local_worker % servers_per_gpu))
            worker_port=$((base_port + gpu_slot * servers_per_gpu + server_slot))
            echo "Part ${i}: gpu=${gpu_id}, worker=${local_worker}, start=$current_start, end=$current_end ([$current_start, $current_end)), shard=${current_shard}/${current_num_shards}, host=${server_host}, port=${worker_port}, use_server=${use_server}"

            sleep 2

            ( run_part "${task_suite_name}" "${stride}" "${result_tag}" "${failed_manifest}" "${current_start}" "${current_end}" \
                "${current_shard}" "${current_num_shards}" "${i}" "${gpu_id}" "${worker_port}" ) &
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

workers_rc=0
for pid in "${gpu_launcher_pids[@]}"; do
    wait "${pid}" || workers_rc=1
done
if [ "${workers_rc}" -ne 0 ]; then
    n_failed=$(wc -l < "${failed_manifest}" 2>/dev/null || echo 0)
    echo "[ERROR] ${n_failed} shard(s) failed. NOT aggregating: a success rate over a subset"
    echo "        would silently understate the real one."
    echo "        Every other shard completed and is on disk -- that work is not lost."
    echo "        Failed ranges (start end shard num_shards): ${failed_manifest}"
    echo "        Re-run the SAME command with STARVLA_RESUME_EVAL=1 to retry only those;"
    echo "        completed shards are skipped, so the episode partitioning is unchanged."
    exit 1
fi

# # =============== Aggregate results ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py

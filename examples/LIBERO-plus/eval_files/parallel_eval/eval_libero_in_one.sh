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
# The NVIDIA EGL stack occasionally fails context initialization when several
# simulator processes start together. A shard is safe to repeat because its
# result filenames are deterministic and are overwritten on success.
worker_max_attempts="${worker_max_attempts:-3}"

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

num_workers=$((num_gpu_slots * workers_per_gpu))

total=$((end_idx - start_idx))
chunk_size=$((total / num_workers))
remainder=$((total % num_workers))
current_start=$start_idx

if [ "${tasks_per_gpu}" -gt 0 ]; then
    expected_total=$((tasks_per_gpu * num_gpu_slots))
    if [ "${expected_total}" -ne "${total}" ]; then
        echo "[WARN] Range size ${total} does not match visible_gpus * tasks_per_gpu = ${expected_total}. Using the explicit start/end range."
    fi
fi

# ── Precompute every worker's task range up front (cheap, pure arithmetic) ──
# so the actual (costly, EGL-staggered) launch loop below can run one
# independent stream per GPU in parallel instead of one big serial queue.
worker_start=(); worker_end=()
for ((i=0; i<num_workers; i++)); do
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
gpu_launcher_pids=()
for ((gpu_slot=0; gpu_slot<num_gpu_slots; gpu_slot++)); do
    gpu_id=${gpu_ids[$gpu_slot]}
    (
        pids=()
        for ((local_worker=0; local_worker<workers_per_gpu; local_worker++)); do
            i=$((gpu_slot * workers_per_gpu + local_worker))
            [ "$i" -ge "$num_active_workers" ] && break

            current_start=${worker_start[$i]}
            current_end=${worker_end[$i]}
            server_slot=$((local_worker % servers_per_gpu))
            worker_port=$((base_port + gpu_slot * servers_per_gpu + server_slot))
            echo "Part ${i}: gpu=${gpu_id}, worker=${local_worker}, start=$current_start, end=$current_end ([$current_start, $current_end)), host=${server_host}, port=${worker_port}, use_server=${use_server}"

            sleep 2

            (
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
                        --gripper_encoding "${gripper_encoding:-auto}" \
                        --object_perturb_m "${object_perturb_m:-0.0}" \
                        --object_perturb_roles "${object_perturb_roles:-source,target}" \
                        --object_perturb_seed "${object_perturb_seed:-20260812}" \
                        --save_video "${save_video:-False}" \
                        --overlay_trace "${overlay_trace:-False}"; then
                        exit 0
                    else
                        rc=$?
                    fi
                    echo "[WARN] Part ${i} [${current_start},${current_end}) failed (rc=${rc}, attempt ${attempt}/${worker_max_attempts})."
                    [ "${attempt}" -lt "${worker_max_attempts}" ] && sleep $((attempt * 10))
                done
                echo "[ERROR] Part ${i} [${current_start},${current_end}) exhausted ${worker_max_attempts} attempts."
                exit "${rc}"
            ) &
            pids+=($!)
        done
        launcher_rc=0
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

# # =============== Aggregate results ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py

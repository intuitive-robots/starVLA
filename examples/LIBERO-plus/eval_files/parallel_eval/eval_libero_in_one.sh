#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)"
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

conda activate "${LIBERO_PLUS_CONDA_ENV}"

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

    gpu_slot=$((i / workers_per_gpu))
    local_worker=$((i % workers_per_gpu))
    server_slot=$((local_worker % servers_per_gpu))
    gpu_id=${gpu_ids[$gpu_slot]}
    worker_port=$((base_port + gpu_slot * servers_per_gpu + server_slot))
    echo "Part $((i)): gpu=${gpu_id}, worker=${local_worker}, start=$current_start, end=$current_end ([$current_start, $current_end)), host=${server_host}, port=${worker_port}, use_server=${use_server}"
    # Stagger EGL init: NVIDIA libnvidia-eglcore.so has a global-state race when
    # multiple processes call eglGetPlatformDisplayEXT simultaneously (driver 590.x).
    # A 2s gap per worker serializes the init window without meaningful throughput loss.
    sleep $((i * 2))
    CUDA_VISIBLE_DEVICES=$gpu_id MUJOCO_EGL_DEVICE_ID=$gpu_id python ./examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
        --pretrained_path "$your_ckpt" \
        --task_suite_name "$task_suite_name" \
        --num_trials_per_task "$num_trials_per_task" \
        --output_dir "$output_dir" \
        --host "$server_host" \
        --port "$worker_port" \
        --use_server "$use_server" \
        --start_idx "$current_start" \
        --end_idx "$current_end" &

    current_start=$current_end

    if [ $current_start -ge $end_idx ]; then
        break
    fi
done


wait

# # =============== Aggregate results ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py

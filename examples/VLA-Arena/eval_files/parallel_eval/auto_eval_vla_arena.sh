#!/bin/bash
# auto_eval_vla_arena.sh
#
# Single-node parallel VLA-Arena evaluation: launches policy servers on each
# local GPU, then runs sim workers against them. Mirrors
# examples/LIBERO-plus/eval_files/parallel_eval/auto_eval_libero_plus.sh.
#
# Work unit = (suite, level). Unlike LIBERO-plus -- where a suite is thousands
# of perturbed task instances and is split by task index -- eval_vla_arena.py
# has no task-range flag and always runs a whole suite, so (suite x level) is
# the finest partition available without modifying it. With 12 suites x N
# levels there are plenty of units to spread across nodes and GPUs.
#
# Units are dealt round-robin across the global worker list so that every node
# and GPU gets an interleaved mix of suites rather than one node landing on all
# the long-horizon ones.
#
# Invoked per node by eval_vla_arena_slurm.sh; runnable standalone too.
#
# Env vars (all optional unless noted):
#   your_ckpt              REQUIRED path to pytorch_model.pt
#   output_dir             results root (default <ckpt>/../../results/vla-arena)
#   suites                 space/comma list, or "all"      (default all)
#   levels                 space/comma list                (default "0 1 2")
#   num_gpus               GPUs on this node               (default nvidia-smi -L)
#   gpu_ids_csv            explicit GPU ids, e.g. "0,1,2,3"
#   workers_per_gpu        concurrent sim workers per GPU  (default 2)
#   servers_per_gpu        policy servers per GPU          (default 1)
#   num_trials             rollouts per task               (default 10)
#   max_batch_size         server-side request batching    (default 32)
#   max_wait_time          batch fill timeout, seconds     (default 1.0)
#   partition_idx          this node's index               (default 0)
#   num_partitions         total nodes                     (default 1)
#   base_port              first server port               (default 10093)
#   save_video_mode        all|first_success_failure|none  (default first_success_failure)
#   overlay_trace          "true" to draw the model's generated trace on video
#   SKIP_AGGREGATE         "true" to leave merging to the caller
#   VLA_ARENA_HOME         REQUIRED root of the VLA-Arena repo
#   VLA_ARENA_python       REQUIRED python with vla_arena installed
#   POLICY_SERVER_CONDA_ENV conda env for the policy server (default starVLA)

set -eo pipefail

your_ckpt="${your_ckpt:-}"
if [ -z "${your_ckpt}" ]; then echo "[ERROR] set your_ckpt"; exit 1; fi
if [ ! -f "${your_ckpt}" ]; then echo "[ERROR] checkpoint not found: ${your_ckpt}"; exit 1; fi

: "${VLA_ARENA_HOME:?set VLA_ARENA_HOME to the VLA-Arena repo root}"
: "${VLA_ARENA_python:?set VLA_ARENA_python to a python with vla_arena installed}"

starVLA_HOME="$(pwd)"
export starVLA_HOME
export PYTHONPATH="${VLA_ARENA_HOME}/vla_arena:${starVLA_HOME}:${PYTHONPATH}"
# Headless offscreen rendering. Without this MuJoCo silently fails to create a
# GL context on compute nodes -- the shipped eval_vla_arena.sh never sets it.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

POLICY_SERVER_CONDA_ENV="${POLICY_SERVER_CONDA_ENV:-starVLA}"
workers_per_gpu="${workers_per_gpu:-2}"
servers_per_gpu="${servers_per_gpu:-1}"
num_trials="${num_trials:-10}"
max_batch_size="${max_batch_size:-32}"
max_wait_time="${max_wait_time:-1.0}"
partition_idx="${partition_idx:-0}"
num_partitions="${num_partitions:-1}"
base_port="${base_port:-10093}"
save_video_mode="${save_video_mode:-first_success_failure}"
overlay_trace="${overlay_trace:-false}"
server_idle_timeout="${server_idle_timeout:--1}"

ALL_SUITES=(
    safety_static_obstacles safety_cautious_grasp safety_hazard_avoidance
    safety_state_preservation safety_dynamic_obstacles
    distractor_static_distractors distractor_dynamic_distractors
    extrapolation_preposition_combinations extrapolation_task_workflows
    extrapolation_unseen_objects long_horizon
)
suites="${suites:-all}"
if [ "${suites}" = "all" ]; then SUITES=("${ALL_SUITES[@]}"); else IFS=', ' read -r -a SUITES <<< "${suites}"; fi
IFS=', ' read -r -a LEVELS <<< "${levels:-0 1 2}"

if [ -z "${gpu_ids_csv}" ]; then
    num_gpus="${num_gpus:-$(nvidia-smi -L | wc -l)}"
    gpu_ids=(); for ((g=0; g<num_gpus; g++)); do gpu_ids+=("$g"); done
else
    IFS=',' read -r -a gpu_ids <<< "${gpu_ids_csv}"
    num_gpus="${#gpu_ids[@]}"
fi

if [ -z "${output_dir}" ]; then
    output_dir="$(dirname "$(dirname "${your_ckpt}")")/results/vla-arena"
fi
shard_dir="${output_dir}/shards"
log_dir="${output_dir}/logs"
server_log_dir="${output_dir}/server_logs"
mkdir -p "${shard_dir}" "${log_dir}" "${server_log_dir}"

# ── Build the global work-unit list, then take this node's slice ────────────
#
# A unit is (suite, level, task-range). tasks_per_unit controls granularity:
#   0  → one unit per (suite, level)  — 11 x 3 = 33 units, coarse
#   1  → one unit per task            — 170 units, maximum parallelism
# Suites have 5 tasks per level, except long_horizon L0 which has 10. We deal
# ranges up to the 10-task maximum and let eval_vla_arena.py clamp: a range
# past the end of a 5-task suite is skipped there, costing nothing.
tasks_per_unit="${tasks_per_unit:-1}"
MAX_TASKS_PER_SUITE_LEVEL=10

UNITS=()
for s in "${SUITES[@]}"; do
    for l in "${LEVELS[@]}"; do
        if [ "${tasks_per_unit}" -le 0 ]; then
            UNITS+=("${s}:${l}:0:-1")   # 0:-1 == "whole suite" to eval_vla_arena.py
        else
            for ((t=0; t<MAX_TASKS_PER_SUITE_LEVEL; t+=tasks_per_unit)); do
                # Suites with 5 tasks simply skip the ranges at/after 5.
                if [ "${s}" != "long_horizon" ] || [ "${l}" != "0" ]; then
                    if [ "${t}" -ge 5 ]; then continue; fi
                fi
                UNITS+=("${s}:${l}:${t}:$(( t + tasks_per_unit ))")
            done
        fi
    done
done
MY_UNITS=()
for ((i=0; i<${#UNITS[@]}; i++)); do
    if (( i % num_partitions == partition_idx )); then MY_UNITS+=("${UNITS[$i]}"); fi
done

echo "=========================================="
echo " VLA-Arena Auto Eval"
echo "=========================================="
echo " Checkpoint      : ${your_ckpt}"
echo " Output dir      : ${output_dir}"
echo " Suites          : ${#SUITES[@]}   Levels: ${LEVELS[*]}"
echo " Total units     : ${#UNITS[@]}  (suite x level x task-range, tasks_per_unit=${tasks_per_unit})"
echo " This partition  : ${partition_idx}/${num_partitions} -> ${#MY_UNITS[@]} units"
echo " GPUs            : ${gpu_ids[*]}"
echo " Workers / GPU   : ${workers_per_gpu}   Servers / GPU: ${servers_per_gpu}"
echo " Trials / task   : ${num_trials}"
echo " Max batch       : ${max_batch_size} (max_wait_time=${max_wait_time}s)"
echo " MUJOCO_GL       : ${MUJOCO_GL}"
echo "=========================================="
if [ "${#MY_UNITS[@]}" -eq 0 ]; then echo "No units for this partition; exiting."; exit 0; fi

# ── Launch policy servers ───────────────────────────────────────────────────
server_pids=(); ports=()
for ((g=0; g<num_gpus; g++)); do
    gpu_id="${gpu_ids[$g]}"
    for ((s=0; s<servers_per_gpu; s++)); do
        port=$((base_port + partition_idx * 1000 + g * servers_per_gpu + s))
        ports+=("${port}")
        lf="${server_log_dir}/p${partition_idx}_gpu${gpu_id}_port${port}.log"
        echo "Launching policy server gpu=${gpu_id} port=${port}"
        setsid bash -lc "CUDA_VISIBLE_DEVICES=${gpu_id} conda run --no-capture-output -n \"${POLICY_SERVER_CONDA_ENV}\" \
            python deployment/model_server/server_policy.py --ckpt_path \"${your_ckpt}\" --port \"${port}\" \
            --use_bf16 --idle_timeout \"${server_idle_timeout}\" \
            --max_batch_size \"${max_batch_size}\" --max_wait_time \"${max_wait_time}\"" \
            > "${lf}" 2>&1 &
        server_pids+=($!)
    done
done

cleanup() { echo "Stopping policy servers: ${server_pids[*]}"; kill ${server_pids[@]} 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for servers to come up..."
for port in "${ports[@]}"; do
    for _ in $(seq 1 180); do
        if bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then echo "  port ${port} ready"; break; fi
        sleep 5
    done
done

# ── Run sim workers: deal units round-robin over (gpu x worker slot) ────────
n_slots=$(( num_gpus * workers_per_gpu ))
pids=()
for ((u=0; u<${#MY_UNITS[@]}; u++)); do
    IFS=':' read -r suite level t_start t_end <<< "${MY_UNITS[$u]}"
    slot=$(( u % n_slots ))
    gpu_id="${gpu_ids[$(( slot / workers_per_gpu ))]}"
    port="${ports[$(( slot % ${#ports[@]} ))]}"
    tag="p${partition_idx}_${suite}_L${level}"
    # Task-sliced units need distinct tags, else several workers share one shard
    # path and one video dir and overwrite each other.
    if [ "${t_end}" -ge 0 ]; then tag="${tag}_T${t_start}-${t_end}"; fi
    # Worker output goes to BOTH the per-unit file and this script's own
    # stdout/stderr, so `sbatch` .out/.err show live progress the way
    # LIBERO-plus's eval_libero_in_one.sh does (it never redirects at all).
    # stdout and stderr are teed separately -- tqdm writes progress bars to
    # stderr, so merging them with 2>&1 would bury the bars in .out and leave
    # .err empty, which is the opposite of what we want. `tee -a` plus an
    # explicit truncate avoids the two process substitutions clobbering each
    # other's writes to the shared file.
    : > "${log_dir}/${tag}.log"
    (
        # MUJOCO_EGL_DEVICE_ID must be set alongside CUDA_VISIBLE_DEVICES: EGL does
        # NOT honor CUDA_VISIBLE_DEVICES, so without it every worker renders on EGL
        # device 0. The resulting contention makes some contexts silently return
        # all-black frames -- which reach the policy, not just the saved video.
        # Same pairing as auto_eval_libero.sh / eval_libero_in_one.sh.
        CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" "${VLA_ARENA_python}" \
            examples/VLA-Arena/eval_files/eval_vla_arena.py \
            --args.port "${port}" --args.host 127.0.0.1 \
            --args.task_suite_name "${suite}" --args.task_level "${level}" \
            --args.task_start "${t_start}" --args.task_end "${t_end}" \
            --args.num_trials_per_task "${num_trials}" \
            --args.save_video_mode "${save_video_mode}" \
            $( [ "${overlay_trace}" = "true" ] && echo "--args.overlay_trace" ) \
            --args.video_out_path "${output_dir}/videos/${tag}" \
            --args.results_json "${shard_dir}/${tag}.json" \
            --args.job_name "${tag}" \
            > >(stdbuf -oL sed "s/^/[${tag}] /" | tee -a "${log_dir}/${tag}.log") \
            2> >(stdbuf -oL sed "s/^/[${tag}] /" | tee -a "${log_dir}/${tag}.log" >&2)
    ) &
    pids+=($!)
    # Stagger the initial fill: creating n_slots EGL contexts in the same instant
    # is what produces the burst of blank-framebuffer contexts right after start.
    # Only the first wave needs spacing -- after that the throttle below paces us.
    if [ "${u}" -lt "${n_slots}" ]; then sleep "${worker_launch_delay:-3}"; fi
    # Throttle to n_slots concurrent workers.
    while [ "$(jobs -rp | wc -l)" -ge "${n_slots}" ]; do sleep 5; done
done
wait_fail=0
for p in "${pids[@]}"; do wait "$p" || wait_fail=1; done
[ "${wait_fail}" -ne 0 ] && echo "[WARN] at least one worker exited non-zero (see ${log_dir})"

echo "Partition ${partition_idx} finished (${#MY_UNITS[@]} units)."
if [ "${SKIP_AGGREGATE}" != "true" ]; then
    python examples/VLA-Arena/eval_files/parallel_eval/aggregate_vla_arena_results.py --root_path "${output_dir}"
fi

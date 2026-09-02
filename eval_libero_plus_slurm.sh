#!/bin/bash
#SBATCH --job-name=starvla_eval_libero_plus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --output=slurm_logs/eval_libero_plus_%j.out
#SBATCH --error=slurm_logs/eval_libero_plus_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
# Uncomment and set partition/account as needed:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

# Parallel, optionally MULTI-NODE LIBERO-plus evaluation. Each node runs its
# own self-contained copy of the single-node pipeline (policy servers + sim
# workers, all node-local -- no cross-node networking involved) against a
# distinct 1/num_nodes slice of the suite; results land in the same shared
# output_dir (every shard file is named by its own [start,end) range, so
# concurrent nodes never collide), then this script aggregates once at the
# end. Wraps examples/LIBERO-plus/eval_files/parallel_eval/auto_eval_libero_plus.sh.
#
# LIBERO-plus suites have thousands of perturbed task instances (not just
# 10-90 tasks like vanilla LIBERO); tasks_per_gpu is auto-sized (see
# auto_eval_libero_plus.sh) so the suite is always split evenly across
# num_nodes * num_gpus_per_node, whatever those are.
#
# Usage:
#   sbatch eval_libero_plus_slurm.sh --ckpt /path/to/pytorch_model.pt
#       -> single node (default #SBATCH --nodes=1), evaluates ALL suites
#
#   sbatch --nodes=4 eval_libero_plus_slurm.sh --ckpt /path/to/pytorch_model.pt --suite libero_10
#       -> 4 nodes, each independently evaluating its own 1/4 slice of libero_10
#
#   sbatch --nodes=2 --gres=gpu:4 eval_libero_plus_slurm.sh \
#       --ckpt /path/to/pytorch_model.pt --suite libero_10 --servers_per_gpu 3 --workers_per_gpu 6
#       -> 2 nodes x 4 GPUs, 3 policy servers per GPU (12/node), 6 sim workers per GPU
#
# Node count is set at SUBMISSION time via `sbatch --nodes=N` (SBATCH
# directives are parsed before this script runs, so it can't be a runtime
# flag) -- same reasoning as GPU count via `sbatch --gres=gpu:N`.
#
# Flags:
#   --ckpt <path>             path to pytorch_model.pt (required; or set $your_ckpt env var)
#   --suite <name[,name...]>  one or more comma-separated suites, or all (default: all)
#   --num_gpus <n>            GPUs PER NODE, default: GPUs visible to the job (nvidia-smi -L)
#   --workers_per_gpu <n>     sim workers per GPU, default: 1 (or $workers_per_gpu env var)
#   --servers_per_gpu <n>     policy servers per GPU, default: 1 (or $servers_per_gpu env var).
#                             Each server is a full model copy on that GPU but gives real
#                             inference concurrency to workers sharing it -- best throughput
#                             when workers_per_gpu is a multiple of servers_per_gpu.
#   --num_trials <n>          trials per task, default: 1 (or $num_trials env var)
#   --max_tasks_per_suite <n> evaluate ~n task instances per suite instead of the full
#                             ~2400-2600. NOT a prefix -- each suite's task indices are
#                             grouped into large contiguous blocks by perturbation category
#                             (e.g. libero_10 index 0-499 is entirely "table" perturbations,
#                             500-999 is "view", ...), so this computes a stride across the
#                             FULL index range instead, sampling proportionally from every
#                             category. Deterministic (same n -> same indices), not random.
#                             The established one-node 8-worker/GPU comparison protocol with
#                             n=50 evaluates 64 deterministic instances per suite because each
#                             of the 32 worker shards begins its own stride. Keep this behavior
#                             for direct comparability with the completed G/D/D-rand results.
#                             0 (default) = full suite. E.g. 500 across the 4 default
#                             suites = ~2000 total instead of ~10000.
#   --exact_tasks_per_suite <n> evaluate exactly n evenly spaced task indices per suite.
#                             This is mutually exclusive with --max_tasks_per_suite and is
#                             preferred for larger matched comparisons with exact denominators.
#   --gpu_ids <csv>           explicit GPU id list (per node), e.g. "0,1,2,3"
#   --max_batch_size <n>      batch concurrent requests into one predict_action() call,
#                             default: 32 (or $max_batch_size env var). 1 = old unbatched
#                             behavior. See auto_eval_libero_plus.sh for measured speedups.
#   --max_wait_time <s>       ceiling on how long the dispatcher waits to fill a batch before
#                             running it under-full, default: 1.0 (or $max_wait_time env var).
#   --sim-runtime <mode>      apptainer|auto|conda (default: apptainer). auto uses the SIF when
#                             Apptainer and the image are available, otherwise the old conda path.
#   --sif <path>              LIBERO-Plus simulator SIF. The StarVLA model server always remains
#                             in POLICY_SERVER_CONDA_ENV on the host.
#   --save_video              save every rollout MP4 (off by default; JSON is sufficient for rates)
#
# Env vars forwarded as-is to auto_eval_libero_plus.sh: output_dir, server_host, base_port,
# server_idle_timeout, LIBERO_HOME, LIBERO_PLUS_CONDA_ENV, POLICY_SERVER_CONDA_ENV,
# LIBERO_PLUS_RUNTIME, LIBERO_PLUS_SIF.

set -euo pipefail

AUTO_EVAL_SCRIPT="./examples/LIBERO-plus/eval_files/parallel_eval/auto_eval_libero_plus.sh"
AGGREGATE_SCRIPT="./examples/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py"

DEFAULT_SUITES=(libero_10 libero_goal libero_object libero_spatial)

# ── Parse flags ────────────────────────────────────────────────────────────
CKPT="${your_ckpt:-}"
SUITE="${suite:-all}"
NUM_TRIALS="${num_trials:-1}"
WORKERS_PER_GPU="${workers_per_gpu:-1}"
SERVERS_PER_GPU="${servers_per_gpu:-1}"
GPU_IDS_CSV="${gpu_ids_csv:-}"
NUM_GPUS="${num_gpus:-}"
MAX_TASKS_PER_SUITE="${max_tasks_per_suite:-0}"
EXACT_TASKS_PER_SUITE="${exact_tasks_per_suite:-0}"
MAX_BATCH_SIZE="${max_batch_size:-32}"
MAX_WAIT_TIME="${max_wait_time:-1.0}"
SAVE_VIDEO="${save_video:-False}"
OBJECT_PERTURB_M="${object_perturb_m:-0.0}"
OBJECT_PERTURB_ROLES="${object_perturb_roles:-source,target}"
OBJECT_PERTURB_SEED="${object_perturb_seed:-20260812}"
SIM_RUNTIME="${LIBERO_PLUS_RUNTIME:-apptainer}"
SIM_SIF="${LIBERO_PLUS_SIF:-$(pwd)/playground/sims/sif/libero-plus-v0.5.0-arm64.sif}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --suite) SUITE="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --workers_per_gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --servers_per_gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --max_tasks_per_suite) MAX_TASKS_PER_SUITE="$2"; shift 2 ;;
        --exact_tasks_per_suite) EXACT_TASKS_PER_SUITE="$2"; shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_wait_time) MAX_WAIT_TIME="$2"; shift 2 ;;
        --sim-runtime) SIM_RUNTIME="$2"; shift 2 ;;
        --sif) SIM_SIF="$2"; shift 2 ;;
        --save_video) SAVE_VIDEO="True"; shift 1 ;;
        --overlay_trace) OVERLAY_TRACE="True"; shift 1 ;;
        --gpu_ids) GPU_IDS_CSV="$2"; shift 2 ;;
        --object_perturb_m) OBJECT_PERTURB_M="$2"; shift 2 ;;
        --object_perturb_roles) OBJECT_PERTURB_ROLES="$2"; shift 2 ;;
        --object_perturb_seed) OBJECT_PERTURB_SEED="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "${MAX_TASKS_PER_SUITE}" -gt 0 ] && [ "${EXACT_TASKS_PER_SUITE}" -gt 0 ]; then
    echo "[ERROR] --max_tasks_per_suite and --exact_tasks_per_suite are mutually exclusive."
    exit 1
fi

SIM_SIF=$(realpath -m "${SIM_SIF}")

if [ -z "${CKPT}" ]; then
    echo "[ERROR] Please pass --ckpt /path/to/pytorch_model.pt (or set \$your_ckpt)."
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
fi
your_ckpt="${CKPT}"

# Resolve once, shared by every node and the final aggregation (matches
# auto_eval_libero_plus.sh's own default so all nodes agree without
# coordinating).
if [ -z "${output_dir:-}" ]; then
    ckpt_parent_dir=$(dirname "$(dirname "${your_ckpt}")")
    output_dir="${ckpt_parent_dir}/results/libero-plus"
fi
mkdir -p "${output_dir}"

# ── Environment modules ───────────────────────────────────────────────────────
ml load CUDA

# ── Conda ─────────────────────────────────────────────────────────────────────
source ~/blank4/envs/miniforge3/etc/profile.d/conda.sh
set -u

# ── Library paths ─────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# ── HuggingFace / cache dirs ──────────────────────────────────────────────────
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
# USE_TF=0: transformers otherwise probes for TensorFlow; that scan over site-packages
# can hit a stale NFS handle and kill the process with SIGBUS (signal 7) before any
# traceback. Same fix as train_libero_slurm.sh -- job 1379419 died exactly that way.
export USE_TF=0 USE_TORCH=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_NODES="${SLURM_NNODES:-1}"

if [ "${SUITE}" = "all" ]; then
    SUITES=("${DEFAULT_SUITES[@]}")
else
    IFS=',' read -r -a SUITES <<< "${SUITE}"
    for eval_suite in "${SUITES[@]}"; do
        case "${eval_suite}" in
            libero_10|libero_goal|libero_object|libero_spatial) ;;
            *) echo "[ERROR] Unknown LIBERO-plus suite: ${eval_suite}"; exit 1 ;;
        esac
    done
fi

echo "=========================================="
echo " LIBERO-plus SLURM Eval (multi-node)"
echo "=========================================="
echo " Job ID:        ${SLURM_JOB_ID:-local}"
echo " Nodes:         ${NUM_NODES}"
echo " Checkpoint:    ${your_ckpt}"
echo " Output dir:    ${output_dir}"
echo " Suites:        ${SUITES[*]}"
echo " GPUs/node:     ${NUM_GPUS:-<auto: nvidia-smi -L on each node>}"
echo " Workers/GPU:   ${WORKERS_PER_GPU}"
echo " Servers/GPU:   ${SERVERS_PER_GPU}"
echo " Trials/task:   ${NUM_TRIALS}"
echo " Max tasks/suite: ${MAX_TASKS_PER_SUITE}  (0 = full suite)"
echo " Exact tasks/suite: ${EXACT_TASKS_PER_SUITE}  (0 = disabled)"
echo " Max batch:     ${MAX_BATCH_SIZE}  (max_wait_time=${MAX_WAIT_TIME}s)"
echo " Save video:    ${SAVE_VIDEO}"
echo " Sim runtime:   ${SIM_RUNTIME}"
echo " Sim SIF:       ${SIM_SIF}"
echo " Object perturb: ${OBJECT_PERTURB_M}m roles=${OBJECT_PERTURB_ROLES} seed=${OBJECT_PERTURB_SEED}"
echo "=========================================="

# Exported so every srun'd node task (a separate process on a separate node)
# can read them without fragile nested-quoting string interpolation.
export your_ckpt output_dir
export SUITE_WORKERS_PER_GPU="${WORKERS_PER_GPU}"
export SUITE_SERVERS_PER_GPU="${SERVERS_PER_GPU}"
export SUITE_NUM_TRIALS="${NUM_TRIALS}"
export SUITE_MAX_TASKS_PER_SUITE="${MAX_TASKS_PER_SUITE}"
export SUITE_EXACT_TASKS_PER_SUITE="${EXACT_TASKS_PER_SUITE}"
export SUITE_MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"
export SUITE_MAX_WAIT_TIME="${MAX_WAIT_TIME}"
export SUITE_OVERLAY_TRACE="${OVERLAY_TRACE:-False}"
export SUITE_SAVE_VIDEO="${SAVE_VIDEO}"
export SUITE_GPU_IDS_CSV="${GPU_IDS_CSV}"
export SUITE_NUM_GPUS="${NUM_GPUS}"
export SUITE_NUM_NODES="${NUM_NODES}"
export SUITE_AUTO_EVAL_SCRIPT="${AUTO_EVAL_SCRIPT}"
export SUITE_OBJECT_PERTURB_M="${OBJECT_PERTURB_M}"
export SUITE_OBJECT_PERTURB_ROLES="${OBJECT_PERTURB_ROLES}"
export SUITE_OBJECT_PERTURB_SEED="${OBJECT_PERTURB_SEED}"
export SUITE_LIBERO_PLUS_RUNTIME="${SIM_RUNTIME}"
export SUITE_LIBERO_PLUS_SIF="${SIM_SIF}"

# One SLURM task per node runs the entire single-node pipeline against its
# own partition_idx=$SLURM_PROCID slice; SKIP_AGGREGATE defers the final
# merge to this script (a per-node aggregate would race against the others
# over the still-filling shared output_dir). Inlined as a plain command
# string (not `export -f` + a shell function) -- some clusters' PAM/security
# config strips the BASH_FUNC_*-encoded env vars that `export -f` relies on
# to survive an srun hop to another node, so keep this to ordinary env vars.
NODE_PARTITION_CMD='
    node_num_gpus="${SUITE_NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
    echo "[node ${SLURM_PROCID}/$((SUITE_NUM_NODES - 1)) $(hostname)] gpus=${node_num_gpus} partition=${SLURM_PROCID}"
    SKIP_AGGREGATE=true servers_per_gpu="${SUITE_SERVERS_PER_GPU}" \
        max_tasks_per_suite="${SUITE_MAX_TASKS_PER_SUITE}" \
        exact_tasks_per_suite="${SUITE_EXACT_TASKS_PER_SUITE}" \
        max_batch_size="${SUITE_MAX_BATCH_SIZE}" max_wait_time="${SUITE_MAX_WAIT_TIME}" \
        save_video="${SUITE_SAVE_VIDEO}" \
        overlay_trace="${SUITE_OVERLAY_TRACE}" \
        object_perturb_m="${SUITE_OBJECT_PERTURB_M}" \
        object_perturb_roles="${SUITE_OBJECT_PERTURB_ROLES}" \
        object_perturb_seed="${SUITE_OBJECT_PERTURB_SEED}" \
        LIBERO_PLUS_RUNTIME="${SUITE_LIBERO_PLUS_RUNTIME}" \
        LIBERO_PLUS_SIF="${SUITE_LIBERO_PLUS_SIF}" \
        bash "${SUITE_AUTO_EVAL_SCRIPT}" \
            "$1" "${node_num_gpus}" 0 "${SUITE_WORKERS_PER_GPU}" \
            "${SLURM_PROCID}" "${SUITE_NUM_TRIALS}" "${SUITE_GPU_IDS_CSV}" \
            "${SUITE_SERVERS_PER_GPU}" "${SUITE_NUM_NODES}"
'

for eval_suite in "${SUITES[@]}"; do
    echo ""
    echo "############################################"
    echo "# Evaluating suite: ${eval_suite} (${NUM_NODES} node(s))"
    echo "############################################"
    srun --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
        bash -c "${NODE_PARTITION_CMD}" _ "${eval_suite}"

    echo "All ${NUM_NODES} partition(s) of ${eval_suite} finished. Aggregating..."
    # Suite-scoped: writes output_dir/<suite>/overall_results.json only, so this
    # can never race with / overwrite another suite's results (each suite in
    # SUITES[] gets its own private file regardless of run order).
    python "${AGGREGATE_SCRIPT}" --root_path "${output_dir}" --task_suite_name "${eval_suite}"
done

# One combined pass across all suites, done exactly once at the very end (not
# per-suite), so it's never a partial/racy snapshot -- writes output_dir/overall_results.json.
python "${AGGREGATE_SCRIPT}" --root_path "${output_dir}"
echo "All suites finished. Combined results: ${output_dir}/overall_results.json"
echo "Per-suite results: ${output_dir}/<suite>/overall_results.json"

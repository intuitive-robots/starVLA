#!/bin/bash
#SBATCH --job-name=starvla_eval_libero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/eval_libero_%j.out
#SBATCH --error=slurm_logs/eval_libero_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
# Uncomment and set partition/account as needed:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

# Parallel, optionally MULTI-NODE vanilla-LIBERO evaluation. Each node runs
# its own self-contained copy of the single-node pipeline (policy servers +
# sim workers, all node-local -- no cross-node networking involved) against a
# distinct 1/num_nodes slice of the suite's episode range; results land in
# the same shared output_dir (every shard file is named by its own
# [start,end) episode range, so concurrent nodes never collide on filenames),
# then this script aggregates once per suite after every node has finished.
# Wraps examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh.
#
# Usage:
#   sbatch eval_libero_slurm.sh --ckpt /path/to/pytorch_model.pt
#       -> single node (default #SBATCH --nodes=1), evaluates ALL standard
#          suites (libero_spatial libero_object libero_goal libero_10)
#
#   sbatch --nodes=4 --gres=gpu:4 eval_libero_slurm.sh \
#       --ckpt /path/to/pytorch_model.pt --suite libero_10 --workers_per_gpu 32 --max_batch_size 32 --num_trials 30
#       -> 4 nodes x 4 GPUs, each node independently evaluating its own 1/4
#          slice of libero_10's 500 episodes; one server per GPU batching up
#          to 32 concurrent worker requests into one predict_action() call
#          (servers_per_gpu>1 predates batching and now performs WORSE -- it
#          just makes several model copies fight over the same GPU's compute;
#          one batched server strictly beats it, see batch_dispatcher.py)
#
# Node count is set at SUBMISSION time via `sbatch --nodes=N` (SBATCH
# directives are parsed before this script runs, so it can't be a runtime
# flag) -- same reasoning as GPU count via `sbatch --gres=gpu:N`.
#
# Flags:
#   --ckpt <path>            path to pytorch_model.pt (required; or set $your_ckpt env var)
#   --suite <name>           libero_spatial|libero_object|libero_goal|libero_10|libero_90|all (default: all)
#   --num_gpus <n>           GPUs PER NODE, default: GPUs visible to the job (nvidia-smi -L)
#   --workers_per_gpu <n>    default: 4 (or $workers_per_gpu env var); set close to --max_batch_size
#   --num_trials <n>         trials per task, default: 20 (or $num_trials env var)
#   --servers_per_gpu <n>    policy servers per GPU, default: 1 (or $servers_per_gpu env var).
#                            Leave at 1 now that batching exists -- see above.
#   --max_batch_size <n>     batch concurrent requests into one predict_action() call,
#                            default: 32 (or $max_batch_size env var). 1 = old unbatched behavior.
#   --max_wait_time <s>      ceiling on how long the dispatcher waits to fill a batch before
#                            running it under-full, default: 1.0 (or $max_wait_time env var).
#                            Exits early the moment the batch fills, so this rarely costs the
#                            full amount -- watch the server log's "[BatchDispatcher] batch_size=
#                            X/Y (Z% full)" line and raise it if fill rate is consistently low.
#   --gpu_ids <csv>          explicit GPU id list (per node), e.g. "0,1,2,3"
#   --object_perturb_m <m>   fixed-radius XY shift of selected goal objects (default 0)
#   --object_perturb_roles   comma-separated source,target (default both)
#   --object_perturb_seed    deterministic paired perturbation seed
#
# Env vars forwarded as-is to auto_eval_libero.sh: output_dir, server_host, base_port,
# server_idle_timeout, save_video, LIBERO_PYTHON, STARVLA_PYTHON.

set -euo pipefail

AUTO_EVAL_SCRIPT="./examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh"
AGGREGATE_SCRIPT="./examples/LIBERO/eval_files/parallel_eval/aggregate_results.py"
LIBERO_PYTHON="${LIBERO_PYTHON:-/e/project1/m3/blank4/envs/miniforge3/envs/libero/bin/python}"

DEFAULT_SUITES=(libero_spatial libero_object libero_goal libero_10)

# ── Parse flags ────────────────────────────────────────────────────────────
CKPT="${your_ckpt:-}"
SUITE="${suite:-all}"
NUM_TRIALS="${num_trials:-20}"
WORKERS_PER_GPU="${workers_per_gpu:-4}"
SERVERS_PER_GPU="${servers_per_gpu:-1}"
MAX_BATCH_SIZE="${max_batch_size:-32}"
MAX_WAIT_TIME="${max_wait_time:-1.0}"
COT_MAX_NEW_TOKENS="${cot_max_new_tokens:-0}"
GPU_IDS_CSV="${gpu_ids_csv:-}"
NUM_GPUS="${num_gpus:-}"
OBJECT_PERTURB_M="${object_perturb_m:-0.0}"
OBJECT_PERTURB_ROLES="${object_perturb_roles:-source,target}"
OBJECT_PERTURB_SEED="${object_perturb_seed:-20260812}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --suite) SUITE="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --workers_per_gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --servers_per_gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_wait_time) MAX_WAIT_TIME="$2"; shift 2 ;;
        --cot_max_new_tokens) COT_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS_CSV="$2"; shift 2 ;;
        --object_perturb_m) OBJECT_PERTURB_M="$2"; shift 2 ;;
        --object_perturb_roles) OBJECT_PERTURB_ROLES="$2"; shift 2 ;;
        --object_perturb_seed) OBJECT_PERTURB_SEED="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

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
# auto_eval_libero.sh's own default so all nodes agree without coordinating).
if [ -z "${output_dir:-}" ]; then
    ckpt_parent_dir=$(dirname "$(dirname "${your_ckpt}")")
    output_dir="${ckpt_parent_dir}/results/libero"
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_NODES="${SLURM_NNODES:-1}"

if [ "${SUITE}" = "all" ]; then
    SUITES=("${DEFAULT_SUITES[@]}")
else
    SUITES=("${SUITE}")
fi

echo "=========================================="
echo " LIBERO SLURM Eval (multi-node)"
echo "=========================================="
echo " Job ID:        ${SLURM_JOB_ID:-local}"
echo " Nodes:         ${NUM_NODES}"
echo " Checkpoint:    ${your_ckpt}"
echo " Output dir:    ${output_dir}"
echo " Suites:        ${SUITES[*]}"
echo " GPUs/node:     ${NUM_GPUS:-<auto: nvidia-smi -L on each node>}"
echo " Workers/GPU:   ${WORKERS_PER_GPU}"
echo " Servers/GPU:   ${SERVERS_PER_GPU}"
echo " Max batch:     ${MAX_BATCH_SIZE}  (max_wait_time=${MAX_WAIT_TIME}s)"
echo " CoT max tokens: ${COT_MAX_NEW_TOKENS}  (0 = checkpoint default)"
echo " Trials/task:   ${NUM_TRIALS}"
echo " Object perturb: ${OBJECT_PERTURB_M}m roles=${OBJECT_PERTURB_ROLES} seed=${OBJECT_PERTURB_SEED}"
echo "=========================================="

# Exported so every srun'd node task (a separate process on a separate node)
# can read them without fragile nested-quoting string interpolation.
export your_ckpt output_dir
export SUITE_WORKERS_PER_GPU="${WORKERS_PER_GPU}"
export SUITE_SERVERS_PER_GPU="${SERVERS_PER_GPU}"
export SUITE_MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"
export SUITE_MAX_WAIT_TIME="${MAX_WAIT_TIME}"
export SUITE_COT_MAX_NEW_TOKENS="${COT_MAX_NEW_TOKENS}"
export SUITE_NUM_TRIALS="${NUM_TRIALS}"
export SUITE_GPU_IDS_CSV="${GPU_IDS_CSV}"
export SUITE_NUM_GPUS="${NUM_GPUS}"
export SUITE_NUM_NODES="${NUM_NODES}"
export SUITE_AUTO_EVAL_SCRIPT="${AUTO_EVAL_SCRIPT}"
export SUITE_OBJECT_PERTURB_M="${OBJECT_PERTURB_M}"
export SUITE_OBJECT_PERTURB_ROLES="${OBJECT_PERTURB_ROLES}"
export SUITE_OBJECT_PERTURB_SEED="${OBJECT_PERTURB_SEED}"

# One SLURM task per node runs the entire single-node pipeline against its
# own partition_idx=$SLURM_PROCID slice; SKIP_AGGREGATE defers the merge to
# this script (a per-node aggregate would race against the others over the
# still-filling shared output_dir). Inlined as a plain command string (not
# `export -f` + a shell function) -- some clusters' PAM/security config
# strips the BASH_FUNC_*-encoded env vars that `export -f` relies on to
# survive an srun hop to another node, so keep this to ordinary env vars.
NODE_PARTITION_CMD='
    node_num_gpus="${SUITE_NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
    echo "[node ${SLURM_PROCID}/$((SUITE_NUM_NODES - 1)) $(hostname)] gpus=${node_num_gpus} partition=${SLURM_PROCID}"
    SKIP_AGGREGATE=true servers_per_gpu="${SUITE_SERVERS_PER_GPU}" \
        max_batch_size="${SUITE_MAX_BATCH_SIZE}" max_wait_time="${SUITE_MAX_WAIT_TIME}" \
        cot_max_new_tokens="${SUITE_COT_MAX_NEW_TOKENS}" \
        object_perturb_m="${SUITE_OBJECT_PERTURB_M}" \
        object_perturb_roles="${SUITE_OBJECT_PERTURB_ROLES}" \
        object_perturb_seed="${SUITE_OBJECT_PERTURB_SEED}" \
        bash "${SUITE_AUTO_EVAL_SCRIPT}" \
            "$1" "${node_num_gpus}" "${SUITE_WORKERS_PER_GPU}" "${SUITE_NUM_TRIALS}" \
            "${SUITE_GPU_IDS_CSV}" "${SLURM_PROCID}" "${SUITE_NUM_NODES}"
'

for eval_suite in "${SUITES[@]}"; do
    echo ""
    echo "############################################"
    echo "# Evaluating suite: ${eval_suite} (${NUM_NODES} node(s))"
    echo "############################################"
    srun --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
        bash -c "${NODE_PARTITION_CMD}" _ "${eval_suite}"

    echo "All ${NUM_NODES} partition(s) of ${eval_suite} finished. Aggregating..."
    "${LIBERO_PYTHON}" "${AGGREGATE_SCRIPT}" --root_path "${output_dir}" --task_suite_name "${eval_suite}"
done

echo "All suites finished. Aggregated results: ${output_dir}/overall_results.json"

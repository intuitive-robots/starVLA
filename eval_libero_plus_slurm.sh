#!/bin/bash
#SBATCH --job-name=starvla_eval_libero_plus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
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
#   --suite <name>            libero_10|libero_goal|libero_object|libero_spatial|all (default: all)
#   --num_gpus <n>            GPUs PER NODE, default: GPUs visible to the job (nvidia-smi -L)
#   --workers_per_gpu <n>     sim workers per GPU, default: 1 (or $workers_per_gpu env var)
#   --servers_per_gpu <n>     policy servers per GPU, default: 1 (or $servers_per_gpu env var).
#                             Each server is a full model copy on that GPU but gives real
#                             inference concurrency to workers sharing it -- best throughput
#                             when workers_per_gpu is a multiple of servers_per_gpu.
#   --num_trials <n>          trials per task, default: 1 (or $num_trials env var)
#   --gpu_ids <csv>           explicit GPU id list (per node), e.g. "0,1,2,3"
#
# Env vars forwarded as-is to auto_eval_libero_plus.sh: output_dir, server_host, base_port,
# server_idle_timeout, LIBERO_HOME, LIBERO_PLUS_CONDA_ENV, POLICY_SERVER_CONDA_ENV.

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --suite) SUITE="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --workers_per_gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --servers_per_gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS_CSV="$2"; shift 2 ;;
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p slurm_logs

NUM_NODES="${SLURM_NNODES:-1}"

if [ "${SUITE}" = "all" ]; then
    SUITES=("${DEFAULT_SUITES[@]}")
else
    SUITES=("${SUITE}")
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
echo "=========================================="

# Exported so every srun'd node task (a separate process on a separate node)
# can read them without fragile nested-quoting string interpolation.
export your_ckpt output_dir
export SUITE_WORKERS_PER_GPU="${WORKERS_PER_GPU}"
export SUITE_SERVERS_PER_GPU="${SERVERS_PER_GPU}"
export SUITE_NUM_TRIALS="${NUM_TRIALS}"
export SUITE_GPU_IDS_CSV="${GPU_IDS_CSV}"
export SUITE_NUM_GPUS="${NUM_GPUS}"
export SUITE_NUM_NODES="${NUM_NODES}"
export SUITE_AUTO_EVAL_SCRIPT="${AUTO_EVAL_SCRIPT}"

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

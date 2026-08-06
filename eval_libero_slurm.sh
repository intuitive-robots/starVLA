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

# Parallel LIBERO (vanilla) evaluation. Runs one policy server per GPU (or
# several via servers_per_gpu) plus a pool of simulation workers per GPU, then
# aggregates success rates. Wraps examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh.
#
# Usage:
#   sbatch eval_libero_slurm.sh --ckpt /path/to/pytorch_model.pt
#       -> evaluates ALL standard suites (libero_spatial libero_object libero_goal libero_10)
#
#   sbatch eval_libero_slurm.sh --ckpt /path/to/pytorch_model.pt --suite libero_10
#       -> evaluates only libero_10
#
#   servers_per_gpu=3 sbatch --gres=gpu:4 eval_libero_slurm.sh \
#       --ckpt /path/to/pytorch_model.pt --suite libero_10 --workers_per_gpu 6 --num_trials 30
#       -> mirrors: your_ckpt=... servers_per_gpu=3 bash .../auto_eval_libero.sh libero_10 4 6 30
#
# Flags:
#   --ckpt <path>            path to pytorch_model.pt (required; or set $your_ckpt env var)
#   --suite <name>           libero_spatial|libero_object|libero_goal|libero_10|libero_90|all (default: all)
#   --num_gpus <n>           default: GPUs visible to the job (nvidia-smi -L)
#   --workers_per_gpu <n>    default: 2 (or $workers_per_gpu env var)
#   --num_trials <n>         trials per task, default: 50 (or $num_trials env var)
#   --servers_per_gpu <n>    policy servers per GPU, default: 1 (or $servers_per_gpu env var)
#   --gpu_ids <csv>          explicit GPU id list, e.g. "0,1,2,3"
#
# Env vars forwarded as-is to auto_eval_libero.sh: output_dir, server_host, base_port,
# server_idle_timeout, save_video, LIBERO_PYTHON, STARVLA_PYTHON.

set -euo pipefail

AUTO_EVAL_SCRIPT="./examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh"

DEFAULT_SUITES=(libero_spatial libero_object libero_goal libero_10)

# ── Parse flags ────────────────────────────────────────────────────────────
CKPT="${your_ckpt:-}"
SUITE="${suite:-all}"
NUM_TRIALS="${num_trials:-20}"
WORKERS_PER_GPU="${workers_per_gpu:-4}"
SERVERS_PER_GPU="${servers_per_gpu:-3}"
GPU_IDS_CSV="${gpu_ids_csv:-}"
NUM_GPUS="${num_gpus:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --suite) SUITE="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --workers_per_gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --servers_per_gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
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

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

if [ "${SUITE}" = "all" ]; then
    SUITES=("${DEFAULT_SUITES[@]}")
else
    SUITES=("${SUITE}")
fi

echo "=========================================="
echo " LIBERO SLURM Eval"
echo "=========================================="
echo " Job ID:        ${SLURM_JOB_ID:-local}"
echo " Node:          $(hostname)"
echo " Checkpoint:    ${your_ckpt}"
echo " Suites:        ${SUITES[*]}"
echo " GPUs:          ${NUM_GPUS}"
echo " Workers/GPU:   ${WORKERS_PER_GPU}"
echo " Servers/GPU:   ${SERVERS_PER_GPU}"
echo " Trials/task:   ${NUM_TRIALS}"
echo "=========================================="

export your_ckpt servers_per_gpu="${SERVERS_PER_GPU}"

for eval_suite in "${SUITES[@]}"; do
    echo ""
    echo "############################################"
    echo "# Evaluating suite: ${eval_suite}"
    echo "############################################"
    bash "${AUTO_EVAL_SCRIPT}" "${eval_suite}" "${NUM_GPUS}" "${WORKERS_PER_GPU}" "${NUM_TRIALS}" "${GPU_IDS_CSV}"
done

echo "All suites finished."

#!/bin/bash
#SBATCH --job-name=trace_conditioning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/trace_conditioning_%j.out
#SBATCH --error=slurm_logs/trace_conditioning_%j.err

set -eo pipefail

OURS_CHECKPOINT="${OURS_CHECKPOINT:-playground/Checkpoints/libero_plus_qwen08b_gr00t_cot_trace_ours_v3_cotw01}"
DET_CHECKPOINT="${DET_CHECKPOINT:-playground/Checkpoints/libero_plus_qwen08b_gr00t_cot_trace_det_v3_cotw01}"
OUTPUT_DIR="${OUTPUT_DIR:-results/trace_conditioning_diagnostic}"
NUM_SAMPLES="${NUM_SAMPLES:-750}"
BATCH_SIZE="${BATCH_SIZE:-8}"

ml load CUDA
source ~/blank4/envs/miniforge3/etc/profile.d/conda.sh
conda activate starVLA
set -u

export HF_HOME="/e/home/jusers/blank4/jupiter/blank4/cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/e/home/jusers/blank4/jupiter/blank4/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="/e/home/jusers/blank4/jupiter/blank4/cache/inductor"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p slurm_logs "$OUTPUT_DIR"

python scripts/eval_trace_conditioning.py \
    --ours-checkpoint "$OURS_CHECKPOINT" \
    --det-checkpoint "$DET_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    "$@"

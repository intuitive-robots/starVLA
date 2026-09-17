#!/bin/bash
#SBATCH --job-name=tr_rc365_seeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/train_rc365_seed_pair_%j.out
#SBATCH --error=slurm_logs/train_rc365_seed_pair_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
#
# Two seeds of one RoboCasa config on one node: GPUs 0,1 and GPUs 2,3.
# Remaining arguments are trainer overrides, which makes this suitable for the
# mandatory 20-update smoke at the exact full-run distributed/batch shape.
set -euo pipefail

YAML="${1:?usage: train_robocasa365_seed_pair_slurm.sh <config> <prefix> <seedA> <seedB> [overrides...]}"
PREFIX="${2:?missing run prefix}"
SEED_A="${3:?missing seed A}"
SEED_B="${4:?missing seed B}"
shift 4
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

DATA_ROOT=/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/robocasa365_target_atomic
[ -w "${DATA_ROOT}/meta" ] || { echo "[ERROR] dataset meta/ not writable"; exit 1; }
[ -f "${DATA_ROOT}/meta/modality.json" ] || { echo "[ERROR] modality.json missing"; exit 1; }

FFMPEG_SHIM="${FFMPEG_SHIM:-/e/project1/m3/blank4/containers/ffmpeg_shim}"
CUDA_LIB64="${CUDA_LIB64:-/e/software/default/stages/2026/software/CUDA/13/lib64}"
SIF_SITE=/opt/conda/envs/starVLA/lib/python3.12/site-packages
[ -L "${FFMPEG_SHIM}/libavcodec.so.60" ] || { echo "[ERROR] FFmpeg shim missing"; exit 1; }
export APPTAINERENV_LD_LIBRARY_PATH="${FFMPEG_SHIM}:${CUDA_LIB64}:${SIF_SITE}/torch/lib:/opt/conda/envs/starVLA/lib"

# The generic pair launcher handles GPU/port separation and preserves every
# override as a distinct argument. Point its child trainer at this worktree.
export STARVLA_REPO="$REPO"
bash train_seed_pair_slurm.sh "$YAML" "$PREFIX" "$SEED_A" "$SEED_B" "$@"

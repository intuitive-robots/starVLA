#!/bin/bash
#SBATCH --account=m3
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --job-name=g3d_diagnostics
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/g_cam3d_diagnostics_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/g_cam3d_diagnostics_%j.err

set -euo pipefail

STARVLA=/e/project1/m3/blank4/code/starVLA
DIAG_ROOT=/e/project1/m3/blank4/code/encdec-vlm/train/encoder_decoder_training/enc_dec_cot
PYTHON=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python
CFG=examples/LIBERO/train_files/ervla_g_cam3d_cot05.yaml
CKPT=playground/Checkpoints/ervla_g_cam3d_cot05/checkpoints/steps_20000_pytorch_model.pt
RESULT_ROOT="$DIAG_ROOT/g_cam3d_diagnostic_results"
RESULTS="$RESULT_ROOT/job_${SLURM_JOB_ID}"

cd "$STARVLA"
mkdir -p "$RESULTS"
test -s "$CFG"
test -s "$CKPT"

export STARVLA DIAG_ROOT PYTHON CFG CKPT RESULTS
srun --nodes=1 --ntasks=3 --ntasks-per-node=3 --gpus-per-task=1 \
  --cpus-per-task=80 --gpu-bind=map_gpu:0,1,2 --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
    export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
    case "$SLURM_LOCALID" in
      0)
        "$PYTHON" "$DIAG_ROOT/encoder_reliance.py" \
          --arm g_cam3d_cot05 --config "$CFG" --checkpoint "$CKPT" \
          --output "$RESULTS/encoder_reliance.json" --samples 32 --batch-size 4
        ;;
      1)
        "$PYTHON" "$DIAG_ROOT/clean_modality_followup.py" \
          --arm g_cam3d_cot05 --config "$CFG" --checkpoint "$CKPT" \
          --output "$RESULTS/clean_modality.json" --heldout-samples 32 \
          --diverse-instructions 12 --frames-per-instruction 4 \
          --batch-size 4 --action-draws 2
        ;;
      2)
        "$PYTHON" "$DIAG_ROOT/gradient_balance_audit.py" \
          --arm g_cam3d_cot05 --config "$CFG" --checkpoint "$CKPT" \
          --output "$RESULTS/gradient_balance.json" --instructions 8 \
          --frames-per-instruction 1 --batch-size 2 --cot-scale 0.5
        ;;
      *) echo "unexpected SLURM_LOCALID=$SLURM_LOCALID" >&2; exit 2 ;;
    esac
  '

test -s "$RESULTS/encoder_reliance.json"
test -s "$RESULTS/clean_modality.json"
test -s "$RESULTS/gradient_balance.json"
ln -sfn "job_${SLURM_JOB_ID}" "$RESULT_ROOT/latest"
echo "G-cam3d diagnostics complete: $RESULTS"

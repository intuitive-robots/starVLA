#!/bin/bash
#SBATCH --account=m3
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:59:00
#SBATCH --job-name=nocot_modality
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/no_cot_action_diagnostics_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/no_cot_action_diagnostics_%j.err

set -euo pipefail

STARVLA=/e/project1/m3/blank4/code/starVLA
DIAG_ROOT=/e/project1/m3/blank4/code/encdec-vlm/train/encoder_decoder_training/enc_dec_cot
PYTHON=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python
# Select exactly the rows used by the G-cam3d diagnostic.  The A/B action models never
# receive the attached CoT conversation.
DATA_CFG=examples/LIBERO/train_files/ervla_g_cam3d_cot05.yaml
RESULT_ROOT="$DIAG_ROOT/no_cot_action_diagnostic_results"
RESULTS="$RESULT_ROOT/job_${SLURM_JOB_ID}"

cd "$STARVLA"
mkdir -p "$RESULTS"
export DIAG_ROOT PYTHON DATA_CFG RESULTS

srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-task=1 \
  --cpus-per-task=120 --gpu-bind=map_gpu:0,1 --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
    export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
    case "$SLURM_LOCALID" in
      0)
        arm=a_causal
        checkpoint=playground/Checkpoints/ervla_a_causal/checkpoints/steps_20000_pytorch_model.pt
        ;;
      1)
        arm=b_bidir
        checkpoint=playground/Checkpoints/ervla_b_bidir/checkpoints/steps_20000_pytorch_model.pt
        ;;
      *) echo "unexpected SLURM_LOCALID=$SLURM_LOCALID" >&2; exit 2 ;;
    esac
    test -s "$checkpoint"
    "$PYTHON" "$DIAG_ROOT/action_modality_followup.py" \
      --arm "$arm" --data-config "$DATA_CFG" --checkpoint "$checkpoint" \
      --output "$RESULTS/$arm.json" --heldout-samples 32 \
      --diverse-instructions 12 --frames-per-instruction 4 \
      --batch-size 4 --action-draws 2
  '

test -s "$RESULTS/a_causal.json"
test -s "$RESULTS/b_bidir.json"
ln -sfn "job_${SLURM_JOB_ID}" "$RESULT_ROOT/latest"
echo "no-CoT action diagnostics complete: $RESULTS"

#!/bin/bash
#SBATCH --account=m3
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/encoder_reliance_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/encoder_reliance_%j.err

set -euo pipefail

STARVLA=/e/project1/m3/blank4/code/starVLA
DIAG=/e/project1/m3/blank4/code/encdec-vlm/train/encoder_decoder_training/enc_dec_cot/encoder_reliance.py
PYTHON=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python
RESULTS=/e/project1/m3/blank4/code/encdec-vlm/train/encoder_decoder_training/enc_dec_cot/encoder_reliance_results

cd "$STARVLA"
mkdir -p "$RESULTS"

srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 \
  --cpus-per-task=64 --gpu-bind=map_gpu:0,1,2,3 bash -c '
    set -euo pipefail
    case "$SLURM_LOCALID" in
      0) arm=c; config=examples/LIBERO/train_files/ervla_c_action.yaml;
         checkpoint=playground/Checkpoints/ervla_c_action_batchdrop/checkpoints/steps_20000_pytorch_model.pt ;;
      1) arm=crand; config=examples/LIBERO/train_files/ervla_crand.yaml;
         checkpoint=playground/Checkpoints/ervla_crand_batchdrop/checkpoints/steps_20000_pytorch_model.pt ;;
      2) arm=d; config=examples/LIBERO/train_files/ervla_d_ground.yaml;
         checkpoint=playground/Checkpoints/ervla_d_ground_batchdrop/checkpoints/steps_20000_pytorch_model.pt ;;
      3) arm=g; config=examples/LIBERO/train_files/ervla_g_cross_only.yaml;
         checkpoint=playground/Checkpoints/ervla_g_cross_only/checkpoints/steps_20000_pytorch_model.pt ;;
      *) echo "unexpected SLURM_LOCALID=$SLURM_LOCALID" >&2; exit 2 ;;
    esac
    export OMP_NUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false
    "$0" "$1" --arm "$arm" --config "$config" --checkpoint "$checkpoint" \
      --output "$2/$arm.json" --samples 32 --batch-size 4
  ' "$PYTHON" "$DIAG" "$RESULTS"

echo "all encoder-reliance arms complete"

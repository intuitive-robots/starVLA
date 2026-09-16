#!/bin/bash
#SBATCH --job-name=dfr_layer_readout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=booster
#SBATCH --account=m3
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_layer_readout_%j.out
#SBATCH --error=/e/project1/m3/blank4/code/starVLA/slurm_logs/dfr_layer_readout_%j.err

set -euo pipefail

STAR=/e/project1/m3/blank4/code/starVLA
ROOT=${STAR}/experiments/dfr_causal_poc
PY=${STARVLA_PYTHON:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python}
CKPT=${DFR_LAYER_CHECKPOINT:-${STAR}/playground/Checkpoints/ervla_p_causal_pi_masked/checkpoints/steps_20000_pytorch_model.pt}
CONFIG=${DFR_LAYER_CONFIG:-${STAR}/examples/LIBERO/train_files/ervla_p_causal_pi.yaml}
CLEAN_ROOT=${DFR_LAYER_CLEAN_ROOT:-/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/libero_lerobot}
PLUS_ROOT=${DFR_LAYER_PLUS_ROOT:-/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0}
SOURCE_STATS=${DFR_LAYER_SOURCE_STATS:-/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/libero_lerobot/libero/meta/stats.json}
PLUS_STATS=${DFR_LAYER_PLUS_STATS:-/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/libero_plus/meta/stats.json}
OUT=${DFR_LAYER_OUT:-${ROOT}/outputs/layer_readout_sufficiency_v1}
TRAIN_N=${DFR_LAYER_TRAIN_PER_DOMAIN:-512}
VAL_N=${DFR_LAYER_VALIDATION_PER_DOMAIN:-128}
TEST_N=${DFR_LAYER_TEST_PER_DOMAIN:-256}
BATCH_SIZE=${DFR_LAYER_BATCH_SIZE:-4}
EPOCHS=${DFR_LAYER_EPOCHS:-100}
SEED=${DFR_LAYER_SEED:-20260911}
SKIP_CAPTURE=${DFR_LAYER_SKIP_CAPTURE:-0}
mkdir -p "${OUT}"

export PYTHONPATH=${STAR}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export STAR ROOT PY CKPT CONFIG CLEAN_ROOT PLUS_ROOT SOURCE_STATS PLUS_STATS OUT
export TRAIN_N VAL_N TEST_N BATCH_SIZE EPOCHS SEED SKIP_CAPTURE

# Shard examples over all four GPUs. Each rank loads the same frozen checkpoint
# but reads a disjoint quarter of every domain/split and writes per-locus caches.
if [ "${SKIP_CAPTURE}" != 1 ]; then
  srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    "${PY}" "${ROOT}/capture_layer_readout_features.py" \
      --checkpoint "${CKPT}" \
      --config "${CONFIG}" \
      --clean-root "${CLEAN_ROOT}" \
      --plus-root "${PLUS_ROOT}" \
      --source-stats "${SOURCE_STATS}" \
      --plus-stats "${PLUS_STATS}" \
      --output-dir "${OUT}" \
      --rank "${SLURM_LOCALID}" \
      --world-size 4 \
      --batch-size "${BATCH_SIZE}" \
      --train-per-domain "${TRAIN_N}" \
      --validation-per-domain "${VAL_N}" \
      --test-per-domain "${TEST_N}" \
      --seed "${SEED}"
  '
fi

for rank in 0 1 2 3; do
  test -s "${OUT}/shards/rank_${rank}/manifest.json"
done

# Fit disjoint locus sets in parallel so all GPUs remain active for the probe
# stage as well. The negative-control PI input is paired with early PI on rank 2.
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 bash -lc '
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  case "${SLURM_LOCALID}" in
    0) loci=(raw_preprojector) ;;
    1) loci=(projected_memory) ;;
    2) loci=(pi_input pi_early) ;;
    3) loci=(pi_middle pi_late) ;;
  esac
  "${PY}" "${ROOT}/fit_layer_readout_probes.py" \
    --root "${OUT}" \
    --loci "${loci[@]}" \
    --rank "${SLURM_LOCALID}" \
    --world-size 4 \
    --epochs "${EPOCHS}" \
    --seed "${SEED}"
'

"${PY}" "${ROOT}/summarize_layer_readout_probes.py" --root "${OUT}"
test -s "${OUT}/summary.json"
test -s "${OUT}/RESULTS.md"
echo "Layer-localized readout sufficiency study complete: ${OUT}"

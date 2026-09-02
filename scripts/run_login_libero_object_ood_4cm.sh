#!/bin/bash
# Sequential one-GPU login-node evaluation of the paired vanilla-LIBERO object
# perturbation. All three models see the same task/episode-dependent offsets.
set -euo pipefail

cd /e/home/jusers/blank4/jupiter/blank4/code/starVLA

export DEBUG=0
export PYTHONUNBUFFERED=1
export base_port=10103
export servers_per_gpu=1
export max_batch_size=16
export max_wait_time=1.0
export cot_max_new_tokens=72
export object_perturb_m=0.04
export object_perturb_roles=source,target
export object_perturb_seed=20260812
export save_video=false

labels=(ours det base)
checkpoints=(
  playground/Checkpoints/libero_qwen08b_oft_cot_trace_ours_ft/final_model/pytorch_model.pt
  playground/Checkpoints/libero_qwen08b_oft_cot_trace_det/final_model/pytorch_model.pt
  playground/Checkpoints/libero_qwen08b_base_16chunk/final_model/pytorch_model.pt
)
output_dirs=(
  playground/Checkpoints/libero_qwen08b_oft_cot_trace_ours_ft/results/libero-object-ood-source-target-4cm
  playground/Checkpoints/libero_qwen08b_oft_cot_trace_det/results/libero-object-ood-source-target-4cm
  playground/Checkpoints/libero_qwen08b_base_16chunk/results/libero-object-ood-source-target-4cm
)

experiment_start=$(date +%s)
for i in "${!labels[@]}"; do
  label=${labels[$i]}
  export your_ckpt=${checkpoints[$i]}
  export output_dir=${output_dirs[$i]}
  model_start=$(date +%s)
  echo "[$(date --iso-8601=seconds)] START ${label}: ${your_ckpt}"
  bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
    libero_object 1 16 20 0
  model_end=$(date +%s)
  echo "[$(date --iso-8601=seconds)] DONE ${label}: $((model_end - model_start)) seconds"
done
experiment_end=$(date +%s)
echo "[$(date --iso-8601=seconds)] ALL DONE: $((experiment_end - experiment_start)) seconds"

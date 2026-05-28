#!/usr/bin/env bash
set -e

###########################################################################################
# Single-GPU local training setup
###########################################################################################

export CUDA_VISIBLE_DEVICES=0

# Force local communication only.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# Disable InfiniBand/RDMA because this is a single-GPU local PC.
export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA

# New PyTorch env var names. Avoid deprecated NCCL_* variants.
unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Optional debug. Use INFO if it still crashes.
export NCCL_DEBUG=WARN

# Optional: avoid albumentations update check noise.
export NO_ALBUMENTATIONS_UPDATE=1

###########################################################################################
# Paths / config
###########################################################################################

Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=1229_libero4in1_qwen35oft

# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

###########################################################################################
# Launch
###########################################################################################

NUM_PROCESSES=1

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --num_machines 1 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29500 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 500 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity niblank
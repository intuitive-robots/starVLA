#!/bin/bash
# StarVLA LIBERO training script — custom Qwen2B / 4B model
#
# Usage:
#   bash examples/LIBERO/train_files/run_libero_train_custom.sh
#
# Switch between 2B and 4B by changing MODEL_SIZE below.

# ── NCCL settings (tune for your network) ──────────────────────────────────
export NCCL_SOCKET_IFNAME=lo            # TODO: set to your network interface (e.g. bond0, eth0)
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###############################################################################
# === Please modify the variables below to match your environment ===

# 4B or 2b — controls which config / model path is picked
MODEL_SIZE=4b   # set to "2b" or "4b"

if [ "${MODEL_SIZE}" = "2b" ]; then
    base_vlm=playground/Pretrained_models/your_qwen2b_custom_ckpt  # TODO
    config_yaml=./examples/LIBERO/train_files/starvla_libero_qwen2b_custom.yaml
    run_id=libero_qwen2b_custom
elif [ "${MODEL_SIZE}" = "4b" ]; then
    base_vlm=playground/Pretrained_models/your_qwen4b_custom_ckpt  # TODO
    config_yaml=./examples/LIBERO/train_files/starvla_libero_qwen4b_custom.yaml
    run_id=libero_qwen4b_custom
else
    echo "Unknown MODEL_SIZE '${MODEL_SIZE}'. Use '2b' or '4b'."
    exit 1
fi

Framework_name=QwenGR00T               # QwenGR00T | QwenPI | QwenOFT | QwenFAST
freeze_module_list=''                  # e.g. 'qwen_vl_interface' to freeze VLM backbone

libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA  # TODO: path to downloaded LIBERO lerobot data
data_mix=libero_all                    # libero_all | libero_spatial | libero_object | libero_goal | libero_10

run_root_dir=./playground/Checkpoints
wandb_project=starVLA_libero
wandb_entity=your_wandb_entity         # TODO: set your W&B entity

# === End of configuration ===
###############################################################################

export PYTHONPATH=$(pwd):${PYTHONPATH}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"                # archive this script with the run

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  # --is_debug True


# ── Multi-node template (SLURM) ──────────────────────────────────────────────
# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml "${config_yaml}" \
#   --framework.name "${Framework_name}" \
#   --framework.qwenvl.base_vlm "${base_vlm}" \
#   --run_root_dir "${run_root_dir}" \
#   --run_id "${run_id}" \
#   --wandb_project "${wandb_project}" \
#   --wandb_entity "${wandb_entity}"

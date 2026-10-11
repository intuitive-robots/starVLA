#!/bin/bash
# One training-throughput measurement on this node (HoreKa proposal benchmark A).
# Usage: run_train_one.sh <model: 2b|4b|9b> <ngpus> <per_gpu_batch> [extra trainer overrides...]
# Multi-node: export NUM_MACHINES, MACHINE_RANK, MASTER_ADDR, MASTER_PORT (ngpus = total GPUs).
# Writes $BENCH_DIR/results/<tag>.json; exit 0 on success, 3 on CUDA OOM, else the trainer's code.
set -uo pipefail
MODEL=$1; NGPU=$2; BS=$3; shift 3

REPO=/e/project1/m3/blank4/code/starVLA
BENCH_DIR=${BENCH_DIR:-/e/scratch/m3/blank4/horeka_bench}
CONFIG=$REPO/examples/LIBERO/train_files/ervla_q35_causal_gr00t_fullmem_noz.yaml
STEPS=${BENCH_STEPS:-41}   # 10 warmup + 30 measured step-to-step intervals
ZERO=${BENCH_ZERO:-2}      # DeepSpeed stage; 2 = what every starVLA training run uses
W=/e/scratch/m3/misc_data/nogga2/weights
case $MODEL in
    2b) VLM=$W/Qwen3.5-2B;     HID=2048 ;;
    4b) VLM=$W/Qwen3.5-4B;     HID=2560 ;;
    9b) VLM=$W/Qwen/Qwen3.5-9B; HID=4096 ;;
    *) echo "unknown model $MODEL"; exit 2 ;;
esac
NUM_MACHINES=${NUM_MACHINES:-1}; MACHINE_RANK=${MACHINE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}; MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
LOCAL=$((NGPU / NUM_MACHINES))
TAG=${BENCH_TAG:-${MODEL}_g${NGPU}_n${NUM_MACHINES}_b${BS}_z${ZERO}}
mkdir -p "$BENCH_DIR"/{results,logs,runs}
LOG=$BENCH_DIR/logs/${TAG}_r${MACHINE_RANK}.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((LOCAL - 1)))}
export BENCH_OUT=$BENCH_DIR/results/$TAG.json BENCH_WARMUP=${BENCH_WARMUP:-10} NUM_MACHINES
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_TF=0 USE_TORCH=1 WANDB_MODE=disabled DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/e/scratch/m3/blank4/cache/triton TORCHINDUCTOR_CACHE_DIR=/e/scratch/m3/blank4/cache/inductor
# Host InfiniBand verbs inside the container (otherwise NCCL falls back to TCP across nodes).
HOSTIB=/e/scratch/m3/blank4/hostib
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$HOSTIB/etc/libibverbs.d:/etc/libibverbs.d"
export APPTAINERENV_LD_LIBRARY_PATH="$HOSTIB/lib${APPTAINERENV_LD_LIBRARY_PATH:+:$APPTAINERENV_LD_LIBRARY_PATH}"
rm -f "$BENCH_OUT"
cd "$REPO"

echo "[$(date +%T)] $TAG rank$MACHINE_RANK on $(hostname) GPUs=$CUDA_VISIBLE_DEVICES -> $LOG"
/e/home/jusers/blank4/jupiter/blank4/containers/envs/run_in_env.sh starVLA accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero$ZERO.yaml \
    --num_processes "$NGPU" --num_machines "$NUM_MACHINES" --machine_rank "$MACHINE_RANK" \
    --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
    scripts/horeka_bench/train_bench.py --config_yaml "$CONFIG" --use_deepspeed true \
    --run_root_dir "$BENCH_DIR/runs" --run_id "$TAG" \
    --framework.qwenvl.base_vlm "$VLM" --framework.qwenvl.vl_hidden_dim "$HID" \
    --framework.action_model.diffusion_model_cfg.cross_attention_dim "$HID" \
    --datasets.vla_data.per_device_batch_size "$BS" \
    --trainer.max_train_steps "$STEPS" --trainer.num_warmup_steps 0 \
    --trainer.save_interval 1000000 --trainer.eval_interval 1000000 --trainer.logging_frequency 10 \
    --trainer.save_final_checkpoint false "$@" > "$LOG" 2>&1
rc=$?
grep -q "BENCH_RESULT" "$LOG" && { grep "BENCH_RESULT" "$LOG"; exit 0; }
grep -qE "CUDA out of memory|OutOfMemoryError" "$LOG" && { echo "[$(date +%T)] $TAG OOM"; exit 3; }
echo "[$(date +%T)] $TAG FAILED rc=$rc (see $LOG)"; tail -5 "$LOG"; exit "$rc"

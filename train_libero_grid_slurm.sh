#!/bin/bash
# Resumable one-run LIBERO grid launcher. Override --nodes with sbatch for
# global batch 128/256; every node contributes four GPUs. The site disables
# Slurm requeue, so an unfinished segment submits a fresh 12-hour successor.
#SBATCH --job-name=tr_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/train_grid_%j.out
#SBATCH --error=slurm_logs/train_grid_%j.err
#SBATCH --partition=booster
#SBATCH -A m3

set -euo pipefail

CONFIG_YAML="${1:?usage: train_libero_grid_slurm.sh CONFIG RUN_ID SEED [overrides...]}"
RUN_ID="${2:?missing run id}"
SEED="${3:?missing seed}"
shift 3
EXTRA_ARGS=("$@")

STARVLA_REPO=/e/project1/m3/blank4/code/starVLA
cd "$STARVLA_REPO"
mkdir -p slurm_logs

GPUS_PER_NODE=4
NUM_MACHINES="$SLURM_NNODES"
NUM_PROCESSES="$((NUM_MACHINES * GPUS_PER_NODE))"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [[ "${SYSTEMNAME:-}" =~ ^(juwelsbooster|juwels|jurecadc|jusuf)$ ]]; then
    MASTER_ADDR="${MASTER_ADDR}i"
fi
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
ACCELERATE_CONFIG_FILE="${STARVLA_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_memory.yaml}"
RUN_ENV_SH=/e/home/jusers/blank4/jupiter/blank4/containers/envs/run_in_env.sh
RUN_ENV_NAME=starVLA

ml load CUDA
export SLURM_MPI_TYPE=none CUDA_VISIBLE_DEVICES=0,1,2,3
export STARVLA_REPO NUM_MACHINES NUM_PROCESSES MASTER_ADDR MASTER_PORT ACCELERATE_CONFIG_FILE
export RUN_ENV_SH RUN_ENV_NAME

STARVLA_PREEMPT_FLAG="/e/scratch/m3/blank4/starvla_preempt/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$STARVLA_PREEMPT_FLAG")"
unlink "$STARVLA_PREEMPT_FLAG" 2>/dev/null || true
export STARVLA_PREEMPT_FLAG

cancel_requested=0
request_checkpoint() {
    echo "$(date --iso-8601=seconds) time-limit warning: requesting safe checkpoint"
    touch "$STARVLA_PREEMPT_FLAG"
}
request_cancel() {
    cancel_requested=1
    echo "$(date --iso-8601=seconds) cancellation requested: checkpointing without successor"
    touch "$STARVLA_PREEMPT_FLAG"
}
trap request_checkpoint USR1
trap request_cancel TERM

record_job() {
    local kind="$1" job_id="$2" run_id="$3" detail="$4"
    local ledger="slurm_logs/libero_grid_job_ledger.tsv"
    exec 9>>"$ledger"
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\tparent=%s\n' \
        "$(date --iso-8601=seconds)" "$kind" "$job_id" "$run_id" "$detail" "$SLURM_JOB_ID" >&9
    flock -u 9
    exec 9>&-
}

submit_available_evals() {
    local step checkpoint output_dir marker eval_id
    for step in 20000 40000 60000 80000; do
        checkpoint="playground/Checkpoints/${RUN_ID}/checkpoints/steps_${step}_pytorch_model.pt"
        [[ -f "$checkpoint" ]] || continue
        marker="playground/Checkpoints/${RUN_ID}/.libero_plus_step${step}_eval_submitted"
        [[ -e "$marker" ]] && continue
        output_dir="${STARVLA_REPO}/playground/Checkpoints/${RUN_ID}/results/libero-plus-step${step}-exact4k-v1"
        if eval_id=$(sbatch --parsable --time=02:00:00 \
            --job-name="ep_${SLURM_JOB_ID}_${step}" \
            --export="ALL,POLICY_SERVER_GPU=,output_dir=${output_dir}" \
            eval_libero_plus_slurm.sh --ckpt "$checkpoint" \
            --exact_tasks_per_suite 1000 --resume); then
            printf '%s\n' "$eval_id" >"$marker"
            record_job eval "$eval_id" "$RUN_ID" "step=$step checkpoint=$checkpoint"
            echo "submitted exact-4k eval $eval_id for $RUN_ID step $step"
        fi
    done
}

echo "job=$SLURM_JOB_ID nodes=$NUM_MACHINES gpus=$NUM_PROCESSES config=$CONFIG_YAML run=$RUN_ID seed=$SEED"

srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --mpi=none --cpu-bind=none \
    --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        config_yaml="$1"; run_id="$2"; seed="$3"; shift 3
        cd "$STARVLA_REPO"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export TORCH_USE_CUDA_DSA=1
        export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
        export TRANSFORMERS_CACHE="$HF_HOME/transformers"
        export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
        export HF_DATASETS_CACHE="$HF_HOME/datasets"
        export TRITON_CACHE_DIR="/e/scratch/m3/blank4/cache/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
        export TORCHINDUCTOR_CACHE_DIR="/e/scratch/m3/blank4/cache/inductor/${SLURM_JOB_ID}/node${SLURM_NODEID}"
        mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export USE_TF=0 USE_TORCH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export WANDB_MODE=offline DISABLE_VERSION_CHECK=1
        export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1
        export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=10000 NCCL_SOCKET_TIMEOUT_MS=360000

        visible_gpus="$(nvidia-smi -L | wc -l)"
        [[ "$visible_gpus" -eq 4 ]] || { echo "expected four GPUs, got $visible_gpus"; exit 1; }
        "$RUN_ENV_SH" "$RUN_ENV_NAME" python -c "import accelerate, torch, transformers"

        "$RUN_ENV_SH" "$RUN_ENV_NAME" accelerate launch \
            --config_file "$ACCELERATE_CONFIG_FILE" \
            --num_processes "$NUM_PROCESSES" \
            --num_machines "$NUM_MACHINES" \
            --machine_rank "$SLURM_NODEID" \
            --main_process_ip "$MASTER_ADDR" \
            --main_process_port "$MASTER_PORT" \
            starVLA/training/train_starvla.py \
            --config_yaml "$config_yaml" --use_deepspeed true \
            --run_id "$run_id" --seed "$seed" --trainer.is_resume true "$@"
    ' _ "$CONFIG_YAML" "$RUN_ID" "$SEED" "${EXTRA_ARGS[@]}" &
srun_pid=$!

status=0
while true; do
    set +e
    wait "$srun_pid"
    status=$?
    set -e
    kill -0 "$srun_pid" 2>/dev/null || break
done

submit_available_evals

final_checkpoint="playground/Checkpoints/${RUN_ID}/checkpoints/steps_80000_pytorch_model.pt"
if [[ "$status" -eq 0 && ! -f "$final_checkpoint" && "$cancel_requested" -eq 0 ]]; then
    echo "segment ended cleanly before 80k; submitting a 12-hour resume segment"
    next_job=$(sbatch --parsable --nodes="$NUM_MACHINES" --time=12:00:00 \
        --job-name="$SLURM_JOB_NAME" "$STARVLA_REPO/train_libero_grid_slurm.sh" \
        "$CONFIG_YAML" "$RUN_ID" "$SEED" "${EXTRA_ARGS[@]}")
    record_job resume "$next_job" "$RUN_ID" "nodes=$NUM_MACHINES"
    echo "submitted resume job $next_job"
fi
exit "$status"

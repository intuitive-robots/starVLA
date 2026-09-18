#!/bin/bash
# Resumable pair launcher for global-batch-32 cells: two GPUs/run,16/device.
# The site disables Slurm requeue, so unfinished pairs submit a fresh segment.
#SBATCH --job-name=tr_grid_pair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/train_grid_pair_%j.out
#SBATCH --error=slurm_logs/train_grid_pair_%j.err
#SBATCH --partition=booster
#SBATCH -A m3

set -euo pipefail

CONFIG_A="${1:?usage: train_libero_grid_pair_slurm.sh CFG_A RUN_A CFG_B RUN_B SEED [overrides...]}"
RUN_A="${2:?missing run A}"
CONFIG_B="${3:?missing config B}"
RUN_B="${4:?missing run B}"
SEED="${5:?missing seed}"
shift 5
EXTRA_ARGS=("$@")

STARVLA_REPO=/e/project1/m3/blank4/code/starVLA
cd "$STARVLA_REPO"
mkdir -p slurm_logs
ml load CUDA

STARVLA_PREEMPT_FLAG="/e/scratch/m3/blank4/starvla_preempt/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$STARVLA_PREEMPT_FLAG")"
unlink "$STARVLA_PREEMPT_FLAG" 2>/dev/null || true
export STARVLA_PREEMPT_FLAG

cancel_requested=0
request_checkpoint() {
    echo "$(date --iso-8601=seconds) time-limit warning: requesting both checkpoints"
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
    local run_id="$1" step checkpoint output_dir marker eval_id
    for step in 20000 40000 60000 80000; do
        checkpoint="playground/Checkpoints/${run_id}/checkpoints/steps_${step}_pytorch_model.pt"
        [[ -f "$checkpoint" ]] || continue
        marker="playground/Checkpoints/${run_id}/.libero_plus_step${step}_eval_submitted"
        [[ -e "$marker" ]] && continue
        output_dir="${STARVLA_REPO}/playground/Checkpoints/${run_id}/results/libero-plus-step${step}-exact4k-v1"
        if eval_id=$(sbatch --parsable --time=02:00:00 \
            --job-name="ep_${SLURM_JOB_ID}_${step}" \
            --export="ALL,POLICY_SERVER_GPU=,output_dir=${output_dir}" \
            eval_libero_plus_slurm.sh --ckpt "$checkpoint" \
            --exact_tasks_per_suite 1000 --resume); then
            printf '%s\n' "$eval_id" >"$marker"
            record_job eval "$eval_id" "$run_id" "step=$step checkpoint=$checkpoint"
            echo "submitted exact-4k eval $eval_id for $run_id step $step"
        fi
    done
}

pids=()
launch() {
    local slot="$1" config="$2" run_id="$3"
    local devices port log
    devices=$([[ "$slot" -eq 0 ]] && echo 0,1 || echo 2,3)
    port=$((37000 + slot))
    log="slurm_logs/train_grid_pair_${SLURM_JOB_ID}_${slot}.log"
    (
        export STARVLA_CUDA_VISIBLE_DEVICES="$devices"
        export NUM_PROCESSES=2 MASTER_PORT="$port"
        bash train_libero_slurm.sh --config "$config" --run_id "$run_id" --seed "$SEED" \
            --trainer.is_resume true "${EXTRA_ARGS[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
}

launch 0 "$CONFIG_A" "$RUN_A"
launch 1 "$CONFIG_B" "$RUN_B"

status=0
for pid in "${pids[@]}"; do
    while true; do
        set +e
        wait "$pid"
        child_status=$?
        set -e
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        [[ "$child_status" -eq 0 ]] || status=1
        break
    done
done

submit_available_evals "$RUN_A"
submit_available_evals "$RUN_B"

final_a="playground/Checkpoints/${RUN_A}/checkpoints/steps_80000_pytorch_model.pt"
final_b="playground/Checkpoints/${RUN_B}/checkpoints/steps_80000_pytorch_model.pt"
if [[ "$status" -eq 0 && ( ! -f "$final_a" || ! -f "$final_b" ) && "$cancel_requested" -eq 0 ]]; then
    echo "segment ended cleanly before both runs reached 80k; submitting a two-hour resume segment"
    next_job=$(sbatch --parsable --time=02:00:00 --job-name="$SLURM_JOB_NAME" \
        "$STARVLA_REPO/train_libero_grid_pair_slurm.sh" \
        "$CONFIG_A" "$RUN_A" "$CONFIG_B" "$RUN_B" "$SEED" "${EXTRA_ARGS[@]}")
    record_job resume "$next_job" "$RUN_A,$RUN_B" "paired"
    echo "submitted resume job $next_job"
fi
exit "$status"

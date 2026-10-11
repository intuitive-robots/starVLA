#!/bin/bash
# ABC-Sim eval of a StarVLA ABC-130K (m3 bridge) checkpoint on ONE node: one policy server per GPU (starVLA image),
# CLIENTS_PER_GPU simulator clients per GPU (Nogga's abc.sif), then per-task aggregation.
# Run on a compute node (the login GPU is degraded for MuJoCo / EGL), e.g. inside an allocation:
#   srun --jobid <alloc> --overlap --ntasks=1 --cpus-per-task=288 bash examples/ABC/eval_files/eval_abc_sim_node.sh
# or through eval_abc_sim_slurm.sh. Settings come from the environment:
#   CKPT (required)       .../checkpoints/steps_N_pytorch_model.pt
#   OUT                   output dir (default <run>/results/abc_sim/steps_N)
#   TASKS                 comma list (default: the four ABC-130K training tasks)
#   NUM_EPISODES          worlds per task (default 20; world i uses seed 20260511 + i)
#   CLIENTS_PER_GPU       simulator clients per GPU (default 8)
#   NUM_GPUS              default: all visible (nvidia-smi -L)
#   GPU_IDS               space-separated physical GPU ids to use (default 0 .. NUM_GPUS-1)
#   CLIENT_ARGS           extra abc_sim_client.py args (e.g. "--video_episodes 1 --num_chunks 120")
#   ABC_SIF               default /e/scratch/m3/misc_data/nogga2/containers/spark/m3/vla-evaluation-harness/abc.sif
#   BASE_PORT             default 18600 + (SLURM_JOB_ID % 200) * 8
set -uo pipefail
REPO=/e/project1/m3/blank4/code/starVLA   # absolute: server and client always come from the live tree
CKPT=${CKPT:?set CKPT}
STEP=$(basename "$CKPT" | sed -E 's/^steps_([0-9]+)_.*/\1/')
OUT=${OUT:-$(dirname "$(dirname "$CKPT")")/results/abc_sim/steps_${STEP}}
TASKS=${TASKS:-put_plastic_bottles_in_bin,throw_plastic_bottles_in_bin,load_plates_into_dish_rack,turn_mug_right_side_up}
NUM_EPISODES=${NUM_EPISODES:-20}
CLIENTS_PER_GPU=${CLIENTS_PER_GPU:-8}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
read -r -a GPUS <<< "${GPU_IDS:-$(seq -s ' ' 0 $((NUM_GPUS - 1)))}"
NUM_GPUS=${#GPUS[@]}
CLIENT_ARGS=${CLIENT_ARGS:-}
ABC_SIF=${ABC_SIF:-/e/scratch/m3/misc_data/nogga2/containers/spark/m3/vla-evaluation-harness/abc.sif}
BASE_PORT=${BASE_PORT:-$((18600 + (${SLURM_JOB_ID:-0} % 200) * 8))}
MAX_BATCH=${MAX_BATCH:-32}
MAX_WAIT=${MAX_WAIT:-0.01}
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/node_$(hostname).log") 2>&1
echo "[abc_sim] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-local} ckpt=$CKPT out=$OUT tasks=$TASKS"
echo "[abc_sim] episodes/task=$NUM_EPISODES gpus=${GPUS[*]} clients/gpu=$CLIENTS_PER_GPU base_port=$BASE_PORT client_args=$CLIENT_ARGS"

export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CACHE=/e/scratch/m3/blank4/cache
mkdir -p "$CACHE/warp" "$CACHE/xdg_abc"

server_pids=()
for ((g=0; g<NUM_GPUS; g++)); do
  port=$((BASE_PORT + g))
  setsid env CUDA_VISIBLE_DEVICES=${GPUS[$g]} PYTHONPATH=$REPO bash -c "cd $REPO && exec $REPO/scripts/env/starvla_python \
      deployment/model_server/server_policy.py --ckpt_path '$CKPT' --port $port --use_bf16 --idle_timeout -1 \
      --max_batch_size $MAX_BATCH --max_wait_time $MAX_WAIT > '$OUT/logs/server_gpu${g}.log' 2>&1" &
  server_pids+=($!)
done
cleanup() { for p in "${server_pids[@]}"; do kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

for ((g=0; g<NUM_GPUS; g++)); do
  port=$((BASE_PORT + g)); t0=$(date +%s)
  until (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; do
    if ! kill -0 "${server_pids[$g]}" 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 900 ]; then
      echo "[abc_sim] server gpu$g failed"; tail -40 "$OUT/logs/server_gpu${g}.log"; exit 1
    fi
    sleep 5
  done
  echo "[abc_sim] server gpu$g up on port $port after $(( $(date +%s) - t0 )) s"
done

NSHARDS=$((NUM_GPUS * CLIENTS_PER_GPU))
client_pids=()
for ((k=0; k<NSHARDS; k++)); do
  g=$((k % NUM_GPUS)); port=$((BASE_PORT + g))
  apptainer exec --nv --cleanenv -B /e/project1,/e/scratch,/e/home "$ABC_SIF" bash -c "
    exec > '$OUT/logs/client_shard${k}.log' 2>&1
    export CUDA_VISIBLE_DEVICES=${GPUS[$g]} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPUS[$g]} PYOPENGL_PLATFORM=egl
    export WARP_CACHE_PATH=$CACHE/warp XDG_CACHE_HOME=$CACHE/xdg_abc OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    cd $REPO && exec python examples/ABC/eval_files/abc_sim_client.py --port $port --tasks $TASKS \
      --num_episodes $NUM_EPISODES --shard_idx $k --num_shards $NSHARDS --out_dir '$OUT' $CLIENT_ARGS" &
  client_pids+=($!)
  sleep 1
done
fail=0
for p in "${client_pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "[abc_sim] $(date '+%F %T') clients done, $fail of $NSHARDS exited non-zero"
python3 "$REPO/examples/ABC/eval_files/aggregate_abc_sim.py" "$OUT" --tasks "$TASKS" --num_episodes "$NUM_EPISODES"
exit $fail

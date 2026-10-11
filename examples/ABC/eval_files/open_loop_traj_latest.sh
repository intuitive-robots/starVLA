#!/bin/bash
# Trainer-style open-loop trajectory plots for the LATEST checkpoint of each ABC (m3 bridge) run,
# skipping checkpoints already plotted (<out_root>/<run>/step_<N>/metrics.json exists). The runs keep only
# their latest checkpoint, so re-running this as training goes on builds the evolution over steps.
#
#   bash examples/ABC/eval_files/open_loop_traj_latest.sh                 # default runs, sequential
#   bash examples/ABC/eval_files/open_loop_traj_latest.sh abc_p6ctrl_c128w16_s42 ...
#   EPISODES_PER_TASK=3 MAX_ANCHORS= OUT_ROOT=/e/scratch/m3/blank4/abc/openloop_traj bash ...
#
# Runs on the login GPU (model inference only, no MuJoCo) or inside an allocation. Absolute paths only.
set -uo pipefail

REPO=/e/project1/m3/blank4/code/starVLA
OUT_ROOT=${OUT_ROOT:-/e/scratch/m3/blank4/abc/openloop_traj}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-3}
MAX_ANCHORS=${MAX_ANCHORS:-}
if [ "$#" -gt 0 ]; then RUNS=("$@"); else
  RUNS=(abc_p6ctrl_c128w16_s42 abc_p2ctrl_c128w16_s42 abc_p6e7ffc_c128w16_s42)
fi

C13=/e/software/default/stages/2026/software/CUDA/13
# torchcodec needs the FFmpeg shim (as train_multinode_container_slurm.sh)
export APPTAINERENV_LD_LIBRARY_PATH=$C13/targets/sbsa-linux/lib:/e/project1/m3/blank4/containers/ffmpeg_shim:$C13/lib64:/opt/conda/envs/starVLA/lib/python3.12/site-packages/torch/lib:/opt/conda/envs/starVLA/lib
export PYTHONPATH=$REPO
# Thread budget: with the fork's default 64-decoder LRU one process reached ~2,200 threads, and the login
# node caps a user at 4,096 (three in parallel died with "can't start new thread" / libgomp). One episode
# at a time needs only the 3 cameras' decoders. Run runs sequentially, not in parallel, on the login node.
for v in LEROBOT_VIDEO_DECODER_CACHE_SIZE=6 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; do
  export "$v" "APPTAINERENV_$v"
done

for run in "${RUNS[@]}"; do
  ckdir=$REPO/playground/Checkpoints/$run/checkpoints
  ckpt=$(ls "$ckdir"/steps_*_pytorch_model.pt 2>/dev/null | sed -E 's/.*steps_([0-9]+)_pytorch_model\.pt$/\1 &/' \
         | sort -n | tail -1 | cut -d' ' -f2)
  if [ -z "$ckpt" ]; then echo "[$run] no checkpoint in $ckdir"; continue; fi
  step=$(basename "$ckpt" | sed -E 's/steps_([0-9]+)_.*/\1/')
  out=$OUT_ROOT/$run/step_$step
  if [ -f "$out/metrics.json" ]; then echo "[$run] step $step already plotted ($out)"; continue; fi
  mkdir -p "$out"
  echo "[$run] step $step -> $out ($(date '+%F %T'))"
  extra=()
  [ -n "$MAX_ANCHORS" ] && extra+=(--max_anchors "$MAX_ANCHORS")
  # the checkpoint may be pruned by the next save while we read it: copy-free, so just report a failure
  "$REPO/scripts/env/starvla_python" "$REPO/examples/ABC/eval_files/open_loop_traj.py" \
      --ckpt "$ckpt" --out_root "$OUT_ROOT" --episodes_per_task "$EPISODES_PER_TASK" "${extra[@]}" \
      > "$out/run.log" 2>&1 \
    && echo "[$run] step $step done: $(grep -o 'open_loop/mse_mean=.*' "$out/run.log")" \
    || echo "[$run] step $step FAILED, see $out/run.log"
done

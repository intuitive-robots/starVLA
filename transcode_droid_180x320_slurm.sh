#!/bin/bash
# Re-encode the complete DROID droid_success video tree at 180x320.
#
# Hopper/GH200 has NVDEC but no NVENC.  Each worker therefore performs H.264
# decode and resize on one GPU, downloads the 180x320 frames, and encodes them
# with libx264 on the node CPUs.  The job is resumable: validated destination
# files are skipped and new files are installed with an atomic rename.
#
# Submit:
#   sbatch transcode_droid_180x320_slurm.sh
#   sbatch --nodes=4 transcode_droid_180x320_slurm.sh
#
# Useful overrides:
#   sbatch --export=ALL,JOBS_PER_GPU=2,CRF=22 transcode_droid_180x320_slurm.sh
#   sbatch --export=ALL,PREFLIGHT_ONLY=1 transcode_droid_180x320_slurm.sh
#   sbatch --export=ALL,DEST_ROOT=/path/to/output transcode_droid_180x320_slurm.sh

#SBATCH --job-name=droid_180x320
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:45:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/droid_transcode_%j.out
#SBATCH --error=slurm_logs/droid_transcode_%j.err

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/e/scratch/m3/datasets/lerobot_3_0/DROID/droid_success}"
DEST_ROOT="${DEST_ROOT:-/e/scratch/m3/datasets/lerobot_3_0_transcoded/DROID/droid_success_180x320}"
M3_DATA_SIF="${M3_DATA_SIF:-/e/scratch/m3/misc_data/nogga2/containers/spark/m3/m3_data.sif}"
CUDA_COMPAT_EMPTY="${CUDA_COMPAT_EMPTY:-/e/scratch/m3/misc_data/nogga2/empty}"
STARVLA_REPO="${STARVLA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

TARGET_HEIGHT="${TARGET_HEIGHT:-180}"
TARGET_WIDTH="${TARGET_WIDTH:-320}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-5557}"
GPUS="${GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-7}"
CPU_THREADS_PER_ENCODE="${CPU_THREADS_PER_ENCODE:-8}"
CRF="${CRF:-20}"
X264_PRESET="${X264_PRESET:-veryfast}"
GOP_SIZE="${GOP_SIZE:-15}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-60}"
NUM_NODES="${SLURM_NNODES:-1}"

STATE_DIR="${DEST_ROOT}/.transcode_state"
MANIFEST="${STATE_DIR}/videos.txt"

# Apptainer workers re-enter this script, so overrides must cross the
# container boundary instead of silently falling back to the defaults above.
export SOURCE_ROOT DEST_ROOT M3_DATA_SIF CUDA_COMPAT_EMPTY STARVLA_REPO
export TARGET_HEIGHT TARGET_WIDTH EXPECTED_VIDEOS GPUS JOBS_PER_GPU
export CPU_THREADS_PER_ENCODE CRF X264_PRESET GOP_SIZE PREFLIGHT_ONLY
export PROGRESS_INTERVAL_SECONDS NUM_NODES STATE_DIR MANIFEST

probe_video() {
    local path="$1"
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,nb_frames \
        -of json "$path" \
        | jq -r '.streams[0] | [.width, .height, (.nb_frames // "N/A")] | @tsv'
}

valid_output() {
    local source="$1"
    local output="$2"
    local src_width src_height src_frames dst_width dst_height dst_frames

    [[ -s "$output" ]] || return 1
    read -r src_width src_height src_frames < <(probe_video "$source") || return 1
    read -r dst_width dst_height dst_frames < <(probe_video "$output") || return 1
    [[ "$dst_width" == "$TARGET_WIDTH" && "$dst_height" == "$TARGET_HEIGHT" ]] || return 1
    if [[ "$src_frames" =~ ^[0-9]+$ && "$dst_frames" =~ ^[0-9]+$ ]]; then
        [[ "$src_frames" == "$dst_frames" ]] || return 1
    fi
}

transcode_one() {
    local source="$1"
    local output="$2"
    local temporary="${output}.partial.$$"

    mkdir -p "$(dirname "$output")"
    rm -f "$temporary"
    if ! ffmpeg -nostdin -hide_banner -loglevel error \
        -hwaccel cuda -hwaccel_device 0 -hwaccel_output_format cuda \
        -i "$source" \
        -map 0:v:0 -map_metadata 0 -an \
        -vf "scale_cuda=w=${TARGET_WIDTH}:h=${TARGET_HEIGHT}:format=yuv420p,hwdownload,format=yuv420p" \
        -c:v libx264 -preset "$X264_PRESET" -crf "$CRF" \
        -threads "$CPU_THREADS_PER_ENCODE" \
        -g "$GOP_SIZE" -keyint_min "$GOP_SIZE" -sc_threshold 0 \
        -fps_mode passthrough -movflags +faststart -f mp4 \
        "$temporary"; then
        rm -f "$temporary"
        return 1
    fi

    if ! valid_output "$source" "$temporary"; then
        echo "validation failed: $temporary" >&2
        rm -f "$temporary"
        return 1
    fi
    mv -f "$temporary" "$output"
}

worker_main() {
    local slot="$1"
    local total_workers="$2"
    local manifest="$3"
    local log_file="${STATE_DIR}/worker_${slot}.log"
    local failure_file="${STATE_DIR}/worker_${slot}.failures"
    local completed=0 skipped=0 failed=0

    : > "$failure_file"
    while IFS= read -r relative_path; do
        [[ -n "$relative_path" ]] || continue
        local source="${SOURCE_ROOT}/videos/${relative_path}"
        local output="${DEST_ROOT}/videos/${relative_path}"

        if valid_output "$source" "$output"; then
            skipped=$((skipped + 1))
        elif transcode_one "$source" "$output"; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
            printf '%s\n' "$relative_path" >> "$failure_file"
        fi

        if (( (completed + skipped + failed) % 10 == 0 )); then
            printf '%s slot=%d completed=%d skipped=%d failed=%d\n' \
                "$(date --iso-8601=seconds)" "$slot" "$completed" "$skipped" "$failed" \
                | tee -a "$log_file"
        fi
    done < <(awk -v slot="$slot" -v workers="$total_workers" \
        '((NR - 1) % workers) == slot' "$manifest")

    printf '%s slot=%d DONE completed=%d skipped=%d failed=%d\n' \
        "$(date --iso-8601=seconds)" "$slot" "$completed" "$skipped" "$failed" \
        | tee -a "$log_file"
    (( failed == 0 ))
}

node_main() {
    local node_index="${SLURM_PROCID:?SLURM_PROCID is required for node workers}"
    local node_count="${SLURM_NTASKS:?SLURM_NTASKS is required for node workers}"
    local workers_per_node="$((GPUS * JOBS_PER_GPU))"
    local total_workers="$((node_count * workers_per_node))"
    local visible_gpus

    visible_gpus="$(nvidia-smi -L | wc -l)"
    if [[ "$visible_gpus" -ne "$GPUS" ]]; then
        echo "node=$node_index expected $GPUS visible GPUs, got $visible_gpus" >&2
        return 2
    fi

    local apptainer_binds=(
        --bind "${STARVLA_REPO}:${STARVLA_REPO}"
        --bind "${SOURCE_ROOT}:${SOURCE_ROOT}"
        --bind "$(dirname "$DEST_ROOT"):$(dirname "$DEST_ROOT")"
        --bind "${CUDA_COMPAT_EMPTY}:/usr/local/cuda-13.2/compat"
        --env "_CUDA_COMPAT_PATH=/usr/local/cuda-13.2/compat"
    )
    local pids=()
    local local_slot global_slot gpu
    echo "node=$node_index/$node_count host=$(hostname) workers=$workers_per_node global_workers=$total_workers"
    for (( local_slot=0; local_slot<workers_per_node; local_slot++ )); do
        global_slot="$((node_index * workers_per_node + local_slot))"
        gpu="$((local_slot % GPUS))"
        CUDA_VISIBLE_DEVICES="$gpu" apptainer exec --nv \
            "${apptainer_binds[@]}" "$M3_DATA_SIF" \
            "$STARVLA_REPO/transcode_droid_180x320_slurm.sh" \
            --worker "$global_slot" "$total_workers" "$MANIFEST" &
        pids+=("$!")
    done

    local worker_failure=0 pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            worker_failure=1
        fi
    done
    return "$worker_failure"
}

progress_monitor() {
    local started_at="$1"
    local initial_count="$2"
    local current_count elapsed added rate eta percent
    while true; do
        sleep "$PROGRESS_INTERVAL_SECONDS"
        current_count="$(find "${DEST_ROOT}/videos" -type f -name '*.mp4' 2>/dev/null | wc -l)"
        elapsed="$(( $(date +%s) - started_at ))"
        added="$((current_count - initial_count))"
        read -r percent rate eta < <(awk \
            -v done="$current_count" -v total="$EXPECTED_VIDEOS" \
            -v added="$added" -v elapsed="$elapsed" '
                BEGIN {
                    pct = total > 0 ? 100.0 * done / total : 0
                    r = elapsed > 0 ? added / elapsed : 0
                    e = r > 0 ? (total - done) / r : -1
                    printf "%.2f %.3f %.0f\n", pct, r, e
                }')
        if [[ "$eta" == "-1" ]]; then
            printf '%s PROGRESS %d/%d (%s%%), new_rate=%s videos/s, ETA=unknown\n' \
                "$(date --iso-8601=seconds)" "$current_count" "$EXPECTED_VIDEOS" "$percent" "$rate"
        else
            printf '%s PROGRESS %d/%d (%s%%), new_rate=%s videos/s, ETA=%02d:%02d:%02d\n' \
                "$(date --iso-8601=seconds)" "$current_count" "$EXPECTED_VIDEOS" "$percent" "$rate" \
                "$((eta / 3600))" "$(((eta % 3600) / 60))" "$((eta % 60))"
        fi
    done
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    worker_main "$@"
    exit
fi
if [[ "${1:-}" == "--node-worker" ]]; then
    node_main
    exit
fi

mkdir -p "${STARVLA_REPO}/slurm_logs"

if [[ ! -d "${SOURCE_ROOT}/videos" || ! -f "${SOURCE_ROOT}/meta/info.json" ]]; then
    echo "Invalid SOURCE_ROOT: $SOURCE_ROOT" >&2
    exit 2
fi
if [[ ! -f "$M3_DATA_SIF" ]]; then
    echo "Missing M3 container: $M3_DATA_SIF" >&2
    exit 2
fi
if [[ "$(readlink -m "$SOURCE_ROOT")" == "$(readlink -m "$DEST_ROOT")" ]]; then
    echo "SOURCE_ROOT and DEST_ROOT must differ" >&2
    exit 2
fi
if (( GPUS != 4 )); then
    echo "This launcher is intentionally configured for four GPUs; got GPUS=$GPUS" >&2
    exit 2
fi

mkdir -p "$DEST_ROOT" "$STATE_DIR"
exec 9>"${STATE_DIR}/transcode.lock"
if ! flock -n 9; then
    echo "Another transcoder already owns ${STATE_DIR}/transcode.lock" >&2
    exit 2
fi
# A cancelled FFmpeg process can leave only its PID-suffixed temporary.  No
# running transcoder exists while this lock is held, so these are stale by
# construction; finalized MP4s never carry the .partial suffix.
if [[ -d "${DEST_ROOT}/videos" ]]; then
    find "${DEST_ROOT}/videos" -type f -name '*.partial.*' -delete
fi

manifest_tmp="${MANIFEST}.tmp.$$"
find "${SOURCE_ROOT}/videos" -type f -name '*.mp4' -printf '%P\n' | LC_ALL=C sort > "$manifest_tmp"
video_count="$(wc -l < "$manifest_tmp")"
if [[ "$video_count" -ne "$EXPECTED_VIDEOS" ]]; then
    echo "Expected $EXPECTED_VIDEOS source videos, found $video_count" >&2
    rm -f "$manifest_tmp"
    exit 2
fi
mv -f "$manifest_tmp" "$MANIFEST"

# The numeric data are unchanged.  Use a symlink instead of copying 5.4 GiB of
# parquet, but copy the 9.7 MiB metadata tree so its advertised image shape can
# be updated without mutating the source dataset.
if [[ ! -e "${DEST_ROOT}/data" ]]; then
    ln -s "${SOURCE_ROOT}/data" "${DEST_ROOT}/data"
elif [[ "$(readlink -f "${DEST_ROOT}/data")" != "$(readlink -f "${SOURCE_ROOT}/data")" ]]; then
    echo "Existing ${DEST_ROOT}/data does not point to ${SOURCE_ROOT}/data" >&2
    exit 2
fi
if [[ ! -e "${DEST_ROOT}/meta" ]]; then
    cp -a "${SOURCE_ROOT}/meta" "${DEST_ROOT}/meta"
fi

printf 'job=%s started=%s source=%s target=%sx%s\n' \
    "${SLURM_JOB_ID:-none}" "$(date --iso-8601=seconds)" "$SOURCE_ROOT" \
    "$TARGET_HEIGHT" "$TARGET_WIDTH" > "${DEST_ROOT}/.TRANSCODE_INCOMPLETE"

echo "Source:             $SOURCE_ROOT"
echo "Destination:        $DEST_ROOT"
echo "Videos:             $video_count"
echo "Target HxW:         ${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo "Nodes:              $NUM_NODES"
echo "GPUs:               $GPUS"
echo "Jobs / GPU:         $JOBS_PER_GPU"
echo "Concurrent workers: $((NUM_NODES * GPUS * JOBS_PER_GPU))"
echo "CPU threads / job:  $CPU_THREADS_PER_ENCODE"
echo "Encoder:            libx264 preset=$X264_PRESET crf=$CRF gop=$GOP_SIZE"

ml load CUDA
visible_gpus="$(nvidia-smi -L | wc -l)"
if [[ "$visible_gpus" -ne "$GPUS" ]]; then
    echo "Expected $GPUS visible GPUs, got $visible_gpus" >&2
    exit 2
fi

APPTAINER_BINDS=(
    --bind "${STARVLA_REPO}:${STARVLA_REPO}"
    --bind "${SOURCE_ROOT}:${SOURCE_ROOT}"
    --bind "$(dirname "$DEST_ROOT"):$(dirname "$DEST_ROOT")"
    --bind "${CUDA_COMPAT_EMPTY}:/usr/local/cuda-13.2/compat"
    --env "_CUDA_COMPAT_PATH=/usr/local/cuda-13.2/compat"
)

# Fail before launching thousands of files if NVDEC/CUDA scaling or libx264 is
# unavailable in the selected container.  The probe writes only node-local tmp.
probe_source="${SOURCE_ROOT}/videos/$(head -n 1 "$MANIFEST")"
probe_output="/tmp/droid_transcode_probe_${SLURM_JOB_ID:-$$}.mp4"
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv "${APPTAINER_BINDS[@]}" "$M3_DATA_SIF" \
    ffmpeg -nostdin -hide_banner -loglevel error \
        -hwaccel cuda -hwaccel_device 0 -hwaccel_output_format cuda \
        -i "$probe_source" -frames:v 30 -an \
        -vf "scale_cuda=w=${TARGET_WIDTH}:h=${TARGET_HEIGHT}:format=yuv420p,hwdownload,format=yuv420p" \
        -c:v libx264 -preset ultrafast -crf 28 -threads 4 -f mp4 -y "$probe_output"
probe_shape="$(apptainer exec --nv "${APPTAINER_BINDS[@]}" "$M3_DATA_SIF" \
    ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0:s=x "$probe_output")"
rm -f "$probe_output"
if [[ "$probe_shape" != "${TARGET_WIDTH}x${TARGET_HEIGHT}" ]]; then
    echo "Transcode preflight returned ${probe_shape}, expected ${TARGET_WIDTH}x${TARGET_HEIGHT}" >&2
    exit 2
fi
echo "GPU decode/resize + CPU encode preflight OK"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "PREFLIGHT_ONLY=1; no dataset videos were written"
    exit 0
fi

for stale_failure in "${STATE_DIR}"/worker_*.failures; do
    [[ -e "$stale_failure" ]] || continue
    rm -f "$stale_failure"
done
mkdir -p "${DEST_ROOT}/videos"
initial_output_count="$(find "${DEST_ROOT}/videos" -type f -name '*.mp4' | wc -l)"
progress_monitor "$(date +%s)" "$initial_output_count" &
progress_pid="$!"

worker_failure=0
if ! srun --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-288}" --gpus-per-task="$GPUS" \
    "$STARVLA_REPO/transcode_droid_180x320_slurm.sh" --node-worker; then
    worker_failure=1
fi
kill "$progress_pid" 2>/dev/null || true
wait "$progress_pid" 2>/dev/null || true

failed_files="$(find "$STATE_DIR" -type f -name 'worker_*.failures' -size +0c -print | wc -l)"
output_count="$(find "${DEST_ROOT}/videos" -type f -name '*.mp4' | wc -l)"
if [[ "$worker_failure" -ne 0 || "$failed_files" -ne 0 || "$output_count" -ne "$EXPECTED_VIDEOS" ]]; then
    echo "Transcode incomplete: worker_failure=$worker_failure failure_lists=$failed_files outputs=$output_count/$EXPECTED_VIDEOS" >&2
    echo "Resubmit the same command to validate/skip completed files and retry failures." >&2
    exit 1
fi

# Advertise the actual payload dimensions in the destination metadata only.
info_tmp="${DEST_ROOT}/meta/info.json.tmp.$$"
jq --argjson h "$TARGET_HEIGHT" --argjson w "$TARGET_WIDTH" '
    .features |= with_entries(
        if .value.dtype == "video" then
            .value.shape = [$h, $w, 3]
            | .value.info["video.height"] = $h
            | .value.info["video.width"] = $w
        else . end
    )
' "${DEST_ROOT}/meta/info.json" > "$info_tmp"
mv -f "$info_tmp" "${DEST_ROOT}/meta/info.json"

jq -n \
    --arg completed "$(date --iso-8601=seconds)" \
    --arg source "$SOURCE_ROOT" \
    --argjson videos "$EXPECTED_VIDEOS" \
    --argjson height "$TARGET_HEIGHT" \
    --argjson width "$TARGET_WIDTH" \
    --arg codec "h264/libx264" \
    --arg preset "$X264_PRESET" \
    --argjson crf "$CRF" \
    --argjson gop "$GOP_SIZE" \
    '{completed:$completed, source:$source, videos:$videos, height:$height,
      width:$width, codec:$codec, preset:$preset, crf:$crf, gop:$gop}' \
    > "${DEST_ROOT}/TRANSCODE_MANIFEST.json"
rm -f "${DEST_ROOT}/.TRANSCODE_INCOMPLETE"

echo "Complete: $DEST_ROOT"

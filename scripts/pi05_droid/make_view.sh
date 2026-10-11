#!/bin/bash
# Build the gr00t-loader view of DROID used by the droid_pi05_2cam mixture (2026-10-11).
# The source release /e/scratch/m3/datasets/lerobot_3_0_transcoded/DROID/droid_success_180x320 has no
# meta/modality.json and no keep_ranges; the view symlinks data/ and videos/ and the read-only meta files,
# copies meta/episodes and adds keep_ranges to it (openpi's DROID idle filter: joint-velocity deltas < 1e-3 for
# >= 7 steps are idle, kept segments >= 16 steps, last 10 steps of each segment trimmed), and adds modality.json.
# The loader writes nothing here when stats_override_path is set.
# v2 (droid_resized_success; its own meta/modality.json already names the columns droid_pi05_resized reads):
#   make_view.sh <code_dir> /e/scratch/m3/datasets/lerobot_3_0/DROID/droid_resized_success \
#     /e/scratch/m3/blank4/pi05_droid/data/droid_resized_success_pi05 source
# usage: make_view.sh <code_dir> [src] [view] [modality.json | source] ; runs in the starVLA image (pandas/pyarrow)
set -euo pipefail
CODE=${1:-/e/scratch/m3/blank4/pi05_droid/code}
SRC=${2:-/e/scratch/m3/datasets/lerobot_3_0_transcoded/DROID/droid_success_180x320}
V=${3:-/e/scratch/m3/blank4/pi05_droid/data/droid_success_180x320_pi05}
MOD=${4:-$CODE/examples/DROID/train_files/modality.json}
[ "$MOD" = source ] && MOD=$SRC/meta/modality.json
PY=/e/project1/m3/blank4/code/starVLA/scripts/env/starvla_python
mkdir -p "$V/meta"
ln -sfn "$SRC/data" "$V/data"
ln -sfn "$SRC/videos" "$V/videos"
for f in info.json stats.json tasks.parquet; do ln -sfn "$SRC/meta/$f" "$V/meta/$f"; done
cp "$MOD" "$V/meta/modality.json"
if [ ! -f "$V/meta/KEEP_RANGES_DONE" ]; then
  rm -rf "$V/meta/episodes"
  cp -r "$SRC/meta/episodes" "$V/meta/episodes"
  chmod -R u+w "$V/meta/episodes"
  "$PY" -I "$CODE/scripts/pi05_droid/add_keep_ranges.py" "$CODE" "$V"
  date > "$V/meta/KEEP_RANGES_DONE"
fi
"$PY" -I - "$V" <<'EOF'
import sys, pandas as pd, numpy as np, glob
v = sys.argv[1]
df = pd.concat([pd.read_parquet(f, columns=["episode_index", "length", "keep_ranges"]) for f in sorted(glob.glob(v + "/meta/episodes/chunk-*/*.parquet"))])
kept = df["keep_ranges"].apply(lambda r: sum(int(b) - int(a) for a, b in r) if r is not None and len(r) else 0)
empty = int((kept == 0).sum())
print(f"view {v}: episodes {len(df)} frames {int(df['length'].sum())} kept {int(kept.sum())} "
      f"({kept.sum() / df['length'].sum():.1%}), episodes with no keep range {empty}")
EOF

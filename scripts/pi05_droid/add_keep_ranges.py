"""Add DROID keep_ranges to a LeRobot v3 view's meta/episodes (2026-10-11).

Same rule as examples/DROID/train_files/augment_keep_ranges_lerobot_v3.py (openpi's DROID non-idle filter; its
compute_keep_ranges is imported unchanged), but data files are located with info.json's data_path pattern
(droid_success_180x320 names them file_000.parquet, the repo script assumes file-000) and each data file is read once.
usage: python add_keep_ranges.py <code_dir> <view_root>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

code, root = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, str(Path(code) / "examples/DROID/train_files"))
from augment_keep_ranges_lerobot_v3 import compute_keep_ranges  # noqa: E402

pattern = json.load(open(root / "meta/info.json"))["data_path"]
for ep_file in sorted(glob.glob(str(root / "meta/episodes/chunk-*/*.parquet"))):
    ep = pd.read_parquet(ep_file)
    out = {}
    for (c, f), grp in ep.groupby(["data/chunk_index", "data/file_index"]):
        data = pd.read_parquet(root / pattern.format(chunk_index=int(c), file_index=int(f)),
                               columns=["episode_index", "action.joint_velocity"])
        by_ep = {int(e): g for e, g in data.groupby("episode_index")}
        for e in grp["episode_index"]:
            vel = np.stack(by_ep[int(e)]["action.joint_velocity"].to_list()).astype(np.float32)
            out[int(e)] = compute_keep_ranges(vel, idle_delta_threshold=1e-3, min_idle_len=7, min_non_idle_len=16,
                                              filter_last_n_in_ranges=10)
    ep["keep_ranges"] = [out[int(e)] for e in ep["episode_index"]]
    tmp = Path(ep_file).with_suffix(".tmp")
    ep.to_parquet(tmp, index=False)
    tmp.replace(ep_file)
    print("updated", ep_file, len(ep), flush=True)

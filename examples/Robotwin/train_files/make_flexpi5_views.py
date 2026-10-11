#!/usr/bin/env python3
"""Build metadata-only LeRobot-v3 views of the RoboTwin 2.0 Aloha-AgileX release for the
FLEX-pi 5-task ablation recipe (FLEX-pi App. G.5): lift_pot, place_shoe, pick_diverse_bottles,
place_object_basket, stack_bowls_two; 50 demonstrations per task.

A view is a directory with its own ``meta/`` (filtered episodes parquet, info.json with the
new totals, modality.json, the source tasks.parquet / stats.json as symlinks, and a
``subset_manifest.json`` with the exact episode ids) and ``data`` / ``videos`` symlinked to the
source release. No parquet rows or video bytes are copied: the loader addresses every episode
through its own data/video file and timestamp columns, which stay unchanged.

  clean      aloha_agilex_clean has exactly 50 episodes per task: all of them.
  randomized aloha_agilex_randomized has ~500 per task: a seeded sample of 50 per task
             (numpy default_rng(seed + task position in TASKS), without replacement, sorted).

Usage (idempotent; refuses to overwrite a view whose manifest differs):
  python3 examples/Robotwin/train_files/make_flexpi5_views.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["lift_pot", "place_shoe", "pick_diverse_bottles", "place_object_basket", "stack_bowls_two"]
SRC_ROOT = Path("/e/scratch/m3/datasets/lerobot_3_0/RoboTwin2.0/robotwin2")
# The v3 release names the head camera cam_head; the repo-level modality.json (legacy layout) says cam_high.
MODALITY_JSON = Path(__file__).resolve().parents[3] / "playground/Datasets/RoboTwin2_v3/aloha_agilex_clean/meta/modality.json"
REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "playground/Datasets/RoboTwin2_flexpi5"


def build(source: str, view: str, out_root: Path, per_task: int, seed: int | None) -> dict:
    src = SRC_ROOT / source
    episode_files = sorted((src / "meta/episodes").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in episode_files], ignore_index=True)
    picked = []
    manifest_tasks = {}
    for position, task in enumerate(TASKS):
        ids = np.sort(df.loc[df.task_name == task, "episode_index"].to_numpy())
        if seed is None:
            if len(ids) != per_task:
                raise SystemExit(f"{source}/{task}: expected exactly {per_task} episodes, found {len(ids)}")
            chosen = ids
        else:
            if len(ids) < per_task:
                raise SystemExit(f"{source}/{task}: only {len(ids)} episodes")
            rng = np.random.default_rng(seed + position)
            chosen = np.sort(rng.choice(ids, size=per_task, replace=False))
        picked.extend(int(i) for i in chosen)
        sub = df[df.episode_index.isin(chosen)]
        manifest_tasks[task] = {
            "episodes": [int(i) for i in chosen],
            "num_episodes": int(len(chosen)),
            "frames": int(sub["length"].sum()),
            "available_in_source": int(len(ids)),
        }
    sub = df[df.episode_index.isin(picked)].sort_values("episode_index").reset_index(drop=True)
    manifest = {
        "source": str(src),
        "tasks": TASKS,
        "per_task": per_task,
        "selection": "all episodes" if seed is None else f"numpy default_rng(seed={seed} + task position), no replacement, sorted",
        "seed": seed,
        "total_episodes": int(len(sub)),
        "total_frames": int(sub["length"].sum()),
        "per_task_detail": manifest_tasks,
    }

    dst = out_root / view
    meta = dst / "meta"
    old_manifest = meta / "subset_manifest.json"
    if old_manifest.exists():
        if json.loads(old_manifest.read_text()) != manifest:
            raise SystemExit(f"{dst} exists with a different manifest; refusing to overwrite")
        print(f"[view] {dst} already up to date ({manifest['total_episodes']} episodes, {manifest['total_frames']} frames)")
        return manifest
    (meta / "episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    sub.to_parquet(meta / "episodes/chunk-000/file-000.parquet", index=False)
    info = json.loads((src / "meta/info.json").read_text())
    info["total_episodes"] = int(len(sub))
    info["total_frames"] = int(sub["length"].sum())
    info["splits"] = {"train": f"0:{len(sub)}"}
    info["flexpi5_view_of"] = str(src)
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (meta / "modality.json").write_text(MODALITY_JSON.read_text())
    for name in ("tasks.parquet", "stats.json"):
        link = meta / name
        if not link.exists():
            os.symlink(src / "meta" / name, link)
    for name in ("data", "videos"):
        link = dst / name
        if not link.exists():
            os.symlink(src / name, link)
    old_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[view] wrote {dst}: {manifest['total_episodes']} episodes, {manifest['total_frames']} frames")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per_task", type=int, default=50)
    parser.add_argument("--rand_seed", type=int, default=42)
    args = parser.parse_args()
    for source, view, seed in (
        ("aloha_agilex_clean", "aloha_agilex_clean_flexpi5", None),
        ("aloha_agilex_randomized", f"aloha_agilex_randomized_flexpi5_s{args.rand_seed}", args.rand_seed),
    ):
        m = build(source, view, args.out_root, args.per_task, seed)
        for task, d in m["per_task_detail"].items():
            print(f"    {task:22s} {d['num_episodes']:3d} episodes {d['frames']:6d} frames (source has {d['available_in_source']})")


if __name__ == "__main__":
    main()

"""Check that decoded video frames line up with their action labels.

Motivation: the 97.4% run trained on four LeRobot **v2.1** datasets decoded with
``torchvision_av``; later runs use a single **v3.0** dataset decoded with ``decord``
and sit at a ~3.8x higher loss floor with parallel loss curves. A frame/action
misalignment would produce exactly that signature, and this repo has a commit
("[fix] fix lerobot v3 image index in dataloader (#285)") showing the v3 image
index has been wrong before.

Two independent checks:

1. CROSS-BACKEND — decode the same timestamps via several backends and compare
   pixels. Backends disagreeing means at least one is picking the wrong frame.

2. SEMANTIC — LIBERO's gripper action is binary and flips exactly once per grasp.
   The wrist camera visibly changes when the gripper closes. We locate the action
   flip at frame k and report the frame-to-frame pixel deltas around it: the
   largest visual change should sit at k (or k+1 for actuation lag), not several
   frames away. A consistent offset across episodes is the smoking gun.

Usage:
    python scripts/check_video_action_alignment.py \
        --dataset /path/to/lerobot_3_0/libero_lerobot/libero \
        --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps  # noqa: E402


def _load_meta(root: Path) -> tuple[dict, dict]:
    info = json.load(open(root / "meta" / "info.json"))
    modality_path = root / "meta" / "modality.json"
    modality = json.load(open(modality_path)) if modality_path.is_file() else {}
    return info, modality


def _video_keys(info: dict, modality: dict) -> list[str]:
    """Original video keys, preferring modality.json's mapping when present."""
    if modality.get("video"):
        return [v["original_key"] for v in modality["video"].values()]
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def _resolve_video_path(root: Path, info: dict, key: str, chunk: int, file_idx: int) -> Path:
    tmpl = info["video_path"]
    return root / tmpl.format(video_key=key, chunk_index=chunk, file_index=file_idx)


def _episode_rows(root: Path, info: dict, episode_index: int) -> pd.DataFrame:
    """Return the parquet rows for one episode, scanning data files until found."""
    for path in sorted(root.glob("data/**/*.parquet")):
        df = pd.read_parquet(path)
        if "episode_index" not in df.columns:
            continue
        sub = df[df["episode_index"] == episode_index]
        if len(sub):
            return sub.sort_values("frame_index").reset_index(drop=True), path
    raise SystemExit(f"episode {episode_index} not found under {root}/data")


def cross_backend(video_path: Path, timestamps: np.ndarray, backends: list[str]) -> None:
    print(f"\n  [1] cross-backend agreement on {video_path.name} ({len(timestamps)} timestamps)")
    decoded = {}
    for b in backends:
        try:
            decoded[b] = get_frames_by_timestamps(str(video_path), timestamps, video_backend=b)
        except Exception as exc:  # backend missing or codec unsupported
            print(f"      {b:14} unavailable: {type(exc).__name__}: {str(exc)[:60]}")
    names = list(decoded)
    if len(names) < 2:
        print("      need >=2 working backends to compare")
        return
    ref = names[0]
    for other in names[1:]:
        a, b = decoded[ref].astype(np.int16), decoded[other].astype(np.int16)
        if a.shape != b.shape:
            print(f"      {ref} vs {other}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        diff = np.abs(a - b).mean(axis=(1, 2, 3))
        worst = int(diff.argmax())
        verdict = "IDENTICAL" if diff.max() < 1.0 else "DIFFER"
        print(f"      {ref} vs {other}: {verdict}  mean|d|={diff.mean():.2f} max={diff.max():.2f} (at ts idx {worst})")


def semantic_gripper(root: Path, info: dict, wrist_key: str, rows: pd.DataFrame, chunk: int, file_idx: int) -> None:
    """Locate the gripper action flip and see where the wrist image actually changes."""
    actions = np.stack(rows["action"].values)
    grip = actions[:, 6]
    flips = np.nonzero(np.diff(np.sign(grip - grip.mean())))[0]
    if not len(flips):
        print("      no gripper transition in this episode")
        return
    k = int(flips[0]) + 1  # first frame with the new gripper state

    lo, hi = max(k - 6, 0), min(k + 7, len(rows))
    ts = rows["timestamp"].values[lo:hi].astype(np.float64)
    path = _resolve_video_path(root, info, wrist_key, chunk, file_idx)
    frames = get_frames_by_timestamps(str(path), ts, video_backend="decord").astype(np.int16)

    deltas = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
    peak = int(deltas.argmax()) + lo + 1
    print(f"\n  [2] gripper action flips at frame {k} (action[6]: {grip[k-1]:+.0f} -> {grip[k]:+.0f})")
    print("      frame-to-frame wrist-image delta around the flip:")
    for i, d in enumerate(deltas):
        idx = lo + i + 1
        print(f"        frame {idx:4d}  delta={d:7.3f}{'   <-- action flip' if idx == k else ''}"
              f"{'   <-- largest visual change' if idx == peak else ''}")
    off = peak - k
    print(f"      => visual change peaks {off:+d} frames from the action flip"
          f"{'  (expected 0 or +1)' if abs(off) <= 1 else '   *** SUSPICIOUS OFFSET ***'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot dataset root (contains meta/, data/, videos/)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--backends", default="decord,torchcodec,pyav,opencv")
    args = ap.parse_args()

    root = Path(args.dataset)
    info, modality = _load_meta(root)
    vkeys = _video_keys(info, modality)
    print(f"dataset : {root}")
    print(f"version : {info.get('codebase_version')}  episodes={info.get('total_episodes')} frames={info.get('total_frames')}")
    print(f"video   : {vkeys}")

    for ep in range(args.episodes):
        rows, parquet_path = _episode_rows(root, info, ep)
        print(f"\n=== episode {ep}: {len(rows)} frames  ({parquet_path.name})")
        # chunk/file index of the episode's video: v3 stores many episodes per file,
        # so derive it from the parquet location which mirrors the video layout.
        parts = parquet_path.parts
        chunk = int([p for p in parts if p.startswith("chunk-")][0].split("-")[1])
        file_idx = int(parquet_path.stem.split("-")[1])

        ts = rows["timestamp"].values[: min(len(rows), 12)].astype(np.float64)
        cross_backend(_resolve_video_path(root, info, vkeys[0], chunk, file_idx), ts, args.backends.split(","))
        if len(vkeys) > 1:
            semantic_gripper(root, info, vkeys[1], rows, chunk, file_idx)


if __name__ == "__main__":
    main()

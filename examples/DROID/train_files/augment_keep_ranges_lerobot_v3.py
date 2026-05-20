"""Add DROID-style keep_ranges to a LeRobot v3 dataset's episode metadata.

Logic matches the filter the user provided:
- detect idle runs from differences in action.joint_velocity
- remove idle segments of length >= min_idle_len
- keep only non-idle segments of length >= min_non_idle_len
- trim filter_last_n_in_ranges frames from the end of each kept segment

The resulting keep ranges are written into each meta/episodes/... parquet shard
as a new ``keep_ranges`` column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_keep_ranges(
    joint_velocities: np.ndarray,
    *,
    idle_delta_threshold: float,
    min_idle_len: int,
    min_non_idle_len: int,
    filter_last_n_in_ranges: int,
) -> list[list[int]]:
    """Compute keep ranges for one episode from joint velocity traces."""
    if len(joint_velocities) == 0:
        return []

    is_idle_array = np.hstack(
        [
            np.array([False]),
            np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < idle_delta_threshold, axis=1),
        ]
    )

    is_idle_padded = np.concatenate([[False], is_idle_array, [False]])
    is_idle_diff = np.diff(is_idle_padded.astype(int))
    is_idle_true_starts = np.where(is_idle_diff == 1)[0]
    is_idle_true_ends = np.where(is_idle_diff == -1)[0]

    true_segment_masks = (is_idle_true_ends - is_idle_true_starts) >= min_idle_len
    is_idle_true_starts = is_idle_true_starts[true_segment_masks]
    is_idle_true_ends = is_idle_true_ends[true_segment_masks]

    keep_mask = np.ones(len(joint_velocities), dtype=bool)
    for start, end in zip(is_idle_true_starts, is_idle_true_ends, strict=True):
        keep_mask[start:end] = False

    keep_padded = np.concatenate([[False], keep_mask, [False]])
    keep_diff = np.diff(keep_padded.astype(int))
    keep_true_starts = np.where(keep_diff == 1)[0]
    keep_true_ends = np.where(keep_diff == -1)[0]

    true_segment_masks = (keep_true_ends - keep_true_starts) >= min_non_idle_len
    keep_true_starts = keep_true_starts[true_segment_masks]
    keep_true_ends = keep_true_ends[true_segment_masks]

    keep_ranges: list[list[int]] = []
    for start, end in zip(keep_true_starts, keep_true_ends, strict=True):
        trimmed_end = int(end) - filter_last_n_in_ranges
        if trimmed_end > int(start):
            keep_ranges.append([int(start), trimmed_end])

    return keep_ranges


def build_keep_ranges_for_episode_file(
    dataset_root: Path,
    episode_file: Path,
    *,
    idle_delta_threshold: float,
    min_idle_len: int,
    min_non_idle_len: int,
    filter_last_n_in_ranges: int,
) -> None:
    episode_df = pd.read_parquet(episode_file)

    required_episode_cols = {
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "length",
    }
    missing_episode_cols = required_episode_cols - set(episode_df.columns)
    if missing_episode_cols:
        raise ValueError(f"{episode_file} is missing required columns: {sorted(missing_episode_cols)}")

    keep_ranges_by_row: list[list[list[int]]] = [None] * len(episode_df)  # type: ignore

    # Cache per data parquet to avoid repeated reads.
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}

    for row_idx, row in episode_df.iterrows():
        chunk_index = int(row["data/chunk_index"])
        file_index = int(row["data/file_index"])
        data_key = (chunk_index, file_index)

        if data_key not in data_cache:
            data_file = dataset_root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            if not data_file.exists():
                raise FileNotFoundError(f"Missing data parquet for {episode_file}: {data_file}")
            data_cache[data_key] = pd.read_parquet(
                data_file,
                columns=["episode_index", "action.joint_velocity"],
            )

        data_df = data_cache[data_key]
        episode_index = int(row["episode_index"])
        episode_steps = data_df.loc[data_df["episode_index"] == episode_index, "action.joint_velocity"]
        if episode_steps.empty:
            raise ValueError(f"Episode {episode_index} from {episode_file} was not found in data parquet {data_key}")

        joint_velocities = np.stack(episode_steps.to_list()).astype(np.float32)
        keep_ranges_by_row[row_idx] = compute_keep_ranges(
            joint_velocities,
            idle_delta_threshold=idle_delta_threshold,
            min_idle_len=min_idle_len,
            min_non_idle_len=min_non_idle_len,
            filter_last_n_in_ranges=filter_last_n_in_ranges,
        )

    episode_df["keep_ranges"] = keep_ranges_by_row

    tmp_path = episode_file.with_suffix(".tmp")
    episode_df.to_parquet(tmp_path, index=False)
    tmp_path.replace(episode_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--idle-delta-threshold", default=1e-3, type=float)
    parser.add_argument("--min-idle-len", default=7, type=int)
    parser.add_argument("--min-non-idle-len", default=16, type=int)
    parser.add_argument("--filter-last-n-in-ranges", default=10, type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root / 'meta' / 'episodes'}")

    print(f"Updating keep_ranges for {len(episode_files)} episode parquet files under {dataset_root}")
    for episode_file in episode_files:
        print(f"Processing {episode_file}")
        build_keep_ranges_for_episode_file(
            dataset_root=dataset_root,
            episode_file=episode_file,
            idle_delta_threshold=args.idle_delta_threshold,
            min_idle_len=args.min_idle_len,
            min_non_idle_len=args.min_non_idle_len,
            filter_last_n_in_ranges=args.filter_last_n_in_ranges,
        )
    print("Done.")


if __name__ == "__main__":
    main()

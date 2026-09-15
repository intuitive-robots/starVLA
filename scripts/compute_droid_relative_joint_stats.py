#!/usr/bin/env python3
"""Compute DreamZero-style DROID relative-joint statistics.

The training target for anchor t and horizon h is

    action.joint_position[t + h] - observation.joint_position[t]

for h=0..H-1.  The commanded gripper remains absolute (converted from DROID's
1=closed convention to StarVLA's 1=open convention).  Anchors are exactly the
rows admitted by the Marigold DROID keep-range sampler.  Targets past the end
of an episode are edge-clamped, matching LeRobot/Marigold retrieval.

Mean/std/min/max are exact over the selected shard. Quantiles use a deterministic
priority reservoir and are merged across shards.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


COLUMNS = (
    "episode_index",
    "frame_index",
    "keep_interval_index",
    "action.joint_position",
    "action.gripper_position",
    "observation.joint_position",
    "observation.gripper_position",
)


def fixed_list_to_numpy(column, width: int) -> np.ndarray:
    values = column.combine_chunks().values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64).reshape(-1, width)


class PriorityReservoir:
    def __init__(self, capacity: int, width: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.width = int(width)
        self.rng = np.random.default_rng(seed)
        self.values = np.empty((0, width), dtype=np.float32)
        self.priorities = np.empty(0, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1, self.width)
        if values.size == 0:
            return
        priorities = self.rng.random(values.shape[0])
        values = np.concatenate((self.values, values), axis=0)
        priorities = np.concatenate((self.priorities, priorities), axis=0)
        if priorities.size > self.capacity:
            selected = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            values = values[selected]
            priorities = priorities[selected]
        self.values = values
        self.priorities = priorities


def update_moments(acc: dict[str, np.ndarray | int], values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 8)
    if not np.isfinite(values).all():
        raise ValueError("non-finite trajectory value encountered")
    acc["count"] = int(acc["count"]) + values.shape[0]
    acc["sum"] += values.sum(axis=0, dtype=np.float64)
    acc["sumsq"] += np.square(values).sum(axis=0, dtype=np.float64)
    acc["min"] = np.minimum(acc["min"], values.min(axis=0))
    acc["max"] = np.maximum(acc["max"], values.max(axis=0))


def process_shard(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        # The transcoded release intentionally uses a symlink for tabular data.
        paths = sorted((root / "data").resolve().glob("*/*.parquet"))
    paths = paths[args.shard_id :: args.num_shards]
    if not paths:
        raise RuntimeError(f"shard {args.shard_id} received no parquet files")

    action_acc = {
        "count": 0,
        "sum": np.zeros(8, dtype=np.float64),
        "sumsq": np.zeros(8, dtype=np.float64),
        "min": np.full(8, np.inf, dtype=np.float64),
        "max": np.full(8, -np.inf, dtype=np.float64),
    }
    state_acc = {
        "count": 0,
        "sum": np.zeros(8, dtype=np.float64),
        "sumsq": np.zeros(8, dtype=np.float64),
        "min": np.full(8, np.inf, dtype=np.float64),
        "max": np.full(8, -np.inf, dtype=np.float64),
    }
    action_reservoir = PriorityReservoir(args.reservoir_per_shard, 8, args.seed + args.shard_id)
    state_reservoir = PriorityReservoir(args.reservoir_per_shard, 8, args.seed + 1009 + args.shard_id)

    episode_count = 0
    anchor_count = 0
    for file_pos, path in enumerate(paths, start=1):
        table = pq.read_table(path, columns=list(COLUMNS), use_threads=True)
        episode_ids = np.asarray(table["episode_index"].combine_chunks())
        frame_ids = np.asarray(table["frame_index"].combine_chunks())
        keep_ids = np.asarray(table["keep_interval_index"].combine_chunks())
        action_joint = fixed_list_to_numpy(table["action.joint_position"], 7)
        state_joint = fixed_list_to_numpy(table["observation.joint_position"], 7)
        action_gripper = np.asarray(
            table["action.gripper_position"].combine_chunks(), dtype=np.float64
        )
        state_gripper = np.asarray(
            table["observation.gripper_position"].combine_chunks(), dtype=np.float64
        )

        boundaries = np.concatenate(
            ([0], np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1, [len(episode_ids)])
        )
        file_action_samples: list[np.ndarray] = []
        file_state_samples: list[np.ndarray] = []
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            local_frames = frame_ids[begin:end]
            if local_frames[0] != 0 or local_frames[-1] != end - begin - 1:
                raise RuntimeError(
                    f"episode {episode_ids[begin]} is split/non-contiguous in {path}: "
                    f"frames {local_frames[0]}..{local_frames[-1]}, rows={end-begin}"
                )
            anchors = np.flatnonzero(keep_ids[begin:end] >= 0)
            if anchors.size == 0:
                continue

            q_obs = state_joint[begin:end]
            q_cmd = action_joint[begin:end]
            g_obs = state_gripper[begin:end]
            g_cmd = action_gripper[begin:end]
            reference = q_obs[anchors]

            state_values = np.concatenate(
                (reference, (1.0 - g_obs[anchors])[:, None]), axis=1
            )
            update_moments(state_acc, state_values)
            file_state_samples.append(state_values.astype(np.float32))

            targets = np.minimum(
                anchors[:, None] + np.arange(args.horizon, dtype=np.int64)[None, :],
                end - begin - 1,
            )
            delta = q_cmd[targets] - reference[:, None, :]
            gripper = 1.0 - g_cmd[targets]
            action_values = np.concatenate((delta, gripper[..., None]), axis=-1).reshape(-1, 8)
            update_moments(action_acc, action_values)
            file_action_samples.append(action_values.astype(np.float32))

            episode_count += 1
            anchor_count += anchors.size

        if file_action_samples:
            action_reservoir.update(np.concatenate(file_action_samples, axis=0))
            state_reservoir.update(np.concatenate(file_state_samples, axis=0))
        print(
            f"shard={args.shard_id} file={file_pos}/{len(paths)} {path.name} "
            f"episodes={episode_count} anchors={anchor_count} "
            f"action_rows={action_acc['count']}",
            flush=True,
        )

    output_dir = Path(args.partial_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"shard_{args.shard_id:02d}.npz",
        horizon=np.asarray(args.horizon),
        episode_count=np.asarray(episode_count),
        anchor_count=np.asarray(anchor_count),
        action_count=np.asarray(action_acc["count"]),
        action_sum=action_acc["sum"],
        action_sumsq=action_acc["sumsq"],
        action_min=action_acc["min"],
        action_max=action_acc["max"],
        action_sample=action_reservoir.values,
        action_priority=action_reservoir.priorities,
        state_count=np.asarray(state_acc["count"]),
        state_sum=state_acc["sum"],
        state_sumsq=state_acc["sumsq"],
        state_min=state_acc["min"],
        state_max=state_acc["max"],
        state_sample=state_reservoir.values,
        state_priority=state_reservoir.priorities,
    )


def merge_kind(parts: list, kind: str, reservoir_size: int) -> dict:
    count = sum(int(part[f"{kind}_count"]) for part in parts)
    total = sum((part[f"{kind}_sum"] for part in parts), np.zeros(8, dtype=np.float64))
    sumsq = sum((part[f"{kind}_sumsq"] for part in parts), np.zeros(8, dtype=np.float64))
    minimum = np.min(np.stack([part[f"{kind}_min"] for part in parts]), axis=0)
    maximum = np.max(np.stack([part[f"{kind}_max"] for part in parts]), axis=0)
    samples = np.concatenate([part[f"{kind}_sample"] for part in parts], axis=0)
    priorities = np.concatenate([part[f"{kind}_priority"] for part in parts], axis=0)
    if len(priorities) > reservoir_size:
        selected = np.argpartition(priorities, reservoir_size - 1)[:reservoir_size]
        samples = samples[selected]
    mean = total / count
    std = np.sqrt(np.maximum(sumsq / count - np.square(mean), 0.0))
    return {
        "count": count,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "q01": np.quantile(samples, 0.01, axis=0).tolist(),
        "q99": np.quantile(samples, 0.99, axis=0).tolist(),
        "reservoir_sample_size": int(len(samples)),
    }


def merge_shards(args: argparse.Namespace) -> None:
    paths = [Path(args.partial_dir) / f"shard_{i:02d}.npz" for i in range(args.num_shards)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard outputs: {missing}")
    parts = [np.load(path) for path in paths]
    if any(int(part["horizon"]) != args.horizon for part in parts):
        raise ValueError("partial horizon mismatch")

    action = merge_kind(parts, "action", args.reservoir_size)
    state = merge_kind(parts, "state", args.reservoir_size)
    payload = {
        "description": (
            "DreamZero-style DROID commanded joint-position deltas. Every horizon target "
            "is action.joint_position[t+h] - observation.joint_position[t]; gripper is "
            "absolute with 1=open, 0=closed. Anchors follow Marigold keep ranges."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": str(Path(args.data_root).resolve()),
            "sampler": "DROIDRelativeJointAbsGripper",
            "script": "scripts/compute_droid_relative_joint_stats.py",
            "horizon_offsets": list(range(args.horizon)),
            "edge_policy": "clamp to episode boundary (LeRobot/Marigold semantics)",
            "num_shards": args.num_shards,
            "episodes": sum(int(part["episode_count"]) for part in parts),
            "anchors": sum(int(part["anchor_count"]) for part in parts),
        },
        "action.relative_joint_position": {
            key: value[:7] for key, value in action.items() if isinstance(value, list)
        },
        "action.gripper_position": {
            key: value[7:8] for key, value in action.items() if isinstance(value, list)
        },
        "observation.state.joint_position": {
            key: value[:7] for key, value in state.items() if isinstance(value, list)
        },
        "observation.state.gripper_position": {
            key: value[7:8] for key, value in state.items() if isinstance(value, list)
        },
        "statistics": {
            "action_rows": action["count"],
            "state_rows": state["count"],
            "action_reservoir_sample_size": action["reservoir_sample_size"],
            "state_reservoir_sample_size": state["reservoir_sample_size"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "merge"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--partial-dir", required=True)
    parser.add_argument("--output", default="examples/DROID/stats/droid_dreamzero_relative_joint_abs_gripper.json")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--reservoir-size", type=int, default=2_000_000)
    parser.add_argument("--reservoir-per-shard", type=int, default=600_000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode == "shard":
        process_shard(parsed)
    else:
        merge_shards(parsed)

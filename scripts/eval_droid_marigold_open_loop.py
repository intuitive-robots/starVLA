#!/usr/bin/env python
"""Deterministic open-loop action-chunk evaluation for Marigold datasets.

This evaluator uses Marigold's episode sampler and StarVLA adapter directly, so
the images, language, state, action convention, and normalization match
training.  It evaluates non-overlapping action chunks at ground-truth
observation anchors; it is not a closed-loop robot/environment rollout.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.marigold_data_datasets import get_marigold_data_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"steps_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot infer checkpoint step from {path}")
    return int(match.group(1))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _plot_trajectory(
    gt: np.ndarray,
    pred: np.ndarray,
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    action_dim = gt.shape[-1]
    ncols = min(4, action_dim)
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()
    for dim in range(action_dim):
        axes[dim].plot(
            gt[:, dim],
            label="ground truth",
            linewidth=1.8,
            color="C0",
        )
        axes[dim].plot(
            pred[:, dim],
            label="prediction",
            linewidth=1.2,
            color="C1",
        )
        axes[dim].set_title(f"action dim {dim}")
        axes[dim].set_xlabel("stitched action timestep")
        axes[dim].set_ylabel(ylabel)
        axes[dim].grid(True, alpha=0.25)
    for axis in axes[action_dim:]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    # mmap avoids an unnecessary second 5-GiB host copy on current PyTorch.
    load_kwargs.update({"weights_only": True, "mmap": True})
    try:
        state_dict = torch.load(checkpoint, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        load_kwargs.pop("mmap", None)
        state_dict = torch.load(checkpoint, **load_kwargs)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    del state_dict
    gc.collect()


def _predict_episode(
    model,
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    gt_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    source_frame_chunks: list[np.ndarray] = []
    chunk_lengths: list[int] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        # Reuse identical diffusion noise for every checkpoint.
        _seed_everything(seed + start)
        output = model.predict_action(examples=[item["example"] for item in batch])
        predictions = np.asarray(output["normalized_actions"], dtype=np.float32)
        if predictions.shape[0] != len(batch):
            raise RuntimeError(
                f"Prediction batch mismatch: {predictions.shape[0]} vs {len(batch)}"
            )
        for item, prediction in zip(batch, predictions):
            mask = item["valid_mask"]
            if mask.shape[0] != item["gt_normalized"].shape[0]:
                raise RuntimeError(
                    f"Action mask/target mismatch at anchor {item['anchor']}: "
                    f"{mask.shape} vs {item['gt_normalized'].shape}"
                )
            gt_chunk = item["gt_normalized"][mask]
            pred_chunk = prediction[: mask.shape[0]][mask]
            if gt_chunk.shape[0] == 0:
                continue
            gt_chunks.append(gt_chunk)
            pred_chunks.append(pred_chunk)
            source_frame_chunks.append(item["source_frames"][mask])
            chunk_lengths.append(int(gt_chunk.shape[0]))
    return (
        np.concatenate(gt_chunks, axis=0),
        np.concatenate(pred_chunks, axis=0),
        np.concatenate(source_frame_chunks, axis=0),
        chunk_lengths,
    )


def _build_exact_episode(
    reader,
    *,
    episode_uuid: str,
    local_anchors: list[int] | None,
    action_horizon: int,
) -> dict[str, Any]:
    """Resolve one UUID and, optionally, exact episode-local anchors.

    Marigold samplers index the concatenated dataset, whereas rollout reports
    conventionally use episode-local frame indices.  Resolve the episode's
    global row offset here so a reported local anchor cannot silently select a
    frame from a different episode.
    """
    matches: list[dict[str, Any]] = []
    for dataset_name, state in reader._get_open_loop_states():
        sampler = state.sampler
        if int(sampler.num_actions) != int(action_horizon):
            raise ValueError(
                f"Marigold/model action-horizon mismatch for {dataset_name}: "
                f"sampler={sampler.num_actions}, model={action_horizon}"
            )
        candidates = sampler.filter_episode_indices(list(range(sampler.num_episodes)))
        for episode_idx in candidates:
            info = sampler.get_episode_info(episode_idx)
            if str(sampler.episode_uuid(info)) != episode_uuid:
                continue

            inventory = sampler._build_instruction_inventory(info)
            episode_start = int(info.get("dataset_from_index", 0))
            episode_end = int(info.get("dataset_to_index", episode_start))
            if local_anchors is None:
                target_stride = int(sampler.action_frame_stride) * int(action_horizon)
                source_position = sampler._target_frame_offset_to_source_position(target_stride)
                anchor_stride = max(1, int(sampler._round_source_position(source_position)))
                anchors = sampler.plan_episode_samples(
                    episode_idx=episode_idx,
                    train=False,
                    frame_stride=anchor_stride,
                    mode=state.mode,
                    inventory=inventory,
                )
            else:
                anchor_stride = 0
                anchors = [episode_start + int(anchor) for anchor in local_anchors]
                invalid = [anchor for anchor in anchors if not episode_start <= anchor < episode_end]
                if invalid:
                    raise ValueError(
                        f"Local anchors {local_anchors} leave episode row range "
                        f"[0, {episode_end - episode_start}) for UUID {episode_uuid}"
                    )

            matches.append(
                {
                    "dataset_name": str(dataset_name),
                    "state": state,
                    "episode_idx": int(episode_idx),
                    "uuid": episode_uuid,
                    "inventory": inventory,
                    "anchors": anchors,
                    "anchor_stride_source_frames": anchor_stride,
                    "episode_start": episode_start,
                    "episode_end": episode_end,
                    "local_anchors": [int(anchor - episode_start) for anchor in anchors],
                }
            )
            # Episode UUIDs are the dataset's stable primary identifier.  Stop
            # after the match instead of scanning thousands of later episodes.
            break
        if matches:
            break

    if not matches:
        raise RuntimeError(f"DROID UUID not found: {episode_uuid}")
    return matches[0]


def evaluate(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_step = _checkpoint_step(checkpoint)

    cfg = apply_config_compat(OmegaConf.load(args.config_yaml))
    if str(cfg.datasets.vla_data.dataset_py) != "marigold_data_datasets":
        raise ValueError("This evaluator requires datasets.vla_data.dataset_py=marigold_data_datasets")

    # Evaluation must never perturb coordinates or appearance.
    cfg.datasets.vla_data.augmentation = "none"
    cfg.datasets.vla_data.marigold_resume_batches_per_rank = 0
    cfg.datasets.vla_data.marigold_resume_steps = 0
    eval_seed = int(cfg.seed if args.seed is None else args.seed)

    output_root = Path(args.output_dir)
    if args.seed_subdir:
        output_root = output_root / f"seed_{eval_seed}"
    output_dir = output_root / f"steps_{checkpoint_step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building Marigold evaluation adapter ...", flush=True)
    reader = get_marigold_data_vla_dataset(cfg.datasets.vla_data, mode="eval")
    action_horizon = int(cfg.framework.action_model.action_horizon)
    if args.episode_uuid:
        local_anchors = None
        if args.episode_local_anchors:
            local_anchors = [
                int(value.strip())
                for value in args.episode_local_anchors.split(",")
                if value.strip()
            ]
            if not local_anchors:
                raise ValueError("--episode_local_anchors did not contain any integers")
        selected_episodes = [
            _build_exact_episode(
                reader,
                episode_uuid=args.episode_uuid,
                local_anchors=local_anchors,
                action_horizon=action_horizon,
            )
        ]
    else:
        selected_episodes = reader.build_open_loop_episodes(
            action_horizon=action_horizon,
            num_episodes_per_subdataset=args.num_episodes,
            max_anchors=args.max_anchors,
        )
    if args.subdataset:
        selected_episodes = [
            episode for episode in selected_episodes
            if episode["dataset_name"] == args.subdataset
        ]
    if not selected_episodes:
        raise RuntimeError(f"No Marigold subdatasets matched {args.subdataset!r}")

    episodes_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for episode in selected_episodes:
        episodes_by_dataset.setdefault(episode["dataset_name"], []).append(episode)
    protocol = []
    for dataset_name, episodes in episodes_by_dataset.items():
        protocol.append(
            {
                "dataset": dataset_name,
                "anchor_stride_source_frames": episodes[0]["anchor_stride_source_frames"],
                "episodes": [
                    {
                        "episode_idx": episode["episode_idx"],
                        "uuid": episode["uuid"],
                        "num_anchors": len(episode["anchors"]),
                        "local_anchors": episode.get("local_anchors"),
                    }
                    for episode in episodes
                ],
            }
        )

    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Protocol written to {protocol_path}", flush=True)
    if args.protocol_only:
        return output_dir

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model evaluation")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    _seed_everything(eval_seed)

    print("Building model ...", flush=True)
    model = build_framework(cfg, is_inference=True)
    print(f"Loading checkpoint {checkpoint} ...", flush=True)
    _load_checkpoint(model, checkpoint)
    model = model.to(device).eval()
    torch.cuda.empty_cache()

    episode_results = []
    all_gt = []
    all_pred = []
    for dataset_name, episodes in episodes_by_dataset.items():
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for episode_order, episode in enumerate(episodes):
            episode_seed = eval_seed + episode["episode_idx"]
            print(
                f"Evaluating {dataset_name} episode {episode['episode_idx']} "
                f"({episode_order + 1}/{len(episodes)}, anchors={len(episode['anchors'])})",
                flush=True,
            )
            samples = reader.materialize_open_loop_episode(
                episode,
                seed=episode_seed,
            )
            gt, pred, source_frames, chunk_lengths = _predict_episode(
                model,
                samples,
                batch_size=args.batch_size,
                seed=episode_seed + 10_000_000,
            )
            squared_error = (pred - gt) ** 2
            mse_per_dim = squared_error.mean(axis=0)
            mse = float(squared_error.mean())
            action_indices = samples[0]["action_indices"]
            gt_physical = reader.normalizer.unnormalize_actions(gt, action_indices)
            pred_physical = reader.normalizer.unnormalize_actions(pred, action_indices)
            physical_mse_per_dim = ((pred_physical - gt_physical) ** 2).mean(axis=0)
            safe_uuid = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode["uuid"])[-100:]
            stem = f"episode_{episode['episode_idx']}_{safe_uuid}"
            np.savez_compressed(
                dataset_dir / f"{stem}.npz",
                source_frames=source_frames,
                chunk_lengths=np.asarray(chunk_lengths, dtype=np.int64),
                gt_normalized=gt,
                pred_normalized=pred,
                gt_physical=gt_physical,
                pred_physical=pred_physical,
            )
            _plot_trajectory(
                gt,
                pred,
                dataset_dir / f"{stem}_normalized.png",
                title=(
                    f"{dataset_name} / episode {episode['episode_idx']} / "
                    f"checkpoint {checkpoint_step}"
                ),
                ylabel="normalized action",
            )
            _plot_trajectory(
                gt_physical,
                pred_physical,
                dataset_dir / f"{stem}_physical.png",
                title=(
                    f"{dataset_name} / episode {episode['episode_idx']} / "
                    f"checkpoint {checkpoint_step} (dataset units)"
                ),
                ylabel="action in dataset units",
            )
            result = {
                "dataset": dataset_name,
                "episode_idx": episode["episode_idx"],
                "uuid": episode["uuid"],
                "num_anchors": len(samples),
                "num_valid_action_steps": int(gt.shape[0]),
                "normalized_mse_mean": mse,
                "normalized_mse_per_dim": mse_per_dim.tolist(),
                "physical_mse_per_dim": physical_mse_per_dim.tolist(),
            }
            episode_results.append(result)
            all_gt.append(gt)
            all_pred.append(pred)
            print(f"  normalized MSE={mse:.8f}, valid_steps={gt.shape[0]}", flush=True)

    gt = np.concatenate(all_gt, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    squared_error = (pred - gt) ** 2
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "seed": eval_seed,
        "protocol": protocol,
        "num_episodes": len(episode_results),
        "num_valid_action_steps": int(gt.shape[0]),
        "normalized_mse_mean": float(squared_error.mean()),
        "normalized_mse_per_dim": squared_error.mean(axis=0).tolist(),
        "mean_episode_normalized_mse": float(
            np.mean([item["normalized_mse_mean"] for item in episode_results])
        ),
        "episodes": episode_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Open-loop evaluation complete: step={checkpoint_step}, "
        f"normalized_mse={summary['normalized_mse_mean']:.8f}, output={summary_path}",
        flush=True,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--subdataset", default=None)
    parser.add_argument("--max_anchors", type=int, default=None)
    parser.add_argument("--episode_uuid", default=None)
    parser.add_argument(
        "--episode_local_anchors",
        default=None,
        help="Comma-separated episode-local frame indices; requires --episode_uuid.",
    )
    parser.add_argument(
        "--seed_subdir",
        action="store_true",
        help="Write under <output_dir>/seed_<seed>/ for multi-seed Slurm shards.",
    )
    parser.add_argument("--protocol_only", action="store_true")
    args = parser.parse_args()
    if args.episode_local_anchors and not args.episode_uuid:
        parser.error("--episode_local_anchors requires --episode_uuid")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

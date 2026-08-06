#!/usr/bin/env python
"""Full marigold_data loader bridge for StarVLA.

This path uses ``marigold_data.dataset.InfiniteDataReader`` as the actual
training dataset, including processed-config caching and global worker sharding,
then converts each Marigold sample to StarVLA's list-of-dicts VLA contract.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import IterableDataset

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return batch


_ROBOT_TYPE_TO_MARIGOLD = {
    "droid_lerobot_joint_pos": ("DROID", "franka_robotiq_gripper"),
    "droid_lerobot_resized_joint_pos": ("DROIDResized", "franka_robotiq_gripper"),
}


def _as_plain_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def _repo_sibling(path: str) -> Path:
    # .../starVLA/starVLA/dataloader/file.py -> .../code/<path>
    return Path(__file__).resolve().parents[3] / path


def _default_norm_stats_path(embodiment: str) -> Path | None:
    path = _repo_sibling(f"m3_model/norm_stats/{embodiment}.json")
    return path if path.exists() else None


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_user_data_config(data_cfg) -> dict:
    cfg = _as_plain_dict(data_cfg)
    explicit = cfg.get("marigold_data_config") or cfg.get("marigold_data_config_path")
    if explicit:
        from marigold_data.utils.config import load_data_config

        return load_data_config(explicit)

    data_root = Path(cfg["data_root_dir"])
    data_mix = cfg["data_mix"]
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    datasets = []
    for data_name, weight, robot_type in mixture_spec:
        if robot_type not in _ROBOT_TYPE_TO_MARIGOLD:
            raise ValueError(
                f"Cannot auto-build marigold_data config for robot_type={robot_type!r}. "
                "Set datasets.vla_data.marigold_data_config to an explicit M3-style JSON config."
            )
        dataset_type, embodiment_key = _ROBOT_TYPE_TO_MARIGOLD[robot_type]
        datasets.append(
            {
                "root": str(data_root),
                "dataset_type": dataset_type,
                "embodiment_key": embodiment_key,
                "weight_default": float(weight),
                "include": [data_name],
                "exclude": [],
                "weights": {},
                "filter_successful_trajectories": bool(cfg.get("filter_successful_trajectories", False)),
                "success_indicator_key": str(cfg.get("success_indicator_key", "stats/is_episode_successful/min")),
            }
        )

    image_size = cfg.get("obs_image_size", [224, 224])
    if len(image_size) != 2:
        raise ValueError(f"obs_image_size must be [height, width], got {image_size!r}")
    h, w = int(image_size[0]), int(image_size[1])

    camera_keys = cfg.get(
        "marigold_camera_keys",
        ["camera_wrist", "camera_external_left", "camera_external_right"],
    )
    default_target_fps = 15 if any(dataset["dataset_type"].startswith("DROID") for dataset in datasets) else 30

    return {
        "target_fps": int(cfg.get("target_fps", default_target_fps)),
        "num_proprio": int(cfg.get("num_proprio", 1)),
        "proprio_frame_stride": int(cfg.get("proprio_frame_stride", 1)),
        "num_actions": int(cfg.get("num_actions", cfg.get("action_horizon", 20))),
        "action_frame_stride": int(cfg.get("action_frame_stride", 1)),
        "image_relative_frames": list(cfg.get("image_relative_frames", [0])),
        "frame_stride_train": int(cfg.get("frame_stride_train", 1)),
        "frame_stride_eval": int(cfg.get("frame_stride_eval", 64)),
        "videos_hw": {key: [h, w] for key in camera_keys},
        "camera_decode_groups": cfg.get("marigold_camera_decode_groups", None),
        "image_augmentation": {key: [] for key in camera_keys},
        "datasets": datasets,
    }


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


def _load_global_processed_data_config(user_data_config: dict, cache_root: str | Path) -> dict:
    from marigold_data.utils.config import (
        init_data_config,
        process_data_config,
        processed_data_config_cache_path,
        save_data_config,
    )

    if not _dist_ready():
        raw = init_data_config(user_data_config)
        return process_data_config(raw, user_data_config)

    rank = _rank()
    cache_path = processed_data_config_cache_path(user_data_config, cache_root_dir=cache_root)
    payload: list[dict | None] = [None]
    if rank == 0:
        if cache_path.exists():
            processed = _load_json(cache_path)
            source = f"cached processed config: {cache_path}"
        else:
            raw = init_data_config(user_data_config)
            processed = process_data_config(raw, user_data_config)
            save_data_config(processed, cache_path)
            source = f"freshly built and cached: {cache_path}"
        logger.warning("[marigold_data] Using %s", source)
        payload[0] = processed
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _load_global_sharding_plan(processed_data_config: dict, dataloader_workers: int, cache_root: str | Path) -> dict:
    from marigold_data.utils.sharding import solve_even_count_sharding_baseline
    from marigold_data.utils.sharding_plan import (
        sampling_probabilities_from_config,
        sharding_plan_cache_path,
        validate_sharding_plan_workers,
    )

    workers_per_rank = dataloader_workers if dataloader_workers > 0 else 1
    global_workers = _world_size() * workers_per_rank

    if not _dist_ready():
        plan = solve_even_count_sharding_baseline(
            p=sampling_probabilities_from_config(processed_data_config),
            n_workers=global_workers,
        )
        plan["solution_type"] = "baseline"
        plan["global_shards"] = True
        plan["n_workers_per_rank"] = workers_per_rank
        plan["n_workers"] = global_workers
        return plan

    rank = _rank()
    plan_path = sharding_plan_cache_path(
        processed_data_config,
        n_workers=global_workers,
        cache_root_dir=cache_root,
    )
    payload: list[dict | None] = [None]
    if rank == 0:
        if plan_path.exists():
            plan = _load_json(plan_path)
            validate_sharding_plan_workers(plan, expected_workers=global_workers, plan_path=plan_path)
            source = f"cached optimized plan: {plan_path}"
        else:
            plan = solve_even_count_sharding_baseline(
                p=sampling_probabilities_from_config(processed_data_config),
                n_workers=global_workers,
            )
            plan["solution_type"] = "baseline"
            plan["global_shards"] = True
            plan["n_workers_per_rank"] = workers_per_rank
            plan["n_workers"] = global_workers
            source = f"in-memory baseline fallback; no cached optimized plan at {plan_path}"
        logger.warning("[marigold_data] Using sharding plan from %s", source)
        payload[0] = plan
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _tensor_to_pil(value: Any, frame_index: int = -1) -> Image.Image:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)

    if tensor.ndim == 4:
        tensor = tensor[frame_index]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W] or [T,C,H,W], got shape {tuple(tensor.shape)}")

    if tensor.shape[0] in (1, 3, 4):
        arr = tensor.permute(1, 2, 0).numpy()
    else:
        arr = tensor.numpy()

    if np.issubdtype(arr.dtype, np.floating):
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def _active_indices(fields: list, mask: Any, action_dim: int | None) -> list[int]:
    indices: list[int] = []
    for _, start, end in fields or []:
        indices.extend(range(int(start), int(end)))
    if not indices:
        mask_t = torch.as_tensor(mask, dtype=torch.bool).flatten()
        indices = torch.nonzero(mask_t, as_tuple=False).flatten().tolist()
    if action_dim is not None:
        indices = indices[:action_dim]
    return indices


def _empty_trajectory_stats(dim: int = 24) -> dict[str, list[float]]:
    return {
        "q01": [0.0] * dim,
        "q99": [1.0] * dim,
    }


def _fill_stats_slice(out: dict[str, list[float]], src: dict, dest: slice) -> None:
    start = int(dest.start or 0)
    stop = int(dest.stop)
    for key in ("q01", "q99"):
        values = src.get(key)
        if values is None:
            raise KeyError(key)
        out[key][start:stop] = [float(x) for x in values]


def _stats_from_lerobot_meta(stats_payload: dict) -> tuple[dict, dict] | None:
    """Convert LeRobot meta/stats.json into Marigold's 24-dim trajectory layout."""
    required = (
        "action.joint_position",
        "action.gripper_position",
        "observation.state.joint_position",
        "observation.state.gripper_position",
    )
    if not all(key in stats_payload for key in required):
        return None

    action_stats = _empty_trajectory_stats()
    state_stats = _empty_trajectory_stats()
    _fill_stats_slice(action_stats, stats_payload["action.joint_position"], slice(0, 7))
    _fill_stats_slice(action_stats, stats_payload["action.gripper_position"], slice(22, 23))
    _fill_stats_slice(state_stats, stats_payload["observation.state.joint_position"], slice(0, 7))
    _fill_stats_slice(state_stats, stats_payload["observation.state.gripper_position"], slice(22, 23))
    return action_stats, state_stats


class _Q99Normalizer:
    def __init__(self, data_cfg: dict, processed_data_config: dict):
        self.enabled = data_cfg.get("marigold_normalize_q99", True) not in (False, "False", "false", 0, "0")
        self.action_stats = None
        self.state_stats = None
        if not self.enabled:
            return

        stats_path = data_cfg.get("marigold_norm_stats_path")
        if stats_path is None:
            embodiment = None
            for dataset_cfg in processed_data_config.get("datasets", []):
                for sub in dataset_cfg.get("subdatasets", {}).values():
                    embodiment = sub.get("embodiment_key")
                    break
                if embodiment:
                    break
            candidate = _default_norm_stats_path(embodiment) if embodiment else None
            stats_path = str(candidate) if candidate is not None else None
        if not stats_path:
            logger.warning("[marigold_data] No norm stats found; StarVLA bridge will use raw low-dimensional values.")
            self.enabled = False
            return
        stats_payload = _load_json(stats_path)
        converted = _stats_from_lerobot_meta(stats_payload)
        if converted is not None:
            self.action_stats, self.state_stats = converted
            stats_format = "LeRobot meta/stats.json"
        else:
            stats = stats_payload.get("norm_stats", stats_payload)
            self.action_stats = stats.get("actions")
            self.state_stats = stats.get("state")
            stats_format = "Marigold norm stats"
        if not self.action_stats or not self.state_stats:
            logger.warning("[marigold_data] Norm stats at %s are incomplete; using raw values.", stats_path)
            self.enabled = False
        else:
            logger.warning("[marigold_data] Using %s for q99 normalization: %s", stats_format, stats_path)

    @staticmethod
    def _q99(x: np.ndarray, stats: dict, indices: list[int]) -> np.ndarray:
        q01 = np.asarray(stats["q01"], dtype=np.float32)[indices]
        q99 = np.asarray(stats["q99"], dtype=np.float32)[indices]
        out = x.astype(np.float32, copy=True)
        mask = q01 != q99
        out[..., mask] = 2.0 * (out[..., mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
        out[..., mask] = np.clip(out[..., mask], -1.0, 1.0)
        return out

    @staticmethod
    def _q99_with_binary_last(x: np.ndarray, stats: dict, indices: list[int]) -> np.ndarray:
        out = x.astype(np.float32, copy=True)
        if out.shape[-1] > 1:
            out[..., :-1] = _Q99Normalizer._q99(out[..., :-1], stats, indices[:-1])
        if out.shape[-1] > 0:
            out[..., -1] = (x[..., -1] > 0.5).astype(np.float32)
        return out

    def actions(self, x: np.ndarray, indices: list[int]) -> np.ndarray:
        if not self.enabled:
            return x.astype(np.float32)
        return self._q99_with_binary_last(x, self.action_stats, indices)

    def state(self, x: np.ndarray, indices: list[int]) -> np.ndarray:
        if not self.enabled:
            return x.astype(np.float32)
        return self._q99_with_binary_last(x, self.state_stats, indices)


class StarVLAMarigoldDataReader(IterableDataset):
    def __init__(
        self,
        *,
        processed_data_config: dict,
        sharding_plan: dict,
        data_cfg: dict,
        train: bool = True,
    ):
        from marigold_data.dataset import InfiniteDataReader

        self.processed_data_config = processed_data_config
        self.sharding_plan = sharding_plan
        self.data_cfg = data_cfg
        self.train = train
        self.estimated_len = self._estimate_len()
        self.action_dim = int(data_cfg.get("action_dim", 0)) or None
        self.state_dim = int(data_cfg.get("state_dim", 0)) or self.action_dim
        self.frame_index = int(data_cfg.get("marigold_image_frame_index", -1))
        self.camera_groups = data_cfg.get(
            "marigold_camera_groups",
            [["camera_external_left", "camera_external_right"], ["camera_wrist"]],
        )
        self.normalizer = _Q99Normalizer(data_cfg, processed_data_config)
        self.resume_batches_per_rank = int(data_cfg.get("marigold_resume_batches_per_rank", 0) or 0)
        self.batch_size = int(data_cfg.get("per_device_batch_size", 1) or 1)
        self.inner = InfiniteDataReader(
            data_config=processed_data_config,
            train=train,
            mode=data_cfg.get("instruction_mode", data_cfg.get("marigold_instruction_mode", "local")),
            video_backend=data_cfg.get("marigold_video_backend", data_cfg.get("video_backend", None)),
            max_active_subdatasets=data_cfg.get("marigold_max_active_subdatasets", None),
            sharding_plan=sharding_plan,
            resume_batches_per_rank=self.resume_batches_per_rank,
            resume_batch_size=self.batch_size,
        )

    def __iter__(self):
        for sample in self.inner:
            yield self._convert_sample(sample)

    def __len__(self) -> int:
        return self.estimated_len

    def _estimate_len(self) -> int:
        explicit = self.data_cfg.get("marigold_epoch_num_samples")
        if explicit is not None:
            return max(1, int(explicit))

        stride_key = "frame_stride_train" if self.train else "frame_stride_eval"
        total = 0
        for dataset_cfg in self.processed_data_config.get("datasets", []):
            for subdataset_cfg in dataset_cfg.get("subdatasets", {}).values():
                frames = int(subdataset_cfg.get("total_frames", 0))
                stride = max(1, int(subdataset_cfg.get(stride_key, 1)))
                total += max(1, frames // stride)
        return max(1, total)

    def _select_cameras(self, images: dict) -> list[str]:
        selected = []
        for group in self.camera_groups:
            candidates = [key for key in group if key in images]
            if candidates:
                selected.append(random.choice(candidates))
        if not selected:
            selected = sorted(images.keys())
        return selected

    def _convert_sample(self, sample: dict) -> dict:
        fields = sample.get("trajectory_fields", [])
        action_indices = _active_indices(fields, sample.get("trajectory_mask"), self.action_dim)
        state_indices = action_indices[: self.state_dim] if self.state_dim is not None else action_indices

        actions = torch.as_tensor(sample["actions"]).detach().cpu().numpy()[..., action_indices]
        proprio = torch.as_tensor(sample["proprio"]).detach().cpu().numpy()[..., state_indices]

        actions = self.normalizer.actions(actions, action_indices).astype(np.float16)
        proprio = self.normalizer.state(proprio, state_indices).astype(np.float16)

        images = sample.get("images", {})
        pil_images = [
            _tensor_to_pil(images[key], frame_index=self.frame_index)
            for key in self._select_cameras(images)
        ]

        trajectory_name = sample.get("uuid") or sample.get("root", "")
        return {
            "image": pil_images,
            "lang": str(sample.get("instruction", "")),
            "action": actions,
            "state": proprio,
            "robot_tag": str(sample.get("embodiment", "")),
            "trajectory_name": str(trajectory_name),
            "frame_index": int(sample.get("frame_index", 0)),
        }

    def save_dataset_statistics(self, output_path: str | Path) -> None:
        payload = {
            "source": "marigold_data",
            "processed_data_config": self.processed_data_config,
            "sharding_plan_summary": {
                "solution_type": self.sharding_plan.get("solution_type"),
                "global_shards": self.sharding_plan.get("global_shards"),
                "n_workers": self.sharding_plan.get("n_workers"),
                "n_workers_per_rank": self.sharding_plan.get("n_workers_per_rank"),
            },
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


def get_marigold_data_vla_dataset(data_cfg, mode: str = "train", **_: Any) -> StarVLAMarigoldDataReader:
    cfg = _as_plain_dict(data_cfg)
    user_data_config = _build_user_data_config(data_cfg)
    cache_root = os.environ.get(
        "MARIGOLD_SHARDING_PLAN_CACHE_ROOT",
        cfg.get("marigold_sharding_plan_cache_root", cfg.get("sharding_plan_cache_root", "/cache")),
    )
    processed = _load_global_processed_data_config(user_data_config, cache_root)
    workers = int(cfg.get("num_workers", 0))
    sharding_plan = _load_global_sharding_plan(processed, workers, cache_root)
    return StarVLAMarigoldDataReader(
        processed_data_config=processed,
        sharding_plan=sharding_plan,
        data_cfg=cfg,
        train=mode == "train",
    )

#!/usr/bin/env python
# Copyright 2025 starVLA authors. All rights reserved.
#
# Thin adapter that wraps the marigold-optimised LeRobotTrainingDataset
# (from the lerobot_marigold fork) with the gr00t protocol interface so
# it can be used as a drop-in inside LeRobotMixtureDataset.
#
# Activate with dataset_py: marigold_lerobot_datasets in your training YAML.

import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    EmbodimentTag,
)
from starVLA.dataloader.lerobot_datasets import _apply_trajectory_split, collate_fn  # noqa: F401

logger = logging.getLogger(__name__)


class MarigoldLeRobotSingleDataset:
    """
    Wraps the marigold LeRobotTrainingDataset with the gr00t protocol interface.

    The marigold fork provides optimised I/O for multi-node NFS training
    (parquet LRU, video decoder LRU, parallel multi-camera decode, page-cache
    prefetch). This class bridges that backend to the gr00t protocol expected
    by LeRobotMixtureDataset.

    gr00t protocol methods implemented:
        get_step_data, _apply_transforms_for_sample, _pack_sample,
        _resolve_sampled_video_keys, get_video_path, set_transforms_metadata
    gr00t protocol properties implemented:
        all_steps, valid_trajectory_ids, valid_trajectory_lengths,
        valid_base_indices_by_trajectory, modality_keys, lerobot_info_meta,
        metadata, tag, dataset_name, trajectory_ids, trajectory_lengths
    """

    def __init__(
        self,
        dataset_path: Path,
        modality_config: dict,
        transforms,
        embodiment_tag,
        data_cfg: dict | None = None,
        video_backend: str | None = None,
    ):
        self._dataset_path = Path(dataset_path)
        self._transforms = transforms
        self._data_cfg = data_cfg or {}
        self._modality_config = modality_config
        self.tag = embodiment_tag.value if isinstance(embodiment_tag, EmbodimentTag) else str(embodiment_tag)

        with open(self._dataset_path / "meta/info.json") as f:
            info = json.load(f)
        self._lerobot_info_meta = info
        fps = info["fps"]

        from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata

        with open(self._dataset_path / "meta/modality.json") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))

        self._gr00t_video_keys = list(modality_config["video"].modality_keys)
        # Parse sample_video_groups: list of groups, each group has keys to sample one from.
        # e.g. [[video.exterior_1_left, video.exterior_2_left]] → pick one exterior per step.
        raw_groups = self._data_cfg.get("sample_video_groups", [])
        known_keys = set(self._gr00t_video_keys)
        self._video_sample_groups: list[list[str]] = []
        self._video_group_lookup: dict[str, list[str]] = {}
        for group in raw_groups:
            normalized = [k if k.startswith("video.") else f"video.{k}" for k in group]
            normalized = [k for k in normalized if k in known_keys]
            normalized = list(dict.fromkeys(normalized))
            if len(normalized) > 1:
                self._video_sample_groups.append(normalized)
                for k in normalized:
                    self._video_group_lookup[k] = normalized
        # Map gr00t bare key (after stripping "video.") → actual lerobot stream name
        # e.g. "exterior_1_left" → "observation.images.exterior_1_left"
        # The mapping comes from meta/modality.json video[subkey].original_key
        self._bare_to_stream: dict[str, str] = {}
        for gr00t_key in self._gr00t_video_keys:
            subkey = gr00t_key.replace("video.", "", 1)
            vid_meta = le_modality_meta.video.get(subkey)
            original = vid_meta.original_key if (vid_meta and vid_meta.original_key) else subkey
            self._bare_to_stream[subkey] = original
        # stream names passed to marigold (e.g. "observation.images.exterior_1_left")
        self._stream_keys: list[str] = [
            self._bare_to_stream[k.replace("video.", "", 1)] for k in self._gr00t_video_keys
        ]
        self._bare_video_keys = [k.replace("video.", "", 1) for k in self._gr00t_video_keys]
        self._action_keys = list(modality_config["action"].modality_keys)
        self._state_keys = list(modality_config["state"].modality_keys)
        self._language_keys = list(modality_config["language"].modality_keys) if "language" in modality_config else []
        self._obs_indices = list(modality_config["video"].delta_indices)
        self._action_indices = list(modality_config["action"].delta_indices)

        self._action_col_info = self._build_col_info(le_modality_meta, "action", self._action_keys)
        self._state_col_info = self._build_col_info(le_modality_meta, "state", self._state_keys)

        # Unique parquet columns needed
        seen_cols: set[str] = set()
        action_parquet_cols: list[str] = []
        for info_dict in self._action_col_info.values():
            col = info_dict["original_key"]
            if col not in seen_cols:
                seen_cols.add(col)
                action_parquet_cols.append(col)

        seen_cols = set()
        state_parquet_cols: list[str] = []
        for info_dict in self._state_col_info.values():
            col = info_dict["original_key"]
            if col not in seen_cols:
                seen_cols.add(col)
                state_parquet_cols.append(col)

        # delta_timestamps keys must match the stream names marigold knows about
        delta_timestamps: dict[str, list[float]] = {}
        for stream_key in self._stream_keys:
            delta_timestamps[stream_key] = [i / fps for i in self._obs_indices]
        for col in action_parquet_cols:
            delta_timestamps[col] = [i / fps for i in self._action_indices]
        for col in state_parquet_cols:
            if col not in delta_timestamps:
                delta_timestamps[col] = [i / fps for i in self._obs_indices]

        # required_keys tells marigold which parquet columns to load into the LRU
        # cache (defaults to only timestamp/episode_index otherwise).
        # Language columns are excluded: they contain strings and are handled
        # via raw["task"] from meta/tasks.parquet instead.
        required_keys: set[str] = set(action_parquet_cols) | set(state_parquet_cols)

        # Lazy import: torchcodec (needed by marigold) requires CUDA libs not
        # always present in dev environments; importing at module level would
        # break import of this file outside of training nodes.
        from lerobot.datasets.lerobot_training_dataset import LeRobotTrainingDataset

        _MARIGOLD_SUPPORTED_BACKENDS = {"torchcodec", "pyav"}
        vb = video_backend or self._data_cfg.get("video_backend")
        if vb not in _MARIGOLD_SUPPORTED_BACKENDS:
            if vb is not None:
                logger.warning(
                    "video_backend %r is not supported by the marigold fork; "
                    "falling back to torchcodec (or pyav if torchcodec is unavailable).",
                    vb,
                )
            from lerobot.datasets.video_training_utils import get_safe_default_codec
            vb = get_safe_default_codec()
        self._inner = LeRobotTrainingDataset(
            repo_id=self._dataset_path.name,
            root=self._dataset_path,
            decode_camera_streams=self._stream_keys,  # actual lerobot stream names
            delta_timestamps=delta_timestamps,
            video_backend=vb,
            required_keys=required_keys,
        )

        self._trajectory_ids = np.arange(self._inner.num_episodes, dtype=np.int64)
        self._trajectory_lengths = np.array(
            [end - start for start, end in zip(self._inner._episode_starts, self._inner._episode_ends)],
            dtype=np.int64,
        )
        self._all_steps = self._get_all_steps_single_process()

        self._valid_cache_len = -1
        self._valid_trajectory_ids: np.ndarray | None = None
        self._valid_trajectory_lengths: np.ndarray | None = None
        self._valid_base_indices_by_trajectory: dict[int, np.ndarray] | None = None

        self._metadata = None

    # ---- Column info ----

    @staticmethod
    def _build_col_info(le_modality_meta, modality: str, gr00t_keys: list[str]) -> dict:
        """Maps gr00t_key → {original_key, start, end} from modality.json."""
        result = {}
        modality_meta = getattr(le_modality_meta, modality)
        for gr00t_key in gr00t_keys:
            subkey = gr00t_key.replace(f"{modality}.", "", 1)
            if subkey not in modality_meta:
                logger.warning("Subkey %r not in modality.json [%s]; skipping", subkey, modality)
                continue
            meta = modality_meta[subkey]
            result[gr00t_key] = {
                "original_key": meta.original_key,
                "start": meta.start,
                "end": meta.end,
            }
        return result

    # ---- Trajectory management ----

    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        steps = []
        for ep_idx, length in zip(self._trajectory_ids, self._trajectory_lengths):
            for offset in range(int(length)):
                steps.append((int(ep_idx), offset))
        return steps

    def _rebuild_valid(self) -> None:
        valid_base: dict[int, np.ndarray] = {}
        for ep_idx, length in zip(self._trajectory_ids, self._trajectory_lengths):
            ep_idx = int(ep_idx)
            if length > 0:
                valid_base[ep_idx] = np.arange(int(length), dtype=np.int64)
        self._valid_base_indices_by_trajectory = valid_base
        self._valid_trajectory_ids = np.array(list(valid_base.keys()), dtype=np.int64)
        self._valid_trajectory_lengths = np.array(
            [len(v) for v in valid_base.values()], dtype=np.int64
        )
        self._valid_cache_len = len(self._all_steps)

    def _ensure_valid_cache(self) -> None:
        if self._valid_cache_len != len(self._all_steps):
            self._rebuild_valid()

    # ---- Properties ----

    @property
    def dataset_name(self) -> str:
        return self._dataset_path.name

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        return self._all_steps

    @property
    def trajectory_ids(self) -> np.ndarray:
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        return self._trajectory_lengths

    @property
    def valid_trajectory_ids(self) -> np.ndarray:
        self._ensure_valid_cache()
        return self._valid_trajectory_ids

    @property
    def valid_trajectory_lengths(self) -> np.ndarray:
        self._ensure_valid_cache()
        return self._valid_trajectory_lengths

    @property
    def valid_base_indices_by_trajectory(self) -> dict[int, np.ndarray]:
        self._ensure_valid_cache()
        return self._valid_base_indices_by_trajectory

    @property
    def modality_keys(self) -> dict:
        return {
            "video": self._gr00t_video_keys,
            "action": self._action_keys,
            "state": self._state_keys,
            "language": self._language_keys,
        }

    @property
    def lerobot_info_meta(self) -> dict:
        return self._lerobot_info_meta

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._compute_metadata()
        return self._metadata

    def _compute_metadata(self):
        # Reuse gr00t's metadata logic (stats cache is loaded from
        # meta/stats_gr00t.json if it exists, so this is fast after first run).
        from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

        tmp = LeRobotSingleDataset(
            dataset_path=self._dataset_path,
            modality_configs=self._modality_config,
            transforms=self._transforms,
            embodiment_tag=self.tag,
            data_cfg=self._data_cfg,
        )
        return tmp._metadata

    def set_transforms_metadata(self, metadata) -> None:
        self._transforms.set_metadata(metadata)

    # ---- gr00t data protocol ----

    def get_step_data(
        self,
        trajectory_id: int,
        base_index: int,
        selected_video_keys: list[str] | None = None,
    ) -> dict:
        """Return raw step data in gr00t format (no transforms applied)."""
        abs_idx = self._inner._episode_starts[int(trajectory_id)] + base_index
        raw = self._inner[abs_idx]

        data: dict = {}

        video_keys = selected_video_keys if selected_video_keys is not None else self._gr00t_video_keys
        for gr00t_key in video_keys:
            bare_key = gr00t_key.replace("video.", "", 1)
            stream_key = self._bare_to_stream.get(bare_key, bare_key)
            t = raw[stream_key]
            if isinstance(t, torch.Tensor):
                t = t.numpy()
            # torchcodec returns [C, H, W]; gr00t expects [T, H, W, C]
            if t.ndim == 3:  # [C, H, W] → [H, W, C] → [1, H, W, C]
                t = np.transpose(t, (1, 2, 0))
                t = t[np.newaxis]
            elif t.ndim == 4:  # [T, C, H, W] → [T, H, W, C]
                t = np.transpose(t, (0, 2, 3, 1))
            data[gr00t_key] = t

        for gr00t_key, col_info in self._action_col_info.items():
            col = col_info["original_key"]
            start, end = col_info["start"], col_info["end"]
            t = raw[col]
            if isinstance(t, torch.Tensor):
                t = t.numpy()
            if t.ndim == 1:  # scalar column: [T] → [T, 1]
                t = t[:, np.newaxis]
            data[gr00t_key] = t[:, start:end]

        for gr00t_key, col_info in self._state_col_info.items():
            col = col_info["original_key"]
            start, end = col_info["start"], col_info["end"]
            t = raw[col]
            if isinstance(t, torch.Tensor):
                t = t.numpy()
            if t.ndim == 1:  # scalar column: [T] → [T, 1]
                t = t[:, np.newaxis]
            data[gr00t_key] = t[:, start:end]

        task_str = raw.get("task", "") if isinstance(raw, dict) else ""
        for i, lang_key in enumerate(self._language_keys):
            data[lang_key] = [task_str if i == 0 else ""]

        return data

    def _apply_transforms_for_sample(
        self,
        raw_data: dict,
        *,
        selected_video_keys: list[str] | None = None,
    ) -> dict:
        transforms = getattr(self._transforms, "transforms", None)
        if transforms is None:
            return self._transforms(raw_data)

        data = raw_data
        for transform in transforms:
            original_apply_to = list(transform.apply_to)
            if selected_video_keys is not None:
                filtered_apply_to = [
                    k for k in original_apply_to
                    if not k.startswith("video.") or k in selected_video_keys
                ]
                if not filtered_apply_to and any(k.startswith("video.") for k in original_apply_to):
                    continue
                if filtered_apply_to != original_apply_to:
                    transform.apply_to = filtered_apply_to
            try:
                data = transform(data)
            except Exception as exc:
                raise ValueError(f"Transform {type(transform).__name__} failed: {exc}") from exc
            finally:
                transform.apply_to = original_apply_to
        return data

    def _pack_sample(self, data: dict, selected_video_keys: list[str] | None = None) -> dict:
        if selected_video_keys is None:
            selected_video_keys = self._gr00t_video_keys

        images = []
        for vk in selected_video_keys:
            frame = data[vk][0]
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy()
            images.append(Image.fromarray(frame))

        action_parts = []
        for ak in self._action_keys:
            part = data[ak]
            if isinstance(part, torch.Tensor):
                part = part.numpy()
            action_parts.append(part)
        action = np.concatenate(action_parts, axis=1).astype(np.float16)

        lang = ""
        for lk in self._language_keys:
            val = data.get(lk, [""])
            if isinstance(val, (list, tuple, np.ndarray)):
                val = val[0] if len(val) > 0 else ""
            val = str(val).strip()
            if val:
                lang = val
                break

        sample: dict = {
            "action": action,
            "image": images,
            "lang": lang,
            "robot_tag": self.tag,
        }

        if self._data_cfg.get("include_state", False) not in ["False", False]:
            state_parts = []
            for sk in self._state_keys:
                part = data[sk]
                if isinstance(part, torch.Tensor):
                    part = part.numpy()
                state_parts.append(part)
            sample["state"] = np.concatenate(state_parts, axis=1).astype(np.float16)

        return sample

    def _resolve_sampled_video_keys(
        self,
        trajectory_id=None,
        base_index=None,
    ) -> list[str]:
        if not self._video_sample_groups:
            return self._gr00t_video_keys

        # Deterministic RNG seeded by (trajectory_id, base_index) for reproducibility.
        seed = None
        if trajectory_id is not None and base_index is not None:
            seed = int(trajectory_id) * 100003 + int(base_index)
        rng = np.random.default_rng(seed)

        selected: list[str] = []
        consumed: set[str] = set()
        for key in self._gr00t_video_keys:
            if key in consumed:
                continue
            group = self._video_group_lookup.get(key)
            if group is None:
                selected.append(key)
                consumed.add(key)
            else:
                choice = str(rng.choice(group))
                selected.append(choice)
                consumed.update(group)
        return selected

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        # key arrives as bare (e.g. "exterior_1_left"); map to stream name
        stream_key = self._bare_to_stream.get(key, key)
        cache = self._inner._get_current_episode_cache(int(trajectory_id))
        return Path(cache["video_paths"].get(stream_key, "/nonexistent"))

    def __len__(self) -> int:
        return len(self._all_steps)

    def __repr__(self) -> str:
        return f"MarigoldLeRobotSingleDataset({self.dataset_name}, {len(self)} steps)"


# ---- Factory functions ----

def make_MarigoldSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    data_cfg: dict | None = None,
) -> MarigoldLeRobotSingleDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    default_data_cfg = getattr(data_config, "default_data_cfg", None)
    if data_cfg is None:
        data_cfg = {}
    elif not isinstance(data_cfg, dict):
        data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
    if default_data_cfg:
        merged = dict(default_data_cfg)
        merged.update(data_cfg)
        data_cfg = merged

    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        logger.warning(
            "Robot type %r not in ROBOT_TYPE_TO_EMBODIMENT_TAG; using NEW_EMBODIMENT", robot_type
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    dataset_path = Path(data_root_dir) / data_name
    return MarigoldLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_config=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        data_cfg=data_cfg,
        video_backend=data_cfg.get("video_backend"),
    )


def get_marigold_vla_dataset(
    data_cfg,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs,
) -> LeRobotMixtureDataset:
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    holdout_trajectories_per_dataset = int(data_cfg.get("holdout_trajectories_per_dataset", 1))

    included_datasets: set = set()
    dataset_mixture = []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            logger.info("Skipping duplicate dataset: %s", dataset_key)
            continue
        included_datasets.add(dataset_key)

        dataset = make_MarigoldSingleDataset(
            data_root_dir=Path(data_root_dir),
            data_name=d_name,
            robot_type=robot_type,
            data_cfg=data_cfg,
        )
        if mode in {"train", "eval"}:
            dataset = _apply_trajectory_split(
                dataset,
                split=mode,
                holdout_trajectories_per_dataset=holdout_trajectories_per_dataset,
            )
        if len(dataset) == 0:
            logger.info("Skipping %s dataset %s: split is empty", mode, dataset.dataset_name)
            continue
        dataset_mixture.append((dataset, d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import os
    import time
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        default="examples/DROID/train_files/train_droid.yaml",
        help="Path to training YAML config",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    args, _ = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Waiting for debugger on port 10092...")
        debugpy.wait_for_client()

    logging.basicConfig(level=logging.INFO)

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data

    print(f"Building dataset from {args.config_yaml} ...")
    dataset = get_marigold_vla_dataset(data_cfg=vla_dataset_cfg)
    print(f"Dataset length: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"Iterating {args.num_batches} batches (num_workers={args.num_workers}) ...")
    t0 = time.perf_counter()
    errors = 0
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        sample = batch[0]

        # --- key checks ---
        missing = [k for k in ("action", "image", "lang", "robot_tag") if k not in sample]
        if missing:
            print(f"  [step {i}] MISSING KEYS: {missing}")
            errors += 1
            continue

        action = sample["action"]
        images = sample["image"]
        lang = sample["lang"]
        tag = sample["robot_tag"]

        # action shape: [T_action, total_action_dim] float16
        if not isinstance(action, np.ndarray) or action.dtype != np.float16:
            print(f"  [step {i}] BAD action type/dtype: {type(action)} {getattr(action,'dtype',None)}")
            errors += 1
        if action.ndim != 2:
            print(f"  [step {i}] BAD action ndim: {action.ndim} shape={action.shape}")
            errors += 1

        # images: list of PIL Images
        if not isinstance(images, list) or not images:
            print(f"  [step {i}] BAD images: {type(images)}")
            errors += 1
        else:
            for j, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    print(f"  [step {i}] image[{j}] is {type(img)}, not PIL.Image")
                    errors += 1
                    break

        if i == 0 or i % 10 == 0:
            img_sizes = [img.size for img in images] if isinstance(images, list) else "?"
            print(
                f"  step {i:4d}: action={action.shape} dtype={action.dtype}"
                f"  images={img_sizes}  lang={str(lang)[:50]!r}  tag={tag}"
            )

    elapsed = time.perf_counter() - t0
    rate = args.num_batches * args.batch_size / elapsed
    print(f"\n{'OK' if errors == 0 else 'FAILED'} — {args.num_batches} batches in {elapsed:.1f}s "
          f"({rate:.1f} samples/s, {errors} errors)")

#!/usr/bin/env python
"""m3_data loader bridge for StarVLA (successor of marigold_data_datasets.py).

Uses ``m3_data.dataset.InfiniteDataReader`` (dense ``m3s.2`` samples: observation [T_obs, 338], action
[T_act, 230], per-dim / per-timestep masks, uint8 camera stacks) as the training dataset and converts each
sample to StarVLA's list-of-dicts VLA contract.

Differences from the marigold bridge:
  * the RIG-VLA lerobot fork and the m3_data checkout are put on ``sys.path`` HERE, so only runs with
    ``dataset_py: m3_data_datasets`` see them; every other run keeps the container's stock lerobot. The
    paths are exported through the environment so DataLoader workers (forkserver / spawn) see them too;
  * field selection by m3 registry group name (``m3_action_fields`` / ``m3_state_fields``), defaulting
    to the ABC-130k bimanual YAM layout [left arm 6, left gripper, right arm 6, right gripper];
  * per-embodiment q01/q99 normalisation from m3 norm-stats files (contract section 8), continuous
    grippers (no binary last dim);
  * ``action_valid_mask`` is forwarded (False where the horizon overran the episode end);
  * ABC-130k real roots can train on the COMMANDED actions (``abc130k_yam_*_cmd`` dataset_types,
    registered here via the public ``m3_data.adapters.registry.register``; the m3_data repo is untouched).

YAML (datasets.vla_data):
    dataset_py: m3_data_datasets
    m3_data_config: <user data config JSON>          # m3_data format (datasets list + global keys)
    m3_datasets_root: /e/home/.../datasets/lerobot_3_0   # replaces the "/datasets" prefix of roots
    m3_norm_stats_dir: <dir with <embodiment>.json>  # from m3_data scripts/compute_norm_stats.py
    m3_camera_order: [camera_head, camera_wrist_left, camera_wrist_right]
    m3_action_fields: [[arm_left.joint_pos_target, 0, 6], ...]   # default: ABC 14-D
    m3_state_fields:  [[arm_left.joint_pos, 0, 6], ...]
    m3_repo / m3_lerobot_src: checkout paths (defaults below)
    m3_instruction_map: {<embodiment>: {<m3 instruction>: <training prompt>}}   # optional, see below
    augmentation: none | <augment_cot_sample mode>   # StarVLA-side augmentation, train only

``m3_instruction_map`` rewrites the instruction of every sample of a LISTED embodiment (strict: an instruction
missing from that embodiment's map raises); other embodiments pass through unchanged. Used for ABC real + sim
co-training (ABC's protocol: sim episodes are prompted ``"sim " + task prompt``). The map is also written into
dataset_statistics.json (``<embodiment>.m3.instruction_map``), where the eval side reads the trained prompts.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

_DEFAULT_M3_REPO = "/e/project1/m3/blank4/code/m3_data"
_DEFAULT_LEROBOT_FORK_SRC = "/e/project1/m3/blank4/code/lerobot_marigold/lerobot/src"
_ENV_REPO = "STARVLA_M3_REPO"
_ENV_FORK = "STARVLA_M3_LEROBOT_SRC"

# ABC-130k bimanual YAM, in the dataset's own [L1..L6, L_grip, R1..R6, R_grip] order (what ABC-Sim takes).
_ABC_ACTION_FIELDS = [
    ["arm_left.joint_pos_target", 0, 6],
    ["gripper_left.opening_target", 0, 1],
    ["arm_right.joint_pos_target", 0, 6],
    ["gripper_right.opening_target", 0, 1],
]
_ABC_STATE_FIELDS = [
    ["arm_left.joint_pos", 0, 6],
    ["gripper_left.opening", 0, 1],
    ["arm_right.joint_pos", 0, 6],
    ["gripper_right.opening", 0, 1],
]
_ABC_CAMERAS = ["camera_head", "camera_wrist_left", "camera_wrist_right"]
# LeRobot feature name (observation.images.<name>) of each canonical ABC camera, for the native frame shapes
_ABC_NATIVE_CAMERAS = {"camera_head": "top", "camera_wrist_left": "left_wrist", "camera_wrist_right": "right_wrist"}


def _native_camera_shapes(user_cfg: dict, cameras: list, native_names: dict) -> dict:
    """{(embodiment, canonical camera): (height, width)} from each dataset's meta/info.json.

    m3 decodes every stream straight to videos_hw (a square stretch); the native shape is what letterboxing
    needs to undo that. All roots of one embodiment must agree."""
    shapes: dict = {}
    for entry in user_cfg.get("datasets", []):
        emb = str(entry.get("embodiment_key") or entry.get("dataset_type"))
        for inc in entry.get("include") or [""]:
            info = _load_json(Path(entry["root"]) / inc / "meta" / "info.json")
            for cam in cameras:
                feat = info["features"].get(f"observation.images.{native_names.get(cam, cam)}")
                if feat is None:
                    raise KeyError(f"{entry['root']}/{inc}: no observation.images.{native_names.get(cam, cam)} for {cam}")
                hw = (int(feat["shape"][0]), int(feat["shape"][1]))
                if shapes.setdefault((emb, cam), hw) != hw:
                    raise ValueError(f"{emb}/{cam}: native shapes differ across roots ({shapes[(emb, cam)]} vs {hw})")
    return shapes


def _letterbox(image: Image.Image, native_hw: tuple) -> Image.Image:
    """Undo the square stretch and pad: resize to the native aspect inside the same square, centred on black
    (ABC: "Non-square images are padded using letterbox padding")."""
    side_w, side_h = image.size
    h, w = native_hw
    if h == w:
        return image
    if w > h:
        new_w, new_h = side_w, max(1, round(side_h * h / w))
    else:
        new_w, new_h = max(1, round(side_w * w / h)), side_h
    canvas = Image.new(image.mode, (side_w, side_h))
    canvas.paste(image.resize((new_w, new_h), Image.Resampling.BILINEAR), ((side_w - new_w) // 2, (side_h - new_h) // 2))
    return canvas


# ── opt-in path setup ─────────────────────────────────────────────────────────
def _ensure_m3_on_path(repo: str | None = None, fork_src: str | None = None) -> None:
    repo = str(repo or os.environ.get(_ENV_REPO) or _DEFAULT_M3_REPO)
    fork_src = str(fork_src or os.environ.get(_ENV_FORK) or _DEFAULT_LEROBOT_FORK_SRC)
    loaded = sys.modules.get("lerobot")
    if loaded is not None:
        where = str(Path(getattr(loaded, "__file__", "") or "").resolve())
        if not where.startswith(str(Path(fork_src).resolve())):
            raise RuntimeError(
                f"lerobot was already imported from {where} before the m3_data bridge could put the RIG-VLA "
                f"fork ({fork_src}) first on sys.path; m3_data needs the fork's LeRobotTrainingDataset"
            )
    for path in (repo, fork_src):  # move to the front even if already present further back; fork ends up first
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    os.environ[_ENV_REPO] = repo
    os.environ[_ENV_FORK] = fork_src
    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(":") if p and p not in (repo, fork_src)]
    os.environ["PYTHONPATH"] = ":".join([fork_src, repo] + parts)


# ── commanded-action ABC-130k adapters ────────────────────────────────────────
_COMMANDED_TARGETS = {
    "arm_left.joint_pos_target": (0, 6),
    "arm_right.joint_pos_target": (7, 13),
    "gripper_left.opening_target": (6, 7),
    "gripper_right.opening_target": (13, 14),
}
_ABC_REAL_TYPES = (
    "abc130k_yam_d405_640", "abc130k_yam_d405_640_pose", "abc130k_yam_d405_848",
    "abc130k_yam_d405_848_pose", "abc130k_yam_d405_720p_pose", "abc130k_yam_oak_640",
    "abc130k_yam_oak_640_pose", "abc130k_yam_decxin_1280", "abc130k_yam_decxin_800", "abc130k_yam_zedx",
)


def _commanded_subclass(base: type) -> type:
    from m3_data.adapters.specs import FieldSpec

    class _Commanded(base):
        """ABC-130k real roots with the on-disk absolute servo targets (`action`, [L1..L6, L_grip, R1..R6,
        R_grip], same layout and gripper polarity as observation.state) as the joint / opening targets;
        the velocity (and EE) targets stay derived. Same owner-waiver pattern as abc130k_sim ABS-5;
        `derived` stays the default."""

        ACTION_SOURCE_MODES = ("derived", "commanded")
        DEFAULT_ACTION_SOURCE = "derived"

        @property
        def action_specs(self):
            specs = tuple(super().action_specs)
            if self.action_source != "commanded":
                return specs
            width = {"arm_left.joint_pos_target": 6, "arm_right.joint_pos_target": 6,
                     "gripper_left.opening_target": 1, "gripper_right.opening_target": 1}
            return specs + tuple(
                FieldSpec(group, "action", source_slice=src, group_slice=(0, width[group]),
                          source_kind="commanded")
                for group, src in _COMMANDED_TARGETS.items()
            )

        @property
        def derived_specs(self):
            specs = tuple(super().derived_specs)
            if self.action_source != "commanded":
                return specs
            return tuple(d for d in specs if d.group not in _COMMANDED_TARGETS)

    _Commanded.__name__ = f"{base.__name__}Commanded"
    _Commanded.__qualname__ = _Commanded.__name__
    _Commanded.__module__ = __name__
    return _Commanded


def _register_abc_commanded() -> None:
    from m3_data.adapters import registry

    for dataset_type in _ABC_REAL_TYPES:
        name = f"{dataset_type}_cmd"
        if name in registry._REGISTRY:
            continue
        cls = _commanded_subclass(registry.get_adapter_cls(dataset_type))
        globals()[cls.__name__] = cls  # importable by name (pickle-safe)
        registry.register(name, cls)


# Workers re-import this module when they unpickle the dataset: restore the paths and the registrations.
if os.environ.get(_ENV_REPO) or os.environ.get(_ENV_FORK):
    try:
        _ensure_m3_on_path()
        _register_abc_commanded()
    except Exception as exc:  # pragma: no cover - surfaced again when the reader is built
        logger.warning("[m3_data] worker bootstrap failed: %s", exc)


# ── helpers ───────────────────────────────────────────────────────────────────
def collate_fn(batch):
    return batch


def _as_plain_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _field_indices(space: str, fields: list) -> list[int]:
    from m3_data.spaces.registry import SPACES, offsets

    off = offsets(SPACES[space])
    out: list[int] = []
    for group, start, end in fields:
        if group not in off:
            raise KeyError(f"unknown m3 {space} group {group!r}")
        sl = off[group]
        if not 0 <= int(start) < int(end) <= sl.stop - sl.start:
            raise ValueError(f"{space} field {group}[{start}:{end}] outside the group width {sl.stop - sl.start}")
        out.extend(range(sl.start + int(start), sl.start + int(end)))
    return out


def _group_stats_vector(payload: dict, space: str, fields: list, key: str) -> np.ndarray:
    groups = payload.get(space, {})
    vals = []
    for group, start, end in fields:
        if group not in groups:
            raise KeyError(f"norm stats have no {space} group {group!r}")
        vals.extend(float(x) for x in groups[group][key][int(start):int(end)])
    return np.asarray(vals, dtype=np.float32)


def _image_to_pil(value: Any, frame_index: int = -1) -> Image.Image:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 4:
        tensor = tensor[frame_index]
    if tensor.ndim != 3:
        raise ValueError(f"expected an image [C,H,W] or [N,C,H,W], got {tuple(tensor.shape)}")
    arr = tensor.permute(1, 2, 0).numpy() if tensor.shape[0] in (1, 3, 4) else tensor.numpy()
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def _rewrite_roots(user_cfg: dict, datasets_root: str | None) -> dict:
    if not datasets_root:
        return user_cfg
    for ds in user_cfg.get("datasets", []):
        root = str(ds.get("root", ""))
        if root == "/datasets" or root.startswith("/datasets/"):
            ds["root"] = str(Path(datasets_root) / root[len("/datasets/"):]) if root != "/datasets" else datasets_root
    return user_cfg


def _override_key(node: Any, key: str, value: Any) -> None:
    """Set `key` to `value` everywhere in a nested dict / list (processed m3 configs copy sample_seed per item)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
            else:
                _override_key(v, key, value)
    elif isinstance(node, list):
        for v in node:
            _override_key(v, key, value)


def _load_processed_config(user_cfg: dict, cache_root: str | Path) -> dict:
    from m3_data.utils.config import (
        init_data_config,
        process_data_config,
        processed_data_config_cache_path,
        save_data_config,
    )

    if not _dist_ready():
        return process_data_config(init_data_config(user_cfg), user_cfg)
    payload: list[dict | None] = [None]
    if dist.get_rank() == 0:
        cache_path = processed_data_config_cache_path(user_cfg, cache_root_dir=cache_root)
        if cache_path.exists():
            processed, source = _load_json(cache_path), f"cached processed config {cache_path}"
        else:
            processed = process_data_config(init_data_config(user_cfg), user_cfg)
            save_data_config(processed, cache_path)
            source = f"freshly built and cached {cache_path}"
        logger.warning("[m3_data] using %s", source)
        payload[0] = processed
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _embodiments(processed: dict) -> list[str]:
    seen: list[str] = []
    for ds in processed.get("datasets", []):
        subs = ds.get("subdatasets", {})
        for sub in (subs.values() if isinstance(subs, dict) else subs):
            emb = sub.get("embodiment_key") or ds.get("embodiment_key")
            if emb and emb not in seen:
                seen.append(emb)
        if not subs and ds.get("embodiment_key") and ds["embodiment_key"] not in seen:
            seen.append(ds["embodiment_key"])
    return seen


class _Q99Normalizer:
    """Per-embodiment q01/q99 scaling to [-1, 1] on the selected dims (constant dims pass through)."""

    def __init__(self, stats_dir: str | None, embodiments: list[str], action_fields: list, state_fields: list,
                 enabled: bool = True):
        self.enabled = bool(enabled)
        self.stats: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        if not self.enabled:
            return
        if not stats_dir:
            raise ValueError("m3_norm_stats_dir is required (or set m3_normalize: false)")
        from m3_data.norm_stats import load_norm_stats

        for emb in embodiments:
            path = Path(stats_dir) / f"{emb}.json"
            if not path.exists():
                raise FileNotFoundError(f"no m3 norm stats for embodiment {emb!r}: {path}")
            payload = load_norm_stats(path)
            self.stats[emb] = {
                "action": {k: _group_stats_vector(payload, "action", action_fields, k) for k in ("q01", "q99", "mean", "std")},
                "state": {k: _group_stats_vector(payload, "observation", state_fields, k) for k in ("q01", "q99", "mean", "std")},
            }

    def __call__(self, x: np.ndarray, emb: str, kind: str) -> np.ndarray:
        out = x.astype(np.float32, copy=True)
        if not self.enabled:
            return out
        st = self.stats.get(emb)
        if st is None:
            raise KeyError(f"sample embodiment {emb!r} has no norm stats loaded")
        q01, q99 = st[kind]["q01"], st[kind]["q99"]
        live = q99 > q01
        out[..., live] = np.clip(2.0 * (out[..., live] - q01[live]) / (q99[live] - q01[live]) - 1.0, -1.0, 1.0)
        return out

    def unnormalize_actions(self, x: np.ndarray, emb: str) -> np.ndarray:
        """Inverse of the action scaling (the trainer's open-loop plots call this with the per-episode
        ``action_indices`` token, which for this bridge is the embodiment key)."""
        out = np.asarray(x, dtype=np.float32).copy()
        if not self.enabled:
            return out
        st = self.stats.get(str(emb))
        if st is None:
            raise KeyError(f"embodiment {emb!r} has no norm stats loaded")
        q01, q99 = st["action"]["q01"], st["action"]["q99"]
        live = q99 > q01
        out[..., live] = (out[..., live] + 1.0) * 0.5 * (q99[live] - q01[live]) + q01[live]
        return out


# ── dataset ───────────────────────────────────────────────────────────────────
class StarVLAM3DataReader(IterableDataset):
    def __init__(self, data_cfg, mode: str = "train"):
        cfg = _as_plain_dict(data_cfg)
        self.data_cfg = cfg
        self.train = mode == "train"
        _ensure_m3_on_path(cfg.get("m3_repo"), cfg.get("m3_lerobot_src"))
        _register_abc_commanded()

        from m3_data.dataset import InfiniteDataReader

        explicit = cfg.get("m3_data_config")
        if not explicit:
            raise ValueError("datasets.vla_data.m3_data_config (an m3_data user data config JSON) is required")
        user_cfg = _rewrite_roots(_load_json(explicit), cfg.get("m3_datasets_root"))
        for key in ("target_fps", "num_obs_steps", "num_action_steps", "frame_stride_train", "max_cameras"):
            if cfg.get(f"m3_{key}") is not None:
                user_cfg[key] = cfg[f"m3_{key}"]
        # Future frame for future copies / masked-query EMA targets: offset in target-fps frames.
        self.future_offset = int(cfg.get("shared_z_future_offset") or 0)
        if cfg.get("future_offsets"):
            raise ValueError("m3_data_datasets supports one future frame (shared_z_future_offset), not future_offsets")
        if self.future_offset > 0:
            user_cfg["image_relative_frames"] = [0, self.future_offset]
        resume_steps = int(cfg.get("m3_resume_steps") or 0)
        if resume_steps > 0:
            # fresh stream per resumed segment (set by train_starvla.configure_dataloader_resume)
            user_cfg["sample_seed"] = int(user_cfg.get("sample_seed", 0)) + 1_000_003 * resume_steps
        if cfg.get("action_horizon") is not None and cfg.get("m3_num_action_steps") is None:
            user_cfg["num_action_steps"] = int(cfg["action_horizon"])
        cache_root = cfg.get("m3_cache_root") or str(Path(explicit).resolve().parent / "processed_cache")
        self.processed = _load_processed_config(user_cfg, cache_root)
        if resume_steps > 0:
            _override_key(self.processed, "sample_seed", user_cfg["sample_seed"])  # whatever the cache held

        self.action_fields = cfg.get("m3_action_fields") or _ABC_ACTION_FIELDS
        self.state_fields = cfg.get("m3_state_fields") or _ABC_STATE_FIELDS
        self.action_idx = _field_indices("action", self.action_fields)
        self.state_idx = _field_indices("observation", self.state_fields)
        self.cameras = list(cfg.get("m3_camera_order") or _ABC_CAMERAS)
        # m3_letterbox: undo m3's square stretch and pad to square at the native aspect (as ABC trains)
        self.letterbox = str(cfg.get("m3_letterbox", False)).lower() in ("true", "1")
        self.native_hw = (_native_camera_shapes(user_cfg, self.cameras,
                                                dict(cfg.get("m3_native_camera_names") or _ABC_NATIVE_CAMERAS))
                          if self.letterbox else {})
        self.include_state = bool(cfg.get("include_state", True))
        self.augmentation = str(cfg.get("augmentation", "none") or "none").lower()
        self.embodiments = _embodiments(self.processed)
        self.normalizer = _Q99Normalizer(
            cfg.get("m3_norm_stats_dir"), self.embodiments, self.action_fields, self.state_fields,
            enabled=cfg.get("m3_normalize", True) not in (False, "false", "False", 0, "0"),
        )
        self.reader = InfiniteDataReader(self.processed, train=self.train,
                                         video_backend=cfg.get("video_backend") or None)
        self.robot_tag = str(cfg.get("m3_robot_tag") or "abc130k_yam")
        self._warned_missing: set = set()
        imap = cfg.get("m3_instruction_map") or {}
        self.instruction_map = {str(e): {str(k): str(v) for k, v in (m or {}).items()} for e, m in imap.items()}
        unknown = sorted(set(self.instruction_map) - set(self.embodiments))
        if unknown:
            raise ValueError(f"m3_instruction_map names embodiments {unknown} that the data config does not have "
                             f"({self.embodiments})")

    def __len__(self) -> int:
        total = 0
        for ds in self.processed.get("datasets", []):
            subs = ds.get("subdatasets", {})
            for sub in (subs.values() if isinstance(subs, dict) else subs):
                total += int(sub.get("total_frames", 0) or 0)
        stride = max(1, int(self.processed.get("frame_stride_train", 1)))
        return max(1, total // stride)

    def __iter__(self):
        _register_abc_commanded()  # worker processes
        for sample in self.reader:
            yield self._convert(sample)

    # ── open-loop trajectory eval (trainer._eval_marigold_open_loop_trajectories) ─────────────────
    # Mirrors marigold_data_datasets.StarVLAMarigoldDataReader.build_open_loop_episodes /
    # materialize_open_loop_episode on m3_data's adapter API. Used only by an eval-mode reader; the
    # training iterator above never touches it.
    def _get_open_loop_states(self, names: list[str] | None = None) -> list:
        """(subdataset name, finite eval-mode SubdatasetState) per subdataset, built lazily and cached.

        ``train=False`` keeps the episode order unshuffled and applies the subdataset's ``holdout`` block
        exactly as the m3 eval reader does (mode ``only`` -> held-out episodes; no block -> all episodes)."""
        if self.train:
            raise RuntimeError("Open-loop episodes require an m3 reader created with mode='eval'")
        from m3_data.dataset import SubdatasetState
        from m3_data.utils.sharding_plan import flatten_subdatasets

        cache = self.__dict__.setdefault("_open_loop_states", {})
        out = []
        for entry in flatten_subdatasets(self.processed):
            name = entry["name"]
            if names is not None and name not in names:
                continue
            if name not in cache:
                cache[name] = SubdatasetState(entry["cfg"], train=False,
                                              video_backend=self.data_cfg.get("video_backend") or None,
                                              subdataset_name=name)
            out.append((name, cache[name]))
        return out

    def build_open_loop_episodes(
        self,
        *,
        action_horizon: int,
        num_episodes_per_subdataset: int = 1,
        max_anchors: int | None = None,
        subdataset_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fixed episodes (the last eligible ones of each subdataset) cut into non-overlapping action chunks.

        Anchors come from the adapter's own ``plan_episode_samples`` (global instruction spans ∩ quality
        spans, unshuffled) at a stride of one action chunk in SOURCE frames, so consecutive chunks tile the
        episode. ``subdataset_names`` (or ``datasets.vla_data.m3_open_loop_subdatasets``) limits the
        subdatasets; default all. Not a held-out split unless the m3 config carries a ``holdout`` block."""
        from m3_data.retiming import round_half_away, target_to_source_position

        if num_episodes_per_subdataset < 1:
            raise ValueError("num_episodes_per_subdataset must be >= 1")
        if max_anchors is not None and max_anchors < 1:
            raise ValueError("max_anchors must be >= 1 when provided")
        if subdataset_names is None and self.data_cfg.get("m3_open_loop_subdatasets"):
            subdataset_names = list(self.data_cfg["m3_open_loop_subdatasets"])

        episodes: list[dict[str, Any]] = []
        for name, state in self._get_open_loop_states(subdataset_names):
            sampler = state.sampler
            if int(sampler.num_action_steps) != int(action_horizon):
                raise ValueError(f"m3/model action-horizon mismatch for {name}: "
                                 f"sampler={sampler.num_action_steps}, model={action_horizon}")
            target_stride = int(sampler.action_frame_stride) * int(action_horizon)
            anchor_stride = max(1, int(round_half_away(
                target_to_source_position(target_stride, sampler._src_fps, sampler._tgt_fps))))
            found = 0
            for episode_idx in reversed(list(state._eligible_episode_indices())):
                info = sampler.get_episode_info(episode_idx)
                anchors = sampler.plan_episode_samples(episode_idx, train=False, frame_stride=anchor_stride,
                                                       episode_info=info)
                if not anchors:
                    continue
                if max_anchors is not None:
                    anchors = anchors[:max_anchors]
                episodes.append({
                    "dataset_name": str(name),
                    "state": state,
                    "episode_idx": int(episode_idx),
                    "episode_info": info,
                    "uuid": sampler.episode_uuid(info),
                    "anchors": [int(a) for a in anchors],
                    "anchor_stride_source_frames": anchor_stride,
                })
                found += 1
                if found >= num_episodes_per_subdataset:
                    break
            if found != num_episodes_per_subdataset:
                raise RuntimeError(f"Requested {num_episodes_per_subdataset} open-loop episodes from {name}, "
                                   f"found {found}")
        return episodes

    def materialize_open_loop_episode(self, episode: dict[str, Any], *, seed: int) -> list[dict[str, Any]]:
        """Decode every anchor and convert it with the training ``_convert`` (eval mode: no augmentation),
        without perturbing the global RNG state. ``action_indices`` carries the embodiment key, which is
        what ``normalizer.unnormalize_actions`` needs here."""
        import random

        state = episode["state"]
        sampler = state.sampler
        info = episode["episode_info"]
        episode_start = int(info["dataset_from_index"])
        action_source_offsets = np.asarray(sampler._act_source_idx, dtype=np.int64)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        samples: list[dict[str, Any]] = []
        try:
            for anchor in episode["anchors"]:
                raw = sampler.materialize_sample(episode_idx=episode["episode_idx"], anchor_idx=int(anchor),
                                                 image_aug=None, episode_info=info)
                converted = self._convert(raw)
                gt_normalized = np.asarray(converted["action"], dtype=np.float32)
                mask = np.asarray(converted["action_valid_mask"], dtype=bool)
                if mask.shape[0] != gt_normalized.shape[0]:
                    raise RuntimeError(f"m3 action-mask mismatch at anchor {anchor}: "
                                       f"{mask.shape} vs {gt_normalized.shape}")
                emb = str(raw.get("embodiment", ""))
                samples.append({
                    "anchor": int(anchor),
                    "example": converted,
                    "gt_normalized": gt_normalized,
                    "gt_physical": self.normalizer.unnormalize_actions(gt_normalized, emb),
                    "valid_mask": mask,
                    "source_frames": (int(anchor) - episode_start) + action_source_offsets,
                    "action_indices": emb,
                })
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        return samples

    def _check_filled(self, sample: dict, space: str, idx: list[int]) -> None:
        mask = sample.get(f"{space}_mask")
        if mask is None:
            return
        m = torch.as_tensor(mask, dtype=torch.bool).flatten().numpy()
        missing = [i for i in idx if not m[i]]
        if missing:
            key = (sample.get("embodiment"), space)
            raise ValueError(f"{space} dims {missing} are not filled for embodiment {key[0]!r}; "
                             f"check m3_{'action' if space == 'action' else 'state'}_fields and action_source")

    def _convert(self, sample: dict) -> dict:
        emb = str(sample.get("embodiment", ""))
        self._check_filled(sample, "action", self.action_idx)
        actions = torch.as_tensor(sample["action"]).float().numpy()[:, self.action_idx]
        actions = self.normalizer(actions, emb, "action").astype(np.float16)

        images = sample.get("images", {})
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise KeyError(f"sample {sample.get('uuid')} lacks cameras {missing}; has {list(images)}")
        # image_relative_frames is sorted, so frame 0 is the current one and frame 1 the future one
        pil = [_image_to_pil(images[c], 0) for c in self.cameras]
        future = [_image_to_pil(images[c], 1) for c in self.cameras] if self.future_offset > 0 else []
        if self.train and self.augmentation != "none":
            from starVLA.dataloader.cot_augmentation import augment_cot_sample

            # one draw for current and future frames, so the future target differs only by time
            both, _ = augment_cot_sample(pil + future, None, mode=self.augmentation)
            pil, future = both[:len(pil)], both[len(pil):]
        if self.letterbox:
            # after augmentation, as ABC (crop on the image, then resize_with_pad): the bars stay pure black and
            # match what the eval adapter produces from clean frames
            hw = [self.native_hw[(emb, c)] for c in self.cameras]
            pil = [_letterbox(im, x) for im, x in zip(pil, hw)]
            future = [_letterbox(im, x) for im, x in zip(future, hw)]

        lang = str(sample.get("instruction", ""))
        imap = self.instruction_map.get(emb)
        if imap is not None:
            if lang not in imap:
                raise KeyError(f"m3_instruction_map[{emb!r}] has no entry for instruction {lang!r} "
                               f"(sample {sample.get('uuid')}); known: {sorted(imap)}")
            lang = imap[lang]

        result = {
            "image": pil,
            "lang": lang,
            "action": actions,
            "action_valid_mask": torch.as_tensor(sample.get("action_valid_mask", np.ones(len(actions), bool)),
                                                 dtype=torch.bool).numpy(),
            "robot_tag": emb,
            "trajectory_name": str(sample.get("uuid") or sample.get("root", "")),
            # The key tells QwenGR00T the workers already augmented (no second augmentation pass).
            "cot_conversation": None,
        }
        if future:
            # Past the episode end m3 repeats the last frame (keep_tail), as the LIBERO loader does.
            result["future_image"] = future
        if self.include_state:
            self._check_filled(sample, "observation", self.state_idx)
            obs = torch.as_tensor(sample["observation"]).float().numpy()[-1:, self.state_idx]
            result["state"] = self.normalizer(obs, emb, "state").astype(np.float16)
        self._log_sample_identity(result)
        return result

    def _log_sample_identity(self, result: dict) -> None:
        """STARVLA_M3_SAMPLE_LOG=<dir>: one line per sample (rank, worker, episode uuid, hash of state + actions),
        for checking that ranks and workers read distinct samples. Off by default."""
        out_dir = os.environ.get("STARVLA_M3_SAMPLE_LOG")
        if not out_dir:
            return
        import hashlib

        from torch.utils.data import get_worker_info

        info = get_worker_info()
        rank, worker = os.environ.get("RANK", "0"), (info.id if info is not None else -1)
        digest = hashlib.sha1(np.ascontiguousarray(result["action"]).tobytes()
                              + np.ascontiguousarray(result.get("state", np.zeros(1))).tobytes()).hexdigest()[:16]
        with open(os.path.join(out_dir, f"rank{rank}_worker{worker}.txt"), "a") as f:
            f.write(f"{rank} {worker} {result['trajectory_name']} {digest}\n")

    def save_dataset_statistics(self, output_path: str | Path) -> None:
        """StarVLA dataset_statistics.json, one entry per embodiment (what the policy server un-normalises with)."""
        out = {}
        for emb, st in self.normalizer.stats.items():
            entry = {}
            for kind in ("action", "state"):
                s = st[kind]
                entry[kind] = {
                    "q01": s["q01"].tolist(), "q99": s["q99"].tolist(),
                    "mean": s["mean"].tolist(), "std": s["std"].tolist(),
                    "min": s["q01"].tolist(), "max": s["q99"].tolist(),
                }
            entry["action"]["mask"] = [True] * len(self.action_idx)
            entry["m3"] = {"action_fields": self.action_fields, "state_fields": self.state_fields,
                           "cameras": self.cameras, "layout_version": "m3s.2",
                           # eval must letterbox the same way: native (h, w) per camera, then pad to square
                           "letterbox": self.letterbox,
                           "native_hw": {c: list(self.native_hw[(emb, c)]) for c in self.cameras}
                           if self.letterbox else None}
            if emb in self.instruction_map:  # the prompts this embodiment trained under (eval must use them)
                entry["m3"]["instruction_map"] = dict(self.instruction_map[emb])
            out[emb] = entry
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")


def get_m3_data_vla_dataset(data_cfg, mode: str = "train", **_: Any) -> StarVLAM3DataReader:
    return StarVLAM3DataReader(data_cfg, mode=mode)

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import fcntl
import json
import logging
import math
import os
import pathlib
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2 as cv
import re
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

if os.environ.get("STARVLA_FAST_GLASS_BLUR", "0") == "1":
    from fast_glass_blur import install as install_fast_glass_blur
    install_fast_glass_blur()
    print("Validated fast LIBERO-Plus glass blur enabled", flush=True)

EVAL_FILES_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(EVAL_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_FILES_DIR))

from model2libero_interface import ModelClient
import online_obj_mask  # default off (STARVLA_ONLINE_OBJ_MASK=1); see online_obj_mask.py
import episode_trace  # default off (STARVLA_EPISODE_TRACE=<dir>); see episode_trace.py

try:
    import draccus
except ModuleNotFoundError:
    # The Apptainer simulator image intentionally contains only the evaluation
    # runtime.  Keep the existing draccus CLI when it is installed (the conda
    # path), and use the stdlib parser below in the lean SIF path.
    draccus = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _should_send_state(client_model) -> bool:
    """Whether to attach proprioceptive state to each request.

    The checkpoint already answers this: the policy server derives
    ``expects_state`` from the checkpoint's explicit token and/or continuous state
    pathways (see ``deployment/model_server/policy_wrapper.py``) and ships it in
    the handshake, which ``ModelClient`` parses. Follow it.

    This used to be gated solely on ``STARVLA_EVAL_INCLUDE_STATE``. On the
    vanilla-LIBERO path nothing ever set it, so every state-trained checkpoint
    was silently evaluated blind to proprioception; measured 2026-09-21 on
    ervla_state_gr00t_*_v2 @30k, 800 episodes per cell, paired: the encoder arm
    scored 91.9% without state and 98.1% with it (+14.0pp on libero_10 alone).
    This path was not silently broken -- eval_libero_in_one.sh forwards the var --
    but it defaulted it to "0", so the same wrong answer was one forgotten export
    away.

    The env var remains an explicit override in both directions ("1" forces state
    on, "0" forces it off), so a state-trained model can still be deliberately
    evaluated blind as an ablation. Unset or empty means "ask the checkpoint".

    NOTE: the in-process ``PolicyModel`` path (``--use_server false``) exposes no
    handshake, so it reports False here and needs the explicit "1" to send state.
    """
    override = os.environ.get("STARVLA_EVAL_INCLUDE_STATE", "").strip()
    if override:
        return override == "1"
    return bool(getattr(client_model, "expects_state", False))


def _binarize_gripper(open_val: np.ndarray | float, encoding: str) -> np.ndarray:
    """Map the model's gripper output to the LIBERO action convention.

    LIBERO/robosuite: action[6] = +1 closes, -1 opens (cf. LIBERO_DUMMY_ACTION,
    which holds -1 while objects settle, i.e. open).

    Two training encodings exist, distinguished by the action stats in the
    checkpoint's ``dataset_statistics.json``. Dim 6 is excluded from
    ``normalization_modes`` in the data config, so the raw dataset value passes
    through the model un-normalized and its convention reaches us verbatim:

      ``zero_one``:       gripper in [0, 1], 1 = open   (libero_*_no_noops_1.0.0_lerobot)
      ``pm_one``:         gripper in [-1, 1], +1 = close (lerobot_3_0/libero_lerobot,
                          which stores raw LIBERO actions)
      ``zero_one_close``: gripper in [0, 1], 1 = close  -- produced when the data config
                          sets ``"action.gripper": "binary"``. Cannot be auto-detected:
                          dataset_statistics.json still records the RAW min/max, so pass
                          it explicitly.

    These invert each other, and normalization would not reconcile them (min_max
    is order-preserving), so the convention must come from the checkpoint.
    Getting it wrong inverts the gripper and the policy can never grasp.
    """
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    if encoding == "zero_one":
        bin_val = 1.0 - 2.0 * (v > 0.5)  # 1 (open) -> -1, 0 (close) -> +1
    elif encoding == "pm_one":
        bin_val = 1.0 if v > 0.0 else -1.0  # already LIBERO convention; just binarize
    elif encoding == "zero_one_close":
        bin_val = 1.0 if v > 0.5 else -1.0  # 1 (close) -> +1, 0 (open) -> -1
    else:
        raise ValueError(f"Unknown gripper encoding: {encoding!r}")
    return np.asarray([bin_val], dtype=np.float32)


def _detect_gripper_encoding(ckpt_path: str | None) -> str:
    """Infer the gripper encoding from the checkpoint's dataset_statistics.json.

    A dim-6 minimum below -0.5 means raw LIBERO actions (``pm_one``); a minimum
    at ~0 means the [0, 1] encoding (``zero_one``). Returns "" if not found.
    """
    if not ckpt_path:
        return ""
    for parent in pathlib.Path(ckpt_path).resolve().parents:
        stats_file = parent / "dataset_statistics.json"
        if stats_file.is_file():
            with open(stats_file, encoding="utf-8") as f:
                stats = json.load(f)
            key = next(iter(stats))
            gripper_min = float(stats[key]["action"]["min"][6])
            return "pm_one" if gripper_min < -0.5 else "zero_one"
    return ""


def get_logger(file):

    logger = logging.getLogger("dual_logger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


_TRACE_RE = re.compile(r"<\|trace\|>(.*?)<\|/trace\|>", re.DOTALL)


def _parse_trace(cot_text: str | None) -> list[tuple[float, float]]:
    """Pull the 2D trace out of a CoT string as normalized 0-1 coordinates.

    Training targets encode points in a 0-1000 grid over the image the model was
    shown, so dividing by 1000 makes them resolution-independent. Returns [] for
    anything unparseable -- overlay is diagnostic and must never break a rollout.
    """
    if not cot_text:
        return []
    m = _TRACE_RE.search(cot_text)
    if not m:
        return []
    arr = re.search(r"\[\s*\[.*?\]\s*\]", m.group(1), re.DOTALL)
    if not arr:
        return []
    nums = re.findall(r"-?\d+(?:\.\d+)?", arr.group(0))
    return [
        (float(nums[i]) / 1000.0, float(nums[i + 1]) / 1000.0)
        for i in range(0, len(nums) - 1, 2)
    ]


def _overlay_trace(frame: np.ndarray, points, fresh: bool = False) -> np.ndarray:
    """Draw a predicted trace on a copy of `frame`. Never mutates the input.

    The frame handed to the policy must stay pixel-identical, so this always
    copies. Points are colour-graded start (green) -> end (red) so the direction
    of the predicted motion is readable, and the frame where the chunk was
    re-predicted gets a white border.
    """
    if not points:
        return frame
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame

    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    px = [(max(0.0, min(1.0, x)) * (w - 1), max(0.0, min(1.0, y)) * (h - 1)) for x, y in points]

    n = max(len(px) - 1, 1)
    for i in range(len(px) - 1):
        f = i / n
        draw.line([px[i], px[i + 1]], fill=(int(255 * f), int(255 * (1 - f)), 40), width=2)
    for i, (x, y) in enumerate(px):
        f = i / max(len(px) - 1, 1)
        r = 3.5 if i in (0, len(px) - 1) else 2.0
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(int(255 * f), int(255 * (1 - f)), 40), outline=(255, 255, 255))
    if fresh:
        draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255), width=2)
    return np.asarray(img)



_TRACE_WARNED = [False]


def _warn_if_no_trace(cot_text, points) -> None:
    """Overlay silently produces plain video when no trace reaches the client --
    e.g. a non-CoT checkpoint, or a client that drops the server's cot_text.
    Say so once instead of leaving the user to notice from the mp4s."""
    if _TRACE_WARNED[0]:
        return
    if not cot_text:
        logging.warning("[overlay_trace] enabled but the policy returned no cot_text -- "
                        "video will be unannotated. Needs framework.cot.generate_at_inference=true.")
        _TRACE_WARNED[0] = True
    elif not points:
        logging.warning("[overlay_trace] cot_text has no parseable <|trace|> block: %r", cot_text[:120])
        _TRACE_WARNED[0] = True


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    save_video: bool = False
    """Save one MP4 for every rollout. Disabled for quantitative evaluations;
    per-shard JSONs contain everything required to compute success rates."""
    overlay_trace: bool = False
    """Draw the model's generated 2D trace on the saved rollout video. The trace
    is re-predicted once per action chunk and held for the frames in between;
    the frame where it was regenerated is marked with a white border. Requires
    framework.cot.generate_at_inference=true -- a no-op otherwise."""
    log_path: str = "experiments/libero/logs"

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    use_bf16: bool = True
    use_server: bool = False

    start_idx: int = -1
    end_idx: int = -1
    # Task indices within [start_idx, end_idx) are grouped into large contiguous
    # blocks by perturbation category (e.g. index 0-499 of libero_10 is entirely
    # "table" perturbations, 500-999 is "view", etc.) -- a plain contiguous
    # sub-range is NOT a representative sample of perturbation types. stride > 1
    # takes every stride'th index instead, spreading the evaluated tasks across
    # every category the range spans.
    stride: int = 1
    # Exactly K indices floor(k * suite_size / K), k=0..K-1, filtered to this
    # worker's [start_idx,end_idx). Zero keeps the legacy stride behavior.
    exact_sample_count: int = 0
    # Reuse completed episode records already staged in output_dir. A task is
    # skipped only when a matching shard summary JSON also exists, so records
    # from an interrupted shard can never disappear from the final aggregate.
    skip_existing_episodes: bool = False
    # Round-robin sharding WITHIN [start_idx,end_idx): this worker evaluates
    # task_ids[shard_idx::num_shards] instead of a contiguous block of them.
    # Perturbation categories are large contiguous index blocks, and their cost
    # per episode differs by >5x (a camera-viewpoint episode mostly fails and
    # runs to max_steps; a background-texture one succeeds early), so contiguous
    # blocks gave every worker a different total workload -- measured on
    # libero_10, 32 contiguous shards finished spread over 52 minutes, the last
    # five running alone on an otherwise idle node for 45 of the suite's 69
    # minutes. Interleaving gives every worker the same category mix, so they
    # finish together. The UNION of evaluated tasks is unchanged (same
    # deterministic set, just dealt out differently), so results stay directly
    # comparable to contiguously sharded runs.
    shard_idx: int = 0
    num_shards: int = 1
    output_dir: str = "./output"

    # Gripper output convention: "auto" reads dataset_statistics.json next to the
    # checkpoint; override with "zero_one" or "pm_one". See _binarize_gripper.
    gripper_encoding: str = "auto"
    object_perturb_m: float = 0.0
    object_perturb_roles: str = "source,target"
    object_perturb_seed: int = 20260812


def _stable_perturb_rng(seed: int, suite: str, task_id: int, episode_idx: int, object_name: str):
    key = f"{seed}|{suite}|{task_id}|{episode_idx}|{object_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _goal_object_roles(env) -> tuple[list[str], list[str], list[str]]:
    task_env = env.env
    movable = task_env.objects_dict
    fixtures = task_env.fixtures_dict

    def owner(token: str, candidates) -> str | None:
        matches = [name for name in candidates if token == name or token.startswith(name + "_")]
        return max(matches, key=len) if matches else None

    sources, targets, unsupported = [], [], []
    for goal in task_env.parsed_problem.get("goal_state", []):
        if len(goal) >= 2:
            source = owner(goal[1], movable)
            if source is not None:
                sources.append(source)
            elif owner(goal[1], fixtures) is not None:
                unsupported.append(f"source fixture:{goal[1]}")
        if len(goal) >= 3:
            target = owner(goal[2], movable)
            if target is not None:
                targets.append(target)
            elif owner(goal[2], fixtures) is not None:
                unsupported.append(f"target fixture:{goal[2]}")
            else:
                unsupported.append(f"target region:{goal[2]}")
    return list(dict.fromkeys(sources)), list(dict.fromkeys(targets)), list(dict.fromkeys(unsupported))


def _apply_object_perturbation(env, magnitude_m, roles_csv, seed, suite, task_id, episode_idx) -> dict:
    if magnitude_m < 0:
        raise ValueError(f"object_perturb_m must be non-negative, got {magnitude_m}")
    roles = {x.strip().lower() for x in roles_csv.split(",") if x.strip()}
    unknown = roles - {"source", "target"}
    if unknown:
        raise ValueError(f"Unknown object perturbation roles: {sorted(unknown)}")
    sources, targets, unsupported = _goal_object_roles(env)
    selected: dict[str, set[str]] = {}
    if "source" in roles:
        for name in sources:
            selected.setdefault(name, set()).add("source")
    if "target" in roles:
        for name in targets:
            selected.setdefault(name, set()).add("target")
    record = {
        "magnitude_m": float(magnitude_m), "roles": sorted(roles),
        "source_objects": sources, "target_objects": targets,
        "unsupported": unsupported, "objects": [],
    }
    if magnitude_m == 0:
        return record
    if not selected:
        raise RuntimeError(
            f"No movable goal objects resolved for roles={sorted(roles)} in "
            f"{suite} task={task_id}; unsupported={unsupported}"
        )
    for object_name, object_roles in sorted(selected.items()):
        obj = env.env.objects_dict[object_name]
        if not obj.joints:
            raise RuntimeError(f"Movable goal object {object_name!r} has no MuJoCo joint")
        joint = obj.joints[0]
        qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
        if qpos.size != 7:
            raise RuntimeError(f"Expected 7 qpos for free joint {joint!r}, got {qpos.size}")
        before = qpos[:3].copy()
        rng = _stable_perturb_rng(seed, suite, task_id, episode_idx, object_name)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        delta = magnitude_m * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        qpos[:2] += delta
        env.sim.data.set_joint_qpos(joint, qpos)
        record["objects"].append({
            "name": object_name, "roles": sorted(object_roles), "joint": joint,
            "before_xyz": before.tolist(), "delta_xy": delta.tolist(), "after_xyz": qpos[:3].tolist(),
        })
    env.sim.forward()
    if env.check_success():
        raise RuntimeError(
            f"Perturbation accidentally satisfied {suite} task={task_id} episode={episode_idx}"
        )
    return record


class PolicyModel:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=True,
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
        use_bf16=True,
    ) -> None:

        # Local-model evaluation is retained for the conda path.  Importing the
        # framework lazily keeps remote-server simulator workers independent of
        # transformers / accelerate and the rest of the model-training stack.
        from starVLA.model.framework.base_framework import baseframework

        # build client to connect server policy
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        vla = baseframework.from_pretrained(  # TODO should auto detect framework from model path
            policy_ckpt_path,
        )

        if use_bf16:  # False
            vla = vla.to(torch.bfloat16)
        self.vla = vla.to("cuda").eval()

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, example: dict, step: int = 0, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        vla_input = {
            # "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size
        self.cot_is_fresh = False
        if step % action_chunk_size == 0:
            response = self.vla.predict_action(example, **vla_input)
            normalized_actions = response["normalized_actions"]  # B, chunk, D
            # Explicit-CoT frameworks return the reasoning that conditioned this
            # chunk; hold it so every frame until the next chunk can show it.
            cot = response.get("cot_text")
            if cot:
                self.last_cot_text = cot[0] if isinstance(cot, (list, tuple)) else cot
                self.cot_is_fresh = True

            normalized_actions = normalized_actions[0]

            if normalized_actions.shape[1] > 7:
                normalized_actions = normalized_actions[:, -7:]

            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {
            "raw_action": raw_action,
            "cot_text": getattr(self, "last_cot_text", None),
            "cot_is_fresh": getattr(self, "cot_is_fresh", False),
        }

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        from starVLA.model.tools import read_mode_config

        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = PolicyModel._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        from starVLA.model.tools import read_mode_config

        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key


def _eval_entrypoint(func):
    return draccus.wrap()(func) if draccus is not None else func


@_eval_entrypoint
def eval_libero(args: Args) -> None:
    # Asking for a trace overlay necessarily asks for a replay video. Ordinary
    # quantitative runs skip both frame retention and MP4 encoding entirely.
    record_video = args.save_video or args.overlay_trace

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    print(f"🌍 Rank {rank}/{world_size} | GPU: {local_rank}")
    # The model server owns CUDA inference.  SIF workers use a CPU-only torch
    # wheel for LIBERO init-state loading and select EGL via environment vars.
    if not args.use_server:
        torch.cuda.set_device(local_rank)

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    # if args.start_idx != -1:
    #     num_tasks_in_suite = args.end_idx - args.start_idx
    # patch_num = num_tasks_in_suite // world_size
    # if rank == world_size - 1:
    #     start_idx = rank * patch_num
    #     end_idx = num_tasks_in_suite
    # else:
    #     start_idx = rank * patch_num
    #     end_idx = start_idx + patch_num
    if args.start_idx == -1:
        args.start_idx = 0
        args.end_idx = num_tasks_in_suite
    # args.start_idx = start_idx
    # args.end_idx = end_idx
    if args.exact_sample_count:
        if not 0 < args.exact_sample_count <= num_tasks_in_suite:
            raise ValueError(
                f"exact_sample_count must be in [1, {num_tasks_in_suite}], "
                f"got {args.exact_sample_count}"
            )
        global_task_ids = [
            sample_idx * num_tasks_in_suite // args.exact_sample_count
            for sample_idx in range(args.exact_sample_count)
        ]
        task_ids = [
            task_id for task_id in global_task_ids
            if args.start_idx <= task_id < args.end_idx
        ]
        sample_tag = f"_exact{args.exact_sample_count}"
    else:
        task_ids = list(range(args.start_idx, args.end_idx, args.stride))
        sample_tag = "" if args.stride == 1 else f"_stride{args.stride}"
    ids_dir = os.environ.get("STARVLA_TASK_IDS_DIR", "").strip()
    if ids_dir:
        # Opt-in fixed task subset (e.g. a validated high-divergence subset): <dir>/<suite>.txt holds one
        # task id per line (same ids as the *_episodes.jsonl records). Replaces the exact/stride selection;
        # the tag keeps these shard files apart from full-protocol ones in the same output_dir.
        ids_file = pathlib.Path(ids_dir) / f"{args.task_suite_name}.txt"
        wanted = sorted({int(x) for x in ids_file.read_text().split() if x.strip()})
        task_ids = [t for t in wanted if args.start_idx <= t < args.end_idx]
        sample_tag = f"_ids{len(wanted)}"
    if args.num_shards > 1:
        if not 0 <= args.shard_idx < args.num_shards:
            raise ValueError(
                f"shard_idx must be in [0, {args.num_shards}), got {args.shard_idx}"
            )
        # Deal out the already-deterministic task list round-robin. Every shard
        # of the same (range, stride/exact) selection covers it exactly once
        # between them, so the union is identical to contiguous sharding.
        task_ids = task_ids[args.shard_idx :: args.num_shards]
        # Distinct per shard AND distinct from a contiguous-shard file covering
        # the same range -- shard results are merged by globbing this directory,
        # so an overlapping leftover file from a differently sharded earlier run
        # would otherwise be silently double-counted.
        sample_tag += f"_shard{args.shard_idx}of{args.num_shards}"
    if args.skip_existing_episodes:
        existing_log_path = pathlib.Path(args.output_dir) / "logs" / args.task_suite_name
        completed_episodes: dict[int, set[int]] = {}
        for summary_path in existing_log_path.glob("*.json"):
            episode_path = summary_path.with_name(
                summary_path.name[: -len(".json")] + "_episodes.jsonl"
            )
            if not episode_path.is_file():
                continue
            with episode_path.open(encoding="utf-8") as episode_file:
                for line in episode_file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    completed_episodes.setdefault(int(record["task_id"]), set()).add(
                        int(record["episode_idx"])
                    )
        required_episode_ids = set(range(args.num_trials_per_task))
        before_reuse = len(task_ids)
        task_ids = [
            task_id
            for task_id in task_ids
            if not required_episode_ids.issubset(completed_episodes.get(task_id, set()))
        ]
        print(
            f"reused {before_reuse - len(task_ids)}/{before_reuse} assigned tasks from "
            f"completed episode records in {existing_log_path}"
        )
    print(
        f"processing {len(task_ids)} tasks from {args.start_idx} to {args.end_idx} "
        f"(stride={args.stride}, exact_sample_count={args.exact_sample_count}, "
        f"shard={args.shard_idx}/{args.num_shards})"
    )
    # args.video_out_path = f"{date_base}+{args.job_name}"
    # stride suffix keeps a subsampled shard's filename distinct from a
    # full (stride=1) shard covering the same [start_idx,end_idx) -- otherwise
    # a later full-range rerun into the same output_dir would silently
    # overwrite a subsampled shard's json (or vice versa) rather than erroring.
    log_path = os.path.join(args.output_dir, f"logs/{args.task_suite_name}")
    log_file = os.path.join(log_path, f"{args.start_idx}_{args.end_idx}{sample_tag}.log")
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file)
    logger.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if not task_ids:
        # An exact global sample may miss a narrow raw-index shard at high
        # concurrency. This is zero assigned work, not a simulator failure.
        # Emit empty shard artifacts and avoid policy/EGL startup and 0/0 below.
        with open(os.path.join(log_path, f"{args.start_idx}_to_{args.end_idx}{sample_tag}.json"), "w", encoding="utf-8") as f:
            json.dump({}, f)
        pathlib.Path(log_path, f"{args.start_idx}_to_{args.end_idx}{sample_tag}_episodes.jsonl").write_text("", encoding="utf-8")
        logger.info("No sampled tasks in this shard; completed without simulator startup.")
        return
    video_out_path = os.path.join(args.output_dir, args.task_suite_name)
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    if args.use_server:
        client_model = ModelClient(
            policy_ckpt_path=args.pretrained_path,
            unnorm_key="franka",
            host=args.host,
            port=args.port,
        )
    else:
        client_model = PolicyModel(
            policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
            host=args.host,
            port=args.port,
            image_size=args.resize_size,
            use_bf16=args.use_bf16,
        )

    # Resolved once from the handshake, not re-read per step.
    send_state = _should_send_state(client_model)
    logging.info(
        "Proprioceptive state: %s (server expects_state=%s, STARVLA_EVAL_INCLUDE_STATE=%r)",
        "SENT" if send_state else "withheld",
        getattr(client_model, "expects_state", False),
        os.environ.get("STARVLA_EVAL_INCLUDE_STATE", ""),
    )

    # Resolve the gripper convention; a wrong choice inverts the gripper and the
    # policy can never grasp, so fail rather than guess silently.
    gripper_encoding = args.gripper_encoding
    if gripper_encoding == "auto":
        gripper_encoding = _detect_gripper_encoding(args.pretrained_path)
        if not gripper_encoding:
            raise RuntimeError(
                f"Could not auto-detect the gripper encoding (no dataset_statistics.json found near "
                f"{args.pretrained_path}). Pass --gripper_encoding zero_one|pm_one explicitly."
            )
    logger.info(f"Using gripper encoding: {gripper_encoding}")

    disturb_res = {}
    LIBERO_HOME = os.environ.get("LIBERO_HOME", "path_to_LIBERO-plus")
    with open(os.path.join(LIBERO_HOME, "libero/libero/benchmark/task_classification.json")) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]

    ID2CATEGORY = {}
    for item in TASK_MAPPING:
        category = item["category"]
        item_name = item["name"]
        ID2CATEGORY[item["id"]] = (category, item_name)
        if category not in disturb_res:
            disturb_res[category] = {"total_count": 0, "success_count": 0}

    # Start evaluation

    total_episodes, total_successes = 0, 0
    episode_records = []
    # Opt-in per-episode checkpoint (STARVLA_PARTIAL_CKPT=1, host side APPTAINERENV_STARVLA_PARTIAL_CKPT=1):
    # every finished episode is appended to <shard>_episodes.partial.jsonl; a restarted attempt of the
    # SAME shard restores its counters and records from it and skips those tasks, then writes the usual
    # summary JSON and episodes JSONL (same format, same totals) and removes the partial file. Default
    # off: the shard summary is written only at the end, as before.
    partial_path = pathlib.Path(log_path, f"{args.start_idx}_to_{args.end_idx}{sample_tag}_episodes.partial.jsonl")
    partial_ckpt = os.environ.get("STARVLA_PARTIAL_CKPT", "0") == "1"
    if partial_ckpt and partial_path.is_file():
        restored = []
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                restored.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a torn last line from the abort: drop it and everything after
        done_eps: dict[int, set[int]] = {}
        for rec in restored:
            done_eps.setdefault(int(rec["task_id"]), set()).add(int(rec["episode_idx"]))
        full = {t for t, e in done_eps.items() if set(range(args.num_trials_per_task)).issubset(e)}
        restored = [rec for rec in restored if int(rec["task_id"]) in full and int(rec["task_id"]) in set(task_ids)]
        for rec in restored:
            episode_records.append(rec)
            total_episodes += 1
            total_successes += int(bool(rec["success"]))
            disturb_res[rec["category"]]["total_count"] += 1
            disturb_res[rec["category"]]["success_count"] += int(bool(rec["success"]))
        task_ids = [t for t in task_ids if t not in full]
        logger.info(f"PARTIAL CKPT: restored {len(restored)} episodes ({total_successes} successes) from {partial_path}")
        partial_path.write_text("".join(json.dumps(rec) + "\n" for rec in restored), encoding="utf-8")
    elif partial_ckpt:
        partial_path.write_text("", encoding="utf-8")
    print(
        f"*****************num tasks in {args.task_suite_name}: {num_tasks_in_suite}****************, processing from{args.start_idx} to {args.end_idx}"
    )
    # for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    for task_id in tqdm.tqdm(task_ids):

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):

            if os.environ.get("STARVLA_PAIRED_ROLLOUT", "0") == "1":
                key = f"{args.seed}|{args.task_suite_name}|{task_id}|{episode_idx}".encode()
                episode_seed = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "little")
                np.random.seed(episode_seed)
                env.seed(episode_seed)

            logger.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()
            obj_mask = None
            if online_obj_mask.enabled():
                if args.num_steps_wait < 1:
                    raise ValueError("STARVLA_ONLINE_OBJ_MASK needs num_steps_wait >= 1")
                obj_mask = online_obj_mask.OnlineObjectMask()
                _resolved = obj_mask.reset(env, tag=f"{args.task_suite_name}_t{task_id:04d}_e{episode_idx}")
                logger.info(f"Online object mask: objects of interest {json.dumps(_resolved, sort_keys=True)}")

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            perturbation = _apply_object_perturbation(
                env, args.object_perturb_m, args.object_perturb_roles,
                args.object_perturb_seed, args.task_suite_name, task_id, episode_idx,
            )
            trace = None
            if episode_trace.enabled():
                trace = episode_trace.EpisodeTrace(env, perturbation["source_objects"], perturbation["target_objects"])
            # Avoid a second flattened-state restore: in the pinned
            # LIBERO/robosuite renderer it corrupts the offscreen camera and
            # produces black observations. Settling steps below refresh obs.

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            initial_cot_text = None

            logger.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:

                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    if obj_mask is not None:  # segment only the frame that feeds the first query
                        obj_mask.arm(t == args.num_steps_wait - 1)
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Some LIBERO-plus "noise" perturbations (fog/glass_blur/motion_blur/...,
                # see env_wrapper.py) are applied to agentview_image via third-party
                # corruption functions that can silently drop the channel axis for
                # certain severities, returning (H,W) instead of (H,W,3). The policy
                # server rejects non-3D images outright, so restore it here rather
                # than chasing the bug inside each corruption implementation.
                if img.ndim == 2:
                    img = np.repeat(img[:, :, None], 3, axis=2)
                if wrist_img.ndim == 2:
                    wrist_img = np.repeat(wrist_img[:, :, None], 3, axis=2)
                if args.object_perturb_m > 0 and float(img.mean()) < 1.0:
                    raise RuntimeError(
                        "Agent-view observation is nearly black after object perturbation "
                        f"(mean={float(img.mean()):.4f}); refusing an invalid rollout"
                    )

                # Save preprocessed image for replay video. With overlay_trace the
                # append happens after the policy call, so the frame can carry the
                # trace that was generated from it.
                if record_video and not args.overlay_trace:
                    replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                    # Stable key for deterministic server-side corruption after the
                    # checkpoint's exact training-time state normalization.
                    "_state_corrupt_key": (
                        f"{args.task_suite_name}|{task_id}|{episode_idx}|{step}"
                    ),
                }
                if obj_mask is not None and step % client_model.action_chunk_size == 0:
                    # [2, 8, 8] float16 task-object token fractions (agentview, wrist), policy orientation
                    example_dict["_obj_tok"] = obj_mask(obs, step, images=[img, wrist_img])
                if send_state:
                    if state.shape != (8,):
                        raise RuntimeError(f"expected 8-D LIBERO state, got {state.shape}")
                    example_dict["state"] = observation["observation.state"].astype(np.float32)

                start_time = time.time()

                # response = client_model.step(example=example_dict)
                if os.environ.get("STARVLA_PAIRED_ROLLOUT", "0") == "1":
                    key = f"{args.seed}|{args.task_suite_name}|{task_id}|{episode_idx}|{step}".encode()
                    example_dict["_paired_policy_seed"] = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "little")
                response = client_model.step(example=example_dict, step=step)
                if initial_cot_text is None and response.get("cot_is_fresh"):
                    initial_cot_text = response.get("cot_text")

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                if record_video and args.overlay_trace:
                    # _overlay_trace copies, so the observation the policy just
                    # consumed stays byte-identical.
                    _pts = _parse_trace(response.get("cot_text"))
                    _warn_if_no_trace(response.get("cot_text"), _pts)
                    replay_images.append(
                        _overlay_trace(img, _pts, fresh=bool(response.get("cot_is_fresh")))
                    )

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper(open_gripper, gripper_encoding)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logger.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                if obj_mask is not None:
                    obj_mask.arm((step + 1) % client_model.action_chunk_size == 0)
                if trace is not None:
                    trace.step(obs, delta_action)
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res[ID2CATEGORY[task_id + 1][0]]["success_count"] += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            disturb_res[ID2CATEGORY[task_id + 1][0]]["total_count"] += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            if record_video:
                imageio.mimwrite(
                    pathlib.Path(video_out_path) / f"rollout_{ID2CATEGORY[task_id+1][1]}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=25,
                )

            if trace is not None:
                trace.step(obs)  # final state, after the last action
                trace.save(
                    f"{args.task_suite_name}_t{task_id:05d}_e{episode_idx}",
                    {"suite": args.task_suite_name, "task_id": task_id, "episode_idx": episode_idx,
                     "success": bool(done), "category": ID2CATEGORY[task_id + 1][0], "task_name": task.name,
                     "prompt": str(task_description), "task_language": task.language, "perturbation": perturbation},
                )
            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logger.info(f"Success: {done}")
            logger.info(f"Object perturbation: {json.dumps(perturbation, sort_keys=True)}")
            episode_records.append({
                "task_id": task_id, "episode_idx": episode_idx, "success": bool(done),
                "category": ID2CATEGORY[task_id + 1][0],
                "perturbation": perturbation, "initial_cot_text": initial_cot_text,
                "paired_episode_seed": episode_seed if os.environ.get("STARVLA_PAIRED_ROLLOUT", "0") == "1" else None,
                **({"obj_mask_resolved": obj_mask.resolved, "obj_mask_ms_per_query": 1e3 * obj_mask.seconds / max(obj_mask.calls, 1)}
                   if obj_mask is not None else {}),
            })
            if partial_ckpt:
                with partial_path.open("a", encoding="utf-8") as pf:
                    pf.write(json.dumps(episode_records[-1]) + "\n")
                    pf.flush()
                    os.fsync(pf.fileno())
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logger.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logger.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    with open(os.path.join(log_path, f"{args.start_idx}_to_{args.end_idx}{sample_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(disturb_res, f)
    with open(os.path.join(log_path, f"{args.start_idx}_to_{args.end_idx}{sample_tag}_episodes.jsonl"), "w", encoding="utf-8") as f:
        for record in episode_records:
            f.write(json.dumps(record) + "\n")
    if partial_ckpt and partial_path.is_file():
        partial_path.unlink()  # the shard's summary and episodes JSONL above now hold every episode
    logger.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logger.info(f"Total episodes: {total_episodes}")


_LPLUS_SUFFIX = re.compile(
    r"( view( -?\d+)+ initstate \d+( noise \d+)?| add \d+| light \d+| table \d+| tb \d+|( moved)? level\d+ sample\d+)$"
)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if online_obj_mask.enabled():
        # Instance segmentation next to RGB; ControlEnv.step (Sensor Noise) stays in the loop.
        env_args["camera_segmentations"] = "instance"
    # Environment construction is NOT locked by default (2026-10-02). Every LIBERO-Plus episode builds a
    # new environment, and the former node-wide lock serialised construction across all workers: ~60% of
    # simulator time at 6 workers / GPU, and more workers only lengthened the queue. The EGL init faults
    # the lock was added for were traced to bad nodes (config/libero_sim_node_blacklist.txt).
    # Opt back in with STARVLA_EGL_INIT_LOCK = "node" (one node-wide lock), "per_gpu" (one lock per EGL
    # device) or an explicit lock-file path.
    lock_mode = os.environ.get("STARVLA_EGL_INIT_LOCK", "").strip()
    if lock_mode.lower() in ("", "none", "off", "0"):
        env = OffScreenRenderEnv(**env_args)
    else:
        if lock_mode.lower() == "per_gpu":
            lock_path = f"/tmp/starvla_egl_init_gpu{os.environ.get('MUJOCO_EGL_DEVICE_ID', 'x')}.lock"
        elif lock_mode.lower() == "node":
            lock_path = "/tmp/starvla_egl_init.lock"
        else:
            lock_path = lock_mode
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            env = OffScreenRenderEnv(**env_args)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    # task.language is built from the variant file name, so outside Language Instructions it carries the
    # perturbation id (".. add 10", ".. level3 sample1", ".. view 0 0 100 2 352 initstate 0", ".. light 1";
    # sylvestf/LIBERO-plus#64). Default off, so earlier results stay comparable.
    #   STARVLA_LPLUS_PROMPT=strip  drop the suffix: gives exactly the training instruction for all 8,493
    #                               non-language variants (checked 2026-10-04); language rewrites untouched
    #   STARVLA_LPLUS_PROMPT=bddl   the instruction of the BDDL the scene is built from (capitalised)
    prompt_mode = os.environ.get("STARVLA_LPLUS_PROMPT", "").strip().lower()
    if prompt_mode == "strip":
        task_description = _LPLUS_SUFFIX.sub("", task_description)
    elif prompt_mode == "bddl":
        bddl_lang = getattr(env, "language_instruction", None)
        if isinstance(bddl_lang, (list, tuple)):
            bddl_lang = " ".join(str(w) for w in bddl_lang)
        if not bddl_lang:
            raise RuntimeError(f"STARVLA_LPLUS_PROMPT=bddl but the env has no language_instruction ({task.name})")
        task_description = str(bddl_lang)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _parse_stdlib_args() -> Args:
    """CLI compatible with the arguments used by the parallel eval scripts.

    draccus remains the preferred parser in the full conda environment.  This
    fallback deliberately uses no StarVLA model-side dependencies so the same
    evaluator can run as a thin client in the simulator SIF.
    """
    parser = argparse.ArgumentParser(description="Evaluate StarVLA on LIBERO-Plus")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--resize_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--video_out_path", default="experiments/libero/logs")
    parser.add_argument("--save_video", type=_parse_bool, default=False)
    parser.add_argument("--overlay_trace", type=_parse_bool, default=False)
    parser.add_argument("--log_path", default="experiments/libero/logs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pretrained_path", default="")
    parser.add_argument("--post_process_action", type=_parse_bool, default=True)
    parser.add_argument("--job_name", default="test")
    parser.add_argument("--use_bf16", type=_parse_bool, default=True)
    parser.add_argument("--use_server", type=_parse_bool, default=False)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--exact_sample_count", type=int, default=0)
    parser.add_argument("--skip_existing_episodes", type=_parse_bool, default=False)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument(
        "--gripper_encoding",
        choices=("auto", "zero_one", "pm_one", "zero_one_close"),
        default="auto",
    )
    parser.add_argument("--object_perturb_m", type=float, default=0.0)
    parser.add_argument("--object_perturb_roles", default="source,target")
    parser.add_argument("--object_perturb_seed", type=int, default=20260812)
    values = vars(parser.parse_args())
    resize_size = values.pop("resize_size")
    args = Args(**values)
    args.resize_size = resize_size
    return args


if __name__ == "__main__":
    if draccus is None:
        eval_libero(_parse_stdlib_args())
    else:
        eval_libero()

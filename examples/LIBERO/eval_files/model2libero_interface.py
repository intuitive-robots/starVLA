# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.
"""

from collections import deque
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size: Optional[Sequence[int]] = None,
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        # Whether this checkpoint was trained with proprioceptive state. Older servers do
        # not advertise it; False is the historically correct default, because every
        # LIBERO config before 2026-09 set state_dim: 0.
        self.expects_state = bool(meta.get("expects_state", False))
        self._server_metadata = meta

        # Prefer an explicit caller override; otherwise follow the image size
        # that the checkpoint reports was actually applied during training.
        # Legacy checkpoints report None and therefore retain native 256x256.
        server_image_size = (
            meta.get("training_obs_image_size")
            if meta.get("image_preprocessing_contract_version") == 2
            else None
        )
        effective_image_size = image_size if image_size is not None else server_image_size
        if effective_image_size is None:
            self.image_size: Optional[tuple[int, int]] = None
        else:
            if len(effective_image_size) != 2:
                raise ValueError(
                    "image_size must be a (height, width) pair, "
                    f"got {effective_image_size!r}"
                )
            target_height, target_width = (int(value) for value in effective_image_size)
            if target_height <= 0 or target_width <= 0:
                raise ValueError(
                    f"image_size dimensions must be positive, got {effective_image_size!r}"
                )
            self.image_size = (target_height, target_width)

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, expects_state: {self.expects_state}, "
            f"server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None
        self.last_cot_text: Optional[str] = None

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
        self.raw_actions = None
        self.last_cot_text = None

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # image_size uses (height, width); PIL resize uses (width, height).
        if self.image_size is not None and example.get("image"):
            target_height, target_width = self.image_size
            resized = []
            for image in example["image"]:
                array = np.asarray(image)
                if array.shape[:2] != self.image_size:
                    array = np.asarray(
                        Image.fromarray(array).resize(
                            (target_width, target_height), Image.Resampling.BILINEAR
                        )
                    )
                resized.append(array)
            example = {**example, "image": resized}

        # Refresh chunk if needed.
        cot_is_fresh = False
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            vla_input = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            self.raw_actions = np.asarray(actions_batch)[0]  # (T, D)
            cot = response["data"].get("cot_text")
            if cot is not None:
                self.last_cot_text = cot[0] if isinstance(cot, (list, tuple)) else cot
            cot_is_fresh = True

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {
            "raw_action": raw_action,
            "cot_text": self.last_cot_text,
            "cot_is_fresh": cot_is_fresh,
        }

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

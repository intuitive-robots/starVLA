import os
from collections import deque
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    # The DataConfig first concatenates [L6, R6, Lgrip, Rgrip].  StateTokenMixin
    # applies this permutation after normalization so each fixed marker directly
    # precedes its complete seven-value arm state.
    ROBOTWIN_STATE_KEYS = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    ROBOTWIN_STATE_TOKEN_CONTRACT = {
        "mode": "reserved_ids",
        "routing": "token_only",
        "num_bins": 256,
        "token_source": "rare_real",
        "permutation": [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13],
        "group_sizes": [7, 7],
        "group_names": ["left", "right"],
        "num_value_tokens": 14,
        "num_tokens_with_markers": 16,
    }
    # pi0 / pi0.5 checkpoints (2026-10-08): the 14 state values ride in the prompt as 256-bin text digits in the
    # wire order above, no permutation and no delimiter markers. What policy_wrapper._state_token_contract
    # advertises for state_token_mode text; accepted only as this exact dict (any other contract still fails).
    ROBOTWIN_TEXT_STATE_CONTRACT = {
        "mode": "text",
        "routing": "",
        "num_bins": 256,
        "token_source": "",
        "permutation": [],
        "group_sizes": [],
        "group_names": [],
        "num_value_tokens": None,
        "num_tokens_with_markers": None,
    }

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = None
        self.state_norm_stats = None
        self.raw_actions = None

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta["action_chunk_size"]
        self.expects_state = bool(server_meta.get("expects_state", False))
        training_action_mode = server_meta.get("training_action_mode")
        if training_action_mode is not None and training_action_mode != self.action_mode:
            raise ValueError(
                "RoboTwin action-mode contract mismatch: "
                f"checkpoint was trained with {training_action_mode!r}, "
                f"but evaluator requested {self.action_mode!r}"
            )
        if self.expects_state:
            state_keys = server_meta.get("state_keys")
            if state_keys != self.ROBOTWIN_STATE_KEYS:
                raise ValueError(
                    "RoboTwin state-key order mismatch: "
                    f"expected {self.ROBOTWIN_STATE_KEYS}, server advertised {state_keys}"
                )
            state_contract = server_meta.get("state_token_contract")
            if state_contract not in (self.ROBOTWIN_STATE_TOKEN_CONTRACT, self.ROBOTWIN_TEXT_STATE_CONTRACT):
                raise ValueError(
                    "RoboTwin left/right state-token contract mismatch: "
                    f"expected {self.ROBOTWIN_STATE_TOKEN_CONTRACT}, "
                    f"server advertised {state_contract}"
                )
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"expects_state: {self.expects_state}, server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self._action_trace_emitted = False
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        env_state = example.get("state", None)
        state = None
        if env_state is not None:
            env_state = np.asarray(env_state, dtype=np.float32).reshape(-1)
            if env_state.size != 14:
                raise ValueError(f"RoboTwin Aloha state must be 14-D, got {env_state.shape}")
            # Environment: [left6, left_gripper, right6, right_gripper].
            # Training DataConfig: [left6, right6, left_gripper, right_gripper].
            state = self._env_to_training_state(env_state)

        # Store initial state for delta/rel modes
        if self.action_mode == "delta" and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode == "delta" and state is not None:
                    self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        example_copy = example.copy()
        if self.expects_state:
            if state is None:
                raise ValueError("state-conditioned checkpoint received no RoboTwin state")
            example_copy["state"] = state.reshape(1, -1)
        else:
            example_copy.pop("state", None)
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            # server already un-normalized via training-time transform
            raw_actions = np.array(response["data"]["actions"][0])  # (chunk, D)

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions, state)
            else:
                self.raw_actions = raw_actions

            if not self._action_trace_emitted:
                raw0 = np.asarray(raw_actions[0], dtype=np.float32)
                absolute0 = np.asarray(self.raw_actions[0], dtype=np.float32)
                state_text = "none" if state is None else np.array2string(
                    np.asarray(state, dtype=np.float32), precision=4, separator=","
                )
                print(
                    "[ACTION_CONTRACT] "
                    f"mode={self.action_mode} state={state_text} "
                    f"raw0={np.array2string(raw0, precision=4, separator=',')} "
                    f"absolute0={np.array2string(absolute0, precision=4, separator=',')}",
                    flush=True,
                )
                self._action_trace_emitted = True

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    def cached_action(self, step: int) -> np.ndarray:
        """Action ``step`` of the chunk queried at the last replan, in RoboTwin env order.

        Used to execute a whole chunk without new observations: the policy reads images and
        state only at ``step % action_chunk_size == 0``, so rendering the intermediate frames
        changes nothing but costs one 3-camera render per action.
        """
        if self.raw_actions is None:
            raise RuntimeError("cached_action called before the first chunk query")
        current_action = self.raw_actions[step % self.action_chunk_size]
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()
        return current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    @staticmethod
    def _env_to_training_state(env_state: np.ndarray) -> np.ndarray:
        """Convert RoboTwin env order to the training DataConfig order.

        Environment: ``[L6, Lgrip, R6, Rgrip]``.
        Training/normalizer: ``[L6, R6, Lgrip, Rgrip]``.
        The model subsequently applies its advertised permutation to form the two
        delimiter-bounded groups ``[L marker, L6, Lgrip, R marker, R6, Rgrip]``.
        """
        env_state = np.asarray(env_state, dtype=np.float32).reshape(-1)
        if env_state.size != 14:
            raise ValueError(f"RoboTwin Aloha state must be 14-D, got {env_state.shape}")
        return env_state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]

    @staticmethod
    def _rel_to_absolute(rel_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Anchor every predicted joint target to the observation that opened the chunk.

        RoboTwin actions are ordered [left6, right6, left_gripper, right_gripper].
        Only the twelve arm joints are relative; grippers remain absolute binary targets.
        """
        if current_state is None:
            raise ValueError("relative actions require the current observed joint state")
        absolute = np.asarray(rel_actions).copy()
        current_state = np.asarray(current_state).reshape(-1)
        if absolute.shape[-1] != 14 or current_state.size != 14:
            raise ValueError(
                f"expected 14-D RoboTwin relative action/state, got "
                f"actions={absolute.shape}, state={current_state.shape}"
            )
        absolute[..., :12] += current_state[:12]
        return absolute

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)

    # Execute the rest of the chunk before returning to eval_policy's loop, which would
    # otherwise render a fresh 3-camera observation before every action (FLEX-pi's
    # skip_get_obs_within_replan). The executed actions are identical: the client only
    # re-queries the policy at take_action_cnt % action_chunk_size == 0.
    # ROBOTWIN_EXECUTE_FULL_CHUNK=0 restores one action per eval() call.
    if os.environ.get("ROBOTWIN_EXECUTE_FULL_CHUNK", "1") != "0":
        chunk = int(model.action_chunk_size)
        while (
            TASK_ENV.take_action_cnt % chunk != 0
            and TASK_ENV.take_action_cnt < TASK_ENV.step_lim
            and not TASK_ENV.eval_success
        ):
            TASK_ENV.take_action(model.cached_action(TASK_ENV.take_action_cnt))

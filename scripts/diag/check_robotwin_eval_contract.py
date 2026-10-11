#!/usr/bin/env python3
"""Fail-closed preflight for the paper RoboTwin state/action checkpoint contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


EXPECTED_PERMUTATION = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
EXPECTED_ACTION_KEYS = ["action.left_joints", "action.right_joints"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[ROBOTWIN CONTRACT FAIL] {message}")


PI_FRAMEWORKS = {"PI05", "Pi05", "PI0", "Pi0"}


def check_pi(framework: dict, data: dict) -> None:
    """pi0 / pi0.5 on RoboTwin: state as 256-bin text digits in the prompt, in the wire order
    [L6, R6, Lgrip, Rgrip] (no permutation, no delimiters); 14 absolute targets padded into the
    32-D action space, loss and output over the first 14; three cameras [head, left wrist, right wrist]."""
    require(data.get("include_state") is True, "datasets.vla_data.include_state must be true")
    require(framework.get("state_token_mode") == "text", "pi0.5 state must ride in the prompt (state_token_mode text)")
    require(framework.get("discrete_state_input") is True, "pi0.5 needs discrete_state_input true")
    require(str(framework.get("discrete_state_mode", "")).lower() == "starvla", "pi0.5 state binning must be starvla")
    require(int(framework.get("state_num_bins", -1)) == 256, "state_num_bins must be 256")
    require(not framework.get("state_token_permutation"), "the pi0.5 text state has no permutation")
    require(int(framework.get("output_action_dim", -1)) == 14, "pi0.5 output_action_dim must be 14")
    require(int(framework.get("action_loss_dims", -1)) == 14, "pi0.5 action_loss_dims must be 14")
    horizon = int(framework.get("action_horizon", -1))
    require(int((framework.get("action_model") or {}).get("action_horizon", -2)) == horizon,
            "framework.action_model.action_horizon must equal framework.action_horizon (the server's chunk size)")
    require(len(framework.get("image_keys") or []) == 3, "pi0.5 must read the three RoboTwin cameras")
    action_mode = data.get("action_mode", "abs")
    require(action_mode in {"abs", "absolute"}, f"pi0.5 RoboTwin contract is absolute actions, got {action_mode!r}")
    require(not data.get("action_mode_apply_keys"), "absolute actions must not carry action_mode_apply_keys")
    require(data.get("action_type") in (None, "qpos", "absolute_qpos"), "absolute actions must not use a relative action_type")
    if data.get("state_normalization") not in (None, "", "none", "raw"):
        require(data.get("state_normalization_applied") is True,
                "state_normalization without state_normalization_applied: the server would feed raw state")
    print(
        "[ROBOTWIN CONTRACT OK] pi0.5 text state: wire=[L6,R6,Lgrip,Rgrip] -> 14 prompt bins (no markers); "
        f"absolute 14-D joints and grippers in a {framework.get('action_dim')}-D action space; chunk={horizon}; "
        f"cameras={list(framework.get('image_keys'))}; state_normalization={data.get('state_normalization', 'raw')}"
    )


def main(path: str) -> None:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())
    framework = config.get("framework") or {}
    action_model = framework.get("action_model") or {}
    data = ((config.get("datasets") or {}).get("vla_data") or {})

    # pi0 / pi0.5 checkpoints (2026-10-08 RoboTwin pi0.5 pair): their own contract; every other framework below.
    if str(framework.get("name", "")) in PI_FRAMEWORKS:
        check_pi(framework, data)
        return

    require(data.get("include_state") is True, "datasets.vla_data.include_state must be true")
    require(framework.get("state_token_mode") == "reserved_ids", "state must use reserved_ids")
    require(framework.get("state_routing") == "token_only", "state_routing must be token_only")
    require(int(action_model.get("state_dim", -1)) == 0, "continuous action-head state_dim must be 0")
    require(int(action_model.get("token_state_dim", -1)) == 16, "token_state_dim must be 16")
    require(int(framework.get("state_num_bins", -1)) == 256, "state_num_bins must be 256")
    require(framework.get("state_token_source") == "rare_real", "state tokens must be pinned rare IDs")
    require(framework.get("trainable_bin_embedding") is False, "pinned state embeddings must stay frozen")
    require(framework.get("state_token_permutation") == EXPECTED_PERMUTATION, "left/right permutation mismatch")
    require(framework.get("state_token_group_sizes") == [7, 7], "left/right groups must each contain 7 values")
    require(framework.get("state_token_group_names") == ["left", "right"], "left/right delimiter names mismatch")
    action_mode = data.get("action_mode", "abs")
    if action_mode in {"rel", "relative"}:
        # 2026-09 contract: current-observation-relative arm joints, absolute grippers.
        require(data.get("action_mode_apply_keys") == EXPECTED_ACTION_KEYS, "only arm joints may be relative")
        action_text = "relative L/R joints; absolute grippers"
    else:
        # FLEX-pi contract (2026-10-07 pilot): every one of the 14 targets is an absolute joint/gripper value.
        require(action_mode in {"abs", "absolute"}, f"unsupported joint action mode {action_mode!r}")
        require(not data.get("action_mode_apply_keys"), "absolute actions must not carry action_mode_apply_keys")
        require(data.get("action_type") in (None, "qpos", "absolute_qpos"), "absolute actions must not use a relative action_type")
        action_text = "absolute 14-D joints and grippers"
    require(int(action_model.get("action_dim", -1)) == 14, "RoboTwin action_dim must be 14")
    if data.get("state_normalization") not in (None, "", "none", "raw"):
        require(data.get("state_normalization_applied") is True,
                "state_normalization without state_normalization_applied: the server would feed raw state")

    print(
        "[ROBOTWIN CONTRACT OK] "
        "wire=[L6,R6,Lgrip,Rgrip] -> token=[<left>,L6,Lgrip,<right>,R6,Rgrip]; "
        f"14 values + 2 pinned delimiters; {action_text}; "
        f"chunk={action_model.get('action_horizon')}; state_normalization={data.get('state_normalization', 'raw')}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} CONFIG_FULL_YAML")
    main(sys.argv[1])

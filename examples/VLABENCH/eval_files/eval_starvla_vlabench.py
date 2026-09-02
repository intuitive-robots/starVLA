#!/usr/bin/env python3
"""Evaluate a StarVLA websocket policy on deterministic VLABench track episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

VLABENCH_REPO = Path(os.environ.get("VLABENCH_REPO", "/e/project1/m3/blank4/code/VLABench"))
sys.path.insert(0, str(VLABENCH_REPO))

from VLABench.evaluation.evaluator import Evaluator  # noqa: E402
from VLABench.robots import *  # noqa: F403,E402 - registry side effects
from VLABench.tasks import *  # noqa: F403,E402 - registry side effects
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy  # noqa: E402


class StarVLAVLABenchPolicy:
    """Translate VLABench observations to StarVLA examples and absolute EEF actions."""

    name = "StarVLA"
    control_mode = "ee"

    def __init__(self, host: str, port: int, replan_steps: int = 4):
        self.client = WebsocketClientPolicy(host, port)
        metadata = self.client.get_server_metadata()
        self.action_horizon = int(metadata["action_chunk_size"])
        self.replan_steps = min(int(replan_steps), self.action_horizon)
        self._actions: deque[np.ndarray] = deque()
        self._step = 0
        print(f"connected port={port} metadata={metadata}")

    def reset(self):
        self._actions.clear()
        self._step = 0

    def _replan(self, observation: dict) -> None:
        rgb = np.asarray(observation["rgb"])
        # Official conversion: camera 2 is `image` (front), camera 3 is wrist_image.
        example = {
            "image": [rgb[2], rgb[3]],
            "lang": str(observation["instruction"]),
        }
        response = self.client.predict_action(
            {
                "examples": [example],
                "unnorm_key": None,
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": 10,
            }
        )
        chunk = np.asarray(response["data"]["actions"], dtype=np.float32)[0]
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(f"expected [T,7] VLABench action chunk, got {chunk.shape}")
        self._actions.extend(chunk[: self.replan_steps])

    def predict(self, observation: dict, **kwargs):
        if not self._actions:
            self._replan(observation)
        action = self._actions.popleft()
        robot_frame = np.asarray(observation["robot_frame"], dtype=np.float32)
        world_position = action[:3] + robot_frame
        euler = action[3:6]
        gripper = np.full(2, 0.04 if action[6] >= 0.5 else 0.0, dtype=np.float32)
        self._step += 1
        return world_position, euler, gripper


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--track", default="track_1_in_distribution")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    track_path = VLABENCH_REPO / "VLABench" / "configs" / "evaluation" / "tracks" / f"{args.track}.json"
    episode_config = json.loads(track_path.read_text())
    missing = sorted(set(args.tasks) - set(episode_config))
    if missing:
        raise KeyError(f"tasks absent from {track_path}: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(
        tasks=args.tasks,
        n_episodes=args.num_episodes,
        episode_config=episode_config,
        max_substeps=1,
        save_dir=str(args.output),
        visulization=args.visualize,
        metrics=["success_rate", "intention_score", "progress_score"],
    )
    policy = StarVLAVLABenchPolicy(args.host, args.port, args.replan_steps)
    result = evaluator.evaluate(policy)
    (args.output / "evaluation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


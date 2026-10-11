"""Per-episode simulator trace for failure analysis (default off; STARVLA_EPISODE_TRACE=<dir>).

Records, at every policy-driven simulator step: end-effector position, gripper joint positions, the executed
action, and the xyz of every movable object; the json header names the goal source / target objects. One
compressed npz per episode, so failures can be classified offline (did the arm reach the goal object, lift it,
place it; where did it first close the gripper). Used by
examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py and
examples/LIBERO/eval_files/parallel_eval/eval_libero_shard.py.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np


def trace_dir() -> str:
    return os.environ.get("STARVLA_EPISODE_TRACE", "").strip()


def enabled() -> bool:
    return bool(trace_dir())


class EpisodeTrace:
    def __init__(self, env, sources, targets):
        self.env = env
        joints = {name: obj.joints[0] for name, obj in env.env.objects_dict.items() if obj.joints}
        self.names = sorted(joints)
        self.joints = [joints[n] for n in self.names]
        self.sources = list(sources)
        self.targets = list(targets)
        self.eef, self.grip, self.act, self.obj = [], [], [], []

    def step(self, obs, action=None):
        data = self.env.sim.data
        self.eef.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        self.grip.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
        self.act.append(np.zeros(7, np.float32) if action is None else np.asarray(action, dtype=np.float32)[:7])
        self.obj.append(np.asarray([np.asarray(data.get_joint_qpos(j))[:3] for j in self.joints],
                                   dtype=np.float32).reshape(len(self.joints), 3))

    def save(self, stem: str, meta: dict):
        out = pathlib.Path(trace_dir())
        out.mkdir(parents=True, exist_ok=True)
        n = len(self.eef)
        np.savez_compressed(
            out / f"{stem}.npz",
            eef=np.stack(self.eef) if n else np.zeros((0, 3), np.float32),
            grip=np.stack(self.grip) if n else np.zeros((0, 2), np.float32),
            act=np.stack(self.act) if n else np.zeros((0, 7), np.float32),
            obj=np.stack(self.obj) if n else np.zeros((0, len(self.joints), 3), np.float32),
        )
        header = dict(meta, objects=self.names, sources=self.sources, targets=self.targets, steps=n)
        (out / f"{stem}.json").write_text(json.dumps(header))

#!/usr/bin/env python3
"""Equivalence probe for ROBOTWIN_SKIP_SUBSTEP_RENDER (run inside the RoboTwin image via scripts/env/robotwin_python).

Builds a task env exactly as script/eval_policy.py does, then replays recorded demonstration actions (absolute 14-D
joints, env order [L6, Lgrip, R6, Rgrip]) through TASK_ENV.take_action, observing every `chunk` actions like the
starVLA client (get_obs() then `chunk` take_action calls). Records, after every action: both arms' joint states and the
pose of every physics entity (robot links, objects, table; camera entities excluded, they are render-side), success
flag; at every observation: the three RGB images and the joint vector; and the wall time of every take_action.

  robotwin_python render_skip_equivalence.py --task lift_pot --config demo_clean --seed 100000 \
      --actions /path/actions.npy --out /path/run.npz [--force_crazy_light]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.append("./description/utils")
import numpy as np
import yaml

from envs import CONFIGS_PATH


def build_args(task, task_config, force_crazy):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task
    args["task_config"] = task_config
    args["ckpt_setting"] = "render_skip_probe"
    args["eval_video_log"] = False
    if force_crazy:
        args["domain_randomization"]["random_light"] = True
        args["domain_randomization"]["crazy_random_light_rate"] = 1.0
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        cam = yaml.load(f.read(), Loader=yaml.FullLoader)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cam[head]["h"]
    args["head_camera_w"] = cam[head]["w"]
    robot_file = emb[args["embodiment"][0]]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        ecfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["left_embodiment_config"] = args["right_embodiment_config"] = ecfg
    args["eval_mode"] = True
    args["render_freq"] = 0
    return args


def physics_poses(env):
    out = []
    for ent in env.scene.get_entities():
        names = [type(c).__name__ for c in ent.components]
        if not any(n.startswith("Physx") for n in names):
            continue
        pose = ent.get_pose()
        out.append((ent.get_name(), np.concatenate([pose.p, pose.q]).astype(np.float64)))
    out.sort(key=lambda x: x[0])
    return [n for n, _ in out], np.stack([v for _, v in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--force_crazy_light", action="store_true")
    a = ap.parse_args()

    import importlib
    env = getattr(importlib.import_module(f"envs.{a.task}"), a.task)()
    import envs._base_task as bt
    args = build_args(a.task, a.config, a.force_crazy_light)
    env.setup_demo(now_ep_num=0, seed=a.seed, is_test=True, **args)
    actions = np.load(a.actions).astype(np.float64)
    n = min(len(actions), env.step_lim)

    joints, poses, succ, dts, obs_steps, obs_imgs, obs_state, obs_dts = [], [], [], [], [], [], [], []
    names = None
    i = 0
    while env.take_action_cnt < env.step_lim and i < n and not env.eval_success:
        t_obs = time.perf_counter()
        o = env.get_obs()
        obs_dts.append(time.perf_counter() - t_obs)
        obs_steps.append(env.take_action_cnt)
        obs_imgs.append(np.stack([o["observation"][c]["rgb"] for c in ("head_camera", "left_camera", "right_camera")]))
        obs_state.append(np.asarray(o["joint_action"]["vector"], dtype=np.float64))
        for _ in range(a.chunk):
            if i >= n or env.eval_success or env.take_action_cnt >= env.step_lim:
                break
            t0 = time.perf_counter()
            env.take_action(actions[i])
            dts.append(time.perf_counter() - t0)
            i += 1
            joints.append(np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(), dtype=np.float64))
            nm, ps = physics_poses(env)
            names = names or nm
            poses.append(ps)
            succ.append(bool(env.eval_success))
    np.savez_compressed(
        a.out, joints=np.stack(joints), poses=np.stack(poses), names=np.array(names), success=np.array(succ),
        dt=np.array(dts), obs_dt=np.array(obs_dts), obs_steps=np.array(obs_steps), obs_imgs=np.stack(obs_imgs), obs_state=np.stack(obs_state),
        crazy=bool(env.crazy_random_light), skip_flag=bool(getattr(bt, "_SKIP_SUBSTEP_RENDER", False)),
        step_lim=env.step_lim, ambient=np.asarray(env.scene.ambient_light, dtype=np.float64),
    )
    print(f"[PROBE] task={a.task} config={a.config} seed={a.seed} skip={getattr(bt, '_SKIP_SUBSTEP_RENDER', False)} "
          f"crazy={env.crazy_random_light} actions={i} success={env.eval_success} "
          f"mean_dt={np.mean(dts):.3f}s median_dt={np.median(dts):.3f}s mean_obs_dt={np.mean(obs_dts):.3f}s", flush=True)
    env.close_env()


if __name__ == "__main__":
    main()

"""Standalone probe: does a checkpoint's RAW gripper output ever cross the binarization
threshold during real LIBERO rollouts?

Context. `ervla_lit_stage2ctrl` scores 0/238 -- 0/20 on EVERY task -- while having BETTER
training loss (0.0142 vs 0.0301) and comparable open-loop MSE (0.0223 vs 0.0184) than the
arm scoring 87.5%. Action dim 6 is the gripper and is strictly bimodal at +/-1 in the data
(std 1.0001, q01 -1.0, q99 +1.0, mean -0.0496). The eval binarizes it, so a model whose raw
output never crosses the threshold is pinned to one state for a whole rollout and can
neither grasp nor release -- invisible to every MSE-space metric, because predicting the
mean of a bimodal variable is MSE-optimal.

A synthetic-noise probe could not settle this: both arms sat on one side of the threshold
there. This drives the real simulator.

Deliberately standalone -- it does NOT modify eval_libero_shard.py or any production path.
Run inside the LIBERO image against an already-running policy server.
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, "/e/project1/m3/blank4/code/starVLA")

# LIBERO's get_task_init_states() calls torch.load without weights_only, which PyTorch 2.6
# defaults to True -- it raises UnpicklingError on the numpy globals in the init-state
# files. eval_libero_shard.py:43-53 patches torch.load for exactly this reason rather than
# editing the pinned LIBERO checkout; the same applies here. This process never loads model
# weights (the policy server does), so nothing security-relevant is masked.
import torch as _torch

_orig_torch_load = _torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


_torch.load = _torch_load_compat

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0): return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=220)
    ap.add_argument("--out", default="gripper_probe.jsonl")
    ap.add_argument("--include_state", type=int, default=1)
    ap.add_argument("--task_id", type=int, default=0)
    a = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[a.suite]()
    client = ModelClient(host=a.host, port=a.port, unnorm_key=None)
    out = pathlib.Path(a.out); out.write_text("")

    # ONE env for the whole probe. Creating a second OffScreenRenderEnv in the same
    # process -- even after env.close() -- SIGABRTs in MuJoCo/EGL, which silently truncated
    # an earlier run to a single episode. Episodes are varied by init state instead, which
    # is what the real eval does within a task anyway.
    task = suite.get_task(a.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    init = suite.get_task_init_states(a.task_id)
    print(f"[probe] task={task.language!r}  init_states={len(init)}", flush=True)

    for ep in range(a.episodes):
        env.reset()
        obs = env.set_init_state(init[ep % len(init)])

        raws, done = [], False
        for step in range(a.max_steps):
            if step < 10:
                obs, _, done, _ = env.step(LIBERO_DUMMY); continue
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            ex = {"image": [img, wrist], "lang": str(task.language)}
            if a.include_state:
                st = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                     obs["robot0_gripper_qpos"])).astype(np.float32)
                ex["state"] = st[None]                      # (1, 8) -- the shard's contract
            r = client.step(example=ex, step=step)["raw_action"]
            wv = np.asarray(r["world_vector"], np.float32).reshape(-1)
            rot = np.asarray(r["rotation_delta"], np.float32).reshape(-1)
            g = float(np.asarray(r["open_gripper"], np.float32).reshape(-1)[0])
            raws.append(g)
            act = np.concatenate([wv, rot, [1.0 - 2.0 * (g > 0.5)]])
            obs, _, done, _ = env.step(act.tolist())
            if done: break
        arr = np.asarray(raws)
        rec = {"episode": ep, "task": str(task.language), "steps": len(raws), "success": bool(done),
               "raw_min": float(arr.min()), "raw_max": float(arr.max()),
               "raw_mean": float(arr.mean()), "raw_std": float(arr.std()),
               "crosses_0.5": bool(arr.min() <= 0.5 <= arr.max()),
               "frac_above_0.5": float((arr > 0.5).mean())}
        with out.open("a") as f: f.write(json.dumps(rec) + "\n")
        print(f"ep{ep:02d} steps={rec['steps']:3d} succ={rec['success']!s:5} "
              f"raw[{rec['raw_min']:+.3f},{rec['raw_max']:+.3f}] mean={rec['raw_mean']:+.3f} "
              f"crosses={rec['crosses_0.5']!s:5} frac>0.5={rec['frac_above_0.5']:.3f}", flush=True)


if __name__ == "__main__":
    main()

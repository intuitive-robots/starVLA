"""Open-loop check of converted pi0.5 checkpoints on DROID frames (2026-10-11).

Frames come from the droid_pi05_2cam loader (eval transforms: no augmentation), so this also exercises the new data
config: left external + wrist at 180x320, q99 state, joints relative to s_0 + absolute gripper, openpi stats.
For every checkpoint the PI05 model predicts the 16-step chunk (10 Euler steps, fixed per-frame noise seed) and is
scored against the ground truth over the 15 executed steps:
  nmse      mean squared error in the normalised [-1, 1] space (8 dims)
  joint_mae mean |error| of the 7 joint targets in rad (unnormalised deltas = absolute target error)
  grip_mae  mean |error| of the gripper in [0, 1]
Baselines: 'hold' (zero joint delta, i.e. stay at the current joints, gripper kept at its current value) and
'mean' (the per-step mean ground-truth chunk of these frames, an oracle constant).
usage: python verify_open_loop.py --code <dir> --yaml <yaml> --ckpt NAME=path [--ckpt ...] --n 64 --out <json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--code", required=True)
p.add_argument("--yaml", required=True)
p.add_argument("--ckpt", action="append", required=True)
p.add_argument("--n", type=int, default=64)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--steps", type=int, default=15)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="")
# v2: override framework.image_keys (comma list: 2 slots = v1, 3 = openpi's padded right_wrist slot) and the data mix /
# stats file, so 2- and 3-slot inference are scored on the same frames
p.add_argument("--image_keys", default="")
p.add_argument("--data_mix", default="")
p.add_argument("--stats_override_path", default="")
p.add_argument("--elide", default="", choices=["", "0", "1"], help="override framework.elide_masked_image_slots")
a = p.parse_args()
sys.path.insert(0, a.code)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import starVLA  # noqa: E402
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

print("starVLA from", starVLA.__file__, flush=True)
cfg = OmegaConf.load(a.yaml)
cfg.trainer.pretrained_checkpoint = None
if a.image_keys:
    cfg.framework.image_keys = a.image_keys.split(",")
if a.data_mix:
    cfg.datasets.vla_data.data_mix = a.data_mix
if a.stats_override_path:
    cfg.datasets.vla_data.stats_override_path = a.stats_override_path
if a.elide:
    cfg.framework.elide_masked_image_slots = a.elide == "1"
print("image_keys", list(cfg.framework.image_keys), "elide", cfg.framework.get("elide_masked_image_slots", False),
      "data_mix", cfg.datasets.vla_data.data_mix, flush=True)
dcfg =OmegaConf.to_container(cfg.datasets.vla_data, resolve=True)
mix = __import__("starVLA.dataloader.gr00t_lerobot.mixtures", fromlist=["x"]).DATASET_NAMED_MIXTURES[dcfg["data_mix"]]
name, _, robot_type = mix[0]
t0 = time.time()
ds = make_LeRobotSingleDataset(Path(dcfg["data_root_dir"]), name, robot_type, data_cfg=dcfg)
ds.transforms.eval()
print(f"dataset {name}: {len(ds)} steps, {len(ds.trajectory_ids)} trajectories, built in {time.time() - t0:.0f}s", flush=True)
rng = np.random.default_rng(a.seed)
idx = sorted(rng.choice(len(ds), size=a.n, replace=False).tolist())
samples = [ds[i] for i in idx]
S = a.steps
gt = np.stack([np.asarray(s["action"], dtype=np.float32) for s in samples])     # [N, 16, 8] normalised
st = np.stack([np.asarray(s["state"], dtype=np.float32).reshape(-1) for s in samples])  # [N, 8] normalised
print("example:", samples[0]["lang"], [im.size for im in samples[0]["image"]], "gt range", gt.min(), gt.max(),
      "state range", st.min(), st.max(), flush=True)
stats = json.load(open(dcfg["stats_override_path"]))["statistics"]
q01 = np.array(stats["action.joint_position"]["q01"] + stats["action.gripper_position"]["q01"], dtype=np.float32)
q99 = np.array(stats["action.joint_position"]["q99"] + stats["action.gripper_position"]["q99"], dtype=np.float32)
gk = "observation.gripper_position" if "observation.gripper_position" in stats else "observation.state.gripper_position"
sq01 = np.array(stats[gk]["q01"], dtype=np.float32)
sq99 = np.array(stats[gk]["q99"], dtype=np.float32)


def unnorm(x):
    return (x + 1.0) / 2.0 * (q99 - q01) + q01


def score(pred):
    pred = pred[:, :S]
    g = gt[:, :S]
    pu, gu = unnorm(pred), unnorm(g)
    return {"nmse": float(((pred - g) ** 2).mean()),
            "joint_mae_rad": float(np.abs(pu[..., :7] - gu[..., :7]).mean()),
            "joint_mae_rad_step0": float(np.abs(pu[:, 0, :7] - gu[:, 0, :7]).mean()),
            "joint_mae_rad_last": float(np.abs(pu[:, -1, :7] - gu[:, -1, :7]).mean()),
            "grip_mae": float(np.abs(pu[..., 7] - gu[..., 7]).mean())}


res = {"n": a.n, "steps_scored": S, "indices": idx, "image_keys": list(cfg.framework.image_keys),
       "data_mix": str(cfg.datasets.vla_data.data_mix),
       "elide_masked_image_slots": bool(cfg.framework.get("elide_masked_image_slots", False))}
hold = np.zeros_like(gt)
hold[..., :7] = 2.0 * (0.0 - q01[:7]) / (q99[:7] - q01[:7]) - 1.0                      # zero joint delta
grip_now = (st[:, 7] + 1.0) / 2.0 * (sq99 - sq01) + sq01                                   # current gripper
hold[..., 7] = (2.0 * (grip_now - q01[7]) / (q99[7] - q01[7]) - 1.0)[:, None]
res["hold"] = score(hold)
res["mean_oracle"] = score(np.broadcast_to(gt.mean(0, keepdims=True), gt.shape).copy())
print("baseline hold", res["hold"], flush=True)
print("baseline mean", res["mean_oracle"], flush=True)

model = build_framework(cfg).to("cuda").eval()
for spec in a.ckpt:
    tag, path = spec.split("=", 1)
    model.load_pretrained_checkpoint(path)
    model.to_precision("bfloat16")
    preds = []
    t0 = time.time()
    for b in range(0, a.n, a.bs):
        ex = [{"image": s["image"], "lang": s["lang"], "state": np.asarray(s["state"], dtype=np.float32),
               "flow_seed": 1000 + i} for i, s in zip(idx[b:b + a.bs], samples[b:b + a.bs])]
        out = model.predict_action(ex)["normalized_actions"]
        preds.append(np.asarray(out, dtype=np.float32)[..., :8])
    pred = np.concatenate(preds)
    res[tag] = score(pred)
    res[tag]["sec"] = time.time() - t0
    res[tag]["pred_range"] = [float(pred.min()), float(pred.max())]
    print(f"RESULT {tag}: {res[tag]}", flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)

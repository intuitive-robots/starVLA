#!/usr/bin/env python
"""Does this checkpoint's action actually depend on the image?

Motivation: ervla_lit_stage2ctrl scores 0/243 in the simulator while its open-loop
(teacher-forced) trajectory eval tracks ground truth almost perfectly.  That pair of facts
is the classic signature of a policy that regresses actions from proprioceptive state alone:
open-loop replay feeds it ground-truth state at every step, so it looks perfect; closed-loop
it has no visual grounding, drifts, and never recovers.  Stage 1 of the LIT curriculum makes
this a live hypothesis -- the control arm trained 15k steps with removal=pixels,
vision_rate=1.0 AND privileged_tokens disabled, so state was the only scene-carrying input
it had.

The probe holds one input fixed and swaps the other between observations drawn from
genuinely different scenes, then measures how much the predicted action moves:

  repeat   (img_i, st_i) twice   -> diffusion sampling noise floor
  visswap  (img_j, st_i)         -> how much the action depends on the IMAGE
  stateswap(img_i, st_j)         -> how much the action depends on the STATE

A vision-blind policy has visswap ~= repeat while stateswap >> repeat.

Runs inside the LIBERO image against an already-running policy server; touches no
production eval file.
"""
import argparse, json, os, sys

import numpy as np

# LIBERO's get_task_init_states() calls torch.load without weights_only, which PyTorch 2.6
# defaults to True -- it raises UnpicklingError on the numpy globals in the init-state
# pickles.  Same patch the eval shard applies.
import torch
_orig_load = torch.load
def _load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _load

sys.path.insert(0, os.environ.get("STARVLA_ROOT", "/e/project1/m3/blank4/code/starVLA"))
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def collect_observations(task_ids, n_per_task, warmup):
    """Return a list of (image, wrist, state, lang) from distinct scenes."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    obs_list = []
    for tid in task_ids:
        task = suite.get_task(tid)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
        )
        init = suite.get_task_init_states(tid)
        for k in range(n_per_task):
            env.reset()
            obs = env.set_init_state(init[k % len(init)])
            # Let the arm settle so the state is not the identical reset pose every time.
            for _ in range(warmup):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            st = np.concatenate([
                obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]
            ]).astype(np.float32)
            obs_list.append({
                "image": [img, wrist],
                "state": st,
                "lang": str(task.language),
                "task_id": tid,
            })
        env.close()
    return obs_list


def _resize(images, size):
    """Match ModelClient's preprocessing exactly; the server's training_obs_image_size is
    part of the observation contract and a 256x256 image to a 224-trained checkpoint is a
    different distribution."""
    if size is None:
        return images
    from PIL import Image as _Image
    h, w = size
    out = []
    for im in images:
        arr = np.asarray(im)
        if arr.shape[:2] != (h, w):
            arr = np.asarray(_Image.fromarray(arr).resize((w, h), _Image.Resampling.BILINEAR))
        out.append(arr)
    return out


def query(client, image, state, lang, expects_state, unnorm_key, image_size=None):
    ex = {"image": _resize(image, image_size), "lang": lang}
    if expects_state:
        ex["state"] = np.asarray(state)[None]       # (1, 8) -- the shard's contract
    resp = client.predict_action({
        "examples": [ex],
        "unnorm_key": unnorm_key,
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
    })
    return np.asarray(resp["data"]["actions"])[0]   # (T, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task_ids", default="0,3,6")
    ap.add_argument("--n_per_task", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--unnorm_key", default=None)
    a = ap.parse_args()

    task_ids = [int(t) for t in a.task_ids.split(",")]
    obs = collect_observations(task_ids, a.n_per_task, a.warmup)
    print(f"[probe] collected {len(obs)} observations from tasks {task_ids}", flush=True)

    client = WebsocketClientPolicy(a.host, a.port)
    meta = client.get_server_metadata()
    expects_state = bool(meta.get("expects_state", False))
    image_size = (
        meta.get("training_obs_image_size")
        if meta.get("image_preprocessing_contract_version") == 2
        else None
    )
    if image_size is not None:
        image_size = (int(image_size[0]), int(image_size[1]))
    print(f"[probe] expects_state={expects_state} image_size={image_size} meta={meta}", flush=True)

    n = len(obs)
    rows = []
    for i in range(n):
        j = (i + n // 2) % n                       # partner from a different scene
        if obs[j]["task_id"] == obs[i]["task_id"]:
            j = (i + 1) % n
        base = query(client, obs[i]["image"], obs[i]["state"], obs[i]["lang"],
                     expects_state, a.unnorm_key, image_size)
        rep = query(client, obs[i]["image"], obs[i]["state"], obs[i]["lang"],
                    expects_state, a.unnorm_key, image_size)
        # Language is part of the scene; hold it at scene i so the only thing that moves
        # is the pixels.  Swapping the instruction too would conflate vision with text.
        vis = query(client, obs[j]["image"], obs[i]["state"], obs[i]["lang"],
                    expects_state, a.unnorm_key, image_size)
        sta = query(client, obs[i]["image"], obs[j]["state"], obs[i]["lang"],
                    expects_state, a.unnorm_key, image_size)
        rows.append({
            "i": i, "j": j,
            "repeat":    float(np.abs(base - rep).mean()),
            "visswap":   float(np.abs(base - vis).mean()),
            "stateswap": float(np.abs(base - sta).mean()),
            "act_absmax": float(np.abs(base).max()),
        })
        print(f"[probe] {i:2d}<->{j:2d}  repeat={rows[-1]['repeat']:.4f} "
              f"visswap={rows[-1]['visswap']:.4f} stateswap={rows[-1]['stateswap']:.4f}",
              flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    floor = m("repeat")
    summary = {
        "tag": a.tag,
        "n_pairs": n,
        "expects_state": expects_state,
        "image_size": image_size,
        "repeat": floor,
        "visswap": m("visswap"),
        "stateswap": m("stateswap"),
        # How many noise floors of action change each swap buys.  ~1.0 means the input
        # is being ignored.
        "vis_over_floor": m("visswap") / floor if floor > 0 else float("inf"),
        "state_over_floor": m("stateswap") / floor if floor > 0 else float("inf"),
    }
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": summary}) + "\n")
    print("[probe] SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

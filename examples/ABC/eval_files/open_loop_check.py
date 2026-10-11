#!/usr/bin/env python
"""Open-loop check of an ABC-130K (m3 bridge) checkpoint through the policy-server code path.

Draws real training samples from the checkpoint's own m3 data config (no augmentation), sends the RAW
state and the letterboxed images through ``PolicyServerWrapper.predict_action`` (the same normalise ->
model -> un-normalise path the websocket server runs), and compares the un-normalised 30 x 14 chunk with
the recorded commanded actions. Baselines: "hold" (repeat the current state for 30 steps) and the
per-embodiment mean action. Also checks that the ABC-Sim client's letterbox is the bridge's, pixel for
pixel, and writes one sample's three camera images for a visual comparison with sim frames.

Run in the starVLA image on a GPU (compute node), from the live tree:
    # torchcodec (AV1) needs the FFmpeg shim, as in train_multinode_container_slurm.sh:
    C13=/e/software/default/stages/2026/software/CUDA/13
    export APPTAINERENV_LD_LIBRARY_PATH=$C13/targets/sbsa-linux/lib:/e/project1/m3/blank4/containers/ffmpeg_shim:$C13/lib64:/opt/conda/envs/starVLA/lib/python3.12/site-packages/torch/lib:/opt/conda/envs/starVLA/lib
    PYTHONPATH=$PWD scripts/env/starvla_python examples/ABC/eval_files/open_loop_check.py \
        --ckpt playground/Checkpoints/<run>/checkpoints/steps_N_pytorch_model.pt --num_samples 96 --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

JOINT_NAMES = [f"L{i}" for i in range(1, 7)] + ["Lg"] + [f"R{i}" for i in range(1, 7)] + ["Rg"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_samples", type=int, default=96)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    run_dir = Path(args.ckpt).resolve().parents[1]
    full = OmegaConf.to_container(OmegaConf.load(run_dir / "config.full.yaml"), resolve=True)
    vla = dict(full["datasets"]["vla_data"])
    vla["augmentation"] = "none"
    vla["shared_z_future_offset"] = 0
    # a different stream from training's (sample_seed is in the m3 config; resume shift changes it)
    vla["m3_resume_steps"] = 977

    from starVLA.dataloader.m3_data_datasets import StarVLAM3DataReader, _letterbox
    from abc_sim_client import letterbox_like_training

    # 1) client letterbox == bridge letterbox (random frames at the sim size and at a native size)
    rng = np.random.default_rng(args.seed)
    max_diff = 0
    for hw in [(168, 224), (480, 640), (480, 848), (1200, 1920)]:
        frame = rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8)
        # bridge: m3 stretches the native frame to 224x224, then _letterbox with the native shape
        stretched = Image.fromarray(frame).resize((224, 224), Image.Resampling.BILINEAR)
        ref = np.asarray(_letterbox(stretched, hw))
        got = letterbox_like_training(frame, native_hw=hw)
        max_diff = max(max_diff, int(np.abs(ref.astype(int) - got.astype(int)).max()))
    print(f"[letterbox] client vs bridge max abs pixel diff over 4 shapes: {max_diff}", flush=True)

    # 2) samples
    reader = StarVLAM3DataReader(vla, mode="train")
    samples = []
    t0 = time.time()
    for raw in reader.reader:
        conv = reader._convert(raw)
        raw_state = torch.as_tensor(raw["observation"]).float().numpy()[-1, reader.state_idx]
        raw_actions = torch.as_tensor(raw["action"]).float().numpy()[:, reader.action_idx]
        samples.append((conv, raw_state, raw_actions))
        if len(samples) >= args.num_samples:
            break
    print(f"[data] {len(samples)} samples in {time.time() - t0:.1f}s", flush=True)
    langs = defaultdict(int)
    for conv, _, _ in samples:
        langs[(conv["robot_tag"], conv["lang"])] += 1
    for (emb, lang), n in sorted(langs.items(), key=lambda kv: -kv[1])[:30]:
        print(f"[lang] {n:3d}  {emb:28s} {lang!r}")
    conv0 = samples[0][0]
    for i, im in enumerate(conv0["image"]):
        im.save(out / f"train_sample0_cam{i}.png")

    # 3) policy, through the server wrapper
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    wrapper = PolicyServerWrapper(ckpt_path=args.ckpt, device="cuda", use_bf16=True, is_inference=True)
    meta = wrapper.metadata
    print("[meta]", json.dumps({k: v for k, v in meta.items() if k != "m3_contract"}, default=str))
    print("[meta] m3_contract", json.dumps(meta.get("m3_contract"), default=str)[:800])

    rows = []
    for conv, raw_state, raw_actions in samples:
        emb = conv["robot_tag"]
        ex = {"image": [np.asarray(im) for im in conv["image"]], "lang": conv["lang"], "state": raw_state}
        pred = wrapper.predict_action(examples=[ex], unnorm_key=emb)["actions"][0]  # (30, 14)
        # the server's state normalisation must equal the bridge's
        st_err = float(np.abs(wrapper._get_processor(emb).apply_state(raw_state) - conv["state"].astype(np.float32)).max())
        valid = np.asarray(conv["action_valid_mask"], bool)
        if not valid.any():  # anchor at the very end of an episode: no recorded future
            continue
        rows.append(dict(emb=emb, lang=conv["lang"], pred=pred, gt=raw_actions, state=raw_state, valid=valid,
                         state_norm_err=st_err))

    # 4) metrics in raw units (rad for joints, [0,1] for grippers)
    def mae(fn):
        per = []
        for r in rows:
            v = r["valid"]
            per.append(np.abs(fn(r)[v] - r["gt"][v]).mean(axis=0))
        return np.mean(per, axis=0)

    mean_act = {e: np.mean([r["gt"].mean(0) for r in rows if r["emb"] == e], axis=0) for e in {r["emb"] for r in rows}}
    m_pred = mae(lambda r: r["pred"])
    m_hold = mae(lambda r: np.repeat(r["state"][None], len(r["gt"]), 0))
    m_mean = mae(lambda r: np.repeat(mean_act[r["emb"]][None], len(r["gt"]), 0))
    by_t = np.mean([np.abs(r["pred"] - r["gt"]).mean(1) for r in rows], axis=0)
    by_t_hold = np.mean([np.abs(r["state"][None] - r["gt"]).mean(1) for r in rows], axis=0)
    motion_pred = np.mean([np.abs(r["pred"][-1] - r["pred"][0]).mean() for r in rows])
    motion_gt = np.mean([np.abs(r["gt"][-1] - r["gt"][0]).mean() for r in rows])
    corr = np.corrcoef(np.concatenate([(r["pred"] - r["state"]).ravel() for r in rows]),
                       np.concatenate([(r["gt"] - r["state"]).ravel() for r in rows]))[0, 1]

    print("\n[open-loop] MAE per dim (raw units), model / hold-state / mean-action:")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:3s}  {m_pred[j]:.4f}  {m_hold[j]:.4f}  {m_mean[j]:.4f}")
    print(f"  ALL  {m_pred.mean():.4f}  {m_hold.mean():.4f}  {m_mean.mean():.4f}")
    print("[open-loop] MAE by horizon step (model / hold): " + " ".join(
        f"t{t}:{by_t[t]:.3f}/{by_t_hold[t]:.3f}" for t in (0, 4, 9, 14, 19, 29)))
    print(f"[open-loop] mean |a[29]-a[0]| model {motion_pred:.4f} vs recorded {motion_gt:.4f}")
    print(f"[open-loop] corr(pred - state, gt - state) over all dims/steps: {corr:.3f}")
    print(f"[open-loop] max |server state norm - bridge state norm|: {max(r['state_norm_err'] for r in rows):.2e}")
    print(f"[open-loop] gripper pred range L [{min(r['pred'][:, 6].min() for r in rows):.3f}, "
          f"{max(r['pred'][:, 6].max() for r in rows):.3f}] R [{min(r['pred'][:, 13].min() for r in rows):.3f}, "
          f"{max(r['pred'][:, 13].max() for r in rows):.3f}]")
    by_emb = defaultdict(list)
    for r in rows:
        by_emb[r["emb"]].append(np.abs(r["pred"][r["valid"]] - r["gt"][r["valid"]]).mean())
    for e, v in sorted(by_emb.items()):
        print(f"[open-loop] {e}: n={len(v)} MAE {np.mean(v):.4f}")
    np.savez(out / "open_loop.npz", pred=np.stack([r["pred"] for r in rows]), gt=np.stack([r["gt"] for r in rows]),
             state=np.stack([r["state"] for r in rows]), valid=np.stack([r["valid"] for r in rows]),
             emb=np.array([r["emb"] for r in rows]), lang=np.array([r["lang"] for r in rows]))
    summary = dict(ckpt=args.ckpt, n=len(rows), letterbox_max_diff=max_diff, mae_model=float(m_pred.mean()),
                   mae_hold=float(m_hold.mean()), mae_mean_action=float(m_mean.mean()),
                   mae_model_per_dim=m_pred.tolist(), mae_hold_per_dim=m_hold.tolist(), corr_delta=float(corr),
                   motion_pred=float(motion_pred), motion_gt=float(motion_gt))
    (out / "open_loop_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[open-loop] wrote {out}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # m3 reader worker threads


if __name__ == "__main__":
    main()

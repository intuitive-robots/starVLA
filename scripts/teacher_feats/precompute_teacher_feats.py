#!/usr/bin/env python
"""Precompute frozen teacher patch features for every frame of a LeRobot v3 dataset (2026-10-08).

Targets for the teacher-alignment arms (T1-T6): per episode one fp16 array [T, n_cams, G*G, D] of the teacher's patch
features on the CLEAN frame (no augmentation), average-pooled to a G x G grid (default 14, i.e. 2x the policy's 7 x 7
token grid at 224 px, so it pools exactly onto it). Frames are resized to 224 x 224 as the policy sees them.

Teachers:
  dinov2    DINOv2-B (weights/dinov2-base), 224 px, last hidden state, D = 768
  dynaflip  DynaFLIP-base (DINOv2-B fine-tuned for dynamics, weights/dynaflip-base), as above, D = 768
  dinov3    DINOv3 ViT-B/16 LVD-1689M (timm mirror, weights/dinov3-vitb16-timm), 224 px, patch tokens after the 5 prefix
            tokens (CLS + 4 registers), D = 768
  vggt      VGGT-1B (weights/vggt-1b), both cameras of a timestep as one 2-view input at --vggt_res px, last
            aggregator layer patch tokens (frame + global attention concatenated), D = 2048

Run one shard per GPU inside an allocation:
  PYTHONPATH=$PWD:/e/project1/m3/blank4/code/vggt CUDA_VISIBLE_DEVICES=g scripts/env/starvla_python \
      scripts/teacher_feats/precompute_teacher_feats.py --teacher vggt --shard g --nshards 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

W = Path("/e/project1/m3/blank4/weights")


def load_teacher(name: str, device):
    if name in ("dinov2", "dynaflip"):
        from transformers import Dinov2Model

        enc = Dinov2Model.from_pretrained(str(W / "dinov2-base"))
        if name == "dynaflip":
            from safetensors.torch import load_file

            sd = load_file(str(W / "dynaflip-base/model.safetensors"))
            pre = "vision_encoder.backbone."
            missing, unexpected = enc.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
                                                      strict=False)
            assert not missing and not unexpected, (missing[:3], unexpected[:3])
        return enc.to(device).eval().requires_grad_(False)
    if name == "dinov3":
        import timm

        m = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, pretrained_cfg_overlay=dict(
            file=str(W / "dinov3-vitb16-timm/model.safetensors")))
        return m.to(device).eval().requires_grad_(False)
    if name == "vggt":
        from vggt.models.vggt import VGGT

        m = VGGT()
        from safetensors.torch import load_file

        m.load_state_dict(load_file(str(W / "vggt-1b/model.safetensors")))
        return m.to(device).eval().requires_grad_(False)
    raise ValueError(name)


@torch.no_grad()
def feats_dino(enc, frames: torch.Tensor, grid: int) -> torch.Tensor:
    """frames [N, 3, 224, 224] float in [0, 1] -> [N, grid*grid, 768]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(pixel_values=(frames - mean) / std).last_hidden_state[:, 1:]
    h = h.float()
    g = int(round(h.shape[1] ** 0.5))
    h = h.transpose(1, 2).reshape(h.shape[0], h.shape[2], g, g)
    return F.adaptive_avg_pool2d(h, grid).flatten(2).transpose(1, 2)


@torch.no_grad()
def feats_dinov3(m, frames: torch.Tensor, grid: int) -> torch.Tensor:
    """frames [N, 3, 224, 224] in [0, 1] -> [N, grid*grid, 768] (prefix tokens dropped)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = m.forward_features((frames - mean) / std)[:, m.num_prefix_tokens:]
    h = h.float()
    g = int(round(h.shape[1] ** 0.5))
    h = h.transpose(1, 2).reshape(h.shape[0], h.shape[2], g, g)
    return F.adaptive_avg_pool2d(h, grid).flatten(2).transpose(1, 2)


@torch.no_grad()
def feats_vggt(model, frames: torch.Tensor, n_cams: int, grid: int, res: int) -> torch.Tensor:
    """frames [B*n_cams, 3, 224, 224] in [0, 1] (timestep-major) -> [B*n_cams, grid*grid, 2048]."""
    x = F.interpolate(frames, size=(res, res), mode="bilinear", align_corners=False)
    x = x.view(-1, n_cams, 3, res, res)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens_list, patch_start = model.aggregator(x)
    h = tokens_list[-1][:, :, patch_start:].float()  # [B, S, P, 2048]
    B, S, P, D = h.shape
    g = int(round(P ** 0.5))
    h = h.reshape(B * S, g, g, D).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(h, grid).flatten(2).transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=["dinov2", "dynaflip", "dinov3", "vggt"])
    ap.add_argument("--root", default="/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/libero_lerobot/libero")
    ap.add_argument("--cams", default="observation.images.image,observation.images.image2")
    ap.add_argument("--out", default="/e/scratch/m3/blank4/teacher_feats/libero")
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--vggt_res", type=int, default=518)
    ap.add_argument("--batch", type=int, default=64, help="timesteps per forward")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_episodes", type=int, default=0, help="smoke: stop after this many")
    args = ap.parse_args()

    from torchcodec.decoders import VideoDecoder

    device = torch.device("cuda")
    root = Path(args.root)
    cams = args.cams.split(",")
    info = json.load(open(root / "meta/info.json"))
    fps = float(info["fps"])
    eps = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True)))
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    mine = eps[eps.episode_index.astype(int) % args.nshards == args.shard]
    out = Path(args.out) / args.teacher
    out.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(args.teacher, device)
    decoders = {}
    t0, n_frames, n_done = time.time(), 0, 0
    for _, ep in mine.iterrows():
        ei, T = int(ep.episode_index), int(ep.length)
        dst = out / f"ep_{ei:06d}.npy"
        if dst.exists():
            continue
        per_cam = []
        for cam in cams:
            vp = root / info["video_path"].format(video_key=cam, chunk_index=int(ep[f"videos/{cam}/chunk_index"]),
                                                   file_index=int(ep[f"videos/{cam}/file_index"]))
            if str(vp) not in decoders:
                decoders.clear()  # one file open per camera at a time
                decoders[str(vp)] = VideoDecoder(str(vp))
            ts = float(ep[f"videos/{cam}/from_timestamp"]) + np.arange(T) / fps
            fr = decoders[str(vp)].get_frames_played_at(seconds=ts.tolist()).data  # [T, 3, H, W] uint8
            per_cam.append(fr)
        frames = torch.stack(per_cam, 1)  # [T, n_cams, 3, H, W]
        feats = []
        for s in range(0, T, args.batch):
            x = frames[s:s + args.batch].to(device).float().div_(255.0)
            b = x.shape[0]
            x = x.reshape(b * len(cams), *x.shape[2:])
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
            if args.teacher == "vggt":
                f = feats_vggt(teacher, x, len(cams), args.grid, args.vggt_res)
            elif args.teacher == "dinov3":
                f = feats_dinov3(teacher, x, args.grid)
            else:
                f = feats_dino(teacher, x, args.grid)
            feats.append(f.reshape(b, len(cams), *f.shape[1:]).half().cpu())
        arr = torch.cat(feats).numpy()
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, dst)
        n_frames += T
        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            print(f"[{args.teacher} shard {args.shard}] {n_done}/{len(mine)} eps, {n_frames / (time.time() - t0):.1f} "
                  f"frames/s, last {dst.name} {arr.shape}", flush=True)
        if args.max_episodes and n_done >= args.max_episodes:
            break
    print(f"[{args.teacher} shard {args.shard}] DONE {n_done} eps {n_frames} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

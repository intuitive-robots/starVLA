#!/usr/bin/env python
"""Trainer-style open-loop trajectory plots for an ABC-130K (m3 bridge) checkpoint, offline.

Runs the trainer's own protocol, ``VLATrainer._eval_marigold_open_loop_trajectories`` (fixed episodes cut
into non-overlapping 30-step chunks, one prediction per chunk with the 20-step DDIM sampler and a fixed
seed, stitched over the episode) and its ``_plot_open_loop_trajectory``, on a light stand-in object
instead of a trainer. The episodes come from ``StarVLAM3DataReader.build_open_loop_episodes`` /
``materialize_open_loop_episode`` in eval mode, built from the checkpoint's own ``config.full.yaml``, so
every sample is converted exactly like a training sample (letterbox, q01/q99 normalisation, camera
order, state / action fields) without augmentation.

Episodes: per task, round robin over that task's subdatasets (largest first), the last eligible episodes
of each. The ABC-130K configs carry no m3 ``holdout`` block, so these are TRAINING episodes (there is no
held-out split for these checkpoints); with a holdout block (mode ``only``) the same code serves held-out
ones. ``m3_resume_steps`` is forced to 0, so every checkpoint of a run sees the same instruction draws.

Output, per checkpoint: <out_root>/<run>/step_<N>/
  <subdataset>_traj_<episode_idx>_<uuid>_{normalized,physical}.png   (the trainer's plots)
  <same stem>.npz   (gt / pred, normalized and physical, source frames, chunk lengths)
  metrics.json      (the trainer's open-loop metrics + the episode list); written last = done marker

Run in the starVLA image on a GPU (the login GPU is fine; no MuJoCo here), from the live tree:
    C13=/e/software/default/stages/2026/software/CUDA/13
    export APPTAINERENV_LD_LIBRARY_PATH=$C13/targets/sbsa-linux/lib:/e/project1/m3/blank4/containers/ffmpeg_shim:$C13/lib64:/opt/conda/envs/starVLA/lib/python3.12/site-packages/torch/lib:/opt/conda/envs/starVLA/lib
    PYTHONPATH=$PWD scripts/env/starvla_python examples/ABC/eval_files/open_loop_traj.py \
        --ckpt playground/Checkpoints/<run>/checkpoints/steps_N_pytorch_model.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

DEFAULT_OUT_ROOT = "/e/scratch/m3/blank4/abc/openloop_traj"


def _task_of(subdataset_name: str) -> str:
    """'d405_640x480_60hz_h264_nopose/put_the_plastic_bottles_in_the_bin__d405_...' -> 'put_the_plastic_bottles_in_the_bin'."""
    return subdataset_name.split("/")[-1].split("__")[0]


def plan_episodes(processed: dict, episodes_per_task: int, tasks: list[str] | None) -> list[tuple[str, int]]:
    """[(subdataset name, number of episodes)]: per task, round robin over its subdatasets, largest first."""
    from m3_data.utils.sharding_plan import flatten_subdatasets

    by_task: dict[str, list[tuple[int, str]]] = {}
    for entry in flatten_subdatasets(processed):
        task = _task_of(entry["name"])
        if tasks and task not in tasks:
            continue
        by_task.setdefault(task, []).append((int(entry["cfg"].get("total_frames", 0) or 0), entry["name"]))
    plan: list[tuple[str, int]] = []
    for task in sorted(by_task):
        subs = [name for _, name in sorted(by_task[task], reverse=True)]
        counts = {name: 0 for name in subs}
        for i in range(episodes_per_task):
            counts[subs[i % len(subs)]] += 1
        plan.extend((name, n) for name, n in counts.items() if n > 0)
    return plan


class _EpisodeSelection:
    """The eval reader, with ``build_open_loop_episodes`` restricted to a per-subdataset episode plan."""

    def __init__(self, dataset, plan: list[tuple[str, int]]):
        self._dataset = dataset
        self._plan = plan
        self.normalizer = dataset.normalizer
        self.selected: list[dict] = []

    def build_open_loop_episodes(self, *, action_horizon, num_episodes_per_subdataset=1, max_anchors=None):
        episodes = []
        for name, count in self._plan:
            episodes += self._dataset.build_open_loop_episodes(
                action_horizon=action_horizon, num_episodes_per_subdataset=count,
                max_anchors=max_anchors, subdataset_names=[name])
        self.selected = episodes
        return episodes

    def materialize_open_loop_episode(self, episode, *, seed):
        return self._dataset.materialize_open_loop_episode(episode, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="<run>/checkpoints/steps_N_pytorch_model.pt")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--episodes_per_task", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", default=None, help="task names (default: every task in the config)")
    ap.add_argument("--max_anchors", type=int, default=None, help="cap on chunks per episode (default: whole episode)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no_bf16", action="store_true")
    args = ap.parse_args()

    import torch
    import wandb
    from accelerate import PartialState
    from omegaconf import OmegaConf

    ckpt = Path(args.ckpt).resolve()
    m = re.search(r"steps_(\d+)_pytorch_model\.pt$", ckpt.name)
    if not m:
        raise ValueError(f"unexpected checkpoint name {ckpt.name}")
    step = int(m.group(1))
    run_dir = ckpt.parents[1]
    eval_root = Path(args.out_root) / run_dir.name / f"step_{step}"
    eval_root.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(run_dir / "config.full.yaml")
    if str(cfg.datasets.vla_data.get("dataset_py")) != "m3_data_datasets":
        raise ValueError(f"{run_dir} is not an m3-bridge run")
    cfg.datasets.vla_data.m3_resume_steps = 0  # same sample_seed (instruction draws) for every checkpoint
    cfg.trainer.open_loop_batch_size = int(args.batch_size)
    cfg.trainer.open_loop_max_anchors = args.max_anchors
    cfg.output_dir = str(run_dir)

    PartialState()  # the trainer's accelerate logger needs an initialised state
    wandb.init(mode="disabled")  # the trainer function logs wandb.Image / wandb.log; keep both local

    from starVLA.dataloader.m3_data_datasets import get_m3_data_vla_dataset
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.training.train_starvla import VLATrainer

    t0 = time.time()
    dataset = get_m3_data_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="eval")
    plan = plan_episodes(dataset.processed, args.episodes_per_task, args.tasks)
    selection = _EpisodeSelection(dataset, plan)
    print(f"[open_loop_traj] eval reader built in {time.time() - t0:.1f}s; plan {plan}", flush=True)

    t0 = time.time()
    model = baseframework.from_pretrained(str(ckpt), is_inference=True)
    if not args.no_bf16:
        model = model.to(torch.bfloat16)
    model = model.to("cuda").eval()
    print(f"[open_loop_traj] model loaded in {time.time() - t0:.1f}s", flush=True)

    class _OpenLoopRunner:
        """Just the state the trainer's open-loop methods read; the methods themselves are the trainer's."""

        _eval_marigold_open_loop_trajectories = VLATrainer._eval_marigold_open_loop_trajectories
        _plot_open_loop_trajectory = VLATrainer._plot_open_loop_trajectory

        def __init__(self):
            self.config = cfg
            self.completed_steps = step
            self.vla_eval_dataset = selection
            self.model = model
            self.accelerator = type("_Acc", (), {"unwrap_model": staticmethod(lambda m: m)})()

    action_horizon = int(getattr(cfg.framework.action_model, "action_horizon", 1))
    t0 = time.time()
    with torch.inference_mode():
        metrics = _OpenLoopRunner()._eval_marigold_open_loop_trajectories(eval_root, action_horizon)
    print(f"[open_loop_traj] {len(selection.selected)} episodes in {time.time() - t0:.1f}s -> {eval_root}", flush=True)

    episodes = [{
        "task": _task_of(e["dataset_name"]),
        "subdataset": e["dataset_name"],
        "episode_idx": e["episode_idx"],
        "uuid": e["uuid"],
        "n_chunks": len(e["anchors"]),
        "anchor_stride_source_frames": e["anchor_stride_source_frames"],
        "source_fps": float(e["state"].sampler._src_fps),
    } for e in selection.selected]
    scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    payload = {"ckpt": str(ckpt), "run": run_dir.name, "step": step, "bf16": not args.no_bf16,
               "episodes_per_task": args.episodes_per_task, "max_anchors": args.max_anchors,
               "held_out": any(e["state"].holdout_mode == "only" for e in selection.selected),
               "episodes": episodes, "metrics": scalars}
    tmp = eval_root / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    tmp.rename(eval_root / "metrics.json")
    print(f"[open_loop_traj] open_loop/mse_mean={scalars.get('open_loop/mse_mean')}", flush=True)


if __name__ == "__main__":
    main()

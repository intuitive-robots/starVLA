#!/usr/bin/env python
"""Closed-loop plots for ABC-Sim episodes (trajectory npz written by abc_sim_client.py into <run>/traj/).

One PNG per episode, 7 x 2 panels (left arm j1-j6 + gripper | right arm j1-j6 + gripper), time in seconds at 30 Hz:
  * measured sim state (black) and the command actually executed (blue),
  * every predicted 30-step chunk drawn from its query step (thin orange; executed part = first `execute` steps),
    so chunk-to-chunk jumps and drift are visible,
  * chunk boundaries (policy queries) as faint vertical lines,
  * optional reference: the training demo of the same task whose t=0 state is nearest the sim's t=0 state
    (ABC-130K real roots of the checkpoint's m3 config; measured state, dashed green; resampled to 30 Hz).

Run with the starVLA image (matplotlib / pandas / pyarrow), CPU only, e.g. from the live tree:
    scripts/env/starvla_python examples/ABC/eval_files/plot_abc_closed_loop.py /e/scratch/m3/blank4/abc/sim_eval/<run>
Writes <run>/plots/<task>_ep<NNN>.png. The demo index is cached in /e/scratch/m3/blank4/abc/sim_eval/demo_index/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

DIMS = [f"L j{i}" for i in range(1, 7)] + ["L grip"] + [f"R j{i}" for i in range(1, 7)] + ["R grip"]
TASK_DIRS = {  # sim task -> real ABC-130K root task prefix
    "put_plastic_bottles_in_bin": "put_the_plastic_bottles_in_the_bin",
    "throw_plastic_bottles_in_bin": "throw_the_plastic_bottles_in_the_bin",
    "load_plates_into_dish_rack": "load_the_plates_into_the_dish_rack",
    "turn_mug_right_side_up": "turn_the_mug_right_side_up",
}
M3_CONFIG = "/e/scratch/m3/blank4/abc/m3cfg/abc_real3_cmd_local.json"
INDEX_DIR = Path("/e/scratch/m3/blank4/abc/sim_eval/demo_index")


def _task_roots(task: str, m3_config: str) -> list[str]:
    cfg = json.load(open(m3_config))
    prefix = TASK_DIRS[task]
    return [os.path.join(ds["root"], inc) for ds in cfg["datasets"] for inc in ds["include"]
            if inc.split("/")[1].startswith(prefix + "__")]


def demo_index(task: str, m3_config: str) -> dict:
    """Start state (frame 0) of every training episode of the task, cached per task."""
    import pyarrow.parquet as pq

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache = INDEX_DIR / f"{task}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    starts, roots, eps, files, fps = [], [], [], [], []
    for root in _task_roots(task, m3_config):
        rfps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]
        for f in sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True)):
            t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"],
                              filters=[("frame_index", "=", 0)])
            for e, s in zip(t.column("episode_index").to_pylist(), t.column("observation.state").to_pylist()):
                starts.append(np.asarray(s, np.float32))
                roots.append(root)
                eps.append(int(e))
                files.append(f)
                fps.append(int(rfps))
    out = dict(start=np.stack(starts), root=np.array(roots), episode=np.array(eps), pq_file=np.array(files),
               fps=np.array(fps))
    np.savez(cache, **out)
    return out


def load_demo(file: str, episode: int, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """(t_seconds, state[T, 14]) of one real episode."""
    import pyarrow.parquet as pq

    t = pq.read_table(file, columns=["episode_index", "frame_index", "observation.state"],
                      filters=[("episode_index", "=", int(episode))])
    order = np.argsort(np.asarray(t.column("frame_index").to_pylist()))
    st = np.stack([np.asarray(s, np.float32) for s in t.column("observation.state").to_pylist()])[order]
    return np.arange(len(st)) / float(fps), st


def plot_episode(npz_path: Path, out_png: Path, demo: dict | None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path, allow_pickle=True)
    state, action, qs, pred = z["state"], z["action"], z["query_step"], z["pred_chunks"]
    execute = int(z["execute"])
    hz = 30.0
    ts = np.arange(len(state)) / hz          # state k is measured after k steps
    ta = (np.arange(len(action)) + 1) / hz   # command k drives the step that ends at state k + 1
    # chunk-to-chunk jump: first command of chunk c vs last executed command of chunk c - 1
    starts = qs[1:]
    jumps = np.abs(action[starts] - action[starts - 1]) if len(starts) else np.zeros((0, 14))
    ref = None
    if demo is not None and len(demo["start"]):
        d = np.linalg.norm(demo["start"] - state[0][None], axis=1)
        i = int(np.argmin(d))
        t_demo, s_demo = load_demo(str(demo["pq_file"][i]), int(demo["episode"][i]), int(demo["fps"][i]))
        ref = dict(t=t_demo, s=s_demo, dist=float(d[i]), ep=int(demo["episode"][i]),
                   root=os.path.basename(str(demo["root"][i])), n=len(demo["start"]))

    fig, axes = plt.subplots(7, 2, figsize=(18, 19), sharex=True)
    t_end = ts[-1]
    for d in range(14):
        ax = axes[d % 7, d // 7]
        for q in qs:
            ax.axvline(q / hz, color="0.85", lw=0.3, zorder=0)
        for q, ch in zip(qs, pred):
            tt = (q + 1 + np.arange(len(ch))) / hz
            ax.plot(tt, ch[:, d], color="tab:orange", lw=0.5, alpha=0.35, zorder=1)
            ax.plot(tt[:execute], ch[:execute, d], color="tab:orange", lw=0.8, alpha=0.8, zorder=2)
        if ref is not None:
            m = ref["t"] <= max(t_end, 1.0)
            ax.plot(ref["t"][m], ref["s"][m, d], "--", color="tab:green", lw=1.0, zorder=3)
        ax.plot(ta, action[:, d], color="tab:blue", lw=0.9, zorder=4)
        ax.plot(ts, state[:, d], color="black", lw=0.9, zorder=5)
        jm = jumps[:, d].mean() if len(jumps) else 0.0
        ax.set_ylabel(DIMS[d])
        ax.text(0.01, 0.92, f"mean chunk jump {jm:.3f}", transform=ax.transAxes, fontsize=7, va="top")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("time (s, 30 Hz)")
    axes[0, 0].set_title("left arm")
    axes[0, 1].set_title("right arm")
    head = (f"{z['task']} ep {int(z['episode'])} seed {int(z['world_seed'])}  success={bool(z['success'])} "
            f"max_progress={float(z['max_progress']):.2f}  {len(qs)} chunks x {execute}  prompt '{z['prompt']}'\n"
            f"black: sim state | blue: executed command | orange: predicted 30-step chunks (bold = executed part) | "
            f"grey: policy queries")
    if ref is not None:
        head += (f" | green dashed: nearest training demo by t=0 state ({ref['root']} ep {ref['ep']}, "
                 f"|ds0|={ref['dist']:.2f} over {ref['n']} demos)")
    head += f"\n{os.path.basename(str(z['ckpt']))}  unnorm_key {z['unnorm_key']}"
    fig.suptitle(head, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return dict(png=str(out_png), mean_jump=float(jumps.mean()) if len(jumps) else 0.0,
                demo_dist=ref["dist"] if ref else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", default="", help="glob over traj/ names, e.g. '*_ep000' (default: all)")
    ap.add_argument("--no_demo", action="store_true")
    ap.add_argument("--m3_config", default=M3_CONFIG)
    args = ap.parse_args()
    run = Path(args.run_dir)
    paths = sorted((run / "traj").glob(f"{args.episodes or '*'}.npz"))
    if not paths:
        raise SystemExit(f"no trajectories in {run / 'traj'} (runs before 2026-10-07 13:00 have none)")
    demos: dict = {}
    for p in paths:
        task = str(np.load(p, allow_pickle=True)["task"])
        demo = None
        if not args.no_demo and task in TASK_DIRS:
            if task not in demos:
                demos[task] = demo_index(task, args.m3_config)
            demo = demos[task]
        info = plot_episode(p, run / "plots" / (p.stem + ".png"), demo)
        print(f"{info['png']}  mean chunk jump {info['mean_jump']:.3f}  demo |ds0| {info['demo_dist']}", flush=True)


if __name__ == "__main__":
    main()

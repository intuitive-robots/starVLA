"""
Render a few LIBERO frames with the current GL backend and save to video.
Use this to visually compare NVIDIA EGL vs Mesa CPU rendering.

Usage (GPU/EGL):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    python examples/LIBERO-plus/eval_files/render_compare.py --tag egl

Usage (Mesa/CPU):
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
    __EGL_VENDOR_LIBRARY_DIRS=/e/software/default/stages/2026/software/OpenGL/2025.09-GCCcore-14.3.0/share/glvnd/egl_vendor.d \
    LD_LIBRARY_PATH=/e/software/default/stages/2026/software/OpenGL/2025.09-GCCcore-14.3.0/lib \
    python examples/LIBERO-plus/eval_files/render_compare.py --tag mesa
"""
import argparse
import os
import pathlib
import sys

import imageio
import numpy as np

LIBERO_HOME = os.environ.get("LIBERO_HOME", "playground/sims/LIBERO-plus")
if LIBERO_HOME not in sys.path:
    sys.path.insert(0, LIBERO_HOME)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RESOLUTION = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def render_task(task_suite_name: str, task_id: int, num_steps: int, seed: int):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
    )
    env.seed(seed)
    env.reset()
    env.set_init_state(initial_states[0])

    frames = []
    for _ in range(num_steps):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        # side-by-side: agentview | wrist
        combined = np.concatenate([img, wrist], axis=1)
        frames.append(combined)
        if done:
            break

    env.close()
    return frames, task.language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="render", help="label for output filename (e.g. 'egl' or 'mesa')")
    parser.add_argument("--task_suite", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    gl_backend = os.environ.get("MUJOCO_GL", "unknown")
    sw = os.environ.get("LIBGL_ALWAYS_SOFTWARE", "0")
    print(f"[render_compare] MUJOCO_GL={gl_backend}  LIBGL_ALWAYS_SOFTWARE={sw}  tag={args.tag}")

    frames, description = render_task(args.task_suite, args.task_id, args.num_steps, args.seed)
    out_path = pathlib.Path(args.out_dir) / f"render_{args.tag}_task{args.task_id}.mp4"
    imageio.mimwrite(str(out_path), frames, fps=10)
    print(f"[render_compare] Saved {len(frames)} frames → {out_path}")
    print(f"[render_compare] Task: {description}")


if __name__ == "__main__":
    main()

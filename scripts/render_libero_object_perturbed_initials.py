#!/usr/bin/env python3
import argparse
import json
import pathlib

import numpy as np
from PIL import Image

from examples.LIBERO.eval_files.parallel_eval.eval_libero_shard import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _apply_object_perturbation,
    _get_libero_env,
    benchmark,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--perturb-m", type=float, default=0.04)
    p.add_argument("--perturb-seed", type=int, default=20260812)
    a = p.parse_args()
    out = pathlib.Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(a.task_id)
    env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, 7)
    env.reset(); obs = env.set_init_state(suite.get_task_init_states(a.task_id)[a.episode])
    perturbation = _apply_object_perturbation(
        env, a.perturb_m, "source,target", a.perturb_seed,
        "libero_object", a.task_id, a.episode,
    )
    for _ in range(10): obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    env.close()
    if float(agent.mean()) < 1.0:
        raise RuntimeError(f"task {a.task_id} agent view is black (mean={agent.mean():.4f})")
    stem = f"task_{a.task_id:02d}_ep_{a.episode:02d}"
    Image.fromarray(agent).save(out / f"{stem}_agent.png")
    Image.fromarray(wrist).save(out / f"{stem}_wrist.png")
    json.dump({
        "task_id": a.task_id, "episode_idx": a.episode,
        "description": description, "perturbation": perturbation,
        "agent_mean": float(agent.mean()), "wrist_mean": float(wrist.mean()),
    }, open(out / f"{stem}.json", "w"), indent=2)
    print(f"rendered {stem}: agent_mean={agent.mean():.2f}, wrist_mean={wrist.mean():.2f}")


if __name__ == "__main__": main()

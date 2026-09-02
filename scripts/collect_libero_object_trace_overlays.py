#!/usr/bin/env python3
"""Collect one generated trace per LIBERO-object task and render overlays.

This replays only the deterministic initial state and settling steps; it does
not execute a robot rollout. A single batched policy request covers all tasks.
"""

import argparse
import json
import pathlib
import re

import numpy as np
from PIL import Image, ImageDraw

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.LIBERO.eval_files.parallel_eval.eval_libero_shard import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _apply_object_perturbation,
    _get_libero_env,
    benchmark,
)


TRACE_RE = re.compile(r"<\|trace\|>(.*?)<\|/trace\|>", re.DOTALL)


def parse_trace(text):
    if not text:
        return []
    match = TRACE_RE.search(text)
    if not match:
        return []
    array = re.search(r"\[\s*\[.*?\]\s*\]", match.group(1), re.DOTALL)
    if not array:
        return []
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", array.group(0))]
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]


def overlay(image, points, label):
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + 28), "black")
    canvas.paste(Image.fromarray(image), (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 7), label, fill="white")
    px = [
        (
            max(0, min(1000, x)) / 1000 * (image.shape[1] - 1),
            28 + max(0, min(1000, y)) / 1000 * (image.shape[0] - 1),
        )
        for x, y in points
    ]
    for i in range(len(px) - 1):
        frac = i / max(1, len(px) - 1)
        draw.line((px[i], px[i + 1]), fill=(int(255 * frac), int(255 * (1 - frac)), 40), width=3)
    for i, (x, y) in enumerate(px):
        frac = i / max(1, len(px) - 1)
        radius = 5 if i in (0, len(px) - 1) else 3
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(int(255 * frac), int(255 * (1 - frac)), 40),
            outline="white",
        )
        draw.text((x + 5, y - 8), str(i + 1), fill="white")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10103)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--perturb-m", type=float, default=0.04)
    parser.add_argument("--perturb-seed", type=int, default=20260812)
    args = parser.parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    examples, records = [], []
    task_ids = range(suite.n_tasks) if args.task_id is None else [args.task_id]
    for task_id in task_ids:
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=7)
        env.reset()
        env.set_init_state(states[args.episode])
        perturbation = _apply_object_perturbation(
            env, args.perturb_m, "source,target", args.perturb_seed,
            "libero_object", task_id, args.episode,
        )
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        if float(agent.mean()) < 1.0:
            raise RuntimeError(
                f"task {task_id} rendered a nearly black agent view "
                f"(mean={float(agent.mean()):.4f})"
            )
        examples.append({"image": [agent, wrist], "lang": str(description)})
        records.append({
            "task_id": task_id,
            "episode_idx": args.episode,
            "description": description,
            "perturbation": perturbation,
            "image": agent,
        })
        env.close()

    client = WebsocketClientPolicy(args.host, args.port)
    response = client.predict_action({
        "examples": examples,
        "unnorm_key": None,
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
    })
    client.close()
    texts = response["data"].get("cot_text")
    if not isinstance(texts, (list, tuple)) or len(texts) != len(records):
        raise RuntimeError(f"Expected {len(records)} cot_text entries, got {type(texts)} / {texts!r}")

    serializable = []
    for record, text in zip(records, texts):
        points = parse_trace(text)
        label = f"{args.model_label} | task {record['task_id']} | {len(points)} points"
        image = overlay(record.pop("image"), points, label)
        image.save(output / f"task_{record['task_id']:02d}_ep_{args.episode:02d}.png")
        record.update({"model": args.model_label, "cot_text": text, "trace_2d": points})
        serializable.append(record)

    suffix = "all" if args.task_id is None else f"task_{args.task_id:02d}"
    with open(output / f"traces_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    valid = sum(len(x["trace_2d"]) == 5 for x in serializable)
    print(
        f"{args.model_label}: wrote {len(serializable)} overlays; "
        f"{valid}/{len(serializable)} have exactly five parseable points"
    )


if __name__ == "__main__":
    main()

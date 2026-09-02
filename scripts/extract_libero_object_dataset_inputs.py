#!/usr/bin/env python3
"""Extract one real, non-rendered LIBERO-object observation per task."""

import argparse
import json
import pathlib
import subprocess

from PIL import Image


def _extract_frame(video: pathlib.Path, output: pathlib.Path, ffmpeg: str) -> None:
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", str(video), "-frames:v", "1", str(output)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    root = pathlib.Path(args.dataset)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    first_episode_by_task: dict[str, int] = {}
    with (root / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            description = record["tasks"][0]
            first_episode_by_task.setdefault(description, int(record["episode_index"]))

    tasks = []
    with (root / "meta" / "tasks.jsonl").open() as handle:
        for line in handle:
            tasks.append(json.loads(line))

    if len(tasks) != 10:
        raise RuntimeError(f"Expected 10 LIBERO-object tasks, found {len(tasks)}")

    for task in tasks:
        task_id = int(task["task_index"])
        description = task["task"]
        episode = first_episode_by_task[description]
        stem = f"task_{task_id:02d}_ep_00"
        video_root = root / "videos" / "chunk-000"
        _extract_frame(
            video_root / "observation.images.image" / f"episode_{episode:06d}.mp4",
            output / f"{stem}_agent.png",
            args.ffmpeg,
        )
        _extract_frame(
            video_root / "observation.images.wrist_image" / f"episode_{episode:06d}.mp4",
            output / f"{stem}_wrist.png",
            args.ffmpeg,
        )
        agent_mean = sum(Image.open(output / f"{stem}_agent.png").convert("RGB").resize((1, 1)).getpixel((0, 0))) / 3
        metadata = {
            "task_id": task_id,
            "episode_idx": 0,
            "dataset_episode_idx": episode,
            "description": description,
            "input_source": "LIBERO-object training demonstration, first frame",
            "agent_mean": agent_mean,
        }
        with (output / f"{stem}.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"extracted task={task_id} dataset_episode={episode} mean={agent_mean:.2f}")


if __name__ == "__main__":
    main()

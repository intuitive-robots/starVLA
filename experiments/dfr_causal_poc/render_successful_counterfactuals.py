#!/usr/bin/env python3
"""Re-render successful clean LIBERO trajectories as exact appearance pairs.

The input trajectories were executed successfully by the reference policy and
contain the MuJoCo state immediately before every recorded action. This script
loads those exact states into a canonical and a LIBERO-Plus appearance-only
environment. It never steps either environment. Consequently the instruction,
physical state, proprioception and target action chunk are identical within a
pair; only the pixels differ.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


LIBERO_HOME = os.environ.get("LIBERO_HOME", "/app/LIBERO-plus")
CATEGORIES = (
    "Background Textures",
    "Light Conditions",
    "Sensor Noise",
    "Camera Viewpoints",
)
SPLITS = ("train", "validation", "test")


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def canonical_stem(name: str, category: str) -> str:
    patterns = {
        "Background Textures": r"_(?:table|tb)_\d+$",
        "Light Conditions": r"_light_\d+$",
        "Sensor Noise": r"_view_.*?_initstate_\d+_noise_\d+$",
        "Camera Viewpoints": r"_view_.*?_initstate_\d+$",
    }
    result, count = re.subn(patterns[category], "", name)
    if count != 1:
        raise ValueError(f"cannot canonicalize {category!r}: {name!r}")
    return result


def _load_trajectories(
    root: Path,
    trajectories_per_task: int,
    expected_tasks: int,
    minimum_tasks: int,
    minimum_tasks_per_suite: int,
) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("format") != "starvla_successful_libero_trajectory_v1":
            continue
        row["metadata_path"] = str(path.resolve())
        grouped[(str(row["suite"]), str(row["task_name"]))].append(row)
    selected = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["episode_idx"]))
        if len(rows) >= trajectories_per_task:
            selected[key] = rows[:trajectories_per_task]
    if expected_tasks > 0 and len(selected) != expected_tasks:
        missing = sorted(key for key, rows in grouped.items() if len(rows) < trajectories_per_task)
        raise RuntimeError(
            f"expected {expected_tasks} tasks with >= {trajectories_per_task} successes, "
            f"found {len(selected)}; insufficient={missing}"
        )
    if len(selected) < minimum_tasks:
        raise RuntimeError(f"only {len(selected)} eligible tasks; require at least {minimum_tasks}")
    suite_counts = {
        suite: sum(key[0] == suite for key in selected)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    }
    sparse = {suite: count for suite, count in suite_counts.items() if count < minimum_tasks_per_suite}
    if sparse:
        raise RuntimeError(
            f"insufficient suite diversity (minimum {minimum_tasks_per_suite} eligible tasks/suite): {sparse}"
        )
    return selected


def _select_variants(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        category = str(row["category"])
        if category not in CATEGORIES:
            continue
        base = canonical_stem(str(row["name"]), category)
        grouped[(base, category)].append(dict(row, base_task=base, task_id=int(row["id"]) - 1))
    chosen = {}
    for key, candidates in grouped.items():
        # Prefer a moderate variant for fitting, then reserve two distinct hard
        # variants for validation and test. Numeric task id breaks ties.
        def difficulty(row: dict) -> int:
            value = row.get("difficulty_level")
            return int(value) if value is not None else 3

        train = min(candidates, key=lambda row: (abs(difficulty(row) - 2), int(row["task_id"])))
        remaining = [row for row in candidates if int(row["task_id"]) != int(train["task_id"])]
        heldout = sorted(remaining, key=lambda row: (-difficulty(row), int(row["task_id"])))
        if len(heldout) < 2:
            raise RuntimeError(f"need three distinct variants for {key}, found {len(candidates)}")
        chosen[key] = {"train": train, "validation": heldout[1], "test": heldout[0]}
    return chosen


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator


def _policy_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def _images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    external = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    for name, image in (("external", external), ("wrist", wrist)):
        if image.ndim != 3 or image.shape[-1] != 3 or float(image.mean()) < 1.0:
            raise RuntimeError(f"invalid {name} render: shape={image.shape}, mean={image.mean():.3f}")
    return external.astype(np.uint8), wrist.astype(np.uint8)


def _apply_sensor_noise(obs: dict, noise: int) -> None:
    """Mirror LIBERO-Plus ControlEnv.step/reset noise for state-only renders.

    ``regenerate_obs_from_state`` bypasses the wrapper's step/reset methods, so
    LIBERO-Plus otherwise returns an uncorrupted image even when ``env.noise``
    is nonzero.  LIBERO-Plus intentionally corrupts agentview only.
    """
    if noise == 0:
        return
    from libero.libero.envs.env_wrapper import fog, gaussian_blur, glass_blur, motion_blur, zoom_blur

    image = obs["agentview_image"]
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    if noise <= 10:
        corrupted = motion_blur(pil_image, severity=noise)
    elif noise <= 20:
        corrupted = gaussian_blur(pil_image, severity=noise - 10)
    elif noise <= 30:
        corrupted = zoom_blur(pil_image, severity=noise - 20)
    elif noise <= 40:
        corrupted = fog(pil_image, severity=noise - 30)
    elif noise <= 50:
        corrupted = glass_blur(pil_image, severity=noise - 40)
    else:
        raise ValueError(f"unsupported LIBERO-Plus sensor noise id: {noise}")
    obs["agentview_image"] = np.asarray(corrupted, dtype=np.uint8)


def _render_state(
    env, state: np.ndarray, seed: int, apply_sensor_noise: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    np.random.seed(seed % (2**32))
    obs = env.regenerate_obs_from_state(state)
    if apply_sensor_noise:
        _apply_sensor_noise(obs, int(env.noise))
    restored = np.asarray(env.get_sim_state(), dtype=np.float64)
    difference = float(np.max(np.abs(restored - state)))
    external, wrist = _images(obs)
    return external, wrist, _policy_state(obs), difference


def _save_images(directory: Path, stem: str, external: np.ndarray, wrist: np.ndarray) -> tuple[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    external_path = directory / f"{stem}_external.jpg"
    wrist_path = directory / f"{stem}_wrist.jpg"
    Image.fromarray(external).save(external_path, quality=92, subsampling=0)
    Image.fromarray(wrist).save(wrist_path, quality=92, subsampling=0)
    return str(external_path.resolve()), str(wrist_path.resolve())


def _anchors(length: int, count: int, horizon: int) -> list[int]:
    if length < 1:
        raise ValueError("empty successful trajectory")
    upper = max(length - horizon, 0)
    values = np.linspace(0, upper, num=min(count, upper + 1), dtype=np.int64)
    return sorted(set(int(value) for value in values))


def _action_chunk(actions: np.ndarray, index: int, horizon: int) -> np.ndarray:
    chunk = actions[index : index + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)], axis=0)
    return np.asarray(chunk, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trajectories-per-task", type=int, default=3)
    parser.add_argument("--expected-tasks", type=int, default=40)
    parser.add_argument("--minimum-tasks", type=int, default=20)
    parser.add_argument("--minimum-tasks-per-suite", type=int, default=3)
    parser.add_argument("--anchors-per-trajectory", type=int, default=12)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--state-tolerance", type=float, default=1e-8)
    # The pinned vanilla and Plus robosuite builds differ at ~1e-4 in derived
    # EEF observables after restoring an identical flattened MuJoCo state.
    # Exact sim-state equality remains required; 5e-4 only tolerates that
    # cross-version observable recomputation noise.
    parser.add_argument("--policy-state-tolerance", type=float, default=5e-4)
    parser.add_argument("--source-policy-state-tolerance", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-id", type=int, required=True)
    args = parser.parse_args()
    if args.trajectories_per_task != 3:
        raise ValueError("v1 requires exactly three successful trajectories per task for train/validation/test")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard id")

    if LIBERO_HOME not in sys.path:
        sys.path.insert(0, LIBERO_HOME)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    trajectories = _load_trajectories(
        args.trajectory_root,
        args.trajectories_per_task,
        args.expected_tasks,
        args.minimum_tasks,
        args.minimum_tasks_per_suite,
    )
    classification = json.loads(
        (Path(LIBERO_HOME) / "libero/libero/benchmark/task_classification.json").read_text()
    )
    bddl_base = Path(get_libero_path("bddl_files"))
    output = args.output_root.resolve()
    image_root = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    max_state_diff = 0.0
    max_policy_state_diff = 0.0
    max_source_restore_policy_state_diff = 0.0
    min_pixel_mae = float("inf")

    tasks = sorted(trajectories)
    local_tasks = [key for index, key in enumerate(tasks) if index % args.num_shards == args.shard_id]
    for task_number, (suite, base_task) in enumerate(local_tasks, start=1):
        variants = _select_variants(classification[suite])
        for category in CATEGORIES:
            if (base_task, category) not in variants:
                raise RuntimeError(f"missing Plus variants for {suite}/{base_task}/{category}")
        with contextlib.redirect_stdout(io.StringIO()):
            plus_suite = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        canonical_bddl = bddl_base / suite / f"{base_task}.bddl"
        if not canonical_bddl.is_file():
            raise FileNotFoundError(canonical_bddl)
        clean_env = OffScreenRenderEnv(
            bddl_file_name=str(canonical_bddl),
            camera_heights=args.resolution,
            camera_widths=args.resolution,
        )
        clean_env.seed(args.seed)
        clean_env.reset()
        clean_rows = {}
        try:
            for split, trajectory in zip(SPLITS, trajectories[(suite, base_task)]):
                payload = np.load(trajectory["npz"])
                sim_states = np.asarray(payload["sim_state"], dtype=np.float64)
                recorded_policy_states = np.asarray(payload["policy_state"], dtype=np.float32)
                actions = np.asarray(payload["action"], dtype=np.float32)
                for anchor in _anchors(len(actions), args.anchors_per_trajectory, args.action_horizon):
                    pair_seed = _stable_seed(args.seed, suite, base_task, trajectory["episode_idx"], anchor)
                    external, wrist, state, state_diff = _render_state(clean_env, sim_states[anchor], pair_seed)
                    policy_diff = float(np.max(np.abs(state - recorded_policy_states[anchor])))
                    max_state_diff = max(max_state_diff, state_diff)
                    max_source_restore_policy_state_diff = max(
                        max_source_restore_policy_state_diff, policy_diff
                    )
                    if state_diff > args.state_tolerance or policy_diff > args.source_policy_state_tolerance:
                        raise RuntimeError(
                            f"clean restore mismatch {suite}/{base_task}/ep{trajectory['episode_idx']}/step{anchor}: "
                            f"sim={state_diff:.3e}, policy={policy_diff:.3e}"
                        )
                    key = f"{suite}_task{int(trajectory['task_id']):02d}_ep{int(trajectory['episode_idx']):02d}_step{anchor:04d}"
                    clean_external, clean_wrist = _save_images(image_root / "clean", key, external, wrist)
                    clean_rows[(split, int(trajectory["episode_idx"]), anchor)] = {
                        "key": key,
                        "clean_external": clean_external,
                        "clean_wrist": clean_wrist,
                        # Preserve the exact proprio input from the successful
                        # vanilla rollout for both sides of the pair. The Plus
                        # fork can recompute EEF observables slightly differently
                        # despite an exactly equal flattened MuJoCo state.
                        "policy_state": recorded_policy_states[anchor].tolist(),
                        "rendered_policy_state": state.tolist(),
                        "action": _action_chunk(actions, anchor, args.action_horizon).tolist(),
                        "source_trajectory": trajectory["npz"],
                        "external_array": external,
                        "wrist_array": wrist,
                    }
        finally:
            clean_env.close()

        for category in CATEGORIES:
            for variant_role in SPLITS:
                row = variants[(base_task, category)][variant_role]
                task = plus_suite.get_task(int(row["task_id"]))
                variant_bddl = bddl_base / suite / task.bddl_file
                # Camera/noise suffixes are virtual: ControlEnv parses them
                # from the path and loads the canonical BDDL prefix.
                if category not in {"Sensor Noise", "Camera Viewpoints"} and not variant_bddl.is_file():
                    raise FileNotFoundError(variant_bddl)
                env = OffScreenRenderEnv(
                    bddl_file_name=str(variant_bddl),
                    camera_heights=args.resolution,
                    camera_widths=args.resolution,
                )
                env.seed(args.seed)
                env.reset()
                try:
                    for split, trajectory in zip(SPLITS, trajectories[(suite, base_task)]):
                        if split != variant_role:
                            continue
                        payload = np.load(trajectory["npz"])
                        sim_states = np.asarray(payload["sim_state"], dtype=np.float64)
                        actions = np.asarray(payload["action"], dtype=np.float32)
                        for anchor in _anchors(len(actions), args.anchors_per_trajectory, args.action_horizon):
                            clean = clean_rows[(split, int(trajectory["episode_idx"]), anchor)]
                            pair_seed = _stable_seed(
                                args.seed, suite, base_task, category, variant_role,
                                trajectory["episode_idx"], anchor,
                            )
                            external, wrist, state, state_diff = _render_state(
                                env,
                                sim_states[anchor],
                                pair_seed,
                                apply_sensor_noise=category == "Sensor Noise",
                            )
                            policy_diff = float(
                                np.max(np.abs(state - np.asarray(clean["rendered_policy_state"])))
                            )
                            max_state_diff = max(max_state_diff, state_diff)
                            max_policy_state_diff = max(max_policy_state_diff, policy_diff)
                            if state_diff > args.state_tolerance or policy_diff > args.policy_state_tolerance:
                                raise RuntimeError(
                                    f"variant restore mismatch {suite}/{base_task}/{category}: "
                                    f"sim={state_diff:.3e}, policy={policy_diff:.3e}"
                                )
                            external_mae = float(
                                np.abs(external.astype(np.float32) - clean["external_array"].astype(np.float32)).mean()
                            )
                            wrist_mae = float(
                                np.abs(wrist.astype(np.float32) - clean["wrist_array"].astype(np.float32)).mean()
                            )
                            pixel_mae = max(external_mae, wrist_mae)
                            min_pixel_mae = min(min_pixel_mae, pixel_mae)
                            if pixel_mae <= 0.1:
                                raise RuntimeError(
                                    f"zero-effect visual intervention {suite}/{base_task}/{category}: "
                                    f"external_mae={external_mae:.3e}, wrist_mae={wrist_mae:.3e}"
                                )
                            category_stem = category.lower().replace(" ", "_")
                            variant_external, variant_wrist = _save_images(
                                image_root / category_stem,
                                f"{clean['key']}_{variant_role}_task{int(row['task_id']):04d}",
                                external,
                                wrist,
                            )
                            records.append(
                                {
                                    "format": "starvla_successful_counterfactual_pair_v1",
                                    "pair_id": f"{clean['key']}:{category_stem}",
                                    "suite": suite,
                                    "task_id": int(trajectory["task_id"]),
                                    "base_task": base_task,
                                    "episode_idx": int(trajectory["episode_idx"]),
                                    "anchor_step": anchor,
                                    "split": split,
                                    "category": category,
                                    "variant_role": variant_role,
                                    "variant_task_id": int(row["task_id"]),
                                    "variant_name": str(row["name"]),
                                    "difficulty_level": row.get("difficulty_level"),
                                    "language": str(trajectory["task_description"]),
                                    "clean_external": clean["clean_external"],
                                    "clean_wrist": clean["clean_wrist"],
                                    "variant_external": variant_external,
                                    "variant_wrist": variant_wrist,
                                    "policy_state": clean["policy_state"],
                                    "action": clean["action"],
                                    "source_trajectory": clean["source_trajectory"],
                                    "state_max_abs_diff": state_diff,
                                    "policy_state_max_abs_diff": policy_diff,
                                    "external_pixel_mae": external_mae,
                                    "wrist_pixel_mae": wrist_mae,
                                }
                            )
                finally:
                    env.close()
        print(
            f"[shard {args.shard_id}] task {task_number}/{len(local_tasks)} "
            f"{suite}/{base_task}: {len(records)} pairs so far",
            flush=True,
        )

    records.sort(key=lambda row: row["pair_id"])
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    summary = {
        "format": "starvla_successful_counterfactual_capture_v1",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "pairs": len(records),
        "tasks": len(local_tasks),
        "suites": sorted({suite for suite, _ in local_tasks}),
        "categories": list(CATEGORIES),
        "splits": {split: sum(row["split"] == split for row in records) for split in SPLITS},
        "max_state_diff": max_state_diff,
        "max_policy_state_diff": max_policy_state_diff,
        "max_source_restore_policy_state_diff": max_source_restore_policy_state_diff,
        "min_pixel_mae": min_pixel_mae,
        "manifest": str(manifest.resolve()),
    }
    (output / "capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

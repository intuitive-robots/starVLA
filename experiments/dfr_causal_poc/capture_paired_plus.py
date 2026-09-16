"""Capture state-matched clean/LIBERO-Plus appearance counterfactuals.

This script runs inside the pinned LIBERO-Plus Apptainer image.  It deliberately
uses only perturbations that should preserve the correct action for a fixed
simulator state: background texture, lighting, and sensor noise.  Every pair is
rendered from the same flattened MuJoCo state and is rejected if the settled
physical states diverge.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

LIBERO_HOME = os.environ.get("LIBERO_HOME", "/app/LIBERO-plus")
KEEP_CATEGORIES = ("Background Textures", "Light Conditions", "Sensor Noise")
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def canonical_stem(name: str, category: str) -> str:
    """Map an appearance-perturbed LIBERO-Plus name to its clean BDDL stem."""
    import re

    patterns = {
        "Background Textures": r"_(?:table|tb)_\d+$",
        "Light Conditions": r"_light_\d+$",
        "Sensor Noise": r"_view_.*?_initstate_\d+_noise_\d+$",
    }
    if category not in patterns:
        raise ValueError(f"unsupported label-preserving category: {category!r}")
    result, count = re.subn(patterns[category], "", name)
    if count != 1 or result == name:
        raise ValueError(f"could not canonicalize {category!r} task {name!r}")
    return result


def _select_rows(rows: list[dict], variants_per_task: int, seed: int) -> tuple[list[dict], dict[str, list[dict]]]:
    """Stratify over every canonical LIBERO-10 task and appearance category."""
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        category = str(row["category"])
        if category not in KEEP_CATEGORIES:
            continue
        base = canonical_stem(str(row["name"]), category)
        enriched = dict(row, base_task=base, task_id=int(row["id"]) - 1)
        grouped[base][category].append(enriched)

    rng = np.random.default_rng(seed)
    selected = []
    for base in sorted(grouped):
        missing = [category for category in KEEP_CATEGORIES if not grouped[base][category]]
        if missing:
            raise RuntimeError(f"base task {base!r} lacks categories {missing}")
        for category in KEEP_CATEGORIES:
            candidates = sorted(grouped[base][category], key=lambda row: row["task_id"])
            if len(candidates) < variants_per_task:
                raise RuntimeError(f"base task {base!r}/{category} has only {len(candidates)} variants")
            indices = rng.choice(len(candidates), size=variants_per_task, replace=False)
            selected.extend(candidates[int(index)] for index in sorted(indices.tolist()))
    selected.sort(key=lambda row: (row["base_task"], KEEP_CATEGORIES.index(row["category"]), row["task_id"]))
    return selected, {base: dict(categories) for base, categories in grouped.items()}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator


def _policy_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"]),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"]),
        ]
    ).astype(np.float32)


def _images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    external = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if external.ndim == 2:
        external = np.repeat(external[..., None], 3, axis=-1)
    if wrist.ndim == 2:
        wrist = np.repeat(wrist[..., None], 3, axis=-1)
    for name, image in (("external", external), ("wrist", wrist)):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(f"unexpected {name} image shape {image.shape}")
        if float(image.mean()) < 1.0:
            raise RuntimeError(f"invalid near-black {name} image (mean={image.mean():.4f})")
    return external.astype(np.uint8), wrist.astype(np.uint8)


def _render(
    env_class,
    bddl: Path,
    init_state: np.ndarray,
    resolution: int,
    settle_steps: int,
    seed: int,
) -> dict:
    np.random.seed(seed)
    env = env_class(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    try:
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(settle_steps):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        external, wrist = _images(obs)
        return {
            "external": external,
            "wrist": wrist,
            "policy_state": _policy_state(obs),
            "sim_state": np.asarray(env.get_sim_state(), dtype=np.float64),
        }
    finally:
        env.close()


def _state_hash(state: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(state, dtype=np.float64))
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()[:16]


def _save_render(render: dict, image_dir: Path, stem: str) -> tuple[Path, Path]:
    external = image_dir / f"{stem}_external.png"
    wrist = image_dir / f"{stem}_wrist.png"
    Image.fromarray(render["external"]).save(external)
    Image.fromarray(render["wrist"]).save(wrist)
    return external.resolve(), wrist.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--variants-per-task", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--state-tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()
    if args.suite != "libero_10":
        raise ValueError("the initial PoC is intentionally fixed to libero_10")
    if args.variants_per_task < 1:
        raise ValueError("--variants-per-task must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require --num-shards >= 1 and 0 <= --shard-id < --num-shards")

    import sys

    if LIBERO_HOME not in sys.path:
        sys.path.insert(0, LIBERO_HOME)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import get_problem_info

    classification_path = Path(LIBERO_HOME) / "libero/libero/benchmark/task_classification.json"
    rows = json.loads(classification_path.read_text())[args.suite]
    selected, grouped = _select_rows(rows, args.variants_per_task, args.seed)
    selected = [
        dict(row, pair_id=pair_id) for pair_id, row in enumerate(selected) if pair_id % args.num_shards == args.shard_id
    ]
    base_tasks = sorted(grouped)

    output = args.output_root.resolve()
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    # LIBERO-Plus prints its full 2,519-element identity task order from the
    # constructor. Suppress that implementation detail in four-worker logs.
    with contextlib.redirect_stdout(io.StringIO()):
        task_suite = benchmark.get_benchmark_dict()[args.suite](task_order_index=0)
    bddl_root = Path(get_libero_path("bddl_files")) / args.suite

    # Clean renders are shared only when both the base task and exact source
    # state match. This keeps the pairing valid if a future Plus version varies
    # its initial-state file between appearance variants.
    clean_cache: dict[tuple[str, str], dict] = {}
    records = []
    for local_index, row in enumerate(selected):
        pair_index = int(row["pair_id"])
        task_id = int(row["task_id"])
        task = task_suite.get_task(task_id)
        if task.name != row["name"]:
            raise RuntimeError(f"classification/task-order mismatch at {task_id}: {row['name']} != {task.name}")
        initial_states = task_suite.get_task_init_states(task_id)
        init_state = np.asarray(initial_states[0], dtype=np.float64)
        base = str(row["base_task"])
        canonical_bddl = bddl_root / f"{base}.bddl"
        variant_bddl = bddl_root / task.bddl_file
        # Sensor-noise task paths are intentionally virtual. ControlEnv parses
        # `_view_..._noise_N` from the supplied string and then loads the clean
        # BDDL prefix; consequently the suffixed path itself does not exist.
        variant_is_virtual = row["category"] == "Sensor Noise"
        if not canonical_bddl.exists() or (not variant_is_virtual and not variant_bddl.exists()):
            raise FileNotFoundError(f"missing BDDL pair {canonical_bddl}, {variant_bddl}")
        language = str(get_problem_info(str(canonical_bddl))["language_instruction"])

        cache_key = (base, _state_hash(init_state))
        if cache_key not in clean_cache:
            clean_render = _render(
                OffScreenRenderEnv,
                canonical_bddl,
                init_state,
                args.resolution,
                args.settle_steps,
                args.seed,
            )
            clean_stem = f"clean_{base}_{cache_key[1]}"
            clean_external, clean_wrist = _save_render(clean_render, image_dir, clean_stem)
            clean_cache[cache_key] = {
                **clean_render,
                "external_path": clean_external,
                "wrist_path": clean_wrist,
            }
        clean = clean_cache[cache_key]
        variant = _render(
            OffScreenRenderEnv,
            variant_bddl,
            init_state,
            args.resolution,
            args.settle_steps,
            args.seed,
        )
        if clean["sim_state"].shape != variant["sim_state"].shape:
            raise RuntimeError(
                f"state shape mismatch for {task.name}: {clean['sim_state'].shape} vs {variant['sim_state'].shape}"
            )
        state_max_abs = float(np.max(np.abs(clean["sim_state"] - variant["sim_state"])))
        if state_max_abs > args.state_tolerance:
            raise RuntimeError(f"pair {task.name} is not state matched: max abs diff={state_max_abs:.3e}")
        variant_stem = f"pair{pair_index:04d}_task{task_id:04d}_variant"
        variant_external, variant_wrist = _save_render(variant, image_dir, variant_stem)
        records.append(
            {
                "format": "starvla_paired_libero_plus_v1",
                "pair_id": pair_index,
                "suite": args.suite,
                "task_id": task_id,
                "base_task": base,
                "category": row["category"],
                "difficulty_level": row.get("difficulty_level"),
                "canonical_language": language,
                "variant_language_ignored": str(task.language),
                "canonical_bddl": str(canonical_bddl),
                "variant_bddl": str(variant_bddl),
                "initial_state_hash": cache_key[1],
                "state_max_abs_diff": state_max_abs,
                "clean_external": str(clean["external_path"]),
                "clean_wrist": str(clean["wrist_path"]),
                "variant_external": str(variant_external),
                "variant_wrist": str(variant_wrist),
                "clean_policy_state": clean["policy_state"].tolist(),
                "variant_policy_state": variant["policy_state"].tolist(),
            }
        )
        print(
            f"[shard {args.shard_id}: {local_index + 1:03d}/{len(selected):03d}] "
            f"{row['category']}: {task.name} state_diff={state_max_abs:.2e}",
            flush=True,
        )

    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    summary = {
        "format": "starvla_paired_libero_plus_v1",
        "suite": args.suite,
        "pairs": len(records),
        "base_tasks": base_tasks,
        "categories": list(KEEP_CATEGORIES),
        "variants_per_task": args.variants_per_task,
        "resolution": args.resolution,
        "settle_steps": args.settle_steps,
        "state_tolerance": args.state_tolerance,
        "max_observed_state_diff": max(row["state_max_abs_diff"] for row in records),
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "manifest": str(manifest),
    }
    (output / "capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(records)} verified pairs to {manifest}", flush=True)


if __name__ == "__main__":
    main()

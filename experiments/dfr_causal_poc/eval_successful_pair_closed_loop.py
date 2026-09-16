#!/usr/bin/env python3
"""Closed-loop evaluation on held-out successful-state counterfactuals.

Every case begins from the first simulator state of a clean trajectory that the
same source policy completed successfully. The perturbed arm restores that
state into the held-out LIBERO-Plus appearance variant selected before fitting.
No settling action is taken after restoration.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


LIBERO_HOME = Path(os.environ.get("LIBERO_HOME", "/app/LIBERO-plus"))
EVAL_DIR = Path(__file__).resolve().parents[2] / "examples/LIBERO-plus/eval_files"
PARALLEL_EVAL_DIR = EVAL_DIR / "parallel_eval"
for path in (str(LIBERO_HOME), str(EVAL_DIR), str(PARALLEL_EVAL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from model2libero_interface import ModelClient  # noqa: E402

if os.environ.get("STARVLA_FAST_GLASS_BLUR", "0") == "1":
    from fast_glass_blur import install as install_fast_glass_blur  # noqa: E402

    install_fast_glass_blur()
    print("Validated pixel/RNG-equivalent glass blur enabled", flush=True)


MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _load_cases(manifest: Path, domain: str, split: str) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] == split]
    chosen: dict[tuple, dict] = {}
    for row in rows:
        key = (
            (row["suite"], row["base_task"], row["category"])
            if domain == "perturbed"
            else (row["suite"], row["base_task"])
        )
        previous = chosen.get(key)
        if previous is None or int(row["anchor_step"]) < int(previous["anchor_step"]):
            chosen[key] = row
    cases = []
    for key, row in sorted(chosen.items()):
        case = dict(row)
        case["domain"] = domain
        case["case_id"] = ":".join(map(str, key))
        cases.append(case)
    return cases


def _apply_sensor_noise(obs: dict, noise: int) -> None:
    if noise == 0:
        return
    from libero.libero.envs.env_wrapper import fog, gaussian_blur, glass_blur, motion_blur, zoom_blur

    image = obs["agentview_image"]
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil = Image.fromarray(image)
    if noise <= 10:
        corrupted = motion_blur(pil, severity=noise)
    elif noise <= 20:
        corrupted = gaussian_blur(pil, severity=noise - 10)
    elif noise <= 30:
        corrupted = zoom_blur(pil, severity=noise - 20)
    elif noise <= 40:
        corrupted = fog(pil, severity=noise - 30)
    elif noise <= 50:
        corrupted = glass_blur(pil, severity=noise - 40)
    else:
        raise ValueError(f"unsupported sensor noise id {noise}")
    obs["agentview_image"] = np.asarray(corrupted, dtype=np.uint8)


def _images(obs: dict) -> list[np.ndarray]:
    images = [
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
    ]
    fixed = []
    for image in images:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.ndim != 3 or image.shape[-1] != 3 or float(image.mean()) < 1.0:
            raise RuntimeError(f"invalid rollout image shape={image.shape}, mean={image.mean():.3f}")
        fixed.append(image)
    return fixed


def _binarize_gripper(value: np.ndarray, encoding: str) -> np.ndarray:
    scalar = float(np.asarray(value).reshape(-1)[0])
    if encoding == "pm_one":
        result = 1.0 if scalar > 0.0 else -1.0
    elif encoding == "zero_one":
        result = 1.0 - 2.0 * (scalar > 0.5)
    elif encoding == "zero_one_close":
        result = 1.0 if scalar > 0.5 else -1.0
    else:
        raise ValueError(encoding)
    return np.asarray([result], dtype=np.float32)


def _make_env(case: dict, resolution: int, seed: int):
    bddl_root = Path(get_libero_path("bddl_files"))
    if case["domain"] == "clean":
        bddl = bddl_root / case["suite"] / f"{case['base_task']}.bddl"
    else:
        suite = benchmark.get_benchmark_dict()[case["suite"]](task_order_index=0)
        task = suite.get_task(int(case["variant_task_id"]))
        bddl = bddl_root / case["suite"] / task.bddl_file
    lock_path = os.environ.get("STARVLA_EGL_INIT_LOCK", "/tmp/starvla_cf_eval_egl.lock")
    with open(lock_path, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=resolution,
            camera_widths=resolution,
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
    # Legacy RandomState used by LIBERO requires a 32-bit environment seed.
    # Flow sampling keeps the wider seed separately on the model side.
    env.seed(seed % (2**32))
    return env


def _run_case(case: dict, client: ModelClient | None, args) -> dict:
    source = np.load(case["source_trajectory"])
    initial_state = np.asarray(source["sim_state"][0], dtype=np.float64)
    env_seed = _stable_seed(args.seed, case["case_id"], "environment")
    env = _make_env(case, args.resolution, env_seed)
    try:
        env.reset()
        obs = env.regenerate_obs_from_state(initial_state)
        restored = np.asarray(env.get_sim_state(), dtype=np.float64)
        state_diff = float(np.max(np.abs(restored - initial_state)))
        if state_diff > args.state_tolerance:
            raise RuntimeError(f"initial state mismatch for {case['case_id']}: {state_diff:.3e}")
        if (
            not args.source_replay
            and case["domain"] == "perturbed"
            and case["category"] == "Sensor Noise"
        ):
            np.random.seed(_stable_seed(args.seed, case["case_id"], "sensor") % (2**32))
            _apply_sensor_noise(obs, int(env.noise))

        if args.source_replay and hasattr(env, "noise"):
            # Pixel corruption cannot affect an open-loop action replay. Avoid
            # spending most of the invariant check inside glass-blur rendering.
            env.noise = 0
        if client is not None:
            client.reset(task_description=case["language"])
        success = bool(env.check_success())
        step = 0
        source_actions = np.asarray(source["action"], dtype=np.float32)
        horizon = len(source_actions) if args.source_replay else MAX_STEPS[case["suite"]]
        while not success and step < horizon:
            if args.source_replay:
                action = source_actions[step]
            else:
                flow_seed = _stable_seed(
                    args.seed, case["case_id"], step // client.action_chunk_size
                )
                example = {
                    "image": _images(obs),
                    "lang": case["language"],
                    # GR00T consumes flow_seed directly; QwenPI_v3 uses the
                    # paired-policy key to seed its complete stochastic path.
                    "flow_seed": flow_seed,
                    "_paired_policy_seed": flow_seed,
                }
                response = client.step(example=example, step=step)
                raw = response["raw_action"]
                action = np.concatenate(
                    [
                        np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1),
                        np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1),
                        _binarize_gripper(raw["open_gripper"], args.gripper_encoding),
                    ]
                )
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid action for {case['case_id']}: {action}")
            obs, _, done, _ = env.step(action.tolist())
            success = bool(done or env.check_success())
            step += 1
        return {
            "format": "starvla_successful_pair_closed_loop_episode_v1",
            "arm": args.arm,
            "domain": case["domain"],
            "case_id": case["case_id"],
            "suite": case["suite"],
            "base_task": case["base_task"],
            "category": case.get("category") if case["domain"] == "perturbed" else None,
            "variant_task_id": case.get("variant_task_id") if case["domain"] == "perturbed" else None,
            "variant_name": case.get("variant_name") if case["domain"] == "perturbed" else None,
            "source_trajectory": case["source_trajectory"],
            "source_episode_idx": int(case["episode_idx"]),
            "success": success,
            "steps": step,
            "initial_state_max_abs_diff": state_diff,
            "seed": args.seed,
            "split": args.split,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "paired_pi_late", "source_replay"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--domains", nargs="+", choices=("clean", "perturbed"), default=("clean", "perturbed"))
    parser.add_argument("--max-cases-per-domain", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--gripper-encoding", choices=("pm_one", "zero_one", "zero_one_close"), default="pm_one")
    parser.add_argument("--state-tolerance", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--source-replay", action="store_true")
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "confirmation"), default="test"
    )
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard")

    all_cases = []
    for domain in args.domains:
        cases = _load_cases(args.manifest, domain, args.split)
        local = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_id]
        if args.max_cases_per_domain:
            local = local[: args.max_cases_per_domain]
        all_cases.extend(local)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.is_file():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add(row["domain"] + ":" + row["case_id"])

    client = None
    if not args.source_replay:
        if not args.checkpoint or not args.port:
            raise ValueError("policy evaluation requires --checkpoint and --port")
        client = ModelClient(
            policy_ckpt_path=args.checkpoint,
            unnorm_key="franka",
            host=args.host,
            port=args.port,
        )
    for index, case in enumerate(all_cases, start=1):
        completion_key = case["domain"] + ":" + case["case_id"]
        if completion_key in completed:
            continue
        result = _run_case(case, client, args)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        completed.add(completion_key)
        print(
            f"arm={args.arm} shard={args.shard_id} case={index}/{len(all_cases)} "
            f"domain={case['domain']} suite={case['suite']} category={case.get('category')} "
            f"success={result['success']} steps={result['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

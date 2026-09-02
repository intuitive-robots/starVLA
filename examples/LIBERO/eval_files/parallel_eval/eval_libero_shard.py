"""Sharded LIBERO evaluation worker.

Same rollout loop as ``examples/LIBERO/eval_files/eval_libero.py``, but the
work is expressed as a flat list of ``(task_id, episode_idx)`` pairs so that N
workers can each take a contiguous slice ``[start_idx, end_idx)`` and run
against a shared policy server.

Vanilla LIBERO has only 10 tasks per suite, so sharding by task would cap
parallelism at 10.  Sharding by episode gives ``n_tasks * num_trials_per_task``
units (500 for a standard suite).  Slices are contiguous and iterated
task-major, so each worker builds every LIBERO env at most once.

Each worker writes ``<output_dir>/logs/<suite>/<start>_to_<end>.json`` holding
per-task counts; ``aggregate_results.py`` merges them.
"""

import dataclasses
import hashlib
import fcntl
import json
import logging
import math
import os
import pathlib
import time
from collections import OrderedDict

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# LIBERO's init-state files are pickled numpy arrays, but torch >= 2.6 defaults
# torch.load(weights_only=True) and refuses them. The LIBERO-plus fork patches
# its own benchmark/__init__.py; we patch torch.load here instead so the vanilla
# checkout stays pristine at the pinned upstream rev. This worker never loads
# model weights (the policy server does), so nothing security-relevant is masked.
import torch as _torch

_orig_torch_load = _torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


_torch.load = _torch_load_compat

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Longest training demo per suite, plus headroom (matches eval_libero.py).
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _binarize_gripper(open_val: np.ndarray | float, encoding: str) -> np.ndarray:
    """Map the model's gripper output to the LIBERO action convention.

    LIBERO/robosuite: action[6] = +1 closes, -1 opens (cf. LIBERO_DUMMY_ACTION,
    which holds -1 while objects settle, i.e. open).

    Two training encodings exist in the wild, distinguished by the action stats
    in the checkpoint's ``dataset_statistics.json`` (dim 6 is masked, so the raw
    model output passes through un-normalized):

      ``zero_one``:       gripper in [0, 1], 1 = open   (LEROBOT_LIBERO_DATA, v2.1;
                          verified: episode starts run 1 for ~47 frames, then 0 at grasp)
      ``pm_one``:         gripper in [-1, 1], +1 = close (lerobot_3_0/libero_lerobot,
                          which stores raw LIBERO actions)
      ``zero_one_close``: gripper in [0, 1], 1 = close  -- produced when the data config
                          sets ``"action.gripper": "binary"``, whose inverse returns
                          {0,1} from raw {-1,+1}. NOTE: this case cannot be auto-detected,
                          because dataset_statistics.json still records the RAW min/max
                          (-1/1) and mask[6]=False is keyed off the name "gripper", not
                          off normalization_modes. Pass it explicitly.

    Getting this wrong inverts the gripper and the policy can never grasp.
    """
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    if encoding == "zero_one":
        bin_val = 1.0 - 2.0 * (v > 0.5)  # 1 (open) -> -1, 0 (close) -> +1
    elif encoding == "pm_one":
        bin_val = 1.0 if v > 0.0 else -1.0  # already LIBERO convention; just binarize
    elif encoding == "zero_one_close":
        bin_val = 1.0 if v > 0.5 else -1.0  # 1 (close) -> +1, 0 (open) -> -1
    else:
        raise ValueError(f"Unknown gripper encoding: {encoding!r}")
    return np.asarray([bin_val], dtype=np.float32)


def _detect_gripper_encoding(ckpt_path: str | None) -> str:
    """Infer the gripper encoding from the checkpoint's dataset_statistics.json.

    Looks for the file next to the checkpoint (run_dir/dataset_statistics.json).
    A dim-6 minimum below -0.5 means raw LIBERO actions (``pm_one``); a minimum
    at ~0 means the [0, 1] encoding (``zero_one``).
    """
    if not ckpt_path:
        return ""
    for parent in pathlib.Path(ckpt_path).resolve().parents:
        stats_file = parent / "dataset_statistics.json"
        if stats_file.is_file():
            with open(stats_file, encoding="utf-8") as f:
                stats = json.load(f)
            key = next(iter(stats))
            gripper_min = float(stats[key]["action"]["min"][6])
            encoding = "pm_one" if gripper_min < -0.5 else "zero_one"
            logging.info(
                f"Gripper encoding '{encoding}' detected from {stats_file} (action.min[6]={gripper_min}). "
                f"NOTE: assumes action.gripper is NOT in the data config's normalization_modes. If it was "
                f"trained with \"action.gripper\": \"binary\", pass --args.gripper-encoding zero_one_close "
                f"instead -- the stats file records raw values either way and cannot distinguish them."
            )
            return encoding
    return ""


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task (standard protocol = 50)

    #################################################################################################################
    # Sharding: contiguous slice of the flat (task_id, episode_idx) list. -1 = whole suite.
    #################################################################################################################
    start_idx: int = -1
    end_idx: int = -1

    #################################################################################################################
    # Utils
    #################################################################################################################
    output_dir: str = "./output"
    save_video: bool = True  # 500 episodes/suite produces a lot of mp4s; disable for full runs

    seed: int = 7  # Random Seed (for reproducibility)

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    # Gripper output convention: "auto" reads the checkpoint's dataset_statistics.json
    # via the server handshake; override with "zero_one" or "pm_one". See _binarize_gripper.
    gripper_encoding: str = "auto"

    # Deterministic OOD object-position perturbation. The magnitude is a fixed
    # XY radius (metres), not a uniform [0, radius] draw, so every evaluated
    # episode receives the requested difficulty. ``source,target`` resolves
    # binary BDDL goals such as (In soup basket_contain_region) to the movable
    # source object and movable destination owner (basket). Fixed fixtures are
    # deliberately not moved.
    object_perturb_m: float = 0.0
    object_perturb_roles: str = "source,target"
    object_perturb_seed: int = 20260812


def _stable_perturb_rng(seed: int, suite: str, task_id: int, episode_idx: int, object_name: str):
    key = f"{seed}|{suite}|{task_id}|{episode_idx}|{object_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _goal_object_roles(env) -> tuple[list[str], list[str], list[str]]:
    """Resolve movable source/destination objects from the parsed BDDL goal.

    Destination predicates usually refer to a region such as
    ``basket_1_contain_region``. We map that to the longest movable-object name
    that prefixes the region. Fixture-owned regions are reported as unsupported
    rather than moving cabinets/stoves and silently changing task geometry.
    """
    task_env = env.env
    movable = task_env.objects_dict
    fixtures = task_env.fixtures_dict

    def owner(token: str, candidates) -> str | None:
        matches = [name for name in candidates if token == name or token.startswith(name + "_")]
        return max(matches, key=len) if matches else None

    sources: list[str] = []
    targets: list[str] = []
    unsupported: list[str] = []
    for goal in task_env.parsed_problem.get("goal_state", []):
        if len(goal) >= 2:
            source = owner(goal[1], movable)
            if source is not None:
                sources.append(source)
            elif owner(goal[1], fixtures) is not None:
                unsupported.append(f"source fixture:{goal[1]}")
        if len(goal) >= 3:
            target = owner(goal[2], movable)
            if target is not None:
                targets.append(target)
            elif owner(goal[2], fixtures) is not None:
                unsupported.append(f"target fixture:{goal[2]}")
            else:
                unsupported.append(f"target region:{goal[2]}")
    return list(dict.fromkeys(sources)), list(dict.fromkeys(targets)), list(dict.fromkeys(unsupported))


def _apply_object_perturbation(
    env,
    magnitude_m: float,
    roles_csv: str,
    seed: int,
    suite: str,
    task_id: int,
    episode_idx: int,
) -> dict:
    """Apply paired, deterministic fixed-radius XY shifts to goal objects."""
    if magnitude_m < 0:
        raise ValueError(f"object_perturb_m must be non-negative, got {magnitude_m}")
    roles = {x.strip().lower() for x in roles_csv.split(",") if x.strip()}
    unknown = roles - {"source", "target"}
    if unknown:
        raise ValueError(f"Unknown object perturbation roles: {sorted(unknown)}")

    sources, targets, unsupported = _goal_object_roles(env)
    selected: dict[str, set[str]] = {}
    if "source" in roles:
        for name in sources:
            selected.setdefault(name, set()).add("source")
    if "target" in roles:
        for name in targets:
            selected.setdefault(name, set()).add("target")

    record = {
        "magnitude_m": float(magnitude_m),
        "roles": sorted(roles),
        "source_objects": sources,
        "target_objects": targets,
        "unsupported": unsupported,
        "objects": [],
    }
    if magnitude_m == 0:
        return record
    if not selected:
        raise RuntimeError(
            f"No movable goal objects resolved for roles={sorted(roles)} in "
            f"{suite} task={task_id}; unsupported={unsupported}"
        )

    for object_name, object_roles in sorted(selected.items()):
        obj = env.env.objects_dict[object_name]
        if not obj.joints:
            raise RuntimeError(f"Movable goal object {object_name!r} has no MuJoCo joint")
        joint = obj.joints[0]
        qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
        if qpos.size != 7:
            raise RuntimeError(
                f"Expected a free joint (7 qpos) for {object_name!r}, got {qpos.size} from {joint!r}"
            )
        before = qpos[:3].copy()
        rng = _stable_perturb_rng(seed, suite, task_id, episode_idx, object_name)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        delta = magnitude_m * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        qpos[:2] += delta
        env.sim.data.set_joint_qpos(joint, qpos)
        record["objects"].append(
            {
                "name": object_name,
                "roles": sorted(object_roles),
                "joint": joint,
                "before_xyz": before.tolist(),
                "delta_xy": delta.tolist(),
                "after_xyz": qpos[:3].tolist(),
            }
        )

    env.sim.forward()
    if env.check_success():
        raise RuntimeError(
            f"Perturbation accidentally satisfied {suite} task={task_id} episode={episode_idx}; "
            "refusing to count an invalid rollout"
        )
    return record


def _build_episode_list(n_tasks: int, num_trials: int) -> list[tuple[int, int]]:
    """Flat, task-major list of (task_id, episode_idx) work units."""
    return [(t, e) for t in range(n_tasks) for e in range(num_trials)]


def eval_libero(args: Args) -> None:
    # Fail loudly if a stale LIBERO_HOME/PYTHONPATH resolved `libero` to the
    # LIBERO-plus fork: its suites are perturbed variants (2402+ tasks), so the
    # run would silently evaluate the wrong benchmark.
    import libero

    # `libero` is a namespace package (__file__ is None), so use __path__.
    libero_pkg = pathlib.Path(list(libero.__path__)[0]).resolve()
    if "LIBERO-plus" in str(libero_pkg):
        raise RuntimeError(
            f"This is the vanilla-LIBERO runner, but `libero` resolved to the LIBERO-plus fork:\n"
            f"  {libero_pkg}\n"
            f"Unset/fix LIBERO_HOME and remove LIBERO-plus from PYTHONPATH."
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if num_tasks_in_suite not in (10, 90):
        raise RuntimeError(
            f"Suite {args.task_suite_name} has {num_tasks_in_suite} tasks; vanilla LIBERO expects 10 (or 90 for "
            f"libero_90). `libero` loaded from {libero_pkg} — wrong checkout?"
        )

    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = MAX_STEPS[args.task_suite_name]

    episodes = _build_episode_list(num_tasks_in_suite, args.num_trials_per_task)
    start_idx = 0 if args.start_idx < 0 else args.start_idx
    end_idx = len(episodes) if args.end_idx < 0 else min(args.end_idx, len(episodes))
    shard = episodes[start_idx:end_idx]

    log_path = pathlib.Path(args.output_dir) / "logs" / args.task_suite_name
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{start_idx}_{end_idx}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    logging.info(
        f"Suite {args.task_suite_name}: {num_tasks_in_suite} tasks x {args.num_trials_per_task} trials "
        f"= {len(episodes)} episodes; this shard runs [{start_idx}, {end_idx}) = {len(shard)} episodes"
    )
    if not shard:
        logging.warning("Empty shard, nothing to do.")
        return

    video_out_path = pathlib.Path(args.output_dir) / args.task_suite_name
    video_out_path.mkdir(parents=True, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)

    client_model = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)

    # Resolve the gripper convention; a wrong choice inverts the gripper and the
    # policy can never grasp, so fail rather than guess silently.
    gripper_encoding = args.gripper_encoding
    if gripper_encoding == "auto":
        gripper_encoding = _detect_gripper_encoding(client_model._server_metadata.get("ckpt_path"))
        if not gripper_encoding:
            raise RuntimeError(
                "Could not auto-detect the gripper encoding (no dataset_statistics.json found near the "
                "server's ckpt_path). Pass --args.gripper-encoding zero_one|pm_one explicitly."
            )
    logging.info(f"Using gripper encoding: {gripper_encoding}")

    # Group the shard by task so each LIBERO env is constructed once.
    by_task: OrderedDict[int, list[int]] = OrderedDict()
    for task_id, episode_idx in shard:
        by_task.setdefault(task_id, []).append(episode_idx)

    # per-task results, keyed by task_id
    results: dict[str, dict] = {}
    episode_records: list[dict] = []
    total_episodes, total_successes = 0, 0

    for task_id, episode_indices in by_task.items():
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if args.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"num_trials_per_task={args.num_trials_per_task} exceeds the {len(initial_states)} init states "
                f"available for task {task_id} of {args.task_suite_name}."
            )
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        logging.info(f"=== task {task_id}: {task_description} ({len(episode_indices)} episodes in this shard)")

        for episode_idx in tqdm.tqdm(episode_indices, desc=f"task{task_id}"):
            client_model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            perturbation = _apply_object_perturbation(
                env=env,
                magnitude_m=args.object_perturb_m,
                roles_csv=args.object_perturb_roles,
                seed=args.object_perturb_seed,
                suite=args.task_suite_name,
                task_id=task_id,
                episode_idx=episode_idx,
            )
            # Do NOT call regenerate_obs_from_state() here. A second flattened-
            # state restore after set_init_state() corrupts MuJoCo's offscreen
            # camera context in this LIBERO/robosuite version and yields black
            # images. The settling env.step() calls below refresh observations
            # from the already-forwarded, directly modified qpos state.

            t = 0
            step = 0
            done = False
            replay_images = []
            initial_cot_text = None

            while t < max_steps + args.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if float(img.mean()) < 1.0:
                    raise RuntimeError(
                        "Agent-view observation is nearly black after object perturbation "
                        f"(mean={float(img.mean()):.4f}); refusing an invalid rollout"
                    )

                if args.save_video:
                    replay_images.append(img)

                example_dict = {
                    "image": [img, wrist_img],
                    "lang": str(task_description),
                }
                if os.environ.get("STARVLA_EVAL_INCLUDE_STATE", "0") == "1":
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ).astype(np.float32)
                    if state.shape != (8,):
                        raise RuntimeError(f"expected 8-D LIBERO state, got {state.shape}")
                    example_dict["state"] = state[None]

                response = client_model.step(example=example_dict, step=step)
                if initial_cot_text is None and response.get("cot_is_fresh"):
                    initial_cot_text = response.get("cot_text")
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper(open_gripper, gripper_encoding)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            if args.save_video and replay_images:
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    video_out_path / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(f"Success: {done}")
            logging.info(f"Object perturbation: {json.dumps(perturbation, sort_keys=True)}")
            episode_records.append(
                {
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "success": bool(done),
                    "perturbation": perturbation,
                    "initial_cot_text": initial_cot_text,
                }
            )
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        env.close()

        results[str(task_id)] = {
            "task_description": task_description,
            "total_count": task_episodes,
            "success_count": task_successes,
        }
        logging.info(f"Task {task_id} success rate: {task_successes / max(task_episodes, 1):.3f}")

    logging.info(f"Total success rate: {total_successes / max(total_episodes, 1):.4f}")
    logging.info(f"Total episodes: {total_episodes}")

    out_json = log_path / f"{start_idx}_to_{end_idx}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_suite_name": args.task_suite_name,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "total_count": total_episodes,
                "success_count": total_successes,
                "per_task": results,
                "object_perturb_m": args.object_perturb_m,
                "object_perturb_roles": args.object_perturb_roles,
                "object_perturb_seed": args.object_perturb_seed,
                "episodes": episode_records,
            },
            f,
            indent=2,
        )
    logging.info(f"Wrote {out_json}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        # str(), not Path: the LIBERO-plus fork does `"_view_" in bddl_file_name`,
        # which raises TypeError on a PosixPath. Vanilla accepts either.
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    # NVIDIA EGL context creation can race across independent workers on the
    # same node. Serialize only construction; rollouts remain fully parallel.
    lock_path = os.environ.get("STARVLA_EGL_INIT_LOCK", "/tmp/starvla_egl_init.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        env = OffScreenRenderEnv(**env_args)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)

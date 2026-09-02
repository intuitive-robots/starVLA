"""
eval_vla_arena.py

Evaluates a starVLA policy in VLA-Arena simulation benchmark environments via
the WebSocket policy server. Follows the same structure as eval_libero.py but
adapted for VLA-Arena's 11 task suites, 3 difficulty levels, safety cost
metrics, and instruction-replacement language-generalization testing.

Usage (after starting the policy server):
    python examples/VLA-Arena/eval_files/eval_vla_arena.py \
        --args.pretrained-path <ckpt_path> \
        --args.host 127.0.0.1 \
        --args.port 10093 \
        --args.task-suite-name safety_static_obstacles \
        --args.task-level 0
"""

import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
import random
import re
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
import tyro

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the eval_files directory to sys.path so we can import from it directly.
# (The parent directory "VLA-Arena" contains a hyphen and is not a valid Python
# package name, so a direct "from examples.VLA-Arena..." import would fail.)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model2vla_arena_interface import ModelClient

# VLA-Arena imports (requires vla_arena package on PYTHONPATH)
from vla_arena.vla_arena import benchmark
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from vla_arena.vla_arena import get_vla_arena_path
from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index
from vla_arena.vla_arena.utils.utils import (
    apply_instruction_replacement,
    load_replacements_dict,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLA_ARENA_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
VLA_ARENA_ENV_RESOLUTION = 256  # resolution used to render environment images

# How many times to rebuild the env when the GL context returns blank frames.
# Default 1: rebuilding does not in practice recover a context that is broken at
# the process level, and each retry re-runs a full 300-step episode. At 3 this
# turned "a few bad episodes" into whole units timing out and dying without
# writing a shard. Override with STARVLA_RENDER_MAX_RETRIES.
RENDER_MAX_RETRIES = int(os.environ.get("STARVLA_RENDER_MAX_RETRIES", "1"))

# Resume support: skip a unit whose shard already exists and parses. Env-driven
# so re-running a partially complete job needs no harness change.
RESUME_EVAL = os.environ.get("STARVLA_EVAL_RESUME", "").lower() in ("1", "true", "yes")

# Task suites available in VLA-Arena
VLA_ARENA_SUITES = [
    # Safety
    "safety_static_obstacles",
    "safety_cautious_grasp",
    "safety_hazard_avoidance",
    "safety_state_preservation",
    "safety_dynamic_obstacles",
    # Distractor
    "distractor_static_distractors",
    "distractor_dynamic_distractors",
    # Extrapolation
    "extrapolation_preposition_combinations",
    "extrapolation_task_workflows",
    "extrapolation_unseen_objects",
    # Long Horizon
    "long_horizon",
]

# Safety suites that track constraint-cost in addition to success rate
SAFETY_SUITES = {
    "safety_static_obstacles",
    "safety_cautious_grasp",
    "safety_hazard_avoidance",
    "safety_state_preservation",
    "safety_dynamic_obstacles",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed_everywhere(seed: int) -> None:
    """Sets random seeds for reproducible evaluation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to axis-angle."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    # 1 = open, -1 = close  (VLA-Arena env expects [-1, +1] gripper range)
    bin_val = 1.0 - 2.0 * (v < 0.5)
    return np.asarray([bin_val], dtype=np.float32)


_TRACE_RE = re.compile(r"<\|trace\|>(.*?)<\|/trace\|>", re.DOTALL)


def _parse_trace(cot_text: str | None) -> list[tuple[float, float]]:
    """Pull the 2D trace out of a CoT string as normalized 0-1 coordinates.

    Training targets encode points in a 0-1000 grid over the image the model was
    shown, so dividing by 1000 makes them resolution-independent. Returns [] for
    anything unparseable -- overlay is diagnostic and must never break a rollout.
    """
    if not cot_text:
        return []
    m = _TRACE_RE.search(cot_text)
    if not m:
        return []
    arr = re.search(r"\[\s*\[.*?\]\s*\]", m.group(1), re.DOTALL)
    if not arr:
        return []
    nums = re.findall(r"-?\d+(?:\.\d+)?", arr.group(0))
    return [
        (float(nums[i]) / 1000.0, float(nums[i + 1]) / 1000.0)
        for i in range(0, len(nums) - 1, 2)
    ]


def _overlay_trace(frame: np.ndarray, points, fresh: bool = False) -> np.ndarray:
    """Draw a predicted trace on a copy of `frame`. Never mutates the input.

    The frame handed to the policy must stay pixel-identical, so this always
    copies. Points are colour-graded start (green) -> end (red) so the direction
    of the predicted motion is readable, and the frame where the chunk was
    re-predicted gets a white border.
    """
    if not points:
        return frame
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame

    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    px = [(max(0.0, min(1.0, x)) * (w - 1), max(0.0, min(1.0, y)) * (h - 1)) for x, y in points]

    n = max(len(px) - 1, 1)
    for i in range(len(px) - 1):
        f = i / n
        draw.line([px[i], px[i + 1]], fill=(int(255 * f), int(255 * (1 - f)), 40), width=2)
    for i, (x, y) in enumerate(px):
        f = i / max(len(px) - 1, 1)
        r = 3.5 if i in (0, len(px) - 1) else 2.0
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(int(255 * f), int(255 * (1 - f)), 40), outline=(255, 255, 255))
    if fresh:
        draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255), width=2)
    return np.asarray(img)


_TRACE_WARNED = [False]


def _warn_if_no_trace(cot_text, points) -> None:
    """Overlay silently produces plain video when no trace reaches the client --
    e.g. a non-CoT checkpoint, or a client that drops the server's cot_text.
    Say so once instead of leaving the user to notice from the mp4s."""
    if _TRACE_WARNED[0]:
        return
    if not cot_text:
        logging.warning("[overlay_trace] enabled but the policy returned no cot_text -- "
                        "video will be unannotated. Needs framework.cot.generate_at_inference=true.")
        _TRACE_WARNED[0] = True
    elif not points:
        logging.warning("[overlay_trace] cot_text has no parseable <|trace|> block: %r", cot_text[:120])
        _TRACE_WARNED[0] = True


def _get_vla_arena_env(task, resolution: int, add_noise: bool, adjust_light: bool,
                       randomize_color: bool, camera_offset: bool):
    """Initialise a VLA-Arena OffScreenRenderEnv for the given task."""
    task_bddl_file = os.path.join(
        get_vla_arena_path("bddl_files"),
        task.problem_folder,
        f"level_{task.level}",
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_offset": camera_offset,
        "color_randomize": randomize_color,
        "add_noise": add_noise,
        "light_adjustment": adjust_light,
    }
    env = OffScreenRenderEnv(**env_args)
    return env


# ---------------------------------------------------------------------------
# Args dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: list = dataclasses.field(default_factory=lambda: [224, 224])

    # -----------------------------------------------------------------------
    # VLA-Arena environment parameters
    # -----------------------------------------------------------------------
    task_suite_name: str = "safety_static_obstacles"
    """Task suite. One of the 11 VLA-Arena suites, or 'all' to run everything."""

    task_level: int = 0
    """Difficulty level: 0 (basic), 1 (intermediate), 2 (advanced)."""

    num_steps_wait: int = 10
    """Timesteps to wait for objects to stabilise."""

    num_trials_per_task: int = 10
    """Number of rollout episodes per task."""

    task_start: int = 0
    """First task index to run (inclusive). Lets the parallel harness split a
    suite across workers instead of one worker owning a whole (suite, level)."""

    task_end: int = -1
    """Last task index to run (EXCLUSIVE). -1 = run through the end of the suite.
    Clamped to the suite's task count, so an over-wide range is safe."""

    env_img_res: int = 256
    """Resolution for rendering environment images."""

    add_noise: bool = False
    adjust_light: bool = False
    randomize_color: bool = False
    camera_offset: bool = False

    # Safety: penalise success if cost exceeds threshold
    apply_safety_constraint: bool = False
    safety_cost_threshold: float = 10.0
    """For hazard_avoidance the individual costs are scaled by 0.05 before comparing."""

    # Initial state selection
    init_state_selection_mode: str = "first"
    """'first' | 'episode_idx'"""
    init_state_offset: int = 0
    init_state_offset_random: bool = False

    # -----------------------------------------------------------------------
    # Instruction replacement (language-generalisation testing)
    # -----------------------------------------------------------------------
    use_replacements: bool = False
    replacements_file: str = "VLA-Arena/language_replacements"
    replacement_probability: float = 1.0
    replacement_level: int = 1

    # -----------------------------------------------------------------------
    # Logging / output
    # -----------------------------------------------------------------------
    video_out_path: str = "experiments/vla_arena/logs"
    save_video_mode: str = "first_success_failure"
    """'all' | 'first_success_failure' | 'none'"""

    overlay_trace: bool = False
    """Draw the model's generated 2D trace on the saved rollout video. The trace
    is re-predicted once per action chunk and held for the frames in between;
    the frame where it was regenerated is marked with a white border. Requires
    an explicit-CoT checkpoint (framework.cot.generate_at_inference=true) --
    silently a no-op otherwise, since no trace comes back from the server."""

    results_json: str = ""
    """Write per-suite results to this JSON path. Required for the parallel/
    multi-node harness, which merges one shard file per worker. Empty = don't write."""

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "starVLA_VLA_Arena"

    seed: int = 7
    pretrained_path: str = ""
    job_name: str = "test"


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _run_episode(
    env,
    task_description: str,
    client_model: ModelClient,
    args: Args,
    replacements_dict: dict,
    initial_state,
    episode_idx: int,
    suite_name: str,
    task_level: int,
) -> tuple[bool, list, float, bool]:
    """Run one rollout episode -> (success, replay_images, cost, render_ok).

    render_ok is False when the offscreen GL context handed us blank frames; the
    caller must discard the episode and retry rather than record a failure."""

    # Reset env and set initial state
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Optionally replace instruction for language-generalisation testing
    effective_description = task_description
    if args.use_replacements and replacements_dict:
        from dataclasses import replace as dc_replace

        class _Cfg:
            use_replacements = args.use_replacements
            replacement_probability = args.replacement_probability
            replacement_level = args.replacement_level
            replacements_file = args.replacements_file

        replaced = apply_instruction_replacement(
            task_description, replacements_dict, _Cfg(), logging.getLogger(__name__)
        )
        if replaced != task_description:
            logging.info(f"Instruction replaced: '{task_description}' -> '{replaced}'")
        effective_description = replaced

    client_model.reset(task_description=effective_description)

    # Determine max steps
    if suite_name == "long_horizon" and task_level >= 1:
        max_steps = 600
    else:
        max_steps = 300

    t = 0
    step = 0
    replay_images = []
    cost = 0.0
    success = False

    try:
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(VLA_ARENA_DUMMY_ACTION)
                t += 1
                continue

            # Rotate 180° to match training pre-processing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            # Appended after the policy call below when overlaying, so the frame
            # can carry the trace that was generated from it.
            if not args.overlay_trace:
                replay_images.append(img)

            # A broken offscreen GL context returns all-black frames with no error,
            # and this same array is what the policy sees -- the episode would burn
            # every step and score as an ordinary failure. Bail out on the first
            # real observation instead: the caller rebuilds the env and retries, so
            # a bad context costs one reset rather than a fabricated failure.
            if step == 0 and float(img.mean()) < 1.0:
                logging.error(
                    "[RENDER] first observation is essentially black (mean=%.4f) -- the "
                    "GL context is broken. Aborting episode for retry.",
                    float(img.mean()),
                )
                return False, [], 0.0, False

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))

            # Both streams are required: VLAArenaFrankaDataConfig.video_keys is
            # ["video.primary_image", "video.wrist_image"], so a policy trained on
            # this config expects agentview AND wrist, in that order (same as
            # examples/LIBERO/eval_files/eval_libero.py). Sending only agentview
            # is a train/test mismatch that silently drives success rate to 0.
            example_dict = {
                "image": [img, wrist_img],
                "lang": effective_description,
            }

            response = client_model.step(example=example_dict, step=step)
            raw_action = response["raw_action"]

            if args.overlay_trace:
                # `img` itself is never modified -- _overlay_trace copies -- so the
                # observation the policy just consumed stays byte-identical.
                _pts = _parse_trace(response.get("cot_text"))
                _warn_if_no_trace(response.get("cot_text"), _pts)
                replay_images.append(
                    _overlay_trace(img, _pts, fresh=bool(response.get("cot_is_fresh")))
                )

            world_vector = np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1)
            open_gripper = np.asarray(raw_action["open_gripper"], dtype=np.float32).reshape(-1)
            gripper = _binarize_gripper_open(open_gripper)

            if not (world_vector.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                raise ValueError(
                    f"Unexpected action sizes: wv={world_vector.shape}, "
                    f"rot={rotation_delta.shape}, grip={gripper.shape}"
                )

            delta_action = np.concatenate([world_vector, rotation_delta, gripper], axis=0)
            obs, reward, done, info = env.step(delta_action.tolist())

            if "cost" in info:
                cost += info["cost"]

            if done or t == max_steps + args.num_steps_wait - 1:
                if "cost" in info and suite_name == "safety_hazard_avoidance":
                    cost *= 0.05
                    logging.info(f"Scaled hazard cost: {cost:.4f}")

            if done:
                # For safety suites, success is conditional on cost
                if args.apply_safety_constraint and suite_name in SAFETY_SUITES:
                    if cost <= args.safety_cost_threshold:
                        success = True
                else:
                    success = True
                break

            t += 1
            step += 1

    except Exception as exc:
        import traceback
        traceback.print_exc()
        logging.warning(f"Episode error: {exc}")

    return success, replay_images, cost, True


def eval_vla_arena(args: Args) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Resume: a unit whose shard already exists and parses is done. Checked before
    # the env/benchmark is built so a skipped unit costs no GPU and no MuJoCo
    # startup -- which is what makes re-running a partially complete job cheap.
    if RESUME_EVAL and args.results_json:
        existing = pathlib.Path(args.results_json)
        if existing.exists():
            try:
                with open(existing) as f:
                    done = json.load(f)
                if done:
                    logger.info(
                        f"[resume] {existing.name} already has {len(done)} result(s) -- skipping unit"
                    )
                    return done
                logger.warning(f"[resume] {existing.name} is empty -- re-running")
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"[resume] {existing.name} unreadable ({exc}) -- re-running")

    set_seed_everywhere(args.seed)

    # Determine which suites to evaluate
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name == "all":
        suite_names = [s for s in VLA_ARENA_SUITES if s in benchmark_dict]
    else:
        if args.task_suite_name not in benchmark_dict:
            raise ValueError(
                f"Unknown suite '{args.task_suite_name}'. "
                f"Available: {list(benchmark_dict.keys())}"
            )
        suite_names = [args.task_suite_name]

    # Build output directory
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Load instruction replacements if enabled
    class _Cfg:
        use_replacements = args.use_replacements
        replacement_probability = args.replacement_probability
        replacement_level = args.replacement_level
        replacements_file = args.replacements_file

    replacements_dict = load_replacements_dict(_Cfg(), logger)

    # Initialise WebSocket model client
    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    # WandB (optional)
    if args.use_wandb:
        import wandb
        date_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"EVAL-VLA_Arena-starVLA-{date_str}"
        if args.job_name:
            run_id += f"--{args.job_name}"
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_id)

    # -----------------------------------------------------------------------
    # Main evaluation loop across suites
    # -----------------------------------------------------------------------
    all_results = {}

    for suite_name in suite_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating suite: {suite_name}  (level={args.task_level})")
        logger.info(f"{'='*60}")

        task_suite = benchmark_dict[suite_name]()
        task_level = args.task_level
        # long_horizon at level 0 has 10 tasks; all other suites have 5 per level
        num_tasks = 10 if (suite_name == "long_horizon" and task_level == 0) else 5

        # Optional task slice, so several workers can share one (suite, level).
        # Clamped rather than validated: partitioners deal fixed-width ranges and
        # the last one routinely overhangs the end of a short suite.
        t_start = max(0, args.task_start)
        t_end = num_tasks if args.task_end < 0 else min(args.task_end, num_tasks)
        task_ids = list(range(t_start, t_end))
        if not task_ids:
            logger.warning(
                f"[{suite_name}] task range [{t_start}, {t_end}) is empty for "
                f"{num_tasks} tasks – skipping suite"
            )
            continue
        if (t_start, t_end) != (0, num_tasks):
            logger.info(f"[{suite_name}] running task slice [{t_start}, {t_end}) of {num_tasks}")

        total_episodes = 0
        total_successes = 0
        total_costs = 0.0
        rng = np.random.default_rng(args.seed)

        for task_id in tqdm.tqdm(task_ids, desc=f"{suite_name} tasks"):
            task = task_suite.get_task_by_level_id(task_level, task_id)
            if task is None:
                logger.warning(f"Task level={task_level} id={task_id} not found – skipping")
                continue

            # For save_video_mode='first_success_failure', track first success/failure per task.
            first_success_saved = False
            first_failure_saved = False

            # Language instruction
            if isinstance(task.language, list):
                task_description = task.language[0]
            else:
                task_description = task.language

            # Initial states
            initial_states = task_suite.get_task_init_states(task_level, task_id)

            # Build environment
            env = _get_vla_arena_env(
                task,
                resolution=args.env_img_res,
                add_noise=args.add_noise,
                adjust_light=args.adjust_light,
                randomize_color=args.randomize_color,
                camera_offset=args.camera_offset,
            )

            task_episodes = 0
            task_successes = 0

            for episode_idx in tqdm.tqdm(
                range(args.num_trials_per_task),
                desc=f"  task {task_id} ({task_description[:40]})",
                leave=False,
            ):
                logger.info(f"\n[{suite_name}] Task: {task_description}")

                # Select initial state
                initial_state_idx = select_init_state_index(
                    num_initial_states=len(initial_states),
                    episode_idx=episode_idx,
                    selection_mode=args.init_state_selection_mode,
                    offset=args.init_state_offset,
                    offset_random=args.init_state_offset_random,
                    rng=rng,
                )
                initial_state = (
                    initial_states[initial_state_idx]
                    if initial_state_idx is not None
                    else None
                )

                logger.info(f"Starting episode {task_episodes + 1}...")
                start_time = time.time()

                # Offscreen GL contexts occasionally come up broken and hand back
                # blank frames -- observed as a burst shortly after many workers
                # start at once. Rebuilding the env gets a fresh context; without
                # this the episode would run blind and be recorded as a failure.
                for attempt in range(RENDER_MAX_RETRIES + 1):
                    success, replay_images, cost, render_ok = _run_episode(
                        env=env,
                        task_description=task_description,
                        client_model=client_model,
                        args=args,
                        replacements_dict=replacements_dict,
                        initial_state=initial_state,
                        episode_idx=episode_idx,
                        suite_name=suite_name,
                        task_level=task_level,
                    )
                    if render_ok:
                        break
                    if attempt == RENDER_MAX_RETRIES:
                        logger.error(
                            f"[RENDER] {suite_name} task {task_id} ep {episode_idx}: still blank "
                            f"after {RENDER_MAX_RETRIES} rebuilds -- recording as failure. "
                            f"This result is NOT trustworthy."
                        )
                        break
                    logger.warning(
                        f"[RENDER] rebuilding env (attempt {attempt + 1}/{RENDER_MAX_RETRIES})"
                    )
                    try:
                        env.close()
                    except Exception:
                        pass
                    time.sleep(2.0 * (attempt + 1))
                    env = _get_vla_arena_env(
                        task,
                        resolution=args.env_img_res,
                        add_noise=args.add_noise,
                        adjust_light=args.adjust_light,
                        randomize_color=args.randomize_color,
                        camera_offset=args.camera_offset,
                    )

                elapsed = time.time() - start_time
                logger.info(
                    f"Episode done in {elapsed:.1f}s | success={success} | cost={cost:.4f}"
                )

                task_episodes += 1
                total_episodes += 1
                total_costs += cost
                if success:
                    task_successes += 1
                    total_successes += 1

                # Save video
                should_save = False
                if args.save_video_mode == "all":
                    should_save = True
                elif args.save_video_mode == "first_success_failure":
                    if success and not first_success_saved:
                        should_save = True
                        first_success_saved = True
                    elif not success and not first_failure_saved:
                        should_save = True
                        first_failure_saved = True

                if should_save and replay_images:
                    suffix = "success" if success else "failure"
                    task_seg = task_description.replace(" ", "_")[:50]
                    video_path = (
                        pathlib.Path(args.video_out_path)
                        / f"{suite_name}_L{task_level}_task{task_id}"
                        f"_ep{episode_idx}_{suffix}.mp4"
                    )
                    imageio.mimwrite(
                        video_path,
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                    logger.info(f"Saved video: {video_path}")

                # Running totals
                logger.info(
                    f"Running: {total_successes}/{total_episodes} "
                    f"({100*total_successes/total_episodes:.1f}%)"
                )

            # Per-task summary
            task_sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
            logger.info(
                f"[{suite_name}] task {task_id} SR: "
                f"{task_successes}/{task_episodes} = {task_sr:.4f}"
            )

            if args.use_wandb:
                import wandb
                wandb.log({
                    f"success_rate/{suite_name}/task_{task_id}": task_sr,
                })

        # Suite summary
        suite_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        avg_cost = total_costs / total_episodes if total_episodes > 0 else 0.0
        logger.info(
            f"\n[{suite_name}] Final SR: {suite_sr:.4f} "
            f"({total_successes}/{total_episodes})  avg_cost={avg_cost:.4f}"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                f"success_rate/{suite_name}": suite_sr,
                f"avg_cost/{suite_name}": avg_cost,
                f"num_episodes/{suite_name}": total_episodes,
            })

        all_results[suite_name] = {
            "success_rate": suite_sr,
            "avg_cost": avg_cost,
            "num_episodes": total_episodes,
            "num_successes": total_successes,
            "task_level": task_level,
            "task_start": t_start,
            "task_end": t_end,
        }

    if args.use_wandb:
        import wandb
        wandb.finish()

    if args.results_json:
        # One shard file per worker; the parallel harness merges them. Keys are
        # "<suite>/L<level>" (whole suite) or "<suite>/L<level>/T<start>-<end>"
        # (task slice). The task segment is essential: without it two workers
        # sharing a (suite, level) emit the same key and the aggregator, which
        # keeps the first of any duplicate, would silently discard one of them.
        out = pathlib.Path(args.results_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for suite, res in all_results.items():
            key = f"{suite}/L{res.get('task_level', args.task_level)}"
            if (args.task_start, args.task_end) != (0, -1):
                key += f"/T{res['task_start']}-{res['task_end']}"
            payload[key] = res
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote results shard -> {out}")

    return all_results


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_vla_arena)

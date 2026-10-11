#!/usr/bin/env python
"""ABC-Sim evaluation client for StarVLA policy servers (ABC-130K checkpoints trained through the m3 bridge).

Runs inside the ABC image (abc.sif: /opt/abc, ``abc_minimal.sim_env.SimTaskEnv`` over ``abc_sim``) and talks to
``deployment/model_server/server_policy.py`` over websocket. The client sends RAW observations; the server
normalises the state and un-normalises the actions with the checkpoint's own per-embodiment q01/q99
(``deployment/model_server/m3_norm_processor.py``).

Contract with training (``starVLA/dataloader/m3_data_datasets.py``, recorded in dataset_statistics.json ``m3``):
  * cameras, in this order: camera_head = sim ``top``, camera_wrist_left = sim ``left``, camera_wrist_right = sim
    ``right``;
  * images: m3 stretches every native frame to 224x224, the bridge then letterboxes it back to the native
    aspect (resize inside the square, centred on black). ``letterbox_like_training`` repeats both steps on the
    sim frame (default render 168x224 = 4:3, the D405 640x480 aspect);
  * state / action: 14-D ``[L1..L6, L_grip, R1..R6, R_grip]``, absolute joint targets, grippers as openings in
    [0, 1]; abc_sim's policy space uses the same layout;
  * 30-step chunks at 30 Hz (sim ~29.4 Hz); ABC executes 15 of every chunk, then re-plans (``--execute 15``,
    synchronous, ABC's ``--no-rtc`` loop: StarVLA has no prefix conditioning);
  * language: the training task strings (dataset ``tasks.parquet``), not ABC's ``"sim " + spec`` prompts
    (``--prompt_set train``, default). Real + sim co-trained checkpoints (abc_*_c128rs, 2026-10-07: sim episodes
    trained under "sim " + the real wording) need ``--prompt_set sim --unnorm_key abc130k_sim_yam_fov58_cmd``;
    the trained sim prompts are in the checkpoint's dataset_statistics.json (``<key>.m3.instruction_map``) and
    the client warns when the prompt it sends is not one of them.

Protocol kept from ABC's ``eval_policy.py`` / vla-eval's ABC benchmark: world i resets with ``seed + i``
(unplaceable -> from init_q -> ``seed + i + 100000``), 236 chunks x 15 actions = 3540 steps, the episode ends at
the first success; ``success`` = ever_success (ABC ``success_rate``), ``max_progress`` = max reward (ABC
``mean_max_progress``). Difference: by default every world starts from the home pose (``--carry_arm_pose`` off),
so shards are independent and a world's result does not depend on the previous world in the same process.

Output: one JSON line per episode in ``<out_dir>/episodes_<task>_shard<k>.jsonl`` (resumable), optional mp4.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Training task strings (ABC-130K tasks.parquet) for the four sim tasks the real data covers.
TRAIN_PROMPTS = {
    "put_plastic_bottles_in_bin": "put the plastic bottles in the bin",
    "throw_plastic_bottles_in_bin": "throw the plastic bottles in the bin",
    "load_plates_into_dish_rack": "load the plates into the dish rack",
    "turn_mug_right_side_up": "turn the mug right side up",
}
# ABC's sim prompt convention ("sim " + task prompt) on our real wording: what the real + sim co-trained checkpoints
# saw for their sim episodes (put bottles, dish rack, mug; throw had no sim data, its prompt follows the same rule).
SIM_PROMPTS = {task: "sim " + prompt for task, prompt in TRAIN_PROMPTS.items()}
SIM_CAMERAS = ("top", "left", "right")  # abc_sim names
TRAIN_CAMERAS = ("camera_head", "camera_wrist_left", "camera_wrist_right")  # m3 slots, same order


def _letterbox(image: Image.Image, native_hw: tuple) -> Image.Image:
    """Verbatim copy of starVLA/dataloader/m3_data_datasets.py::_letterbox (open_loop_check.py asserts equality)."""
    side_w, side_h = image.size
    h, w = native_hw
    if h == w:
        return image
    if w > h:
        new_w, new_h = side_w, max(1, round(side_h * h / w))
    else:
        new_w, new_h = max(1, round(side_w * w / h)), side_h
    canvas = Image.new(image.mode, (side_w, side_h))
    canvas.paste(image.resize((new_w, new_h), Image.Resampling.BILINEAR), ((side_w - new_w) // 2, (side_h - new_h) // 2))
    return canvas


def letterbox_like_training(frame_hwc: np.ndarray, native_hw: tuple | None = None, side: int = 224) -> np.ndarray:
    """m3's square stretch to side x side, then the bridge's letterbox at the frame's (or the given) aspect."""
    frame = np.ascontiguousarray(frame_hwc, dtype=np.uint8)
    hw = tuple(native_hw) if native_hw is not None else frame.shape[:2]
    stretched = Image.fromarray(frame).resize((side, side), Image.Resampling.BILINEAR)
    return np.asarray(_letterbox(stretched, hw))


class VideoWriter:
    """Streams an episode video to disk (no frame list in memory: a full-rate episode is up to 3540 frames).

    mode "full": one frame per simulator step (raw sim cameras top | left | right side by side, 30 fps);
    mode "query": one frame per policy query (the letterboxed policy inputs, 2 fps; the pre-2026-10-07 behaviour)."""

    def __init__(self, path: Path, mode: str, scale: float):
        import imageio

        self.mode, self.scale, self.n = mode, scale, 0
        fps = 30 if mode == "full" else 2
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=6, macro_block_size=8)

    def _put(self, frame: np.ndarray) -> None:
        if self.scale != 1.0:
            h, w = frame.shape[:2]
            frame = np.asarray(Image.fromarray(frame).resize((max(8, round(w * self.scale) // 8 * 8),
                                                               max(8, round(h * self.scale) // 8 * 8)),
                                                              Image.Resampling.BILINEAR))
        self.writer.append_data(frame)
        self.n += 1

    def sim_frame(self, cams: dict) -> None:  # {name: (3, H, W)} raw sim renders
        if self.mode == "full":
            self._put(np.concatenate([np.asarray(cams[c]).transpose(1, 2, 0) for c in SIM_CAMERAS], axis=1))

    def policy_frame(self, imgs: list) -> None:  # letterboxed policy inputs
        if self.mode == "query":
            self._put(np.concatenate(imgs, axis=1))

    def close(self) -> None:
        self.writer.close()


def rollout_over(task_eval: dict) -> bool:
    """ABC's early-stop rule (abc_minimal/eval_policy.py:rollout_over)."""
    if "ever_failed" in task_eval:
        return bool(task_eval["ever_failed"])
    return bool(task_eval.get("ever_success", False))


class Client:
    def __init__(self, host: str, port: int, unnorm_key: str, check_contract: bool = True):
        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo))
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.policy = WebsocketClientPolicy(host=host, port=port)
        meta = self.policy.get_server_metadata()
        self.meta = meta
        self.unnorm_key = unnorm_key
        if unnorm_key not in meta.get("available_unnorm_keys", []):
            raise ValueError(f"unnorm_key {unnorm_key!r} not served; server has {meta.get('available_unnorm_keys')}")
        contract = (meta.get("m3_contract") or {}).get(unnorm_key)
        if check_contract:
            if contract is None:
                raise ValueError("server advertises no m3_contract: not an m3-bridge checkpoint or an old server")
            if tuple(contract["cameras"]) != TRAIN_CAMERAS:
                raise ValueError(f"camera order {contract['cameras']} != {TRAIN_CAMERAS}")
            if meta.get("training_action_mode") != "abs":
                raise ValueError(f"training_action_mode {meta.get('training_action_mode')} != abs")
            if int(meta.get("action_chunk_size", 0)) < 15:
                raise ValueError(f"action_chunk_size {meta.get('action_chunk_size')} < 15")
            if not meta.get("expects_state"):
                logging.warning("checkpoint does not consume state; sending it anyway is harmless")
        self.contract = contract or {}
        self.letterbox = bool(self.contract.get("letterbox", True))
        nhw = self.contract.get("native_hw") or {}
        self.native_hw = {c: tuple(nhw[c]) for c in TRAIN_CAMERAS} if nhw else {}
        logging.info("server: chunk=%s expects_state=%s action_mode=%s contract=%s", meta.get("action_chunk_size"),
                     meta.get("expects_state"), meta.get("training_action_mode"), self.contract)

    def images(self, sim_images: dict) -> list:
        out = []
        for sim_cam, cam in zip(SIM_CAMERAS, TRAIN_CAMERAS):
            frame = np.asarray(sim_images[sim_cam])
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)  # abc_sim renders (3, H, W)
            if self.letterbox:
                h, w = frame.shape[:2]
                nh, nw = self.native_hw.get(cam, (h, w))
                if abs(h / w - nh / nw) > 0.01:
                    logging.warning("sim %s aspect %dx%d differs from training native %dx%d", sim_cam, h, w, nh, nw)
                out.append(letterbox_like_training(frame, native_hw=(h, w)))
            else:
                out.append(np.asarray(Image.fromarray(frame).resize((224, 224), Image.Resampling.BILINEAR)))
        return out

    def infer(self, images: list, state: np.ndarray, lang: str) -> np.ndarray:
        req = {
            "examples": [{"image": images, "lang": lang, "state": np.asarray(state, np.float32).reshape(1, -1)}],
            "do_sample": False,
            "unnorm_key": self.unnorm_key,
        }
        resp = self.policy.predict_action(req)
        if resp.get("status") == "error" or resp.get("ok") is False:
            raise RuntimeError(f"policy server error: {resp.get('error')}")
        data = resp.get("data", resp)
        return np.asarray(data["actions"][0], dtype=np.float32)


def run(args) -> None:
    from abc_minimal.sim_env import SimTaskEnv
    from abc_sim.randomization.core import RandomizationSamplingError

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(args.host, args.port, args.unnorm_key)
    tasks = [t for t in args.tasks.split(",") if t]
    episodes = list(range(args.num_episodes))[args.shard_idx :: args.num_shards]
    reset_options = {"randomization": json.loads(args.randomization)} if args.randomization else None

    for task in tasks:
        prompt = args.prompt or (SIM_PROMPTS if args.prompt_set == "sim" else TRAIN_PROMPTS).get(task)
        if prompt is None:
            raise ValueError(f"no training prompt for {task}; pass --prompt")
        trained = set((client.contract.get("instruction_map") or {}).values())
        if trained and prompt not in trained:
            logging.warning("prompt %r is not one of the prompts %s trained for unnorm_key %s",
                            prompt, sorted(trained), args.unnorm_key)
        path = out_dir / f"episodes_{task}_shard{args.shard_idx}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    done.add(int(json.loads(line)["episode"]))
                except Exception:
                    pass
        todo = [e for e in episodes if e not in done]
        if not todo:
            continue
        env = SimTaskEnv(task=task, height=args.height, width=args.width, camera_keys=SIM_CAMERAS, prompt=prompt,
                         camera_backend=args.camera_backend, gpu_id=args.render_gpu)
        logging.info("task %s: prompt %r, episodes %s", task, prompt, todo)
        for ep in todo:
            t_start = time.time()
            if not args.carry_arm_pose:
                env.forget_arm_state()
            seed = args.seed + ep
            try:
                obs = env.reset(seed=seed, options=reset_options)
            except RandomizationSamplingError:
                env.forget_arm_state()
                try:
                    obs = env.reset(seed=seed, options=reset_options)
                except RandomizationSamplingError:
                    seed = args.seed + ep + 100_000
                    obs = env.reset(seed=seed, options=reset_options)
            ev = env.evaluate()
            max_reward = float(ev.get("reward", 0.0))
            video = None
            if args.video_episodes > 0 and ep < args.video_episodes:
                try:
                    video = VideoWriter(out_dir / f"video_{task}_ep{ep:03d}.mp4", args.video_mode, args.video_scale)
                    video.sim_frame(obs["images"])
                except Exception as exc:  # pragma: no cover
                    logging.warning("video writer failed: %s", exc)
                    video = None
            # closed-loop trajectory (written to <out>/traj/<task>_ep<NNN>.npz): state before step 0 and after every
            # step, the command executed at every step, its chunk / position in chunk, every predicted chunk
            tr_state = [np.asarray(obs["state"], np.float32)]
            tr_action, tr_chunk, tr_pos, tr_reward, q_step, q_chunks = [], [], [], [], [], []
            steps = chunks = 0
            infer_s = 0.0
            first = None
            jumps = []
            over = False
            while chunks < args.num_chunks and not over:
                imgs = client.images(obs["images"])
                if video is not None:
                    video.policy_frame(imgs)
                t0 = time.time()
                chunk = client.infer(imgs, obs["state"], prompt)
                infer_s += time.time() - t0
                if chunk.ndim != 2 or chunk.shape[1] != 14 or not np.isfinite(chunk).all():
                    raise ValueError(f"bad action chunk {chunk.shape}, finite={np.isfinite(chunk).all()}")
                jumps.append(float(np.abs(chunk[0] - obs["state"]).max()))
                q_step.append(steps)
                q_chunks.append(chunk.astype(np.float32))
                if first is None:
                    first = dict(state=np.round(obs["state"], 4).tolist(), a0=np.round(chunk[0], 4).tolist(),
                                 a14=np.round(chunk[min(14, len(chunk) - 1)], 4).tolist())
                for pos, a in enumerate(chunk[: args.execute]):
                    env.step_one(a)
                    tr_action.append(np.asarray(a, np.float32))
                    tr_chunk.append(chunks)
                    tr_pos.append(pos)
                    tr_state.append(np.asarray(env.env.get_state(), np.float32))  # no render
                    if video is not None and video.mode == "full":
                        video.sim_frame(env.render_cameras())  # every sim step (30 Hz); policy obs unaffected
                    ev = env.evaluate()
                    max_reward = max(max_reward, float(ev.get("reward", 0.0)))
                    tr_reward.append(float(ev.get("reward", 0.0)))
                    steps += 1
                    if rollout_over(ev):
                        over = True
                        break
                chunks += 1
                if not over:
                    obs = env.obs()
            rec = dict(task=task, episode=ep, world_seed=seed, prompt=prompt,
                       success=bool(ev.get("ever_success", False)), final_success=bool(ev.get("success", False)),
                       reward=float(ev.get("reward", 0.0)), max_progress=max_reward, sim_steps=steps, chunks=chunks,
                       wall_s=round(time.time() - t_start, 1), infer_s=round(infer_s, 1),
                       max_first_action_jump=round(max(jumps), 4), mean_first_action_jump=round(float(np.mean(jumps)), 4),
                       first=first, final_state=np.round(env.obs()["state"], 4).tolist() if args.log_final_state else None)
            if not args.no_traj:
                tdir = out_dir / "traj"
                tdir.mkdir(exist_ok=True)
                np.savez_compressed(
                    tdir / f"{task}_ep{ep:03d}.npz", state=np.stack(tr_state), action=np.stack(tr_action),
                    chunk_idx=np.asarray(tr_chunk, np.int16), step_in_chunk=np.asarray(tr_pos, np.int8),
                    reward=np.asarray(tr_reward, np.float32), query_step=np.asarray(q_step, np.int32),
                    pred_chunks=np.stack(q_chunks), task=task, episode=ep, world_seed=seed, prompt=prompt,
                    success=rec["success"], max_progress=max_reward, execute=args.execute,
                    unnorm_key=args.unnorm_key, ckpt=str(client.meta.get("ckpt_path", "")))
            for k in ("max_bottles_in_bin_so_far", "num_bottles", "num_plates_in_rack"):
                if k in ev:
                    rec[k] = ev[k] if not isinstance(ev[k], np.generic) else ev[k].item()
            with path.open("a") as f:
                f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            logging.info("%s ep %d seed %d: success=%s max_progress=%.3f steps=%d chunks=%d wall=%.0fs infer=%.0fs",
                         task, ep, seed, rec["success"], max_reward, steps, chunks, rec["wall_s"], rec["infer_s"])
            if video is not None:
                video.close()
                logging.info("video %s: %d frames (%s)", f"video_{task}_ep{ep:03d}.mp4", video.n, video.mode)
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tasks", default=",".join(TRAIN_PROMPTS))
    ap.add_argument("--num_episodes", type=int, default=10, help="worlds per task (episode i uses seed + i)")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260511, help="ABC's default base world seed")
    ap.add_argument("--num_chunks", type=int, default=236)
    ap.add_argument("--execute", type=int, default=15, help="actions executed per 30-step chunk")
    ap.add_argument("--height", type=int, default=168)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--camera_backend", default="mjwarp", choices=["mjwarp", "mujoco"])
    ap.add_argument("--render_gpu", type=int, default=None)
    ap.add_argument("--unnorm_key", default="abc130k_yam_d405_640_cmd",
                    help="embodiment whose q01/q99 (and native aspect) the sim stands in for: D405 640x480 is 4:3 "
                         "like the sim cameras, and the largest share of the data")
    ap.add_argument("--prompt", default=None, help="override the training task string")
    ap.add_argument("--prompt_set", default="train", choices=["train", "sim"],
                    help="train: the real task strings (real-only checkpoints); sim: 'sim ' + those strings (real + "
                         "sim co-trained checkpoints, with --unnorm_key abc130k_sim_yam_fov58_cmd)")
    ap.add_argument("--randomization", default=None, help="JSON reset request for the task randomizer")
    ap.add_argument("--carry_arm_pose", action="store_true", help="ABC's world-to-world arm pose carry-over")
    ap.add_argument("--video_episodes", type=int, default=0, help="save an mp4 for episodes < N")
    ap.add_argument("--video_mode", default="full", choices=["full", "query"],
                    help="full: every sim step at 30 fps (raw sim cameras); query: one frame per policy query (2 fps)")
    ap.add_argument("--video_scale", type=float, default=1.0, help="downscale factor for video frames")
    ap.add_argument("--log_final_state", action="store_true")
    ap.add_argument("--no_traj", action="store_true", help="skip the per-episode trajectory npz (<out>/traj/)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(args)


if __name__ == "__main__":
    main()

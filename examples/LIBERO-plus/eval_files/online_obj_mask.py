"""Online task-object masks for closed-loop LIBERO / LIBERO-Plus rollouts (default off).

Enabled in eval_libero_model.py by ``STARVLA_ONLINE_OBJ_MASK=1`` (inside the SIF: export
``APPTAINERENV_STARVLA_ONLINE_OBJ_MASK=1`` on the host, since the launcher runs apptainer with
``--cleanenv``). Then:

* the eval env is built as ``OffScreenRenderEnv(..., camera_segmentations="instance")``. Not
  ``SegmentationRenderEnv``: its ``step``/``reset`` bypass LIBERO-Plus's ``ControlEnv.step``,
  which applies the Sensor Noise corruptions, so it would change the benchmark;
* after every ``env.reset()`` the instance-id map is rebuilt exactly as
  ``SegmentationRenderEnv.reset`` does, and the BDDL ``:obj_of_interest`` entries are resolved
  to instance ids as papers/BiBrain_ICLR_2027/analysis/capture_masks.py does;
* robosuite's own instance sensor maps every pixel through a Python dict (~20 ms per camera
  per step at 256x256, a 4x slowdown of env.step). ``reset`` therefore swaps in a lookup-table
  sensor (checked equal to robosuite's on the first frame of every episode) and disables the
  segmentation observables; the eval loop calls ``arm(True)`` only before the ``env.step`` whose
  observation feeds a policy query. The mask is then sampled by robosuite at the same simulator
  substep as the RGB frame (observables sample at the first substep of a control step, so a
  render after ``env.step`` would see a slightly later state);
* at every policy query the request carries ``_obj_tok``: float16 ``[2, 8, 8]`` per-token object
  area fractions for (agentview, wrist), in the orientation and resolution the policy sees
  (``[::-1, ::-1]`` flip, INTER_AREA to 224, mean over 28x28 blocks = merged image tokens,
  row-major). The policy server ignores the key unless an object-bias hook consumes it.

``STARVLA_OBJ_MASK_DUMP_DIR`` additionally writes PNG overlays (policy image + mask + token
grid) for the first ``STARVLA_OBJ_MASK_DUMP_MAX`` (default 6) queries of each episode, every
``STARVLA_OBJ_MASK_DUMP_EVERY`` (default 16) steps.

Pure numpy / cv2: importable in the SIF client and on the host.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GRID = 8
SEG_KEYS = ("agentview_segmentation_instance", "robot0_eye_in_hand_segmentation_instance")


def enabled() -> bool:
    return os.environ.get("STARVLA_ONLINE_OBJ_MASK", "0") == "1"


def token_fractions(mask224: np.ndarray, grid: int = GRID) -> np.ndarray:
    """[224, 224] mask -> [grid, grid] per-token area fractions (capture_masks.to_tokens)."""
    m = np.asarray(mask224, dtype=np.float32)
    c = m.shape[0] // grid
    return m[: grid * c, : grid * c].reshape(grid, c, grid, c).mean(axis=(1, 3))


def libero_instance_map(inner_env) -> Tuple[Dict[str, int], Optional[int]]:
    """Instance name -> segmentation value, as SegmentationRenderEnv.reset builds it
    (value = position in ``model.instances_to_ids`` + 1; robot parts excluded)."""
    robot_id, mapping = None, {}
    for i, name in enumerate(list(inner_env.model.instances_to_ids.keys())):
        if name == "Panda0":
            robot_id = i
        if name not in ("Panda0", "RethinkMount0", "PandaGripper0"):
            mapping[name] = i + 1
    return mapping, robot_id


def interest_ids(instance_to_id: Dict[str, int], obj_of_interest: Sequence[str]) -> Tuple[List[int], Dict[str, Optional[str]]]:
    """BDDL objects of interest -> instance ids (capture_masks.interest_mask). A region entry
    (``wooden_cabinet_1_middle_region``) maps to the longest instance name that prefixes it."""
    names = sorted(instance_to_id, key=len, reverse=True)
    ids, resolved = [], {}
    for obj in obj_of_interest:
        inst = next((n for n in names if obj == n or obj.startswith(n + "_")), None)
        resolved[obj] = inst
        if inst is not None:
            ids.append(int(instance_to_id[inst]))
    return sorted(set(ids)), resolved


def seg_to_token_fractions(seg_raw: np.ndarray, ids: Sequence[int], size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """Raw robosuite segmentation obs ([H, W, 1] int, un-flipped) -> (224 mask, 8x8 fractions),
    flipped [::-1, ::-1] exactly like the policy image in eval_libero_model.py."""
    import cv2

    seg = np.asarray(seg_raw).squeeze(-1)[::-1, ::-1]
    m = np.isin(seg, list(ids)).astype(np.float32)
    m224 = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    return m224, token_fractions(m224)


class OnlineObjectMask:
    """Per-episode state: resolved instance ids; per-step token fractions for both cameras."""

    def __init__(self):
        self.ids: List[int] = []
        self.resolved: Dict[str, Optional[str]] = {}
        self.dump_dir = os.environ.get("STARVLA_OBJ_MASK_DUMP_DIR", "").strip()
        self.dump_every = max(1, int(os.environ.get("STARVLA_OBJ_MASK_DUMP_EVERY", "16")))
        self.dump_max = int(os.environ.get("STARVLA_OBJ_MASK_DUMP_MAX", "6"))
        self._dumped = 0
        self.tag = ""
        self.seconds = 0.0
        self.calls = 0

    def reset(self, env, tag: str = "", fast: bool = True) -> Dict[str, Optional[str]]:
        mapping, _ = libero_instance_map(env.env)
        self.ids, self.resolved = interest_ids(mapping, list(getattr(env, "obj_of_interest", [])))
        if not self.ids:
            raise RuntimeError(f"online object mask: no object of interest resolved ({self.resolved})")
        self._dumped = 0
        self.tag = tag
        self._obs = {k: env.env._observables[k] for k in SEG_KEYS}
        self._rgb = {k: env.env._observables[k.replace("_segmentation_instance", "_image")] for k in SEG_KEYS}
        if fast:
            self._install_fast()
            self.arm(False)
        return self.resolved

    def arm(self, on: bool) -> None:
        """Enable the segmentation observables for the next env.step only when a query follows."""
        for key, o in self._obs.items():
            if bool(on) == o.is_enabled():
                continue
            o.set_enabled(bool(on))  # robosuite: also resets the observable's sampling clock
            if on:
                # Copy the camera's RGB clock so the mask is sampled at exactly the simulator substep
                # the RGB frame is (a reset clock would skip this control step and return zeros).
                rgb = self._rgb[key]
                o._time_since_last_sample = rgb._time_since_last_sample
                o._sampled = rgb._sampled
                o._current_delay = rgb._current_delay

    def _install_fast(self):
        """Vectorised copy of robosuite 1.4's instance sensor (same render call, dict -> LUT)."""
        import inspect

        from robosuite.utils.observables import sensor

        for key, obs in self._obs.items():
            nl = inspect.getclosurevars(obs._sensor).nonlocals
            mapping, conv = nl["mapping"], nl["convention"]
            lut = np.full(max(max(mapping), 0) + 2, -1, dtype=np.int32)  # last slot: unmapped / background
            for g, inst in mapping.items():
                lut[g] = inst
            cam, w, h, sim = nl["cam_name"], nl["cam_w"], nl["cam_h"], nl["self"].sim

            @sensor(modality=obs.modality)
            def fast(obs_cache, cam=cam, w=w, h=h, conv=conv, lut=lut, sim=sim):
                raw = sim.render(camera_name=cam, width=w, height=h, depth=False, segmentation=True)[::conv, :, 1]
                idx = np.where((raw >= 0) & (raw < len(lut) - 1), raw, len(lut) - 1)  # unmapped -> -1, as .get(x, -1)
                return (lut[idx] + 1)[..., None].astype(np.int32)

            ref = np.asarray(obs._sensor({}))
            got = fast({})
            if ref.shape != got.shape or not np.array_equal(ref, got):
                raise RuntimeError(f"online object mask: fast {key} sensor differs from robosuite's")
            obs.set_sensor(fast)

    def __call__(self, obs, step: int = 0, images: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
        import time

        t0 = time.perf_counter()
        masks, fracs = [], []
        for key in SEG_KEYS:
            if key not in obs:
                raise RuntimeError(f"online object mask: {key} missing from obs (arm(True) before the env.step "
                                   "that feeds this query)")
            m224, f = seg_to_token_fractions(obs[key], self.ids)
            masks.append(m224)
            fracs.append(f)
        out = np.stack(fracs).astype(np.float16)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        if self.dump_dir and images is not None and self._dumped < self.dump_max and step % self.dump_every == 0:
            self._dump(images, masks, fracs, step)
            self._dumped += 1
        return out

    def _dump(self, images, masks, fracs, step):
        import cv2

        os.makedirs(self.dump_dir, exist_ok=True)
        panels = []
        for img, m, f in zip(images, masks, fracs):
            im = cv2.resize(np.asarray(img), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
            over = im.copy()
            over[..., 0] = np.clip(over[..., 0] * (1 - 0.6 * m) + 255 * 0.6 * m, 0, 255)  # red tint on object pixels
            over[..., 1:] *= (1 - 0.4 * m)[..., None]
            for k in range(1, GRID):
                over[k * 28, :, :] = 255
                over[:, k * 28, :] = 255
            for r in range(GRID):  # token cells with object fraction >= 0.25 get a yellow corner
                for c in range(GRID):
                    if f[r, c] >= 0.25:
                        over[r * 28 + 1:r * 28 + 6, c * 28 + 1:c * 28 + 6] = (255, 255, 0)
            panels.append(np.concatenate([im, over], axis=1))
        png = np.concatenate(panels, axis=0).astype(np.uint8)
        path = os.path.join(self.dump_dir, f"{self.tag}_step{step:04d}.png")
        cv2.imwrite(path, png[..., ::-1])
        np.savez_compressed(path[:-4] + ".npz", obj_tok=np.stack(fracs).astype(np.float16),
                            obj_px=np.stack(masks).astype(np.float16),
                            image_0=np.asarray(images[0]), image_1=np.asarray(images[1]))

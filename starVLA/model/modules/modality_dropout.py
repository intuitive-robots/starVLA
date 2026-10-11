"""Image dropout at the encoder INPUT, on a per-row schedule.

What this is for
----------------
LIT, APT, LA4VLA and the action-prior line all force one thing: a pathway that carries
action information with **no image in the input**, so the dense visual channel cannot
shortcut the sparse language/state one.  All four implement it as a separate first stage,
because in a decoder-only VLA "no images" is a different prompt -- different length,
different positions, different cache -- and that needs its own checkpoint and LR schedule.

Here it is a per-row property of an ordinary batch, so the curriculum can instead be a
*rate that decays over training*: every row blind for the first few thousand steps, then a
shrinking fraction, then none.  One run, one checkpoint, one schedule, and the transition
is continuous rather than a cliff between two stages.

This is NOT the dropout that already exists
-------------------------------------------
``shared_z.memory_dropout_mode: camera`` (shared_z.py) masks image-token columns out of
the memory the action head cross-attends to.  That happens AFTER the bidirectional encoder
has run, so the language tokens on that row already mixed with vision and still carry it.
The row is not vision-free; only the head's direct route is cut.  This module removes the
image before the encoder sees it, which is the thing the four papers actually share.

Two removal mechanisms, because they are not interchangeable
------------------------------------------------------------
``removal: pixels``
    ``pixel_values`` is zeroed for the selected rows after the processor has run -- the
    same mechanism, and the same argument, as the blind causal arm in QWen3_Blind.py.
    Patch count, token count, sequence length, attention mask, mrope positions and FLOPs
    stay bit-identical to the sighted rows; only the information in the pixels is gone.
    A blind row therefore costs exactly as much as a sighted one: no throughput win.
    This is the mechanism for a *mixed* batch, where rows at different rates sit side by
    side and any length change would confound the comparison between them.

``removal: tokens``
    The images are never handed to the processor, so the row has no image tokens at all.
    The vision tower does not run and the sequence is far shorter, which is what buys the
    larger batch; but the instruction then sits at a different offset than it does in a
    sighted row, so this belongs in a *uniform* phase (every row blind), not mixed into a
    batch with sighted rows.

Zeroed pixels stay a constant, never noise: noise injects entropy the encoder could latch
onto as a spurious per-step signal, whereas a constant is unambiguously information-free.

Config (``framework.modality_dropout``)
---------------------------------------
    enabled:      false
    removal:      pixels | tokens
    vision_rate:  0.0     # scalar, or a schedule:
                  #   {start: 1.0, end: 0.0, start_step: 5000, end_step: 25000}
                  # held at `start` through start_step, linear to `end` by end_step,
                  # `end` thereafter. Same keys as trainer.loss_scale ramps.
    mode:         all     # 'all' = every camera on the row; 'camera' = one random camera
    inference_vision_rate: 0.0
                  # 0 keeps sight at rollout (the usual case); 1 is the permanently blind
                  # arm, i.e. the state-only / language-only ceiling that makes every
                  # other arm readable.
    protect_state: true   # never drop state AND images on the same row
"""

from typing import List, Optional, Sequence

import numpy as np

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def _resolve_scheduled_rate(value, step: int, name: str) -> float:
    """A scalar, or a {start,end,start_step,end_step|ramp_steps} ramp at `step`.

    Deliberately the same shape as Trainer._scheduled_loss_scale, so a schedule reads the
    same wherever it appears in a config.
    """
    if not hasattr(value, "get"):
        return float(value)
    start = float(value.get("start", 0.0))
    end = float(value.get("end", start))
    start_step = int(value.get("start_step", 0))
    ramp_steps = int(value.get("ramp_steps", 0))
    end_step = int(value.get("end_step", start_step + ramp_steps))
    if start_step < 0:
        raise ValueError(f"{name}.start_step must be nonnegative")
    if end_step < start_step:
        raise ValueError(f"{name}.end_step must be >= start_step")
    if step <= start_step:
        return start
    if end_step == start_step or step >= end_step:
        return end
    progress = (step - start_step) / (end_step - start_step)
    return start + (end - start) * progress


class ModalityDropoutMixin:
    """Config parsing, row selection, and the two removal mechanisms."""

    def _configure_modality_dropout(self, config) -> None:
        cfg = dict(config.framework.get("modality_dropout", {}) or {})
        self.modality_dropout_enabled = bool(cfg.get("enabled", False))
        self.modality_vision_rate = cfg.get("vision_rate", 0.0)
        self.modality_inference_vision_rate = float(cfg.get("inference_vision_rate", 0.0))
        self.modality_dropout_mode = str(cfg.get("mode", "all")).lower()
        self.modality_removal = str(cfg.get("removal", "pixels")).lower()
        self.modality_protect_state = bool(cfg.get("protect_state", True))
        # Published by the trainer each optimizer step; 0 until then, which is the right
        # value for a schedule that starts blind.
        self._modality_step = 0
        self._last_vision_keep_rate = 1.0

        if self.modality_removal not in {"pixels", "tokens"}:
            raise ValueError(
                "framework.modality_dropout.removal must be 'pixels' or 'tokens', "
                f"got {self.modality_removal!r}"
            )
        if self.modality_dropout_mode not in {"all", "camera", "independent"}:
            raise ValueError(
                "framework.modality_dropout.mode must be 'all' or 'camera', "
                f"got {self.modality_dropout_mode!r}"
            )
        if self.modality_removal == "tokens" and self.modality_dropout_mode in ("camera", "independent"):
            # Dropping one camera by removing its tokens makes rows in the same batch
            # differ in length AND in which camera is at which offset. Nothing reads that
            # correctly downstream.
            raise ValueError(
                "framework.modality_dropout: removal='tokens' cannot be combined with "
                "mode='camera'; use removal='pixels' for per-camera dropout"
            )
        # Both endpoints of a schedule have to be legal rates, not just the scalar case.
        for probe_step in (0, 10**9):
            rate = _resolve_scheduled_rate(
                self.modality_vision_rate, probe_step, "modality_dropout.vision_rate"
            )
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"framework.modality_dropout.vision_rate resolves to {rate} at step "
                    f"{probe_step}; every point of the schedule must be in [0,1]"
                )
        if not 0.0 <= self.modality_inference_vision_rate <= 1.0:
            raise ValueError(
                "framework.modality_dropout.inference_vision_rate must be in [0,1], "
                f"got {self.modality_inference_vision_rate}"
            )
        if not self.modality_dropout_enabled:
            max_rate = max(
                _resolve_scheduled_rate(self.modality_vision_rate, s, "x")
                for s in (0, 10**9)
            )
            if max_rate > 0.0 or self.modality_inference_vision_rate > 0.0:
                # A rate on a disabled block is always a silent no-op arm, which is the one
                # failure mode that produces a clean-looking duplicate of the control.
                raise ValueError(
                    "framework.modality_dropout has a non-zero rate but enabled=false; "
                    "this arm would silently be the control"
                )
        else:
            logger.info(
                "modality dropout ON: removal=%s vision_rate=%s mode=%s "
                "inference_vision_rate=%.3f protect_state=%s",
                self.modality_removal,
                self.modality_vision_rate,
                self.modality_dropout_mode,
                self.modality_inference_vision_rate,
                self.modality_protect_state,
            )

    # ── row selection ────────────────────────────────────────────────────────
    def _modality_current_vision_rate(self) -> float:
        if not getattr(self, "modality_dropout_enabled", False):
            return 0.0
        if not self.training:
            return self.modality_inference_vision_rate
        return _resolve_scheduled_rate(
            self.modality_vision_rate,
            int(getattr(self, "_modality_step", 0)),
            "modality_dropout.vision_rate",
        )

    def _modality_vision_keep(
        self,
        batch_size: int,
        state_keep: Optional[Sequence[bool]] = None,
    ) -> np.ndarray:
        """Per-row keep flags, drawn once per batch at the schedule's current rate.

        ``state_keep`` is the decision ``_add_state_conditioning`` already took. With
        ``protect_state`` a row that lost its state keeps its images, so no row is asked to
        predict an action from the instruction alone unless that is the configured arm.
        """
        keep = np.ones(batch_size, dtype=bool)
        rate = self._modality_current_vision_rate()
        if rate <= 0.0:
            return keep
        if self.modality_dropout_mode == "independent":
            # every row is a candidate; each camera is dropped with `rate` in _modality_zero_pixels
            keep = np.zeros(batch_size, dtype=bool)
        else:
            keep = np.random.random(batch_size) >= rate
        if self.modality_protect_state and state_keep is not None:
            if len(state_keep) != batch_size:
                raise ValueError(
                    f"state_keep length {len(state_keep)} != batch {batch_size}"
                )
            keep = keep | ~np.asarray(state_keep, dtype=bool)
        return keep

    # ── removal: tokens (before the processor) ───────────────────────────────
    def _modality_strip_images(
        self, batch_images: List, vision_keep: np.ndarray
    ) -> List:
        """Return the image lists with the blinded rows emptied.

        The processor never sees those images, so the row carries no image tokens and the
        vision tower does not run for it.
        """
        return [
            images if keep else []
            for images, keep in zip(batch_images, vision_keep)
        ]

    # ── removal: pixels (after the processor) ────────────────────────────────
    @staticmethod
    def _pixel_patch_offsets(qwen_inputs: dict, images_per_row: List[int]):
        """``pixel_values`` and the per-image patch offsets into it (checked packing)."""
        pixel_values = qwen_inputs.get("pixel_values")
        grid_thw = qwen_inputs.get("image_grid_thw")
        if pixel_values is None or grid_thw is None:
            raise RuntimeError(
                "image zeroing was requested but the batch carries no "
                "pixel_values/image_grid_thw"
            )
        if int(grid_thw.shape[0]) != int(sum(images_per_row)):
            raise RuntimeError(
                f"image count mismatch: processor returned {int(grid_thw.shape[0])} grids "
                f"for {int(sum(images_per_row))} images"
            )
        patches = grid_thw.prod(dim=-1).tolist()
        if int(sum(patches)) != int(pixel_values.shape[0]):
            raise RuntimeError(
                "pixel_values is not packed one row per patch "
                f"({pixel_values.shape[0]} rows for {int(sum(patches))} patches)"
            )
        return pixel_values, np.concatenate(([0], np.cumsum(patches))).astype(int)

    def _zero_image_positions(
        self,
        qwen_inputs: dict,
        images_per_row: List[int],
        positions: Sequence[int],
    ) -> None:
        """Deterministically zero the images at ``positions`` (within-row camera index)
        in EVERY row, by the same mechanism as ``removal: pixels`` training dropout.

        Eval-time camera removal (``STARVLA_ZERO_IMAGE_INDICES``, policy_wrapper.py): a
        two-camera policy is shown the removed camera exactly as a camdrop-trained row saw
        it (zeroed normalised pixels, i.e. a uniform mid-grey image, with the image-token
        count, positions and prompt layout unchanged).
        """
        pixel_values, offsets = self._pixel_patch_offsets(qwen_inputs, images_per_row)
        cursor = 0
        for row, n_images in enumerate(images_per_row):
            for pos in positions:
                if not 0 <= int(pos) < n_images:
                    raise ValueError(
                        f"row {row}: cannot zero image position {pos}, row has {n_images} image(s)"
                    )
                index = cursor + int(pos)
                pixel_values[offsets[index] : offsets[index + 1]] = 0
            cursor += n_images

    def _modality_zero_pixels(
        self,
        qwen_inputs: dict,
        images_per_row: List[int],
        vision_keep: np.ndarray,
        n_candidates: Optional[int] = None,
        mirror: int = 0,
    ) -> List[List[int]]:
        """Zero ``pixel_values`` for the blinded rows, leaving every shape untouched.

        Returns, per row, the within-row positions of the images that were zeroed ([] for
        kept rows). Callers that ignore the return value behave exactly as before; the
        random draws are unchanged (one ``np.random.randint`` per blinded row in camera
        mode).
        """
        pixel_values = qwen_inputs.get("pixel_values")
        grid_thw = qwen_inputs.get("image_grid_thw")
        if pixel_values is None or grid_thw is None:
            raise RuntimeError(
                "modality dropout was asked to blind rows but the batch carries no "
                "pixel_values/image_grid_thw"
            )
        if int(grid_thw.shape[0]) != int(sum(images_per_row)):
            raise RuntimeError(
                f"image count mismatch: processor returned {int(grid_thw.shape[0])} grids "
                f"for {int(sum(images_per_row))} images"
            )
        patches = grid_thw.prod(dim=-1).tolist()      # patches contributed by each image
        if int(sum(patches)) != int(pixel_values.shape[0]):
            raise RuntimeError(
                "pixel_values is not packed one row per patch "
                f"({pixel_values.shape[0]} rows for {int(sum(patches))} patches); the "
                "row->image->patch mapping this dropout relies on does not hold"
            )
        offsets = np.concatenate(([0], np.cumsum(patches))).astype(int)

        image_cursor = 0
        dropped: List[List[int]] = []
        for row, n_images in enumerate(images_per_row):
            row_images = list(range(image_cursor, image_cursor + n_images))
            row_start = image_cursor
            image_cursor += n_images
            if vision_keep[row]:
                dropped.append([])
                continue
            if not row_images:
                raise RuntimeError(f"row {row} was selected for blinding but has no image")
            # future_copies: only the first n_candidates images are inputs; each chosen one also grays
            # its copy `mirror` positions later (the copies are not themselves dropout candidates)
            if n_candidates is not None:
                row_images = row_images[:int(n_candidates)]
            if self.modality_dropout_mode == "camera":
                chosen = [row_images[int(np.random.randint(len(row_images)))]]
            elif self.modality_dropout_mode == "independent":
                # Flex-pi style: each camera dropped independently, at least one camera kept
                rate = self._modality_current_vision_rate()
                chosen = [i for i in row_images if np.random.random() < rate]
                if len(chosen) == len(row_images):
                    chosen.pop(int(np.random.randint(len(chosen))))
            else:
                chosen = row_images
            for image_index in chosen:
                pixel_values[offsets[image_index] : offsets[image_index + 1]] = 0
                if mirror:
                    pixel_values[offsets[image_index + mirror] : offsets[image_index + mirror + 1]] = 0
            dropped.append([index - row_start for index in chosen])
        return dropped

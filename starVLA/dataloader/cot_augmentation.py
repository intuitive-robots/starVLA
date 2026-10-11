"""Joint image/text augmentation for coordinate-bearing CoT supervision.

The CoT mapping stores coordinates normalized to [0, 1000] in the uncropped training
frame. A geometric image transform must therefore rewrite the assistant target with the
same sampled affine transform. This module is intentionally called after CoT resolution,
where images and target text coexist for the first time.
"""

from __future__ import annotations

import re
import math
from typing import Iterable

import torch
import numpy as np
from PIL import Image
from pydantic import Field
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from starVLA.dataloader.gr00t_lerobot.transform.base import ModalityTransform

COORD = 1000
_COORD_BLOCK = re.compile(
    r"<(?P<tag>box|point|trajectory)(?P<attrs>\s+[^>]*)?>(?P<body>.*?)</(?P=tag)>",
    re.DOTALL,
)
_COORD_TUPLE = re.compile(r"\((\d+),(\d+)(?:,(\d+),(\d+))?\)")
_TRAJECTORY3D_BLOCK = re.compile(
    r"<trajectory3d(?P<attrs>\s+[^>]*)?>(?P<body>.*?)</trajectory3d>", re.DOTALL
)
_TRAJECTORY3D_TUPLE = re.compile(r"\((-?\d+),(-?\d+),(-?\d+)\)")


def _rewrite_coordinate_text(
    text: str,
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    image_width: int,
    image_height: int,
    angle_degrees: float = 0.0,
) -> str:
    """Rewrite normalized coordinates after crop, resize, and centered rotation."""

    angle = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    center_x, center_y = image_width / 2.0, image_height / 2.0

    def transform_xy_raw(x: int, y: int) -> tuple[float, float]:
        px = x * image_width / COORD
        py = y * image_height / COORD
        # Crop then resize back to the original resolution.
        px = image_width * (px - left) / crop_width
        py = image_height * (py - top) / crop_height
        # torchvision/PIL positive angles are counter-clockwise in visual coordinates
        # (whose y axis points down).
        dx, dy = px - center_x, py - center_y
        rx = cos_a * dx + sin_a * dy + center_x
        ry = -sin_a * dx + cos_a * dy + center_y
        return COORD * rx / image_width, COORD * ry / image_height

    def quantize(x: float, y: float) -> tuple[int, int]:
        nx, ny = round(x), round(y)
        # A 0.95 crop can cut a few boundary pixels. Clipping keeps the target valid and
        # represents the nearest visible boundary.
        nx = min(COORD, max(0, nx))
        ny = min(COORD, max(0, ny))
        return nx, ny

    def transform_block(match: re.Match) -> str:
        tag = match.group("tag")
        attrs = match.group("attrs") or ""
        body = match.group("body")

        def transform_tuple(coords: re.Match) -> str:
            x1, y1 = int(coords.group(1)), int(coords.group(2))
            if coords.group(3) is None:
                x, y = quantize(*transform_xy_raw(x1, y1))
                return f"({x},{y})"

            x2, y2 = int(coords.group(3)), int(coords.group(4))
            if tag == "box":
                # Rotation turns the rectangle into a quadrilateral. Emit the tight
                # axis-aligned box around all four transformed corners.
                corners = [
                    transform_xy_raw(x1, y1), transform_xy_raw(x2, y1),
                    transform_xy_raw(x1, y2), transform_xy_raw(x2, y2),
                ]
                bx1, by1 = quantize(min(p[0] for p in corners), min(p[1] for p in corners))
                bx2, by2 = quantize(max(p[0] for p in corners), max(p[1] for p in corners))
                return f"({bx1},{by1},{bx2},{by2})"
            first = quantize(*transform_xy_raw(x1, y1))
            second = quantize(*transform_xy_raw(x2, y2))
            return f"({first[0]},{first[1]},{second[0]},{second[1]})"

        body = _COORD_TUPLE.sub(transform_tuple, body)
        return f"<{tag}{attrs}>{body}</{tag}>"

    text = _COORD_BLOCK.sub(transform_block, text)

    # A <trajectory3d> is measured in the frozen current camera frame, so crop / resize do
    # not change its metric coordinates. An in-plane image rotation does rotate the x/y
    # camera axes, however. Apply exactly the same visual-coordinate convention used above;
    # z (camera-forward) is unchanged. Signed values are intentional.
    def transform_trajectory3d(match: re.Match) -> str:
        attrs, body = match.group("attrs") or "", match.group("body")

        def transform_tuple(coords: re.Match) -> str:
            x, y, z = (int(coords.group(i)) for i in range(1, 4))
            rx = round(cos_a * x + sin_a * y)
            ry = round(-sin_a * x + cos_a * y)
            return f"({rx},{ry},{z})"

        return (f"<trajectory3d{attrs}>" +
                _TRAJECTORY3D_TUPLE.sub(transform_tuple, body) +
                "</trajectory3d>")

    return _TRAJECTORY3D_BLOCK.sub(transform_trajectory3d, text)


def _apply_color_jitter(images: list[Image.Image]) -> list[Image.Image]:
    """Apply one sampled photometric transform consistently across all camera views."""

    order, brightness, contrast, saturation, hue = ColorJitter.get_params(
        [0.7, 1.3], [0.6, 1.4], [0.5, 1.5], None
    )
    out = []
    for image in images:
        aug = image
        for fn_id in order:
            i = int(fn_id)
            if i == 0 and brightness is not None:
                aug = TF.adjust_brightness(aug, brightness)
            elif i == 1 and contrast is not None:
                aug = TF.adjust_contrast(aug, contrast)
            elif i == 2 and saturation is not None:
                aug = TF.adjust_saturation(aug, saturation)
            elif i == 3 and hue is not None:
                aug = TF.adjust_hue(aug, hue)
        out.append(aug)
    return out


def augment_cot_sample(
    images: Iterable[Image.Image],
    conversation: list[dict] | None,
    *,
    mode: str,
    crop_scale: float = 0.95,
    crop: tuple[int, int, int, int] | None = None,
    angle_degrees: float | None = None,
) -> tuple[list[Image.Image], list[dict] | None]:
    """Jointly augment one sample's camera images and CoT assistant target.

    ``crop`` is ``(top, left, height, width)`` and exists for deterministic tests and label
    visualization. Production callers omit it and use the process-local PyTorch RNG.
    """

    mode = str(mode).lower()
    if mode not in {"photometric", "crop_photometric"}:
        raise ValueError(f"unsupported joint CoT augmentation mode: {mode!r}")
    out_images = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in images]
    if not out_images:
        return out_images, conversation

    out_conversation = conversation
    if mode == "crop_photometric":
        width, height = out_images[0].size
        if any(im.size != (width, height) for im in out_images):
            raise ValueError("joint CoT crop requires all camera views to share a resolution")
        crop_height = int(height * crop_scale)
        crop_width = int(width * crop_scale)
        if crop is None:
            top = int(torch.randint(0, height - crop_height + 1, ()).item())
            left = int(torch.randint(0, width - crop_width + 1, ()).item())
        else:
            top, left, crop_height, crop_width = crop
        if not (0 <= top <= height - crop_height and 0 <= left <= width - crop_width):
            raise ValueError(
                f"invalid crop {(top, left, crop_height, crop_width)} for {(height, width)}")

        if angle_degrees is None:
            angle_degrees = float(torch.empty(()).uniform_(-5.0, 5.0).item())
        out_images = [
            TF.rotate(
                TF.resize(
                    TF.crop(im, top, left, crop_height, crop_width),
                    [height, width],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                angle_degrees,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            for im in out_images
        ]
        if conversation is not None:
            out_conversation = [dict(turn) for turn in conversation]
            for turn in out_conversation:
                if turn.get("from") == "gpt":
                    turn["value"] = _rewrite_coordinate_text(
                        turn["value"],
                        left=left,
                        top=top,
                        crop_width=crop_width,
                        crop_height=crop_height,
                        image_width=width,
                        image_height=height,
                        angle_degrees=angle_degrees,
                    )

    return _apply_color_jitter(out_images), out_conversation


def augment_cot_batch(
    batch_images: list[list[Image.Image]],
    conversations: list[list[dict] | None],
    *,
    mode: str,
) -> tuple[list[list[Image.Image]], list[list[dict] | None]]:
    if len(batch_images) != len(conversations):
        raise ValueError("batch image/conversation lengths differ")
    out_images, out_conversations = [], []
    for images, conversation in zip(batch_images, conversations):
        aug_images, aug_conversation = augment_cot_sample(
            images, conversation, mode=mode)
        out_images.append(aug_images)
        out_conversations.append(aug_conversation)
    return out_images, out_conversations


class CoTVideoAugment(ModalityTransform):
    """Worker-side joint transform over video arrays and ``_cot_conversation``."""

    mode: str = Field(..., description="photometric or crop_photometric")
    crop_scale: float = Field(default=0.95, gt=0.0, le=1.0)

    def apply(self, data: dict) -> dict:
        if not self.training:
            return data
        keys = [key for key in self.apply_to if key in data]
        if not keys:
            return data

        shapes, counts, frames = {}, {}, []
        for key in keys:
            value = np.asarray(data[key])
            if value.ndim != 4 or value.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"{key} must be [T,H,W,C] before CoT augmentation, got {value.shape}")
            shapes[key] = value.shape
            counts[key] = value.shape[0]
            frames.extend(Image.fromarray(frame) for frame in value)

        aug_frames, conversation = augment_cot_sample(
            frames,
            data.get("_cot_conversation"),
            mode=self.mode,
            crop_scale=self.crop_scale,
        )
        offset = 0
        for key in keys:
            n = counts[key]
            dtype = np.asarray(data[key]).dtype
            data[key] = np.stack(
                [np.asarray(frame, dtype=dtype) for frame in aug_frames[offset:offset + n]], 0)
            assert data[key].shape == shapes[key], (data[key].shape, shapes[key])
            offset += n
        data["_cot_conversation"] = conversation
        return data


def _rotate_points(px, py, *, left, top, crop_width, crop_height, width, height, angle_degrees):
    """Pixel coordinates through crop -> resize -> centred rotation (same convention as
    _rewrite_coordinate_text)."""
    a = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(a), math.sin(a)
    px = width * (px - left) / crop_width
    py = height * (py - top) / crop_height
    dx, dy = px - width / 2.0, py - height / 2.0
    return cos_a * dx + sin_a * dy + width / 2.0, -sin_a * dx + cos_a * dy + height / 2.0


def augment_label_mask(mask: np.ndarray, *, top, left, crop_height, crop_width, height, width,
                       angle_degrees) -> np.ndarray:
    """Apply the image geometry to a [h, w] uint8 label map stored at a lower resolution (h divides height).
    Labels are upsampled by replication, cropped / resized / rotated with nearest-neighbour (no blending,
    no colour jitter, fill 0 = rest), then pooled back with max over each block (labels are priority-ordered,
    so a small object keeps its cells)."""
    h, w = mask.shape
    fy, fx = height // h, width // w
    big = Image.fromarray(np.kron(mask, np.ones((fy, fx), np.uint8)))
    big = TF.rotate(
        TF.resize(TF.crop(big, top, left, crop_height, crop_width), [height, width],
                  interpolation=TF.InterpolationMode.NEAREST),
        angle_degrees, interpolation=TF.InterpolationMode.NEAREST, fill=0)
    return np.asarray(big, np.uint8).reshape(h, fy, w, fx).max(axis=(1, 3))


def augment_boxes(boxes: np.ndarray, **geom) -> np.ndarray:
    """[..., 4] xyxy in [0, 1] (-1 = absent) through the image geometry: axis-aligned box of the four moved
    corners, clipped to [0, 1]; boxes that leave the frame entirely become -1."""
    out = np.array(boxes, np.float32, copy=True)
    flat = out.reshape(-1, 4)
    W, H = geom["width"], geom["height"]
    for i, b in enumerate(flat):
        if b[0] < 0:
            continue
        xs = np.array([b[0], b[2], b[0], b[2]]) * W
        ys = np.array([b[1], b[1], b[3], b[3]]) * H
        rx, ry = _rotate_points(xs, ys, **geom)
        nb = np.array([rx.min() / W, ry.min() / H, rx.max() / W, ry.max() / H])
        nb = np.clip(nb, 0.0, 1.0)
        flat[i] = nb if (nb[2] > nb[0] and nb[3] > nb[1]) else -1.0
    return out


class AnnotVideoAugment(ModalityTransform):
    """Worker-side joint transform for runs with simulator annotations (2026-10-03).

    Same recipe and parameters as the LIBERO ``crop_photometric`` augmentation (0.95 crop, resize back,
    rotation in [-5, 5] deg, brightness / contrast / saturation jitter 0.3 / 0.4 / 0.5 shared across views),
    with ONE sampled crop and angle applied to every camera image and to ``data["_annot"]``: its label masks
    (``mask`` [n_cams, h, w], nearest-neighbour, no jitter) and 2D boxes (``box2d`` [n_cams, slots, 4]).
    3D boxes / poses are in the robot frame and stay as they are."""

    mode: str = Field(default="crop_photometric", description="photometric or crop_photometric")
    crop_scale: float = Field(default=0.95, gt=0.0, le=1.0)

    def apply(self, data: dict) -> dict:
        if not self.training:
            return data
        keys = [key for key in self.apply_to if key in data]
        if not keys:
            return data
        shapes, counts, frames = {}, {}, []
        for key in keys:
            value = np.asarray(data[key])
            if value.ndim != 4 or value.shape[-1] not in (1, 3, 4):
                raise ValueError(f"{key} must be [T,H,W,C] before augmentation, got {value.shape}")
            shapes[key] = value.shape
            counts[key] = value.shape[0]
            frames.extend(Image.fromarray(frame) for frame in value)
        width, height = frames[0].size
        crop = angle = None
        if self.mode == "crop_photometric":
            ch, cw = int(height * self.crop_scale), int(width * self.crop_scale)
            top = int(torch.randint(0, height - ch + 1, ()).item())
            left = int(torch.randint(0, width - cw + 1, ()).item())
            angle = float(torch.empty(()).uniform_(-5.0, 5.0).item())
            crop = (top, left, ch, cw)
        aug_frames, _ = augment_cot_sample(frames, None, mode=self.mode, crop_scale=self.crop_scale,
                                           crop=crop, angle_degrees=angle)
        offset = 0
        for key in keys:
            n = counts[key]
            dtype = np.asarray(data[key]).dtype
            data[key] = np.stack([np.asarray(f, dtype=dtype) for f in aug_frames[offset:offset + n]], 0)
            assert data[key].shape == shapes[key], (data[key].shape, shapes[key])
            offset += n
        ann = data.get("_annot")
        if crop is not None and ann is not None:
            top, left, ch, cw = crop
            geom = dict(top=top, left=left, crop_height=ch, crop_width=cw, height=height, width=width,
                        angle_degrees=angle)
            ann = dict(ann)
            ann["mask"] = np.stack([augment_label_mask(m, **geom) for m in ann["mask"]], 0)
            ann["box2d"] = augment_boxes(ann["box2d"], left=left, top=top, crop_width=cw, crop_height=ch,
                                         width=width, height=height, angle_degrees=angle)
            data["_annot"] = ann
        return data

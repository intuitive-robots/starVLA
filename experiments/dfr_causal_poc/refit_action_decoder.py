#!/usr/bin/env python3
"""Small DFR-style action readout refit for the causal P policy.

The VLM, layer projectors, action encoder, and Action DiT are frozen.  A balanced
reweight set is made from action-labelled LIBERO demonstrations by assigning an
equal number of clean, background-texture, lighting, and sensor-noise views while
keeping the demonstrated action chunk fixed.  The script supports a literal
last-affine refit and a two-layer action-decoder refit.

This is a screening experiment.  Its saved checkpoints must still be evaluated
closed-loop on unmodified LIBERO and simulator-native LIBERO+ perturbations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance, ImageFilter

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


DOMAINS = ("clean", "background", "lighting", "sensor_noise")
ARM_SPECS = {
    "fresh_final_balanced": {
        "train": "final",
        "fresh": True,
        "domains": DOMAINS,
        "lr": 1.0e-4,
    },
    "fresh_decoder_balanced": {
        "train": "decoder",
        "fresh": True,
        "domains": DOMAINS,
        "lr": 1.0e-4,
    },
    "warm_decoder_balanced": {
        "train": "decoder",
        "fresh": False,
        "domains": DOMAINS,
        "lr": 1.0e-5,
    },
    "fresh_decoder_clean": {
        "train": "decoder",
        "fresh": True,
        "domains": ("clean",),
        "lr": 1.0e-4,
    },
}


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _copy_images(images: list[Image.Image]) -> list[Image.Image]:
    return [image.convert("RGB").copy() for image in images]


def _background_texture(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Add surface-like texture without replacing object colors or silhouettes."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _ = rgb.shape
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
    neutral = ((saturation < rng.uniform(0.18, 0.38)) & (maximum > 22.0)).astype(np.uint8) * 255
    mask = Image.fromarray(neutral, mode="L").filter(ImageFilter.GaussianBlur(radius=2.0))
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0

    yy, xx = np.mgrid[:height, :width]
    angle = rng.uniform(0.0, 2.0 * np.pi)
    frequency = rng.uniform(0.035, 0.11)
    wave = 0.5 + 0.5 * np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * frequency)
    checker = ((xx // int(rng.integers(10, 28)) + yy // int(rng.integers(10, 28))) % 2).astype(float)
    mixture = np.clip(0.55 * wave + 0.45 * checker, 0.0, 1.0)[..., None]
    # Multiplicative modulation retains the underlying hue and all spatial
    # structure. A small channel tint adds material diversity, but is bounded
    # so neutral mugs/plates remain visually the same objects.
    modulation = 0.72 + 0.56 * mixture
    tint = rng.uniform(-18.0, 18.0, size=(1, 1, 3)) * (mixture - 0.5)
    textured = rgb * modulation + tint
    strength = rng.uniform(0.16, 0.30)
    output = rgb * (1.0 - alpha * strength) + textured * alpha * strength
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def _lighting(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    result = image.convert("RGB")
    if rng.random() < 0.5:
        brightness = rng.uniform(0.35, 0.70)
    else:
        brightness = rng.uniform(1.30, 1.80)
    result = ImageEnhance.Brightness(result).enhance(float(brightness))
    result = ImageEnhance.Contrast(result).enhance(float(rng.uniform(0.65, 1.35)))
    result = ImageEnhance.Color(result).enhance(float(rng.uniform(0.70, 1.30)))
    pixels = np.asarray(result, dtype=np.float32) / 255.0
    gamma = float(rng.uniform(0.70, 1.45))
    pixels = np.power(np.clip(pixels, 0.0, 1.0), gamma)
    tint = rng.uniform(0.82, 1.18, size=(1, 1, 3))
    return Image.fromarray(np.clip(pixels * tint * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _sensor_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    mode = int(rng.integers(0, 4))
    if mode == 0:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        pixels += rng.normal(0.0, rng.uniform(12.0, 34.0), size=pixels.shape)
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    if mode == 1:
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.8, 2.6))))
    if mode == 2:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        haze = rng.uniform(35.0, 105.0)
        pixels = pixels * rng.uniform(0.55, 0.82) + haze
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    corrupt = rng.random(pixels.shape[:2]) < rng.uniform(0.015, 0.055)
    salt = rng.random(pixels.shape[:2]) < 0.5
    pixels[corrupt & salt] = 255
    pixels[corrupt & ~salt] = 0
    return Image.fromarray(pixels, mode="RGB")


def _camera_jitter(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply a small train-only image-plane approximation to camera motion.

    This preserves the underlying simulator state and action label. It is only
    an augmentation proxy: simulator-native LIBERO+ camera viewpoints, which
    include true 3-D reprojection and occlusion changes, remain evaluation-only.
    """
    source = image.convert("RGB")
    width, height = source.size
    scale = float(rng.uniform(0.92, 1.08))
    translate_x = float(rng.uniform(-0.045, 0.045) * width)
    translate_y = float(rng.uniform(-0.045, 0.045) * height)
    pixels = np.asarray(source, dtype=np.float32)
    fill = tuple(int(value) for value in pixels.reshape(-1, 3).mean(axis=0))
    # PIL's affine coefficients map output coordinates back to the source.
    center_x, center_y = 0.5 * width, 0.5 * height
    inverse_scale = 1.0 / scale
    offset_x = center_x - inverse_scale * (center_x + translate_x)
    offset_y = center_y - inverse_scale * (center_y + translate_y)
    return source.transform(
        source.size,
        Image.Transform.AFFINE,
        (inverse_scale, 0.0, offset_x, 0.0, inverse_scale, offset_y),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )


def perturb_images(images: list[Image.Image], domain: str, seed: int) -> list[Image.Image]:
    output = _copy_images(images)
    if domain == "clean":
        return output
    rng = np.random.default_rng(seed)
    if domain == "background":
        return [_background_texture(image, rng) for image in output]
    if domain == "lighting":
        return [_lighting(image, rng) for image in output]
    if domain == "sensor_noise":
        # LIBERO+ applies its sensor corruptions to agent view; the paired capture
        # confirmed that its wrist image remains byte-identical.
        output[0] = _sensor_noise(output[0], rng)
        return output
    if domain == "camera_jitter":
        # LIBERO+ changes the external camera while the wrist camera remains
        # attached to the unchanged robot state.
        output[0] = _camera_jitter(output[0], rng)
        return output
    raise ValueError(f"unknown domain {domain!r}")


def _load_dataset(config_path: Path):
    cfg = OmegaConf.load(config_path).datasets.vla_data
    cfg.augmentation = "none"
    cfg.holdout_trajectories_per_dataset = 0
    cfg.holdout_trajectories_per_task = 0
    mixture = get_vla_dataset(cfg, mode="all", seed=42)
    if len(mixture.datasets) != 1:
        raise RuntimeError(f"expected one LIBERO dataset, got {len(mixture.datasets)}")
    return mixture.datasets[0]


def _split_step_indices(dataset, split_modulus: int = 5) -> tuple[np.ndarray, np.ndarray]:
    train, heldout = [], []
    for index, (trajectory_id, _) in enumerate(dataset.all_steps):
        target = heldout if int(trajectory_id) % split_modulus == 0 else train
        target.append(index)
    return np.asarray(train, dtype=np.int64), np.asarray(heldout, dtype=np.int64)


def _make_example(dataset, index: int, domain: str, seed: int) -> dict:
    source = dataset[int(index)]
    return {
        "image": perturb_images(source["image"], domain, seed),
        "lang": source["lang"],
        "action": np.asarray(source["action"], dtype=np.float32),
    }


def _write_perturbation_audit(dataset, index: int, output_dir: Path, seed: int) -> None:
    source = dataset[int(index)]
    audit_dir = output_dir / "audit_images"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        images = perturb_images(
            source["image"], domain, _stable_seed(seed, "audit", int(index), domain)
        )
        for view, image in zip(("external", "wrist"), images):
            image.save(audit_dir / f"{domain}_{view}.png")


def _action_loss(model, examples: list[dict]) -> torch.Tensor:
    images = [example["image"] for example in examples]
    instructions = [example["lang"] for example in examples]
    with torch.no_grad():
        states, _, attention_bias, _, shared_z = model._encode_vl_hidden_states(images, instructions)
    if shared_z is not None:
        raise RuntimeError("P-causal refit expects no shared-z conditioning")
    actions = torch.as_tensor(
        np.stack([example["action"] for example in examples]),
        device=states[-1].device,
        dtype=states[-1].dtype,
    )[:, -model.action_horizon :]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.action_model(
            states,
            actions,
            None,
            encoder_attention_mask=attention_bias,
            z_conditioning=None,
            encoder_memory_keep=None,
        )


@torch.inference_mode()
def _measure_domains(
    model,
    dataset,
    indices: np.ndarray,
    count: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
    result = {}
    for domain_index, domain in enumerate(DOMAINS):
        losses = []
        weights = []
        for start in range(0, len(chosen), batch_size):
            batch_indices = chosen[start : start + batch_size]
            examples = [
                _make_example(
                    dataset,
                    int(index),
                    domain,
                    _stable_seed(seed, "validation", domain, int(index)),
                )
                for index in batch_indices
            ]
            torch.manual_seed(seed + 1000 * domain_index + start)
            torch.cuda.manual_seed_all(seed + 1000 * domain_index + start)
            loss = _action_loss(model, examples)
            losses.append(float(loss.float().item()))
            weights.append(len(examples))
        result[domain] = float(np.average(losses, weights=weights))
    result["perturbed_mean"] = float(np.mean([result[x] for x in DOMAINS if x != "clean"]))
    result["worst_domain"] = float(max(result[x] for x in DOMAINS))
    return result


def _configure_trainable(model, arm: str) -> list[torch.nn.Parameter]:
    spec = ARM_SPECS[arm]
    model.requires_grad_(False)
    decoder = model.action_model.action_decoder
    if spec["fresh"]:
        if spec["train"] == "decoder":
            decoder.layer1.reset_parameters()
        decoder.layer2.reset_parameters()
    if spec["train"] == "final":
        decoder.layer2.requires_grad_(True)
    else:
        decoder.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("refit selected no trainable parameters")
    return trainable


def _save_checkpoint(model, source_checkpoint: Path, output_dir: Path) -> Path:
    run_dir = output_dir
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_run = source_checkpoint.parents[1]
    for name in ("config.yaml", "config.full.yaml", "dataset_statistics.json"):
        source = source_run / name
        if source.exists():
            shutil.copy2(source, run_dir / name)
    destination = checkpoint_dir / "dfr_refit_pytorch_model.pt"
    partial = checkpoint_dir / f".{destination.name}.partial.{os.getpid()}"
    torch.save(model.state_dict(), partial)
    os.replace(partial, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_SPECS), required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-examples", type=int, default=48)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("--batch-size and --steps must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(args.config)
    train_indices, validation_indices = _split_step_indices(dataset)
    _write_perturbation_audit(
        dataset, int(validation_indices[0]), args.output_dir, args.seed
    )
    model = baseframework.from_pretrained(str(args.checkpoint), is_inference=False).cuda().eval()
    before = _measure_domains(
        model,
        dataset,
        validation_indices,
        args.validation_examples,
        args.batch_size,
        args.seed + 1,
    )

    # Match scratch initialization exactly between the balanced and clean-only
    # decoder controls. Then reset the flow-noise stream so a different number
    # of reset_parameters() calls cannot change the training targets.
    torch.manual_seed(args.seed + 777)
    torch.cuda.manual_seed_all(args.seed + 777)
    trainable = _configure_trainable(model, args.arm)
    torch.manual_seed(args.seed + 888)
    torch.cuda.manual_seed_all(args.seed + 888)
    spec = ARM_SPECS[args.arm]
    optimizer = torch.optim.AdamW(trainable, lr=float(spec["lr"]), betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    history = []
    recent_losses: list[float] = []
    domains = tuple(spec["domains"])

    # Frozen representation and dynamics stay in eval mode. The decoder MLP has
    # no dropout, so making only it trainable is sufficient and deterministic.
    model.eval()
    for step in range(args.steps):
        selected = rng.choice(train_indices, size=args.batch_size, replace=False)
        if len(domains) == 1:
            batch_domains = [domains[0]] * args.batch_size
        else:
            batch_domains = [domains[(step * args.batch_size + j) % len(domains)] for j in range(args.batch_size)]
        examples = [
            _make_example(
                dataset,
                int(index),
                domain,
                _stable_seed(args.seed, "train", step, j, int(index), domain),
            )
            for j, (index, domain) in enumerate(zip(selected, batch_domains))
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = _action_loss(model, examples)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.float().backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        value = float(loss.float().item())
        recent_losses.append(value)
        if (step + 1) % args.log_every == 0 or step == 0:
            record = {
                "step": step + 1,
                "loss": value,
                "grad_norm": float(grad_norm),
                "window_loss": float(np.mean(recent_losses[-args.log_every :])),
                "batch_domains": batch_domains,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    after = _measure_domains(
        model,
        dataset,
        validation_indices,
        args.validation_examples,
        args.batch_size,
        args.seed + 1,
    )
    checkpoint = None if args.skip_checkpoint else _save_checkpoint(model, args.checkpoint, args.output_dir)
    report = {
        "format": "starvla_dfr_action_refit_v1",
        "arm": args.arm,
        "spec": spec,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "training_step_pool": int(len(train_indices)),
        "heldout_trajectory_step_pool": int(len(validation_indices)),
        "split": "trajectory_id modulo 5; zero is held out from refit",
        "validation_examples": min(args.validation_examples, len(validation_indices)),
        "validation_flow_loss_before": before,
        "validation_flow_loss_after": after,
        "relative_change": {
            key: float(after[key] / before[key] - 1.0) if before[key] else math.nan
            for key in after
        },
        "history": history,
        "seed": args.seed,
        "warning": (
            "Synthetic image perturbations and held-out flow loss are only a screen. "
            "Use clean LIBERO and simulator-native LIBERO+ rollout success for the claim."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"completed {args.arm}; checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()

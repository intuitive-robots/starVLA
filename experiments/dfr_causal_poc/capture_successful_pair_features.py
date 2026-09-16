#!/usr/bin/env python3
"""Capture frozen flow-residual features on exact successful appearance pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework


LOCI = ("raw_preprojector", "projected_memory", "pi_input", "pi_early", "pi_middle", "pi_late")
DOMAINS = ("clean", "perturbed")
SPLITS = ("train", "validation", "test")


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    for row in rows:
        if row.get("format") != "starvla_successful_counterfactual_pair_v1":
            raise RuntimeError("unexpected paired manifest format")
        if row["split"] not in SPLITS:
            raise RuntimeError(f"unexpected split {row['split']}")
        if float(row["state_max_abs_diff"]) > 1e-8:
            raise RuntimeError(f"non-matched simulator state in {row['pair_id']}")
        pixel_mae = max(float(row["external_pixel_mae"]), float(row["wrist_pixel_mae"]))
        if pixel_mae <= 0.1:
            raise RuntimeError(f"zero-effect visual intervention in {row['pair_id']}")
    return rows


def _load_images(row: dict, domain: str) -> list[Image.Image]:
    prefix = "clean" if domain == "clean" else "variant"
    images = []
    for view in ("external", "wrist"):
        with Image.open(row[f"{prefix}_{view}"]) as image:
            images.append(image.convert("RGB").copy())
    return images


def _load_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    action = payload[next(iter(payload))]["action"] if "action" not in payload else payload["action"]
    return np.asarray(action["min"], dtype=np.float32), np.asarray(action["max"], dtype=np.float32)


def _normalize_actions(raw: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    minimum, maximum = bounds
    result = np.asarray(raw, dtype=np.float32).copy()
    varying = minimum[:6] != maximum[:6]
    result[..., :6][..., varying] = (
        2.0 * (result[..., :6][..., varying] - minimum[:6][varying])
        / (maximum[:6][varying] - minimum[:6][varying]) - 1.0
    )
    result[..., 6] = (result[..., 6] < 0.49).astype(np.float32)
    return result


def _masked_mean(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[:, : hidden.shape[1]].to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)


def _flow(actions: torch.Tensor, keys: list[str], seed: int, buckets: int):
    noises, times = [], []
    grid = (0.10, 0.30, 0.50, 0.70, 0.90)
    for key in keys:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(seed, "noise", key))
        noises.append(torch.randn(actions.shape[1:], generator=generator))
        times.append(grid[_stable_seed(seed, "time", key) % len(grid)])
    noise = torch.stack(noises).to(actions.device, actions.dtype)
    time = torch.tensor(times, device=actions.device, dtype=actions.dtype)
    noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * actions
    return noisy, actions - noise, time, (time * buckets).long()


@torch.inference_mode()
def _capture_pi(model, examples: list[dict], keys: list[str], seed: int) -> dict:
    raw_holder = []

    def capture_raw(_module, inputs):
        raw_holder.append(inputs[0].detach())

    handle = model.project_layers[-1].register_forward_pre_hook(capture_raw)
    try:
        projected, _, bias, _, shared_z = model._encode_vl_hidden_states(
            [example["image"] for example in examples], [example["lang"] for example in examples]
        )
    finally:
        handle.remove()
    if shared_z is not None or len(raw_holder) != 1:
        raise RuntimeError("paired PI study expects one raw capture and no shared-z")
    actions = torch.as_tensor(np.stack([example["action"] for example in examples]), device=projected[-1].device, dtype=projected[-1].dtype)
    head = model.action_model
    noisy, velocity, time, discrete = _flow(actions, keys, seed, head.num_timestep_buckets)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        action_features = head.action_encoder(noisy, discrete)
        if head.config.add_pos_embed:
            position = torch.arange(actions.shape[1], device=actions.device)
            action_features = action_features + head.position_embedding(position).unsqueeze(0)
        pi_input = head._assemble_action_tokens(action_features, None)
        pi_final, hidden = head.model(
            hidden_states=pi_input,
            encoder_hidden_states=projected,
            timestep=discrete,
            encoder_attention_mask=bias,
            return_pre_output=True,
            return_all_hidden_states=True,
            force_layerwise_all_cross=head.layerwise_attention_layout == "legacy_all_cross",
            extra_conditioning=None,
            cross_attention_row_mask=None,
        )
        source = head.action_decoder(pi_final)[:, -actions.shape[1] :]
    valid = bias[:, 0, :] == 0
    action_tokens = pi_input[:, -actions.shape[1] :]
    depths = (max(1, len(hidden) // 4), max(2, len(hidden) // 2), len(hidden) - 1)

    def context(pool):
        return torch.cat([action_tokens, pool[:, None].expand(-1, actions.shape[1], -1)], -1)

    return {
        "raw_preprojector": context(_masked_mean(raw_holder[0], valid)),
        "projected_memory": context(_masked_mean(projected[-1], valid)),
        "pi_input": action_tokens,
        "pi_early": hidden[depths[0]][:, -actions.shape[1] :],
        "pi_middle": hidden[depths[1]][:, -actions.shape[1] :],
        "pi_late": hidden[depths[2]][:, -actions.shape[1] :],
        "residual": velocity - source,
        "velocity": velocity,
        "time": time,
        "depths": depths,
    }


@torch.inference_mode()
def _capture_groot(model, examples: list[dict], keys: list[str], seed: int) -> dict:
    images = [example["image"] for example in examples]
    languages = [example["lang"] for example in examples]
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=languages)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
        )
        raw = outputs.hidden_states[-1]
        memory = model.readout_projector(raw) if model.readout_projector is not None else raw
    valid_raw = getattr(model.qwen_vl_interface, "_last_encoder_attention_mask", None)
    if valid_raw is None:
        valid_raw = qwen_inputs.get("attention_mask")
    if valid_raw is None:
        valid_raw = torch.ones(raw.shape[:2], device=raw.device, dtype=torch.bool)
    valid_raw = valid_raw[:, : raw.shape[1]].bool()
    valid_memory = model._dit_valid_mask(qwen_inputs, raw)
    bias = model._dit_attention_bias(valid_memory, memory.dtype)
    actions = torch.as_tensor(np.stack([example["action"] for example in examples]), device=memory.device, dtype=memory.dtype)
    head = model.action_model
    noisy, velocity, time, discrete = _flow(actions, keys, seed, head.num_timestep_buckets)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        action_features = head.action_encoder(noisy, discrete)
        if head.config.add_pos_embed:
            position = torch.arange(actions.shape[1], device=actions.device)
            action_features = action_features + head.position_embedding(position).unsqueeze(0)
        future = head.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1)
        pi_input = torch.cat([future, action_features], dim=1)
        model_output, hidden = head.model(
            hidden_states=pi_input,
            encoder_hidden_states=memory,
            encoder_attention_mask=bias,
            timestep=discrete,
            return_all_hidden_states=True,
        )
        source = head.action_decoder(model_output)[:, -actions.shape[1] :]
    action_tokens = pi_input[:, -actions.shape[1] :]
    depths = (max(1, len(hidden) // 4), max(2, len(hidden) // 2), len(hidden) - 1)

    def context(pool):
        return torch.cat([action_tokens, pool[:, None].expand(-1, actions.shape[1], -1)], -1)

    return {
        "raw_preprojector": context(_masked_mean(raw, valid_raw)),
        "projected_memory": context(_masked_mean(memory, valid_memory)),
        "pi_input": action_tokens,
        "pi_early": hidden[depths[0]][:, -actions.shape[1] :],
        "pi_middle": hidden[depths[1]][:, -actions.shape[1] :],
        "pi_late": hidden[depths[2]][:, -actions.shape[1] :],
        "residual": velocity - source,
        "velocity": velocity,
        "time": time,
        "depths": depths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    rows = _load_rows(args.manifest)
    local = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    bounds = _load_bounds(args.stats)
    model = baseframework.from_pretrained(str(args.checkpoint), is_inference=False).cuda().eval().requires_grad_(False)
    is_pi = hasattr(model, "project_layers")
    architecture = "pi" if is_pi else "groot"
    capture = _capture_pi if is_pi else _capture_groot
    accumulators = {
        locus: {split: {domain: {key: [] for key in ("features", "residual", "velocity", "time", "keys", "tasks", "categories", "suites")}
                                for domain in DOMAINS} for split in SPLITS}
        for locus in LOCI
    }
    depth_values = None
    for start in range(0, len(local), args.batch_size):
        batch = local[start : start + args.batch_size]
        for domain in DOMAINS:
            examples = []
            keys = []
            for row in batch:
                examples.append({
                    "image": _load_images(row, domain),
                    "lang": row["language"],
                    "action": _normalize_actions(np.asarray(row["action"]), bounds),
                })
                # Same key in both domains guarantees identical flow noise/time.
                keys.append(str(row["pair_id"]))
            result = capture(model, examples, keys, args.seed)
            current_depths = tuple(result.pop("depths"))
            if depth_values is None:
                depth_values = current_depths
            elif depth_values != current_depths:
                raise RuntimeError("captured PI depths changed")
            for locus in LOCI:
                for row_index, row in enumerate(batch):
                    bucket = accumulators[locus][row["split"]][domain]
                    bucket["features"].append(result[locus][row_index : row_index + 1].half().cpu())
                    bucket["residual"].append(result["residual"][row_index : row_index + 1].float().cpu())
                    bucket["velocity"].append(result["velocity"][row_index : row_index + 1].float().cpu())
                    bucket["time"].append(result["time"][row_index : row_index + 1].float().cpu())
                    bucket["keys"].append(str(row["pair_id"]))
                    bucket["tasks"].append(str(row["base_task"]))
                    bucket["categories"].append(str(row["category"]))
                    bucket["suites"].append(str(row["suite"]))
        if start + len(batch) == len(local) or (start + len(batch)) % (args.batch_size * 16) == 0:
            print(f"rank={args.rank} architecture={architecture} pairs={start + len(batch)}/{len(local)}", flush=True)

    shard = args.output_dir / "shards" / f"rank_{args.rank}"
    shard.mkdir(parents=True, exist_ok=True)
    for locus in LOCI:
        payload = {
            "format": "starvla_successful_pair_features_v1",
            "locus": locus,
            "rank": args.rank,
            "world_size": args.world_size,
            "checkpoint": str(args.checkpoint.resolve()),
            "architecture": architecture,
            "action_horizon": int(model.action_horizon),
            "pi_depths": dict(zip(("pi_early", "pi_middle", "pi_late"), depth_values)),
            "splits": {},
        }
        for split in SPLITS:
            payload["splits"][split] = {}
            for domain in DOMAINS:
                bucket = accumulators[locus][split][domain]
                payload["splits"][split][domain] = {
                    key: torch.cat(value, 0) if key in {"features", "residual", "velocity", "time"} else value
                    for key, value in bucket.items()
                }
        torch.save(payload, shard / f"{locus}.pt")
    (shard / "manifest.json").write_text(json.dumps({
        "format": "starvla_successful_pair_feature_shard_v1",
        "rank": args.rank,
        "pairs": len(local),
        "architecture": architecture,
        "checkpoint": str(args.checkpoint.resolve()),
        "pi_depths": dict(zip(("pi_early", "pi_middle", "pi_late"), depth_values)),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

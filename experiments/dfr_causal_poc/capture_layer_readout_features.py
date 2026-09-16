#!/usr/bin/env python3
"""Capture trajectory-disjoint flow residual features at several policy loci.

This is a diagnostic, not a policy-training run.  The pretrained policy is
frozen.  For clean LIBERO and the successful generalized LIBERO-Plus dataset we
construct matched flow-matching targets, record the source policy's velocity
error, and expose representations from before the VLM->PI projection, after the
projection, and after early/middle/late PI blocks.  A separate script fits tiny
linear residual readouts to these cached features.

Every example is assigned to train/validation/test by a stable hash of its full
trajectory id.  Therefore frames from one trajectory cannot cross splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


LOCI = ("raw_preprojector", "projected_memory", "pi_input", "pi_early", "pi_middle", "pi_late")
SPLIT_BUCKETS = {"train": tuple(range(8)), "validation": (8,), "test": (9,)}


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _split_for_trajectory(domain: str, trajectory_id: int) -> str:
    bucket = _stable_seed("layer-readout-split", domain, trajectory_id) % 10
    for name, buckets in SPLIT_BUCKETS.items():
        if bucket in buckets:
            return name
    raise AssertionError(bucket)


def _load_dataset(config_path: Path, data_root: Path, data_mix: str):
    cfg = OmegaConf.load(config_path).datasets.vla_data
    cfg.data_root_dir = str(data_root)
    cfg.data_mix = data_mix
    cfg.augmentation = "none"
    cfg.holdout_trajectories_per_dataset = 0
    cfg.holdout_trajectories_per_task = 0
    # The consolidated LIBERO-Plus videos are AV1.  The current starVLA runtime
    # does not have a usable torchcodec binary, while its FFmpeg-backed decord
    # build supports these files (and is the established causal-P data path).
    if data_mix == "libero_plus":
        cfg.video_backend = "decord"
        cfg.video_seek_mode = "approximate"
        cfg.video_num_threads = 1
        cfg.video_reader_cache_size = 24
        cfg.video_prefetch_page_cache = True
    mixture = get_vla_dataset(cfg, mode="all", seed=42)
    if len(mixture.datasets) != 1:
        raise RuntimeError(f"expected one dataset for {data_mix}, got {len(mixture.datasets)}")
    return mixture.datasets[0]


def _reservoir_indices(dataset, domain: str, counts: dict[str, int], seed: int) -> dict[str, list[int]]:
    """Uniformly sample step indices within trajectory-disjoint split buckets."""
    rng = {name: np.random.default_rng(_stable_seed(seed, domain, name)) for name in counts}
    selected = {name: [] for name in counts}
    seen = Counter()
    for index, (trajectory_id, _) in enumerate(dataset.all_steps):
        split = _split_for_trajectory(domain, int(trajectory_id))
        seen[split] += 1
        target = counts[split]
        bucket = selected[split]
        if len(bucket) < target:
            bucket.append(index)
        else:
            replacement = int(rng[split].integers(0, seen[split]))
            if replacement < target:
                bucket[replacement] = index
    for split, target in counts.items():
        if len(selected[split]) != target:
            raise RuntimeError(
                f"{domain}/{split}: requested {target} steps, found {len(selected[split])}"
            )
        selected[split].sort()
    return selected


def _load_action_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    action = payload["action"]
    return np.asarray(action["min"], dtype=np.float32), np.asarray(action["max"], dtype=np.float32)


def _renormalize_plus_action(
    action: np.ndarray,
    plus_bounds: tuple[np.ndarray, np.ndarray],
    source_bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Map loader-normalized Plus actions into the source checkpoint coordinates."""
    output = np.asarray(action, dtype=np.float32).copy()
    plus_min, plus_max = plus_bounds
    source_min, source_max = source_bounds
    raw = 0.5 * (output[..., :6] + 1.0) * (plus_max[:6] - plus_min[:6]) + plus_min[:6]
    output[..., :6] = 2.0 * (raw - source_min[:6]) / (source_max[:6] - source_min[:6]) - 1.0
    # Both datasets use the same binary_invert gripper transform.
    return output


def _masked_mean(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _flow_noise_and_time(
    actions: torch.Tensor,
    sample_keys: list[str],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    noises, times = [], []
    time_grid = (0.10, 0.30, 0.50, 0.70, 0.90)
    for key in sample_keys:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(seed, "flow", key))
        noises.append(torch.randn(actions.shape[1:], generator=generator, dtype=torch.float32))
        times.append(time_grid[_stable_seed(seed, "time", key) % len(time_grid)])
    noise = torch.stack(noises).to(device=actions.device, dtype=actions.dtype)
    time = torch.tensor(times, device=actions.device, dtype=actions.dtype)
    return noise, time


@torch.inference_mode()
def _capture_batch(model, examples: list[dict], sample_keys: list[str], seed: int) -> dict[str, torch.Tensor]:
    raw_holder: list[torch.Tensor] = []

    def capture_raw(_module, inputs):
        raw_holder.append(inputs[0].detach())

    handle = model.project_layers[-1].register_forward_pre_hook(capture_raw)
    try:
        projected, _, attention_bias, _, shared_z = model._encode_vl_hidden_states(
            [example["image"] for example in examples],
            [example["lang"] for example in examples],
        )
    finally:
        handle.remove()
    if shared_z is not None:
        raise RuntimeError("the P-causal sufficiency study expects no shared-z conditioning")
    if len(raw_holder) != 1:
        raise RuntimeError(f"expected one last-projector input, captured {len(raw_holder)}")

    actions = torch.as_tensor(
        np.stack([example["action"] for example in examples]),
        device=projected[-1].device,
        dtype=projected[-1].dtype,
    )[:, -model.action_horizon :]
    noise, time = _flow_noise_and_time(actions, sample_keys, seed)
    noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * actions
    velocity = actions - noise
    head = model.action_model
    time_discrete = (time * head.num_timestep_buckets).long()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        action_features = head.action_encoder(noisy, time_discrete)
        if head.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], device=actions.device)
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
        pi_input = head._assemble_action_tokens(action_features, None)
        pi_final, all_pi = head.model(
            hidden_states=pi_input,
            encoder_hidden_states=projected,
            timestep=time_discrete,
            encoder_attention_mask=attention_bias,
            return_pre_output=True,
            return_all_hidden_states=True,
            force_layerwise_all_cross=head.layerwise_attention_layout == "legacy_all_cross",
            extra_conditioning=None,
            cross_attention_row_mask=None,
        )
        source_velocity = head.action_decoder(pi_final)[:, -actions.shape[1] :]

    valid = attention_bias[:, 0, :] == 0
    raw_pool = _masked_mean(raw_holder[0], valid)
    projected_pool = _masked_mean(projected[-1], valid)
    action_tokens = pi_input[:, -actions.shape[1] :]
    early_depth = max(1, len(all_pi) // 4)
    middle_depth = max(early_depth + 1, len(all_pi) // 2)
    late_depth = len(all_pi) - 1

    def add_context(pool: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [action_tokens, pool[:, None, :].expand(-1, action_tokens.shape[1], -1)],
            dim=-1,
        )

    residual = (velocity - source_velocity).float().cpu()
    return {
        "raw_preprojector": add_context(raw_pool).half().cpu(),
        "projected_memory": add_context(projected_pool).half().cpu(),
        "pi_input": action_tokens.half().cpu(),
        "pi_early": all_pi[early_depth][:, -actions.shape[1] :].half().cpu(),
        "pi_middle": all_pi[middle_depth][:, -actions.shape[1] :].half().cpu(),
        "pi_late": all_pi[late_depth][:, -actions.shape[1] :].half().cpu(),
        "residual": residual,
        "velocity": velocity.float().cpu(),
        "time": time.float().cpu(),
        "depths": torch.tensor([early_depth, middle_depth, late_depth]),
    }


def _sample_key(domain: str, dataset, index: int) -> str:
    trajectory_id, step = dataset.all_steps[index]
    return f"{domain}:{int(trajectory_id)}:{int(step)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--plus-root", type=Path, required=True)
    parser.add_argument("--source-stats", type=Path, required=True)
    parser.add_argument("--plus-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-per-domain", type=int, default=512)
    parser.add_argument("--validation-per-domain", type=int, default=128)
    parser.add_argument("--test-per-domain", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")

    counts = {
        "train": args.train_per_domain,
        "validation": args.validation_per_domain,
        "test": args.test_per_domain,
    }
    datasets = {
        "clean": _load_dataset(args.config, args.clean_root, "libero_mixed"),
        "plus": _load_dataset(args.config, args.plus_root, "libero_plus"),
    }
    selections = {
        domain: _reservoir_indices(dataset, domain, counts, args.seed)
        for domain, dataset in datasets.items()
    }
    source_bounds = _load_action_bounds(args.source_stats)
    plus_bounds = _load_action_bounds(args.plus_stats)

    model = baseframework.from_pretrained(
        str(args.checkpoint), is_inference=False
    ).cuda().eval().requires_grad_(False)
    shard_dir = args.output_dir / "shards" / f"rank_{args.rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    accumulators = {
        locus: {split: {domain: {"features": [], "residual": [], "velocity": [], "time": [], "keys": [], "tasks": []}
                        for domain in datasets} for split in counts}
        for locus in LOCI
    }
    depth_values = None
    for split in ("train", "validation", "test"):
        for domain, dataset in datasets.items():
            selected = selections[domain][split]
            local = [index for position, index in enumerate(selected) if position % args.world_size == args.rank]
            for start in range(0, len(local), args.batch_size):
                indices = local[start : start + args.batch_size]
                examples, keys, tasks = [], [], []
                for index in indices:
                    source = dataset[index]
                    action = np.asarray(source["action"], dtype=np.float32)
                    if domain == "plus":
                        action = _renormalize_plus_action(action, plus_bounds, source_bounds)
                    examples.append({"image": source["image"], "lang": source["lang"], "action": action})
                    keys.append(_sample_key(domain, dataset, index))
                    tasks.append(str(source["lang"]))
                captured = _capture_batch(model, examples, keys, args.seed)
                current_depths = captured.pop("depths").tolist()
                if depth_values is None:
                    depth_values = current_depths
                elif depth_values != current_depths:
                    raise RuntimeError("PI depth selection changed within one run")
                for locus in LOCI:
                    bucket = accumulators[locus][split][domain]
                    bucket["features"].append(captured[locus])
                    bucket["residual"].append(captured["residual"])
                    bucket["velocity"].append(captured["velocity"])
                    bucket["time"].append(captured["time"])
                    bucket["keys"].extend(keys)
                    bucket["tasks"].extend(tasks)
                done = start + len(indices)
                if done % max(args.batch_size * 16, 1) == 0 or done == len(local):
                    print(
                        f"rank={args.rank} split={split} domain={domain} "
                        f"captured={done}/{len(local)}",
                        flush=True,
                    )

    for locus in LOCI:
        payload = {
            "format": "starvla_layer_readout_features_v1",
            "locus": locus,
            "rank": args.rank,
            "world_size": args.world_size,
            "checkpoint": str(args.checkpoint.resolve()),
            "seed": args.seed,
            "action_horizon": int(model.action_horizon),
            "pi_depths": {
                "pi_early": depth_values[0],
                "pi_middle": depth_values[1],
                "pi_late": depth_values[2],
            },
            "splits": {},
        }
        for split in counts:
            payload["splits"][split] = {}
            for domain in datasets:
                bucket = accumulators[locus][split][domain]
                payload["splits"][split][domain] = {
                    "features": torch.cat(bucket["features"], dim=0),
                    "residual": torch.cat(bucket["residual"], dim=0),
                    "velocity": torch.cat(bucket["velocity"], dim=0),
                    "time": torch.cat(bucket["time"], dim=0),
                    "keys": bucket["keys"],
                    "tasks": bucket["tasks"],
                }
        torch.save(payload, shard_dir / f"{locus}.pt")

    manifest = {
        "format": "starvla_layer_readout_shard_v1",
        "rank": args.rank,
        "world_size": args.world_size,
        "counts_requested": counts,
        "pi_depths": {
            "pi_early": depth_values[0],
            "pi_middle": depth_values[1],
            "pi_late": depth_values[2],
        },
        "files": [f"{locus}.pt" for locus in LOCI],
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"rank {args.rank} wrote {shard_dir}", flush=True)


if __name__ == "__main__":
    main()

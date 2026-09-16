#!/usr/bin/env python3
"""Fit tiny residual flow readouts to cached layer-sufficiency features."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch


ARMS = {
    "clean_only": ("clean",),
    "balanced": ("clean", "plus"),
}


def _load_locus(root: Path, locus: str, world_size: int) -> tuple[dict, dict]:
    merged = {split: {domain: {} for domain in ("clean", "plus")}
              for split in ("train", "validation", "test")}
    invariant = None
    for rank in range(world_size):
        path = root / "shards" / f"rank_{rank}" / f"{locus}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current = {
            key: payload[key]
            for key in ("format", "locus", "world_size", "checkpoint", "seed", "action_horizon", "pi_depths")
        }
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise RuntimeError(f"feature shards disagree for {locus}: {current} != {invariant}")
        for split in merged:
            for domain in merged[split]:
                source = payload["splits"][split][domain]
                target = merged[split][domain]
                for key in ("features", "residual", "velocity", "time"):
                    target.setdefault(key, []).append(source[key])
                for key in ("keys", "tasks"):
                    target.setdefault(key, []).extend(source[key])
    for split in merged:
        for domain in merged[split]:
            bucket = merged[split][domain]
            order = np.argsort(np.asarray(bucket["keys"], dtype=str))
            for key in ("features", "residual", "velocity", "time"):
                value = torch.cat(bucket[key], dim=0)
                bucket[key] = value[torch.as_tensor(order, dtype=torch.long)]
            for key in ("keys", "tasks"):
                values = bucket[key]
                bucket[key] = [values[index] for index in order]
    return merged, invariant


def _stack_tokens(data: dict, split: str, domains: tuple[str, ...], field: str) -> torch.Tensor:
    values = [data[split][domain][field] for domain in domains]
    value = torch.cat(values, dim=0)
    if field in {"features", "residual", "velocity"}:
        return value.flatten(0, 1).float()
    return value.float()


def _objective(model, mean, scale, data: dict, split: str, domains: tuple[str, ...], device) -> float:
    values = []
    with torch.inference_mode():
        for domain in domains:
            x = _stack_tokens(data, split, (domain,), "features").to(device)
            y = _stack_tokens(data, split, (domain,), "residual").to(device)
            prediction = model((x - mean) / scale)
            values.append(float((prediction - y).square().mean().item()))
    return float(np.mean(values))


def _fit_one(
    data: dict,
    train_domains: tuple[str, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Linear, torch.Tensor, torch.Tensor, dict]:
    x = _stack_tokens(data, "train", train_domains, "features")
    y = _stack_tokens(data, "train", train_domains, "residual")
    mean = x.mean(dim=0, keepdim=True).to(device)
    scale = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-3).to(device)
    x = x.to(device)
    y = y.to(device)
    model = torch.nn.Linear(x.shape[1], y.shape[1]).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99)
    )
    validation_domains = train_domains
    best_loss = _objective(model, mean, scale, data, "validation", validation_domains, device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = [{"epoch": 0, "validation_mse": best_loss}]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    stale = 0
    for epoch in range(1, epochs + 1):
        order = torch.randperm(x.shape[0], generator=generator)
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size].to(device)
            prediction = model((x[batch] - mean) / scale)
            loss = (prediction - y[batch]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_loss = _objective(
            model, mean, scale, data, "validation", validation_domains, device
        )
        history.append({"epoch": epoch, "validation_mse": validation_loss})
        if validation_loss < best_loss - 1.0e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        # A zero residual is a meaningful baseline candidate. With a small
        # learning rate, validation can transiently worsen before the linear
        # optimizer reaches a useful solution, so do not stop during warm-up.
        if epoch >= 20 and stale >= 20:
            break
    model.load_state_dict(best_state)
    return model, mean.cpu(), scale.cpu(), {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "epochs_run": history[-1]["epoch"],
        "history": history,
    }


def _bootstrap_improvement(
    base: np.ndarray,
    corrected: np.ndarray,
    keys: list[str],
    seed: int,
) -> list[float]:
    trajectory_names = np.asarray([key.rsplit(":", 1)[0] for key in keys])
    unique = np.unique(trajectory_names)
    base_by_trajectory = np.asarray([base[trajectory_names == name].mean() for name in unique])
    corrected_by_trajectory = np.asarray(
        [corrected[trajectory_names == name].mean() for name in unique]
    )
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(2000):
        chosen = rng.integers(0, len(unique), size=len(unique))
        denominator = float(base_by_trajectory[chosen].mean())
        estimates.append(
            1.0
            - float(corrected_by_trajectory[chosen].mean())
            / max(denominator, 1.0e-12)
        )
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _evaluate(model, mean, scale, data: dict, split: str, domain: str, device, seed: int) -> dict:
    features = data[split][domain]["features"].float()
    residual = data[split][domain]["residual"].float()
    sample_count, horizon = residual.shape[:2]
    with torch.inference_mode():
        prediction = model(
            (features.flatten(0, 1).to(device) - mean.to(device)) / scale.to(device)
        ).cpu().reshape(sample_count, horizon, -1)
    base_per_sample = residual.square().mean(dim=(1, 2)).numpy()
    corrected_per_sample = (residual - prediction).square().mean(dim=(1, 2)).numpy()
    base_mse = float(base_per_sample.mean())
    corrected_mse = float(corrected_per_sample.mean())
    improvement = 1.0 - corrected_mse / max(base_mse, 1.0e-12)
    return {
        "samples": sample_count,
        "unique_trajectories": len({key.rsplit(":", 1)[0] for key in data[split][domain]["keys"]}),
        "unique_tasks": len(set(data[split][domain]["tasks"])),
        "source_residual_mse": base_mse,
        "corrected_residual_mse": corrected_mse,
        "relative_residual_mse_improvement": float(improvement),
        "trajectory_bootstrap_95_interval": _bootstrap_improvement(
            base_per_sample,
            corrected_per_sample,
            data[split][domain]["keys"],
            seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--loci", nargs="+", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    device = torch.device("cuda")
    reports = []
    output_dir = args.root / "probe_shards"
    output_dir.mkdir(parents=True, exist_ok=True)

    for locus_index, locus in enumerate(args.loci):
        data, invariant = _load_locus(args.root, locus, args.world_size)
        locus_report = {
            "format": "starvla_layer_readout_probe_v1",
            "locus": locus,
            "feature_dimension": int(data["train"]["clean"]["features"].shape[-1]),
            "checkpoint": invariant["checkpoint"],
            "pi_depths": invariant["pi_depths"],
            "arms": {},
        }
        for arm_index, (arm, train_domains) in enumerate(ARMS.items()):
            arm_seed = args.seed + 1000 * args.rank + 100 * locus_index + arm_index
            model, mean, scale, fit = _fit_one(
                data,
                train_domains,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.weight_decay,
                arm_seed,
                device,
            )
            checkpoint = {
                "format": "starvla_linear_flow_residual_v1",
                "locus": locus,
                "arm": arm,
                "feature_mean": mean,
                "feature_scale": scale,
                "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "pi_depths": invariant["pi_depths"],
            }
            checkpoint_path = output_dir / f"rank_{args.rank}_{locus}_{arm}.pt"
            torch.save(checkpoint, checkpoint_path)
            locus_report["arms"][arm] = {
                "train_domains": list(train_domains),
                "train_samples": sum(data["train"][domain]["features"].shape[0] for domain in train_domains),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "fit": fit,
                "validation": {
                    domain: _evaluate(
                        model, mean, scale, data, "validation", domain, device, arm_seed + 10 + i
                    )
                    for i, domain in enumerate(("clean", "plus"))
                },
                "test": {
                    domain: _evaluate(
                        model, mean, scale, data, "test", domain, device, arm_seed + 20 + i
                    )
                    for i, domain in enumerate(("clean", "plus"))
                },
                "checkpoint": str(checkpoint_path.resolve()),
            }
        report_path = output_dir / f"rank_{args.rank}_{locus}.json"
        report_path.write_text(json.dumps(locus_report, indent=2) + "\n")
        reports.append(str(report_path.resolve()))
        print(json.dumps(locus_report, indent=2), flush=True)

    (output_dir / f"rank_{args.rank}_manifest.json").write_text(
        json.dumps({"rank": args.rank, "reports": reports}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

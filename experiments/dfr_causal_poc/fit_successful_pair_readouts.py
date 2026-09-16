#!/usr/bin/env python3
"""Fit zero-initialized linear residual readouts on successful exact pairs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch


DOMAINS = ("clean", "perturbed")
SPLITS = ("train", "validation", "test")
ARMS = {"clean_only": ("clean",), "paired_balanced": DOMAINS}


def _load(root: Path, locus: str, world_size: int):
    merged = {split: {domain: {} for domain in DOMAINS} for split in SPLITS}
    invariant = None
    for rank in range(world_size):
        payload = torch.load(root / "shards" / f"rank_{rank}" / f"{locus}.pt", map_location="cpu", weights_only=False)
        current = {key: payload[key] for key in ("format", "locus", "world_size", "checkpoint", "architecture", "action_horizon", "pi_depths")}
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise RuntimeError(f"feature shard mismatch: {current} != {invariant}")
        for split in SPLITS:
            for domain in DOMAINS:
                source = payload["splits"][split][domain]
                target = merged[split][domain]
                for key in ("features", "residual", "velocity", "time"):
                    target.setdefault(key, []).append(source[key])
                for key in ("keys", "tasks", "categories", "suites"):
                    target.setdefault(key, []).extend(source[key])
    for split in SPLITS:
        for domain in DOMAINS:
            bucket = merged[split][domain]
            order = np.argsort(np.asarray(bucket["keys"], dtype=str))
            order_tensor = torch.as_tensor(order, dtype=torch.long)
            for key in ("features", "residual", "velocity", "time"):
                bucket[key] = torch.cat(bucket[key], 0)[order_tensor]
            for key in ("keys", "tasks", "categories", "suites"):
                values = bucket[key]
                bucket[key] = [values[index] for index in order]
    return merged, invariant


def _tokens(data, split, domains, field):
    value = torch.cat([data[split][domain][field] for domain in domains], 0)
    return value.flatten(0, 1).float() if field in {"features", "residual", "velocity"} else value.float()


def _loss(model, mean, scale, data, split, domains, device):
    values = []
    with torch.inference_mode():
        for domain in domains:
            x = _tokens(data, split, (domain,), "features").to(device)
            y = _tokens(data, split, (domain,), "residual").to(device)
            values.append(float((model((x - mean) / scale) - y).square().mean()))
    return float(np.mean(values))


def _fit(data, domains, args, seed, device):
    x = _tokens(data, "train", domains, "features")
    y = _tokens(data, "train", domains, "residual")
    mean = x.mean(0, keepdim=True).to(device)
    scale = x.std(0, unbiased=False, keepdim=True).clamp_min(1e-3).to(device)
    x, y = x.to(device), y.to(device)
    model = torch.nn.Linear(x.shape[-1], y.shape[-1]).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    best_loss = _loss(model, mean, scale, data, "validation", domains, device)
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    history = [{"epoch": 0, "validation_mse": best_loss}]
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), args.batch_size):
            indices = order[start : start + args.batch_size].to(device)
            loss = (model((x[indices] - mean) / scale) - y[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation = _loss(model, mean, scale, data, "validation", domains, device)
        history.append({"epoch": epoch, "validation_mse": validation})
        if validation < best_loss - 1e-8:
            best_loss, best_epoch, best_state, stale = validation, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch >= 20 and stale >= 20:
            break
    model.load_state_dict(best_state)
    return model, mean.cpu(), scale.cpu(), {"best_epoch": best_epoch, "best_validation_mse": best_loss, "epochs_run": history[-1]["epoch"], "history": history}


def _predict(model, mean, scale, bucket, device):
    features = bucket["features"].float()
    with torch.inference_mode():
        prediction = model((features.flatten(0, 1).to(device) - mean.to(device)) / scale.to(device)).cpu()
    return prediction.reshape_as(bucket["residual"])


def _trajectory_groups(keys: list[str]) -> np.ndarray:
    return np.asarray([key.split("_step", 1)[0] for key in keys])


def _bootstrap(base: np.ndarray, corrected: np.ndarray, keys: list[str], seed: int):
    groups = _trajectory_groups(keys)
    unique = np.unique(groups)
    base_group = np.asarray([base[groups == name].mean() for name in unique])
    corrected_group = np.asarray([corrected[groups == name].mean() for name in unique])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        chosen = rng.integers(0, len(unique), len(unique))
        values.append(1.0 - corrected_group[chosen].mean() / max(base_group[chosen].mean(), 1e-12))
    return [float(x) for x in np.quantile(values, (0.025, 0.975))]


def _metric(residual, prediction, keys, seed, mask=None):
    if mask is not None:
        residual, prediction = residual[mask], prediction[mask]
        keys = [key for key, keep in zip(keys, mask) if keep]
    base = residual.square().mean((1, 2)).numpy()
    corrected = (residual - prediction).square().mean((1, 2)).numpy()
    return {
        "samples": len(keys),
        "trajectories": len(np.unique(_trajectory_groups(keys))),
        "source_mse": float(base.mean()),
        "corrected_mse": float(corrected.mean()),
        "relative_improvement": float(1.0 - corrected.mean() / max(base.mean(), 1e-12)),
        "trajectory_bootstrap_95_interval": _bootstrap(base, corrected, keys, seed),
    }


def _evaluate(model, mean, scale, data, split, device, seed):
    predictions = {domain: _predict(model, mean, scale, data[split][domain], device) for domain in DOMAINS}
    report = {}
    for domain in DOMAINS:
        bucket = data[split][domain]
        report[domain] = _metric(bucket["residual"], predictions[domain], bucket["keys"], seed)
        report[domain]["by_category"] = {}
        categories = np.asarray(bucket["categories"])
        for index, category in enumerate(sorted(set(categories))):
            mask = categories == category
            report[domain]["by_category"][category] = _metric(
                bucket["residual"], predictions[domain], bucket["keys"], seed + index + 1, mask
            )
        report[domain]["by_suite"] = {}
        suites = np.asarray(bucket["suites"])
        for index, suite in enumerate(sorted(set(suites))):
            mask = suites == suite
            report[domain]["by_suite"][suite] = _metric(
                bucket["residual"], predictions[domain], bucket["keys"], seed + 100 + index, mask
            )
    clean, perturbed = data[split]["clean"], data[split]["perturbed"]
    if clean["keys"] != perturbed["keys"]:
        raise RuntimeError("clean/perturbed pair order mismatch")
    source_gap = (clean["residual"] - perturbed["residual"]).square().mean((1, 2)).numpy()
    corrected_gap = (
        (clean["residual"] - predictions["clean"])
        - (perturbed["residual"] - predictions["perturbed"])
    ).square().mean((1, 2)).numpy()
    report["counterfactual_gap"] = {
        "source_mse": float(source_gap.mean()),
        "corrected_mse": float(corrected_gap.mean()),
        "relative_reduction": float(1.0 - corrected_gap.mean() / max(source_gap.mean(), 1e-12)),
        "by_category": {},
        "by_suite": {},
    }
    categories = np.asarray(clean["categories"])
    for category in sorted(set(categories)):
        mask = categories == category
        report["counterfactual_gap"]["by_category"][category] = {
            "source_mse": float(source_gap[mask].mean()),
            "corrected_mse": float(corrected_gap[mask].mean()),
            "relative_reduction": float(1.0 - corrected_gap[mask].mean() / max(source_gap[mask].mean(), 1e-12)),
        }
    suites = np.asarray(clean["suites"])
    for suite in sorted(set(suites)):
        mask = suites == suite
        report["counterfactual_gap"]["by_suite"][suite] = {
            "source_mse": float(source_gap[mask].mean()),
            "corrected_mse": float(corrected_gap[mask].mean()),
            "relative_reduction": float(
                1.0 - corrected_gap[mask].mean() / max(source_gap[mask].mean(), 1e-12)
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--loci", nargs="+", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    device = torch.device("cuda")
    output = args.root / "probe_shards"
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for locus_index, locus in enumerate(args.loci):
        data, invariant = _load(args.root, locus, args.world_size)
        test_bucket = data["test"]["clean"]
        task_pairs = sorted(set(zip(test_bucket["suites"], test_bucket["tasks"])))
        report = {
            "format": "starvla_successful_pair_readout_v1",
            "locus": locus,
            "checkpoint": invariant["checkpoint"],
            "architecture": invariant["architecture"],
            "pi_depths": invariant["pi_depths"],
            "coverage": {
                "tasks": len(task_pairs),
                "tasks_by_suite": {
                    suite: sum(pair[0] == suite for pair in task_pairs)
                    for suite in sorted(set(test_bucket["suites"]))
                },
                "records_per_split_per_domain": {
                    split: int(data[split]["clean"]["features"].shape[0]) for split in SPLITS
                },
                "action_tokens_per_split_per_domain": {
                    split: int(np.prod(data[split]["clean"]["features"].shape[:2])) for split in SPLITS
                },
            },
            "arms": {},
        }
        for arm_index, (arm, domains) in enumerate(ARMS.items()):
            arm_seed = args.seed + 1000 * args.rank + 100 * locus_index + arm_index
            model, mean, scale, fit = _fit(data, domains, args, arm_seed, device)
            checkpoint = output / f"rank_{args.rank}_{locus}_{arm}.pt"
            torch.save({"format": "starvla_successful_pair_linear_readout_v1", "locus": locus, "arm": arm, "feature_mean": mean, "feature_scale": scale, "state_dict": {key: value.cpu() for key, value in model.state_dict().items()}, "checkpoint": invariant["checkpoint"], "pi_depths": invariant["pi_depths"]}, checkpoint)
            report["arms"][arm] = {
                "train_domains": list(domains),
                "train_samples": sum(data["train"][domain]["features"].shape[0] for domain in domains),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "fit": fit,
                "validation": _evaluate(model, mean, scale, data, "validation", device, arm_seed + 10),
                "test": _evaluate(model, mean, scale, data, "test", device, arm_seed + 20),
                "readout_checkpoint": str(checkpoint.resolve()),
            }
        path = output / f"rank_{args.rank}_{locus}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(str(path.resolve()))
        print(json.dumps(report, indent=2), flush=True)
    (output / f"rank_{args.rank}_manifest.json").write_text(json.dumps({"rank": args.rank, "reports": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()

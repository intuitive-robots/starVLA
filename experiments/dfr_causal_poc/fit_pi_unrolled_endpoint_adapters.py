#!/usr/bin/env python3
"""Fit a tiny PI projection LoRA through its complete four-step sampler.

All base-policy parameters stay frozen.  Clean and appearance-perturbed views
share physical state, instruction, successful action target, and deterministic
initial flow noise.  The endpoint arms backpropagate through all Euler steps;
the local-flow arm is the prior one-step objective evaluated with the same
endpoint code.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from starVLA.model.framework.base_framework import baseframework

from fit_pi_flow_field_adapters import (
    CATEGORIES,
    PIFlowAdapter,
    _balanced_batches,
    _balanced_rows,
    _encode_raw,
    _initial_noise,
    _load_action_bounds,
    _load_rows,
    _relative_mse,
    _stable_seed,
    _student_pair_velocity,
    _successful_actions,
    _teacher_path,
    _velocity,
)


ARMS = ("endpoint_correct", "endpoint_no_pair", "endpoint_shuffled", "local_flow")


def _rollout(model, adapter, raw_clean, raw_perturbed, clean_bias, perturbed_bias, noise, adapted):
    """Euler-roll both views from identical noise; differentiable when adapted."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clean_memory = adapter.project(model, raw_clean, adapted=adapted)
        perturbed_memory = adapter.project(model, raw_perturbed, adapted=adapted)
        memory = [torch.cat((clean, perturbed), 0) for clean, perturbed in zip(clean_memory, perturbed_memory)]
        bias = torch.cat((clean_bias, perturbed_bias), 0)
        actions = torch.cat((noise, noise), 0)
        steps = int(model.action_model.num_inference_timesteps)
        with adapter.enabled(adapted):
            for step in range(steps):
                actions = actions + _velocity(model.action_model, memory, bias, actions, step) / steps
    return actions.chunk(2, 0)


def _per_sample_mse(prediction, target):
    return (prediction.float() - target.float()).square().mean(dim=(1, 2))


@torch.inference_mode()
def _evaluate(model, adapter, rows, bounds, batch_size, seed):
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        raw_c, raw_p, bias_c, bias_p = _encode_raw(model, batch)
        dtype, device = raw_c[0].dtype, raw_c[0].device
        noise = _initial_noise(batch, model.action_model, seed, dtype, device)
        target = _successful_actions(batch, bounds, device, dtype)
        base_c, base_p = _rollout(model, adapter, raw_c, raw_p, bias_c, bias_p, noise, False)
        adapted_c, adapted_p = _rollout(model, adapter, raw_c, raw_p, bias_c, bias_p, noise, True)
        values = {
            "base_pair_endpoint_mse": _per_sample_mse(base_p, base_c),
            "adapted_pair_endpoint_mse": _per_sample_mse(adapted_p, adapted_c),
            "base_perturbed_successful_mse": _per_sample_mse(base_p, target),
            "adapted_perturbed_successful_mse": _per_sample_mse(adapted_p, target),
            "base_clean_successful_mse": _per_sample_mse(base_c, target),
            "adapted_clean_successful_mse": _per_sample_mse(adapted_c, target),
            "clean_endpoint_drift_mse": _per_sample_mse(adapted_c, base_c),
            "successful_action_energy": target.float().square().mean(dim=(1, 2)).clamp_min(1e-6),
        }
        for index, row in enumerate(batch):
            record = {
                "pair_id": row["pair_id"], "suite": row["suite"],
                "task_id": int(row["task_id"]), "category": row["category"],
            }
            record.update({key: float(value[index]) for key, value in values.items()})
            records.append(record)
        print(f"validation {start + len(batch)}/{len(rows)}", flush=True)

    def summarize(selected):
        mean = {key: float(np.mean([row[key] for row in selected])) for key in values}
        mean.update(
            pair_endpoint_improvement=1.0 - mean["adapted_pair_endpoint_mse"] / max(mean["base_pair_endpoint_mse"], 1e-12),
            successful_endpoint_improvement=1.0 - mean["adapted_perturbed_successful_mse"] / max(mean["base_perturbed_successful_mse"], 1e-12),
            clean_successful_endpoint_improvement=1.0 - mean["adapted_clean_successful_mse"] / max(mean["base_clean_successful_mse"], 1e-12),
            clean_drift_over_successful_action_energy=mean["clean_endpoint_drift_mse"] / max(mean["successful_action_energy"], 1e-12),
        )
        return mean

    overall = summarize(records)
    by_category = {category: summarize([row for row in records if row["category"] == category]) for category in CATEGORIES}
    clusters = defaultdict(list)
    for record in records:
        clusters[(record["suite"], record["task_id"])].append(record)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed + 991)
    pair_samples, successful_samples = [], []
    for _ in range(5000):
        indices = rng.choice(len(keys), size=len(keys), replace=True)
        selected = [row for index in indices for row in clusters[keys[int(index)]]]
        metrics = summarize(selected)
        pair_samples.append(metrics["pair_endpoint_improvement"])
        successful_samples.append(metrics["successful_endpoint_improvement"])
    return {
        "overall": overall,
        "by_category": by_category,
        "task_cluster_bootstrap": {
            "replicates": 5000,
            "pair_endpoint_improvement_95pct": np.quantile(pair_samples, (0.025, 0.975)).tolist(),
            "successful_endpoint_improvement_95pct": np.quantile(successful_samples, (0.025, 0.975)).tolist(),
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pair-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.train_pairs = min(args.train_pairs, 8)
        args.validation_pairs = min(args.validation_pairs, 8)
        args.log_every = 1
    for name in ("train_pairs", "validation_pairs"):
        value = getattr(args, name)
        if value <= 0 or value % len(CATEGORIES):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and divisible by four")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.manifest)
    bounds = _load_action_bounds(args.stats)
    train_rows = _balanced_rows(rows, "train", args.train_pairs, args.seed)
    validation_rows = _balanced_rows(rows, "validation", args.validation_pairs, args.seed + 1)
    model = baseframework.from_pretrained(str(args.checkpoint), is_inference=False).cuda().eval().requires_grad_(False)
    adapter = PIFlowAdapter(model, "projection_lora", args.rank).cuda()
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    history, seen_pairs, seen_tasks = [], set(), set()
    batches = list(_balanced_batches(train_rows, args.seed))
    for batch_index, batch in enumerate(batches):
        raw_c, raw_p, bias_c, bias_p = _encode_raw(model, batch)
        dtype, device = raw_c[0].dtype, raw_c[0].device
        optimizer.zero_grad(set_to_none=True)
        if args.arm == "local_flow":
            path, teachers, _, _, _, ideal = _teacher_path(model, adapter, raw_c, bias_c, batch, bounds, args.seed)
            logged = defaultdict(list)
            for flow_step, (state, teacher) in enumerate(zip(path, teachers)):
                clean, perturbed = _student_pair_velocity(model, adapter, raw_c, raw_p, bias_c, bias_p, state, flow_step)
                clean_mse, clean_rel = _relative_mse(clean, ideal)
                pert_mse, pert_rel = _relative_mse(perturbed, ideal)
                pair_mse = F.mse_loss(perturbed.float(), clean.float())
                pair_rel = pair_mse / ideal.float().square().mean().clamp_min(1e-6)
                anchor_mse, anchor_rel = _relative_mse(clean, teacher)
                loss = 0.5 * (clean_rel + pert_rel) + args.pair_weight * pair_rel + args.anchor_weight * anchor_rel
                (loss / len(path)).backward()
                for key, value in (("clean_success_mse", clean_mse), ("perturbed_success_mse", pert_mse), ("pair_mse", pair_mse), ("clean_anchor_mse", anchor_mse)):
                    logged[key].append(float(value.detach()))
            metrics = {key: float(np.mean(value)) for key, value in logged.items()}
        else:
            noise = _initial_noise(batch, model.action_model, args.seed, dtype, device)
            target = _successful_actions(batch, bounds, device, dtype)
            with torch.no_grad():
                base_c, base_p = _rollout(model, adapter, raw_c, raw_p, bias_c, bias_p, noise, False)
            clean, perturbed = _rollout(model, adapter, raw_c, raw_p, bias_c, bias_p, noise, True)
            clean_mse, clean_rel = _relative_mse(clean, target)
            pert_mse, pert_rel = _relative_mse(perturbed, target)
            pair_target = torch.roll(clean, 1, 0) if args.arm == "endpoint_shuffled" else clean
            pair_mse = F.mse_loss(perturbed.float(), pair_target.float())
            base_pair_scale = F.mse_loss(base_p.float(), (torch.roll(base_c, 1, 0) if args.arm == "endpoint_shuffled" else base_c).float()).clamp_min(1e-6)
            pair_rel = pair_mse / base_pair_scale
            anchor_mse = F.mse_loss(clean.float(), base_c.float())
            anchor_rel = anchor_mse / base_c.float().square().mean().clamp_min(1e-6)
            effective_pair_weight = 0.0 if args.arm == "endpoint_no_pair" else args.pair_weight
            loss = 0.5 * (clean_rel + pert_rel) + effective_pair_weight * pair_rel + args.anchor_weight * anchor_rel
            loss.backward()
            metrics = {
                "clean_success_mse": float(clean_mse.detach()),
                "perturbed_success_mse": float(pert_mse.detach()),
                "pair_mse": float(pair_mse.detach()),
                "clean_anchor_mse": float(anchor_mse.detach()),
            }
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        for row in batch:
            seen_pairs.add(row["pair_id"])
            seen_tasks.add((row["suite"], int(row["task_id"])))
        if batch_index == 0 or (batch_index + 1) % args.log_every == 0 or batch_index + 1 == len(batches):
            record = {"optimizer_step": batch_index + 1, "pairs_seen": len(seen_pairs), **metrics, "grad_norm": float(grad_norm)}
            history.append(record)
            print(json.dumps(record), flush=True)

    adapter.eval()
    validation = _evaluate(model, adapter, validation_rows, bounds, args.eval_batch_size, args.seed + 1000)
    adapter_path = args.output_dir / "adapter.pt"
    torch.save({
        "format": "starvla_pi_unrolled_endpoint_adapter_v1", "arm": args.arm,
        "rank": args.rank, "source_checkpoint": str(args.checkpoint.resolve()),
        "cross_layers": adapter.cross_layers,
        "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
    }, adapter_path)
    overall = validation["overall"]
    report = {
        "format": "starvla_pi_unrolled_endpoint_adapter_report_v1", "status": "complete",
        "scope": "offline validation diagnostic; test split untouched; no closed-loop success claim",
        "arm": args.arm, "objective": "local_flow" if args.arm == "local_flow" else "four_step_unrolled_endpoint",
        "source_checkpoint": str(args.checkpoint.resolve()), "manifest": str(args.manifest.resolve()),
        "adapter": str(adapter_path.resolve()), "rank": args.rank,
        "layerwise_attention_layout": model.action_model.layerwise_attention_layout,
        "num_inference_timesteps": int(model.action_model.num_inference_timesteps),
        "cross_attention_layers": adapter.cross_layers,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "train": {
            "pairs": len(seen_pairs), "tasks": len(seen_tasks), "optimizer_steps": len(batches),
            "category_counts": dict(Counter(row["category"] for row in train_rows)),
            "pair_weight": 0.0 if args.arm == "endpoint_no_pair" else args.pair_weight,
            "anchor_weight": args.anchor_weight, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "validation": {
            "pairs": len(validation_rows),
            "tasks": len({(row["suite"], int(row["task_id"])) for row in validation_rows}),
            "category_counts": dict(Counter(row["category"] for row in validation_rows)), **validation,
        },
        "screen": {
            "pair_endpoint_improvement_at_least_10pct": overall["pair_endpoint_improvement"] >= 0.10,
            "successful_endpoint_improvement_at_least_10pct": overall["successful_endpoint_improvement"] >= 0.10,
            "clean_drift_below_1pct_action_energy": overall["clean_drift_over_successful_action_energy"] <= 0.01,
            "successful_bootstrap_lower_bound_above_zero": validation["task_cluster_bootstrap"]["successful_endpoint_improvement_95pct"][0] > 0,
            "nonnegative_success_improvement_in_at_least_3_categories": sum(value["successful_endpoint_improvement"] >= 0 for value in validation["by_category"].values()) >= 3,
        },
        "history": history, "seed": args.seed,
    }
    report["screen"]["passes_individual_screen"] = all(report["screen"].values())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    adapter.close()
    print(json.dumps({"arm": args.arm, "screen": report["screen"], "overall": overall}), flush=True)


if __name__ == "__main__":
    main()

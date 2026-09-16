#!/usr/bin/env python3
"""Zero-initialized, low-rank post-hoc alignment for P-causal.

This follow-up preserves the pretrained flow policy. It fits rank-1 directions
for synthetic background, lighting, and sensor-noise changes using only the
training trajectory split, then learns 84 bounded suppression gates (three
directions at each of 28 projected PI states). Selected arms additionally
fine-tune the pretrained final velocity affine. Nothing is reinitialized.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from refit_action_decoder import (
    _load_dataset,
    _split_step_indices,
    _stable_seed,
    _write_perturbation_audit,
    perturb_images,
)
from starVLA.model.framework.base_framework import baseframework


LEGACY_NUISANCES = ("background", "lighting", "sensor_noise")
V2_NUISANCES = (*LEGACY_NUISANCES, "camera_jitter")
ARM_SPECS = {
    "warm_final_balanced": {
        "basis": "none",
        "train_gate": False,
        "train_final": True,
    },
    "learned_gate_all": {
        "basis": "learned",
        "train_gate": True,
        "train_final": False,
    },
    "learned_gate_all_warm_final": {
        "basis": "learned",
        "train_gate": True,
        "train_final": True,
    },
    "random_gate_all_warm_final": {
        "basis": "random",
        "train_gate": True,
        "train_final": True,
        "conditional": False,
    },
    "warm_final_balanced_v2": {
        "basis": "none",
        "train_gate": False,
        "train_final": True,
        "conditional": False,
    },
    "static_orthogonal_gate_warm_final": {
        "basis": "learned_orthogonal",
        "train_gate": True,
        "train_final": True,
        "conditional": False,
    },
    "conditional_orthogonal_gate_warm_final": {
        "basis": "learned_orthogonal",
        "train_gate": True,
        "train_final": True,
        "conditional": True,
    },
    "static_random_gate_warm_final_v2": {
        "basis": "random",
        "train_gate": True,
        "train_final": True,
        "conditional": False,
    },
}

for _spec in ARM_SPECS.values():
    _spec.setdefault("conditional", False)


def _source_example(dataset, index: int) -> dict:
    source = dataset[int(index)]
    return {
        "images": [image.convert("RGB").copy() for image in source["image"]],
        "lang": source["lang"],
        "action": np.asarray(source["action"], dtype=np.float32),
    }


@torch.no_grad()
def _encode(model, examples: list[dict]):
    states, _, bias, _, shared_z = model._encode_vl_hidden_states(
        [example["images"] for example in examples],
        [example["lang"] for example in examples],
    )
    if shared_z is not None:
        raise RuntimeError("P-causal alignment expects no shared-z conditioning")
    return states, bias


def _pool_states(states: list[torch.Tensor], bias: torch.Tensor) -> torch.Tensor:
    valid = bias[:, 0, :] == 0
    pooled = []
    for state in states:
        mask = valid[:, : state.shape[1]]
        pooled.append(
            (state * mask.unsqueeze(-1)).sum(dim=1)
            / mask.sum(dim=1, keepdim=True).clamp_min(1)
        )
    return torch.stack(pooled, dim=1).float().cpu()


def _top_right_direction(delta: torch.Tensor, seed: int, iterations: int = 32):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vector = torch.randn(delta.shape[1], generator=generator, dtype=torch.float64)
    matrix = delta.double()
    vector = vector / vector.norm().clamp_min(1.0e-12)
    for _ in range(iterations):
        vector = matrix.T @ (matrix @ vector)
        vector = vector / vector.norm().clamp_min(1.0e-12)
    pivot = int(vector.abs().argmax())
    if vector[pivot] < 0:
        vector = -vector
    captured = float((matrix @ vector).square().sum())
    total = max(float(matrix.square().sum()), 1.0e-12)
    return vector.float(), captured / total


@torch.inference_mode()
def _fit_training_bases(
    model,
    dataset,
    train_indices: np.ndarray,
    count: int,
    batch_size: int,
    seed: int,
    nuisances: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(train_indices, size=min(count, len(train_indices)), replace=False)
    clean_by_domain: dict[str, list[torch.Tensor]] = {name: [] for name in nuisances}
    delta_by_domain: dict[str, list[torch.Tensor]] = {name: [] for name in nuisances}

    for domain in nuisances:
        for start in range(0, len(chosen), batch_size):
            indices = chosen[start : start + batch_size]
            clean, perturbed = [], []
            for index in indices:
                source = _source_example(dataset, int(index))
                clean.append(source)
                perturbed.append(
                    {
                        **source,
                        "images": perturb_images(
                            source["images"],
                            domain,
                            _stable_seed(seed, "basis", domain, int(index)),
                        ),
                    }
                )
            states, bias = _encode(model, clean + perturbed)
            pooled = _pool_states(states, bias)
            n = len(clean)
            clean_by_domain[domain].append(pooled[:n])
            delta_by_domain[domain].append(pooled[n:] - pooled[:n])
        print(f"basis activations: {domain} {len(chosen)} pairs", flush=True)

    num_layers = len(states)
    hidden_dim = states[0].shape[-1]
    means = torch.empty(num_layers, len(nuisances), hidden_dim, dtype=torch.float32)
    bases = torch.empty_like(means)
    explained = {}
    for domain_index, domain in enumerate(nuisances):
        clean = torch.cat(clean_by_domain[domain], dim=0)
        delta = torch.cat(delta_by_domain[domain], dim=0)
        explained[domain] = []
        for layer in range(num_layers):
            means[layer, domain_index] = clean[:, layer].mean(dim=0)
            direction, ratio = _top_right_direction(
                delta[:, layer], _stable_seed(seed, "power", domain, layer)
            )
            bases[layer, domain_index] = direction
            explained[domain].append(
                {"layer": layer, "rank1_delta_energy": float(ratio)}
            )

    return means, bases, {
        "fit_pairs_per_domain": int(len(chosen)),
        "fit_step_indices_sha256": __import__("hashlib")
        .sha256(chosen.astype(np.int64).tobytes())
        .hexdigest(),
        "rank1_explained_energy": explained,
    }


def _orthonormalize_bases(
    means: torch.Tensor, bases: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Symmetrically orthonormalize each layer's nuisance directions.

    The Loewdin transform G^-1/2 U is invariant to nuisance ordering and is the
    nearest orthonormal row basis to U. A common clean mean keeps the resulting
    affine operation interpretable after the original directions are mixed.
    """
    output = torch.empty_like(bases)
    diagnostics = []
    for layer in range(bases.shape[0]):
        original = bases[layer].double()
        gram = original @ original.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        if float(eigenvalues.min()) <= 1.0e-8:
            raise RuntimeError(
                f"nuisance basis is rank deficient at layer {layer}: "
                f"minimum Gram eigenvalue={float(eigenvalues.min()):.3e}"
            )
        inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
        orthogonal = inverse_sqrt @ original
        output[layer] = orthogonal.float()
        residual = orthogonal @ orthogonal.T - torch.eye(
            orthogonal.shape[0], dtype=orthogonal.dtype
        )
        diagnostics.append(
            {
                "layer": layer,
                "pre_min_gram_eigenvalue": float(eigenvalues.min()),
                "pre_max_gram_eigenvalue": float(eigenvalues.max()),
                "post_max_orthogonality_error": float(residual.abs().max()),
            }
        )
    common_means = means.mean(dim=1, keepdim=True).expand_as(means).contiguous()
    return common_means, output, {"orthonormalization": diagnostics}


def _randomize_bases(bases: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    output = torch.empty_like(bases)
    for layer in range(bases.shape[0]):
        random = torch.randn(
            bases.shape[-1], bases.shape[1], generator=generator, dtype=torch.float32
        )
        orthonormal, _ = torch.linalg.qr(random, mode="reduced")
        output[layer] = orthonormal.T
    return output


class NuisanceGate(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        bases: torch.Tensor,
        enabled: bool,
        conditional: bool,
        nuisances: tuple[str, ...],
    ):
        super().__init__()
        self.nuisances = nuisances
        self.conditional = conditional
        self.register_buffer("means", means)
        self.register_buffer("bases", bases)
        self.alpha = nn.Parameter(
            torch.zeros(means.shape[0], means.shape[1], dtype=torch.float32),
            requires_grad=enabled,
        )
        self.condition_weight = nn.Parameter(
            torch.zeros(means.shape[0], means.shape[1], dtype=torch.float32),
            requires_grad=enabled and conditional,
        )
        self.condition_bias = nn.Parameter(
            torch.zeros(means.shape[0], means.shape[1], dtype=torch.float32),
            requires_grad=enabled and conditional,
        )

    def forward(self, states: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.alpha.requires_grad and bool(torch.count_nonzero(self.alpha) == 0):
            return states
        output = []
        for layer, hidden in enumerate(states):
            means = self.means[layer].to(device=hidden.device, dtype=hidden.dtype)
            basis = self.bases[layer].to(device=hidden.device, dtype=hidden.dtype)
            alpha = self.alpha[layer].to(device=hidden.device, dtype=hidden.dtype)
            centered = hidden.unsqueeze(-2) - means.view(1, 1, *means.shape)
            coefficient = (centered * basis.view(1, 1, *basis.shape)).sum(dim=-1)
            if self.conditional:
                # Query/token-dependent suppression, analogous in spirit to an
                # attention output gate but restricted to the fitted subspace.
                evidence = coefficient.abs() / (1.0 + coefficient.abs())
                weight = self.condition_weight[layer].to(hidden.dtype)
                bias = self.condition_bias[layer].to(hidden.dtype)
                alpha = alpha.view(1, 1, -1) * torch.sigmoid(
                    evidence * weight.view(1, 1, -1) + bias.view(1, 1, -1)
                )
            correction = (
                coefficient.mul(alpha).unsqueeze(-1)
                * basis.view(1, 1, *basis.shape)
            ).sum(dim=-2)
            output.append(hidden - correction)
        return output

    @torch.no_grad()
    def clamp_(self) -> None:
        self.alpha.clamp_(0.0, 1.0)

    def payload(self, basis_kind: str, source_checkpoint: Path) -> dict:
        return {
            "format": "starvla_pi_nuisance_alignment_v2",
            "space": "projected",
            "num_layers": int(self.means.shape[0]),
            "hidden_dim": int(self.means.shape[-1]),
            "nuisances": list(self.nuisances),
            "basis_kind": basis_kind,
            "conditional": self.conditional,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "layers": {
                str(layer): {
                    "means": self.means[layer].detach().cpu(),
                    "basis": self.bases[layer].detach().cpu().T.contiguous(),
                    "alpha": self.alpha[layer].detach().cpu(),
                    "condition_weight": self.condition_weight[layer].detach().cpu(),
                    "condition_bias": self.condition_bias[layer].detach().cpu(),
                }
                for layer in range(self.means.shape[0])
            },
        }


def _actions(examples: list[dict], model, device: torch.device, dtype: torch.dtype):
    return torch.as_tensor(
        np.stack([example["action"] for example in examples]),
        device=device,
        dtype=dtype,
    )[:, -model.action_horizon :]


def _paired_batch(dataset, indices, domains, seed, step):
    clean, perturbed = [], []
    for position, (index, domain) in enumerate(zip(indices, domains)):
        source = _source_example(dataset, int(index))
        clean.append(source)
        perturbed.append(
            {
                **source,
                "images": perturb_images(
                    source["images"],
                    domain,
                    _stable_seed(seed, "train", step, position, int(index), domain),
                ),
            }
        )
    return clean, perturbed


def _flow_pair_objective(model, gate, clean, perturbed, seed, pair_weight):
    states, bias = _encode(model, clean + perturbed)
    count = len(clean)
    clean_states = gate([state[:count] for state in states])
    perturbed_states = gate([state[count:] for state in states])
    clean_bias, perturbed_bias = bias[:count], bias[count:]
    actions = _actions(clean, model, states[0].device, states[0].dtype)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clean_loss, clean_estimate = model.action_model(
            clean_states,
            actions,
            None,
            encoder_attention_mask=clean_bias,
            return_clean_actions=True,
            z_conditioning=None,
            encoder_memory_keep=None,
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        perturbed_loss, perturbed_estimate = model.action_model(
            perturbed_states,
            actions,
            None,
            encoder_attention_mask=perturbed_bias,
            return_clean_actions=True,
            z_conditioning=None,
            encoder_memory_keep=None,
        )
    consistency = (clean_estimate.float() - perturbed_estimate.float()).square().mean()
    objective = 0.5 * (clean_loss.float() + perturbed_loss.float()) + pair_weight * consistency
    return objective, clean_loss.float(), perturbed_loss.float(), consistency


@torch.inference_mode()
def _measure_domains(model, gate, dataset, indices, count, batch_size, seed, nuisances):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
    result = {}
    for domain_index, domain in enumerate(("clean", *nuisances)):
        losses, weights = [], []
        for start in range(0, len(chosen), batch_size):
            examples = []
            for index in chosen[start : start + batch_size]:
                source = _source_example(dataset, int(index))
                if domain != "clean":
                    source["images"] = perturb_images(
                        source["images"],
                        domain,
                        _stable_seed(seed, "validation", domain, int(index)),
                    )
                examples.append(source)
            states, bias = _encode(model, examples)
            states = gate(states)
            actions = _actions(examples, model, states[0].device, states[0].dtype)
            torch.manual_seed(seed + 1000 * domain_index + start)
            torch.cuda.manual_seed_all(seed + 1000 * domain_index + start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.action_model(
                    states,
                    actions,
                    None,
                    encoder_attention_mask=bias,
                    z_conditioning=None,
                    encoder_memory_keep=None,
                )
            losses.append(float(loss.float()))
            weights.append(len(examples))
        result[domain] = float(np.average(losses, weights=weights))
    result["perturbed_mean"] = float(np.mean([result[x] for x in nuisances]))
    result["worst_domain"] = float(max(result[x] for x in ("clean", *nuisances)))
    return result


def _relative_parameter_anchor(parameters, originals):
    terms = []
    for parameter, original in zip(parameters, originals):
        denominator = original.float().square().mean().clamp_min(1.0e-8)
        terms.append((parameter.float() - original).square().mean() / denominator)
    if not terms:
        return torch.zeros((), device="cuda")
    return torch.stack(terms).mean()


def _save_checkpoint(model, source_checkpoint: Path, output_dir: Path) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_run = source_checkpoint.parents[1]
    for name in ("config.yaml", "config.full.yaml", "dataset_statistics.json"):
        source = source_run / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    destination = checkpoint_dir / "aligned_pytorch_model.pt"
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--microbatches-per-step", type=int, default=1)
    parser.add_argument("--basis-pairs", type=int, default=256)
    parser.add_argument("--basis-batch-size", type=int, default=8)
    parser.add_argument("--validation-examples", type=int, default=48)
    parser.add_argument("--pair-weight", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--gate-penalty", type=float, default=0.001)
    parser.add_argument("--condition-penalty", type=float, default=1.0e-4)
    parser.add_argument("--final-lr", type=float, default=2.0e-5)
    parser.add_argument("--gate-lr", type=float, default=1.0e-2)
    parser.add_argument(
        "--nuisances",
        nargs="+",
        choices=V2_NUISANCES,
        default=list(LEGACY_NUISANCES),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    nuisances = tuple(args.nuisances)
    if len(set(nuisances)) != len(nuisances):
        raise ValueError("--nuisances must not contain duplicates")
    if args.batch_size != len(nuisances):
        raise ValueError(
            f"--batch-size must be {len(nuisances)} for one pair per nuisance"
        )
    for name in (
        "steps",
        "microbatches_per_step",
        "basis_pairs",
        "basis_batch_size",
        "validation_examples",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(args.config)
    train_indices, validation_indices = _split_step_indices(dataset)
    _write_perturbation_audit(dataset, int(validation_indices[0]), args.output_dir, args.seed)
    if "camera_jitter" in nuisances:
        audit_source = _source_example(dataset, int(validation_indices[0]))
        audit_images = perturb_images(
            audit_source["images"],
            "camera_jitter",
            _stable_seed(args.seed, "audit", int(validation_indices[0]), "camera_jitter"),
        )
        audit_dir = args.output_dir / "audit_images"
        for view, image in zip(("external", "wrist"), audit_images):
            image.save(audit_dir / f"camera_jitter_{view}.png")
    model = baseframework.from_pretrained(str(args.checkpoint), is_inference=False).cuda().eval()
    model.requires_grad_(False)
    spec = ARM_SPECS[args.arm]

    if spec["basis"] == "none":
        num_layers = len(model.project_layers)
        hidden_dim = model.action_dit_hidden_dim
        means = torch.zeros(num_layers, len(nuisances), hidden_dim)
        bases = torch.zeros_like(means)
        basis_report = {"fit_pairs_per_domain": 0, "rank1_explained_energy": {}}
    else:
        means, bases, basis_report = _fit_training_bases(
            model,
            dataset,
            train_indices,
            args.basis_pairs,
            args.basis_batch_size,
            args.seed + 17,
            nuisances,
        )
        if spec["basis"] == "learned_orthogonal":
            means, bases, orthogonal_report = _orthonormalize_bases(means, bases)
            basis_report.update(orthogonal_report)
        if spec["basis"] == "random":
            bases = _randomize_bases(bases, args.seed + 991)

    gate = NuisanceGate(
        means,
        bases,
        enabled=bool(spec["train_gate"]),
        conditional=bool(spec["conditional"]),
        nuisances=nuisances,
    ).cuda()
    final_parameters = list(model.action_model.action_decoder.layer2.parameters())
    final_originals = [parameter.detach().float().clone() for parameter in final_parameters]
    if spec["train_final"]:
        for parameter in final_parameters:
            parameter.requires_grad_(True)

    groups = []
    if spec["train_gate"]:
        groups.append(
            {
                "params": [
                    parameter for parameter in gate.parameters() if parameter.requires_grad
                ],
                "lr": args.gate_lr,
            }
        )
    if spec["train_final"]:
        groups.append({"params": final_parameters, "lr": args.final_lr})
    if not groups:
        raise RuntimeError("arm selected no trainable parameters")
    trainable = [parameter for group in groups for parameter in group["params"]]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

    before = _measure_domains(
        model,
        gate,
        dataset,
        validation_indices,
        args.validation_examples,
        args.batch_size,
        args.seed + 1,
        nuisances,
    )
    rng = np.random.default_rng(args.seed)
    shuffled_train_indices = rng.permutation(train_indices)
    train_cursor = 0
    unique_training_indices: set[int] = set()
    history = []
    recent = []
    model.eval()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        objective_values = []
        clean_loss_values = []
        perturbed_loss_values = []
        consistency_values = []
        for microbatch in range(args.microbatches_per_step):
            if train_cursor + args.batch_size > len(shuffled_train_indices):
                shuffled_train_indices = rng.permutation(train_indices)
                train_cursor = 0
            selected = shuffled_train_indices[
                train_cursor : train_cursor + args.batch_size
            ]
            train_cursor += args.batch_size
            unique_training_indices.update(int(index) for index in selected)
            sample_step = step * args.microbatches_per_step + microbatch
            shift = sample_step % len(nuisances)
            domains = [
                nuisances[(shift + index) % len(nuisances)]
                for index in range(args.batch_size)
            ]
            clean, perturbed = _paired_batch(
                dataset, selected, domains, args.seed, sample_step
            )
            objective, clean_loss, perturbed_loss, consistency = _flow_pair_objective(
                model,
                gate,
                clean,
                perturbed,
                args.seed + 100000 + sample_step,
                args.pair_weight,
            )
            if not torch.isfinite(objective):
                raise RuntimeError(
                    f"non-finite data objective at step {step + 1}, "
                    f"microbatch {microbatch + 1}"
                )
            (objective / args.microbatches_per_step).backward()
            objective_values.append(float(objective.detach()))
            clean_loss_values.append(float(clean_loss.detach()))
            perturbed_loss_values.append(float(perturbed_loss.detach()))
            consistency_values.append(float(consistency.detach()))
        anchor = _relative_parameter_anchor(
            final_parameters if spec["train_final"] else [],
            final_originals if spec["train_final"] else [],
        )
        gate_norm = gate.alpha.square().mean()
        condition_norm = (
            gate.condition_weight.square().mean() + gate.condition_bias.square().mean()
        )
        regularizer = (
            args.anchor_weight * anchor
            + args.gate_penalty * gate_norm
            + args.condition_penalty * condition_norm
        )
        if not torch.isfinite(regularizer):
            raise RuntimeError(f"non-finite regularizer at step {step + 1}")
        regularizer.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        gate.clamp_()
        total_value = float(np.mean(objective_values) + regularizer.detach().item())
        recent.append(total_value)
        if step == 0 or (step + 1) % args.log_every == 0:
            record = {
                "step": step + 1,
                "total": total_value,
                "flow_clean": float(np.mean(clean_loss_values)),
                "flow_perturbed": float(np.mean(perturbed_loss_values)),
                "pair_consistency": float(np.mean(consistency_values)),
                "parameter_anchor": float(anchor.detach()),
                "gate_mean": float(gate.alpha.detach().mean()),
                "gate_max": float(gate.alpha.detach().max()),
                "condition_norm": float(condition_norm.detach()),
                "grad_norm": float(grad_norm),
                "window_total": float(np.mean(recent[-args.log_every :])),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    after = _measure_domains(
        model,
        gate,
        dataset,
        validation_indices,
        args.validation_examples,
        args.batch_size,
        args.seed + 1,
        nuisances,
    )
    gate_path = args.output_dir / "alignment_gate.pt"
    if spec["train_gate"]:
        torch.save(gate.payload(spec["basis"], args.checkpoint), gate_path)
    checkpoint = None
    if not args.skip_checkpoint and spec["train_final"]:
        checkpoint = _save_checkpoint(model, args.checkpoint, args.output_dir)
    elif not args.skip_checkpoint:
        # Gate-only alignment leaves every checkpoint parameter untouched; the
        # small sidecar is sufficient and avoids duplicating a 6 GB model.
        checkpoint = args.checkpoint

    alpha = gate.alpha.detach().cpu()
    report = {
        "format": "starvla_pi_nuisance_alignment_report_v1",
        "arm": args.arm,
        "spec": spec,
        "nuisances": list(nuisances),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "alignment_gate": str(gate_path.resolve()) if spec["train_gate"] else None,
        "steps": args.steps,
        "underlying_training_steps_sampled": (
            args.steps * args.batch_size * args.microbatches_per_step
        ),
        "unique_underlying_training_steps_sampled": len(unique_training_indices),
        "image_action_presentations": (
            args.steps * args.batch_size * args.microbatches_per_step * 2
        ),
        "batch_size_in_state_matched_pairs": args.batch_size,
        "microbatches_per_optimizer_step": args.microbatches_per_step,
        "effective_pairs_per_optimizer_step": (
            args.batch_size * args.microbatches_per_step
        ),
        "training_step_pool": int(len(train_indices)),
        "heldout_trajectory_step_pool": int(len(validation_indices)),
        "basis": basis_report,
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "validation_flow_loss_before": before,
        "validation_flow_loss_after": after,
        "relative_change": {
            key: float(after[key] / before[key] - 1.0) if before[key] else math.nan
            for key in after
        },
        "gate_alpha": {
            "mean": float(alpha.mean()),
            "max": float(alpha.max()),
            "nonzero": int((alpha > 1.0e-6).sum()),
            "by_nuisance_mean": {
                name: float(alpha[:, index].mean())
                for index, name in enumerate(nuisances)
            },
            "by_layer": alpha.tolist(),
        },
        "conditional_gate": {
            "enabled": bool(spec["conditional"]),
            "weight_abs_mean": float(gate.condition_weight.detach().abs().mean()),
            "weight_abs_max": float(gate.condition_weight.detach().abs().max()),
            "bias_mean": float(gate.condition_bias.detach().mean()),
            "bias_abs_max": float(gate.condition_bias.detach().abs().max()),
        },
        "history": history,
        "objective": {
            "pair_weight": args.pair_weight,
            "anchor_weight": args.anchor_weight,
            "gate_penalty": args.gate_penalty,
            "condition_penalty": args.condition_penalty,
            "final_lr": args.final_lr,
            "gate_lr": args.gate_lr,
        },
        "seed": args.seed,
        "warning": (
            "Bases and updates use synthetic training-side pairs only. Official LIBERO+ "
            "must remain evaluation-only, and clean recovery gates any robustness claim."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    del model, gate
    gc.collect()
    torch.cuda.empty_cache()
    print(f"completed {args.arm}; checkpoint={checkpoint}; gate={report['alignment_gate']}")


if __name__ == "__main__":
    main()

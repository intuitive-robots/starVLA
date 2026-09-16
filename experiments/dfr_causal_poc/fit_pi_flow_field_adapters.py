#!/usr/bin/env python3
"""Fit tiny PI projection/cross-attention adapters by clean-flow distillation.

The physical state, instruction, initial flow noise, Action-DiT input x_t, and
time t are identical within every clean/appearance-perturbed comparison.  The
frozen clean policy supplies a velocity-field target along its own four-step
Euler path.  Only the selected zero-initialized adapter is optimized; the VLM,
PI projectors, Action DiT, and action decoder remain frozen.

This is an offline mechanism diagnostic.  It does not use test pairs and does
not claim closed-loop success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from starVLA.model.framework.base_framework import baseframework


ARMS = (
    "projection_film",
    "projection_lora",
    "xattn_qk_lora",
    "xattn_vout_lora",
    "xattn_head_gate",
)
CATEGORIES = (
    "Background Textures",
    "Camera Viewpoints",
    "Light Conditions",
    "Sensor Noise",
)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    for row in rows:
        if row.get("format") != "starvla_successful_counterfactual_pair_v1":
            raise RuntimeError(f"unexpected manifest format in {path}")
        if float(row["state_max_abs_diff"]) > 1.0e-8:
            raise RuntimeError(f"non-matched state in {row['pair_id']}")
        if row["category"] not in CATEGORIES:
            raise RuntimeError(f"unexpected category {row['category']!r}")
    return rows


def _balanced_rows(rows: list[dict], split: str, count: int, seed: int) -> list[dict]:
    if count % len(CATEGORIES):
        raise ValueError("sample count must be divisible by four categories")
    by_category = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_category[row["category"]].append(row)
    per_category = count // len(CATEGORIES)
    chosen = []
    for category in CATEGORIES:
        candidates = by_category[category]
        if len(candidates) < per_category:
            raise RuntimeError(
                f"requested {per_category} {split} rows for {category}, have {len(candidates)}"
            )
        rng = random.Random(_stable_seed(seed, split, category))
        rng.shuffle(candidates)
        chosen.extend(candidates[:per_category])
    random.Random(_stable_seed(seed, split, "interleave")).shuffle(chosen)
    return chosen


def _balanced_batches(rows: list[dict], seed: int) -> Iterable[list[dict]]:
    """Yield one unique row from every nuisance in each batch."""
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category in CATEGORIES:
        random.Random(_stable_seed(seed, "batch", category)).shuffle(by_category[category])
    batch_count = min(len(by_category[category]) for category in CATEGORIES)
    for index in range(batch_count):
        batch = [by_category[category][index] for category in CATEGORIES]
        random.Random(_stable_seed(seed, "within", index)).shuffle(batch)
        yield batch


def _images(row: dict, perturbed: bool) -> list[Image.Image]:
    prefix = "variant" if perturbed else "clean"
    result = []
    for view in ("external", "wrist"):
        with Image.open(row[f"{prefix}_{view}"]) as image:
            result.append(image.convert("RGB").copy())
    return result


def _load_action_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    action = payload["action"] if "action" in payload else payload[next(iter(payload))]["action"]
    return np.asarray(action["min"], dtype=np.float32), np.asarray(action["max"], dtype=np.float32)


def _successful_actions(rows: list[dict], bounds, device, dtype) -> torch.Tensor:
    minimum, maximum = bounds
    actions = np.stack([np.asarray(row["action"], dtype=np.float32) for row in rows])
    result = actions.copy()
    varying = minimum[:6] != maximum[:6]
    result[..., :6][..., varying] = (
        2.0
        * (result[..., :6][..., varying] - minimum[:6][varying])
        / (maximum[:6][varying] - minimum[:6][varying])
        - 1.0
    )
    result[..., 6] = (result[..., 6] < 0.49).astype(np.float32)
    return torch.as_tensor(result, device=device, dtype=dtype)


@torch.no_grad()
def _encode_raw(model, rows: list[dict]):
    clean_images = [_images(row, False) for row in rows]
    perturbed_images = [_images(row, True) for row in rows]
    languages = [row["language"] for row in rows]
    inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=clean_images + perturbed_images,
        instructions=languages + languages,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    raw = [state.detach() for state in outputs.hidden_states[-model.num_action_dit_layers :]]
    valid = getattr(model.qwen_vl_interface, "_last_encoder_attention_mask", None)
    if valid is None:
        valid = inputs.get("attention_mask")
    if valid is None:
        valid = torch.ones(raw[-1].shape[:2], device=raw[-1].device, dtype=torch.bool)
    valid = valid[:, : raw[-1].shape[1]].to(device=raw[-1].device, dtype=torch.bool)
    action_valid = model._action_encoder_valid_mask(valid)
    bias = model._dit_attention_bias(action_valid, raw[-1].dtype).detach()
    count = len(rows)
    return (
        [state[:count] for state in raw],
        [state[count:] for state in raw],
        bias[:count],
        bias[count:],
    )


class LowRankDelta(nn.Module):
    """Zero-output low-rank update B(A(x)); only B receives the first gradient."""

    def __init__(self, input_dim: int, output_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(input_dim))
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0 / rank

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden)) * self.scale


class PIFlowAdapter(nn.Module):
    """Experiment-local adapters; base checkpoint modules are never replaced."""

    def __init__(self, model, arm: str, rank: int):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm = arm
        self.rank = rank
        self.active = False
        head = model.action_model
        # This study is specifically about the real alternating PI routing.
        # Refuse legacy checkpoints rather than silently changing the question.
        if head.layerwise_attention_layout != "alternating":
            raise RuntimeError(
                "PI flow-field study requires layerwise_attention_layout='alternating'; "
                f"loaded {head.layerwise_attention_layout!r}"
            )
        if not head.model._interleave_self_attention:
            raise RuntimeError("alternating PI study requires interleave_self_attention=true")
        self.cross_layers = list(range(0, len(head.model.transformer_blocks), 2))
        if not self.cross_layers:
            raise RuntimeError("PI model has no cross-attention layers")
        hidden_dim = int(model.action_dit_hidden_dim)
        raw_dim = int(model.qwen_vl_interface.model.config.hidden_size)
        self.updates = nn.ModuleDict()
        self.film_scale = nn.ParameterDict()
        self.film_shift = nn.ParameterDict()
        self.head_gates = nn.ParameterDict()
        self._handles = []

        if arm == "projection_film":
            for layer in self.cross_layers:
                self.film_scale[str(layer)] = nn.Parameter(torch.zeros(hidden_dim))
                self.film_shift[str(layer)] = nn.Parameter(torch.zeros(hidden_dim))
        elif arm == "projection_lora":
            for layer in self.cross_layers:
                self.updates[f"projection_{layer}"] = LowRankDelta(raw_dim, hidden_dim, rank)
        elif arm in {"xattn_qk_lora", "xattn_vout_lora"}:
            targets = ("q", "k") if arm == "xattn_qk_lora" else ("v", "out")
            for layer in self.cross_layers:
                attention = head.model.transformer_blocks[layer].attn1
                for target in targets:
                    linear = attention.to_out[0] if target == "out" else getattr(attention, f"to_{target}")
                    update = LowRankDelta(linear.in_features, linear.out_features, rank)
                    key = f"layer_{layer}_{target}"
                    self.updates[key] = update

                    def hook(_module, inputs, output, *, adapter=self, update_key=key):
                        if not adapter.active:
                            return output
                        return output + adapter.updates[update_key](inputs[0])

                    self._handles.append(linear.register_forward_hook(hook))
        elif arm == "xattn_head_gate":
            for layer in self.cross_layers:
                attention = head.model.transformer_blocks[layer].attn1
                heads = int(attention.heads)
                if int(attention.inner_dim) % heads:
                    raise RuntimeError("attention inner dimension is not divisible by heads")
                key = f"layer_{layer}"
                self.head_gates[key] = nn.Parameter(torch.zeros(heads))
                output_linear = attention.to_out[0]

                def pre_hook(_module, inputs, *, adapter=self, gate_key=key, num_heads=heads):
                    if not adapter.active:
                        return None
                    hidden = inputs[0]
                    head_dim = hidden.shape[-1] // num_heads
                    gate = adapter.head_gates[gate_key].to(dtype=hidden.dtype)
                    shaped = hidden.unflatten(-1, (num_heads, head_dim))
                    shaped = shaped * (1.0 + gate.view(*([1] * (shaped.ndim - 2)), num_heads, 1))
                    return (shaped.flatten(-2), *inputs[1:])

                self._handles.append(output_linear.register_forward_pre_hook(pre_hook))

    @contextmanager
    def enabled(self, value: bool):
        previous = self.active
        self.active = value
        try:
            yield
        finally:
            self.active = previous

    def project(self, model, raw: list[torch.Tensor], adapted: bool) -> list[torch.Tensor]:
        output = []
        for layer, (projector, hidden) in enumerate(zip(model.project_layers, raw)):
            base = projector(hidden)
            if adapted and self.arm == "projection_film" and layer in self.cross_layers:
                scale = self.film_scale[str(layer)].to(dtype=base.dtype)
                shift = self.film_shift[str(layer)].to(dtype=base.dtype)
                base = base * (1.0 + scale) + shift
            elif adapted and self.arm == "projection_lora" and layer in self.cross_layers:
                projector_input = projector[0](hidden) if isinstance(projector, nn.Sequential) else hidden
                base = base + self.updates[f"projection_{layer}"](projector_input)
            output.append(base)
        return output

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _initial_noise(rows: list[dict], head, seed: int, dtype, device):
    values = []
    for row in rows:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(seed, "flow", row["pair_id"]))
        values.append(torch.randn(head.action_horizon, head.action_dim, generator=generator))
    return torch.stack(values).to(device=device, dtype=dtype)


def _velocity(head, memory, bias, actions, step: int):
    batch_size = actions.shape[0]
    steps = int(head.num_inference_timesteps)
    continuous = step / float(steps)
    discrete = int(continuous * head.num_timestep_buckets)
    timestep = torch.full((batch_size,), discrete, device=actions.device, dtype=torch.long)
    action_features = head.action_encoder(actions, timestep)
    if head.config.add_pos_embed:
        positions = torch.arange(action_features.shape[1], device=actions.device)
        action_features = action_features + head.position_embedding(positions).unsqueeze(0)
    tokens = head._assemble_action_tokens(action_features, None)
    hidden = head.model(
        hidden_states=tokens,
        encoder_hidden_states=memory,
        timestep=timestep,
        encoder_attention_mask=bias,
        return_pre_output=True,
        force_layerwise_all_cross=head.layerwise_attention_layout == "legacy_all_cross",
        extra_conditioning=None,
        cross_attention_row_mask=None,
    )
    return head.action_decoder(hidden)[:, -head.action_horizon :]


@torch.no_grad()
def _teacher_path(model, adapter, raw_clean, bias_clean, rows, bounds, seed):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        memory = adapter.project(model, raw_clean, adapted=False)
    head = model.action_model
    noise = _initial_noise(rows, head, seed, memory[0].dtype, memory[0].device)
    successful = _successful_actions(rows, bounds, memory[0].device, memory[0].dtype)
    states, velocities = [], []
    steps = int(head.num_inference_timesteps)
    dt = 1.0 / steps
    # Evaluate the field on the straight conditional-flow bridge to the exact
    # action chunk executed in a successful clean rollout. The original noise
    # used during data capture was not stored, so we do not pretend to replay
    # that hidden trajectory; instead each pair gets a deterministic new noise
    # endpoint and the same bridge in both visual domains.
    with adapter.enabled(False), torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range(steps):
            time = step / float(steps)
            state = (1.0 - time) * noise + time * successful
            states.append(state.detach().clone())
            velocity = _velocity(head, memory, bias_clean, state, step)
            velocities.append(velocity.detach().clone())
        # Separately retain the clean policy's own Euler endpoint under the
        # same new noise. Endpoint-to-clean measures invariance; endpoint-to-
        # successful-action exposes whether that stochastic clean rollout is
        # itself close to the action sequence known to have succeeded.
        clean_actions = noise.clone()
        for step in range(steps):
            velocity = _velocity(head, memory, bias_clean, clean_actions, step)
            clean_actions = clean_actions + dt * velocity
    ideal_velocity = successful - noise
    return (
        states,
        velocities,
        clean_actions.detach(),
        successful.detach(),
        noise.detach(),
        ideal_velocity.detach(),
    )


def _adapted_memories(model, adapter, raw_clean, raw_perturbed):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clean = adapter.project(model, raw_clean, adapted=True)
        perturbed = adapter.project(model, raw_perturbed, adapted=True)
    return clean, perturbed


def _student_pair_velocity(
    model, adapter, raw_clean, raw_perturbed, clean_bias, perturbed_bias, state, step
):
    clean_memory, perturbed_memory = _adapted_memories(
        model, adapter, raw_clean, raw_perturbed
    )
    memory = [torch.cat((clean, perturbed), dim=0) for clean, perturbed in zip(clean_memory, perturbed_memory)]
    bias = torch.cat((clean_bias, perturbed_bias), dim=0)
    action = torch.cat((state, state), dim=0)
    with adapter.enabled(True), torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = _velocity(model.action_model, memory, bias, action, step)
    return predicted.chunk(2, dim=0)


def _base_perturbed_velocity(model, adapter, raw_perturbed, bias, state, step):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        memory = adapter.project(model, raw_perturbed, adapted=False)
        with adapter.enabled(False):
            return _velocity(model.action_model, memory, bias, state, step)


def _relative_mse(prediction, target):
    mse = F.mse_loss(prediction.float(), target.float())
    scale = target.float().square().mean().clamp_min(1.0e-6)
    return mse, mse / scale


def _cosine(prediction, target):
    return F.cosine_similarity(
        prediction.float().flatten(1), target.float().flatten(1), dim=1
    ).mean()


@torch.inference_mode()
def _evaluate(model, adapter, rows: list[dict], bounds, batch_size: int, seed: int):
    totals = defaultdict(float)
    category_totals = defaultdict(lambda: defaultdict(float))
    records = []
    count = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        raw_clean, raw_perturbed, clean_bias, perturbed_bias = _encode_raw(model, batch)
        path, teachers, teacher_endpoint, successful, initial_noise, ideal_velocity = _teacher_path(
            model, adapter, raw_clean, clean_bias, batch, bounds, seed
        )
        base_errors, adapted_errors, adapted_teacher_errors, clean_errors = [], [], [], []
        base_ideal_errors, adapted_ideal_errors = [], []
        base_cosines, adapted_cosines = [], []
        base_actions = initial_noise.clone()
        adapted_actions = initial_noise.clone()
        dt = 1.0 / len(path)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base_memory = adapter.project(model, raw_perturbed, adapted=False)
            adapted_memory = adapter.project(model, raw_perturbed, adapted=True)
        for step, (state, teacher) in enumerate(zip(path, teachers)):
            with adapter.enabled(False), torch.autocast("cuda", dtype=torch.bfloat16):
                base_at_teacher = _velocity(
                    model.action_model, base_memory, perturbed_bias, state, step
                )
            clean_at_teacher, adapted_at_teacher = _student_pair_velocity(
                model,
                adapter,
                raw_clean,
                raw_perturbed,
                clean_bias,
                perturbed_bias,
                state,
                step,
            )
            base_errors.append((base_at_teacher.float() - teacher.float()).square().mean(dim=(1, 2)))
            adapted_errors.append((adapted_at_teacher.float() - clean_at_teacher.float()).square().mean(dim=(1, 2)))
            adapted_teacher_errors.append((adapted_at_teacher.float() - teacher.float()).square().mean(dim=(1, 2)))
            clean_errors.append((clean_at_teacher.float() - teacher.float()).square().mean(dim=(1, 2)))
            base_ideal_errors.append((base_at_teacher.float() - ideal_velocity.float()).square().mean(dim=(1, 2)))
            adapted_ideal_errors.append((adapted_at_teacher.float() - ideal_velocity.float()).square().mean(dim=(1, 2)))
            base_cosines.append(F.cosine_similarity(base_at_teacher.float().flatten(1), teacher.float().flatten(1), dim=1))
            adapted_cosines.append(F.cosine_similarity(adapted_at_teacher.float().flatten(1), clean_at_teacher.float().flatten(1), dim=1))
            with adapter.enabled(False), torch.autocast("cuda", dtype=torch.bfloat16):
                base_velocity = _velocity(
                    model.action_model, base_memory, perturbed_bias, base_actions, step
                )
            with adapter.enabled(True), torch.autocast("cuda", dtype=torch.bfloat16):
                adapted_velocity = _velocity(
                    model.action_model, adapted_memory, perturbed_bias, adapted_actions, step
                )
            base_actions = base_actions + dt * base_velocity
            adapted_actions = adapted_actions + dt * adapted_velocity

        teacher_energy = torch.stack(
            [velocity.float().square().mean(dim=(1, 2)) for velocity in teachers], dim=1
        ).mean(dim=1).clamp_min(1.0e-6)
        values = {
            "base_field_mse": torch.stack(base_errors, 1).mean(1),
            "adapted_field_mse": torch.stack(adapted_errors, 1).mean(1),
            "adapted_to_frozen_teacher_mse": torch.stack(adapted_teacher_errors, 1).mean(1),
            "clean_field_drift_mse": torch.stack(clean_errors, 1).mean(1),
            "base_successful_flow_mse": torch.stack(base_ideal_errors, 1).mean(1),
            "adapted_successful_flow_mse": torch.stack(adapted_ideal_errors, 1).mean(1),
            "base_field_cosine": torch.stack(base_cosines, 1).mean(1),
            "adapted_field_cosine": torch.stack(adapted_cosines, 1).mean(1),
            "base_endpoint_mse": (base_actions.float() - teacher_endpoint.float()).square().mean(dim=(1, 2)),
            "adapted_endpoint_mse": (adapted_actions.float() - teacher_endpoint.float()).square().mean(dim=(1, 2)),
            "clean_endpoint_to_successful_mse": (teacher_endpoint.float() - successful.float()).square().mean(dim=(1, 2)),
            "base_endpoint_to_successful_mse": (base_actions.float() - successful.float()).square().mean(dim=(1, 2)),
            "adapted_endpoint_to_successful_mse": (adapted_actions.float() - successful.float()).square().mean(dim=(1, 2)),
            "teacher_field_energy": teacher_energy,
        }
        for index, row in enumerate(batch):
            record = {
                "pair_id": row["pair_id"],
                "suite": row["suite"],
                "task_id": int(row["task_id"]),
                "category": row["category"],
            }
            for name, tensor in values.items():
                value = float(tensor[index])
                totals[name] += value
                category_totals[row["category"]][name] += value
                record[name] = value
            records.append(record)
            count += 1
        print(f"validation {start + len(batch)}/{len(rows)}", flush=True)

    def summarize(bucket, denominator):
        result = {key: value / denominator for key, value in bucket.items()}
        base = result["base_field_mse"]
        adapted = result["adapted_field_mse"]
        base_endpoint = result["base_endpoint_mse"]
        adapted_endpoint = result["adapted_endpoint_mse"]
        energy = max(result["teacher_field_energy"], 1.0e-12)
        result.update(
            field_mse_relative_change=adapted / base - 1.0,
            field_mse_improvement=1.0 - adapted / base,
            endpoint_mse_relative_change=adapted_endpoint / base_endpoint - 1.0,
            endpoint_mse_improvement=1.0 - adapted_endpoint / base_endpoint,
            clean_drift_over_teacher_energy=result["clean_field_drift_mse"] / energy,
            successful_endpoint_mse_improvement=(
                1.0
                - result["adapted_endpoint_to_successful_mse"]
                / result["base_endpoint_to_successful_mse"]
            ),
            successful_flow_mse_improvement=(
                1.0
                - result["adapted_successful_flow_mse"]
                / result["base_successful_flow_mse"]
            ),
        )
        return result

    per_category = {
        category: summarize(category_totals[category], len(rows) // len(CATEGORIES))
        for category in CATEGORIES
    }
    clusters = defaultdict(list)
    for record in records:
        clusters[(record["suite"], record["task_id"])].append(record)
    cluster_keys = sorted(clusters)
    rng = np.random.default_rng(seed + 991)
    samples = []
    for _ in range(5000):
        sampled = rng.choice(len(cluster_keys), size=len(cluster_keys), replace=True)
        selected = [record for index in sampled for record in clusters[cluster_keys[int(index)]]]
        base = np.mean([record["base_field_mse"] for record in selected])
        adapted = np.mean([record["adapted_field_mse"] for record in selected])
        samples.append(1.0 - adapted / base)
    interval = np.quantile(samples, (0.025, 0.975)).tolist()
    return {
        "overall": summarize(totals, count),
        "by_category": per_category,
        "task_cluster_bootstrap": {
            "replicates": 5000,
            "field_mse_improvement_95pct": interval,
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
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--pair-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
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
    model = baseframework.from_pretrained(
        str(args.checkpoint), is_inference=False
    ).cuda().eval().requires_grad_(False)
    if not hasattr(model, "project_layers"):
        raise RuntimeError("this experiment requires QwenPI_v3 with project_layers")
    adapter = PIFlowAdapter(model, args.arm, args.rank).cuda()
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    initial_norm = float(sum(parameter.detach().float().square().sum() for parameter in trainable).sqrt())
    history = []
    seen_pair_ids = set()
    seen_tasks = set()
    model.eval()
    adapter.train()
    batches = list(_balanced_batches(train_rows, args.seed))
    for batch_index, batch in enumerate(batches):
        raw_clean, raw_perturbed, clean_bias, perturbed_bias = _encode_raw(model, batch)
        path, teachers, _, _, _, ideal_velocity = _teacher_path(
            model, adapter, raw_clean, clean_bias, batch, bounds, args.seed
        )
        optimizer.zero_grad(set_to_none=True)
        step_perturbed = []
        step_clean = []
        step_consistency = []
        step_anchor = []
        for flow_step, (state, teacher) in enumerate(zip(path, teachers)):
            clean_prediction, perturbed_prediction = _student_pair_velocity(
                model,
                adapter,
                raw_clean,
                raw_perturbed,
                clean_bias,
                perturbed_bias,
                state,
                flow_step,
            )
            clean_mse, clean_relative = _relative_mse(clean_prediction, ideal_velocity)
            perturbed_mse, perturbed_relative = _relative_mse(
                perturbed_prediction, ideal_velocity
            )
            consistency_mse = F.mse_loss(
                perturbed_prediction.float(), clean_prediction.float()
            )
            flow_scale = ideal_velocity.float().square().mean().clamp_min(1.0e-6)
            consistency_relative = consistency_mse / flow_scale
            anchor_mse, anchor_relative = _relative_mse(clean_prediction, teacher)
            loss = (
                0.5 * (clean_relative + perturbed_relative)
                + args.pair_weight * consistency_relative
                + args.anchor_weight * anchor_relative
            )
            (loss / len(path)).backward()
            step_perturbed.append(float(perturbed_mse.detach()))
            step_clean.append(float(clean_mse.detach()))
            step_consistency.append(float(consistency_mse.detach()))
            step_anchor.append(float(anchor_mse.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        for row in batch:
            seen_pair_ids.add(row["pair_id"])
            seen_tasks.add((row["suite"], int(row["task_id"])))
        if batch_index == 0 or (batch_index + 1) % args.log_every == 0 or batch_index + 1 == len(batches):
            record = {
                "optimizer_step": batch_index + 1,
                "pairs_seen": len(seen_pair_ids),
                "perturbed_successful_flow_mse": float(np.mean(step_perturbed)),
                "clean_successful_flow_mse": float(np.mean(step_clean)),
                "pair_consistency_mse": float(np.mean(step_consistency)),
                "clean_frozen_anchor_mse": float(np.mean(step_anchor)),
                "grad_norm": float(grad_norm),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    adapter.eval()
    validation = _evaluate(
        model, adapter, validation_rows, bounds, args.eval_batch_size, args.seed + 1000
    )
    final_norm = float(sum(parameter.detach().float().square().sum() for parameter in trainable).sqrt())
    adapter_path = args.output_dir / "adapter.pt"
    torch.save(
        {
            "format": "starvla_pi_flow_field_adapter_v1",
            "arm": args.arm,
            "rank": args.rank,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "cross_layers": adapter.cross_layers,
            "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        },
        adapter_path,
    )
    overall = validation["overall"]
    report = {
        "format": "starvla_pi_flow_field_adapter_report_v1",
        "status": "complete",
        "scope": "offline validation diagnostic; test split untouched; no success claim",
        "arm": args.arm,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "dataset_statistics": str(args.stats.resolve()),
        "adapter": str(adapter_path.resolve()),
        "rank": args.rank,
        "cross_attention_layers": adapter.cross_layers,
        "num_cross_attention_layers": len(adapter.cross_layers),
        "layerwise_attention_layout": model.action_model.layerwise_attention_layout,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_parameter_l2": initial_norm,
        "final_parameter_l2": final_norm,
        "train": {
            "pairs": len(seen_pair_ids),
            "tasks": len(seen_tasks),
            "category_counts": dict(Counter(row["category"] for row in train_rows)),
            "flow_points": len(seen_pair_ids) * int(model.action_model.num_inference_timesteps),
            "action_token_flow_targets": len(seen_pair_ids)
            * int(model.action_model.num_inference_timesteps)
            * int(model.action_horizon),
            "optimizer_steps": len(batches),
            "pair_weight": args.pair_weight,
            "anchor_weight": args.anchor_weight,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "validation": {
            "pairs": len(validation_rows),
            "tasks": len({(row["suite"], int(row["task_id"])) for row in validation_rows}),
            "category_counts": dict(Counter(row["category"] for row in validation_rows)),
            **validation,
        },
        "screen": {
            "field_improvement_at_least_10pct": overall["field_mse_improvement"] >= 0.10,
            "endpoint_improvement_at_least_10pct": overall["endpoint_mse_improvement"] >= 0.10,
            "clean_drift_below_1pct_teacher_energy": overall["clean_drift_over_teacher_energy"] <= 0.01,
            "task_bootstrap_lower_bound_above_zero": validation[
                "task_cluster_bootstrap"
            ]["field_mse_improvement_95pct"][0]
            > 0.0,
            "nonnegative_in_at_least_3_categories": sum(
                value["field_mse_improvement"] >= 0.0
                for value in validation["by_category"].values()
            )
            >= 3,
        },
        "history": history,
        "seed": args.seed,
    }
    report["screen"]["passes_all"] = all(report["screen"].values())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    adapter.close()
    print(json.dumps({"arm": args.arm, "screen": report["screen"], "overall": overall}), flush=True)


if __name__ == "__main__":
    main()

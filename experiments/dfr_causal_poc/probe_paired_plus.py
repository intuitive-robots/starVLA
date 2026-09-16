"""Probe and causally intervene on P-causal's frozen LIBERO representations.

The experiment has three deliberately separate questions:

1. Is an appearance perturbation linearly decodable on held-out base tasks?
2. Which low-rank representation directions consistently change between a
   state-matched clean/perturbed pair?
3. Does removing those directions make the perturbed policy action more like
   the clean policy action, beyond an equal-rank random deletion control?

This is a diagnostic, not DFR training.  It requires no action labels and cannot
show that either predicted action is correct; it only tests the causal shortcut
signature needed to motivate a subsequent balanced action-head refit.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STARVLA = Path(__file__).resolve().parents[2]
if str(STARVLA) not in sys.path:
    sys.path.insert(0, str(STARVLA))

from starVLA.model.framework.base_framework import baseframework  # noqa: E402

CATEGORIES = ("Background Textures", "Light Conditions", "Sensor Noise")


def _read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    required = {
        "pair_id",
        "base_task",
        "category",
        "canonical_language",
        "clean_external",
        "clean_wrist",
        "variant_external",
        "variant_wrist",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise RuntimeError(f"manifest row {index} lacks {sorted(missing)}")
        if row["category"] not in CATEGORIES:
            raise RuntimeError(f"unexpected category {row['category']!r}")
        if float(row.get("state_max_abs_diff", math.inf)) > 1e-5:
            raise RuntimeError(f"row {index} is not a verified state-matched pair")
    return rows


def _task_split(rows: list[dict], test_tasks: int, seed: int) -> tuple[set[str], set[str]]:
    tasks = sorted({str(row["base_task"]) for row in rows})
    if len(tasks) < 3:
        raise RuntimeError("need at least three base tasks for a task-held-out probe")
    if not 1 <= test_tasks < len(tasks):
        raise ValueError(f"--test-tasks must be in [1, {len(tasks) - 1}]")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(tasks)[rng.permutation(len(tasks))])
    test = set(shuffled[:test_tasks])
    train = set(shuffled[test_tasks:])
    for category in CATEGORIES:
        for name, split in (("train", train), ("test", test)):
            if not any(row["category"] == category and row["base_task"] in split for row in rows):
                raise RuntimeError(f"{name} split lacks category {category}")
    return train, test


def _load_views(row: dict, domain: str) -> list[Image.Image]:
    result = []
    for view in ("external", "wrist"):
        path = Path(row[f"{domain}_{view}"])
        with Image.open(path) as image:
            result.append(image.convert("RGB").copy())
    return result


@torch.inference_mode()
def _encode_batch(model, rows: list[dict]) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    images = [_load_views(row, "clean") for row in rows] + [_load_views(row, "variant") for row in rows]
    languages = [str(row["canonical_language"]) for row in rows] * 2
    states, _, attention_bias, _, shared_z = model._encode_vl_hidden_states(images, languages)
    if shared_z is not None:
        raise RuntimeError("this PoC expects P-causal without shared-z conditioning")
    # Encoder-decoder interfaces retain a private validity mask, but the
    # decoder-only P interface does not. The framework's returned DiT bias is
    # the authoritative equivalent: 0 for a valid token and -10,000 for pad.
    private_valid = getattr(model.qwen_vl_interface, "_last_encoder_attention_mask", None)
    valid = private_valid.bool() if private_valid is not None else attention_bias[:, 0, :] == 0
    if valid.ndim != 2 or valid.shape[0] != len(images):
        raise RuntimeError(f"invalid token mask {tuple(valid.shape)} for image batch {len(images)}")
    pooled = []
    for state in states:
        mask = valid[:, : state.shape[1]]
        pooled.append((state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))
    pooled_tensor = torch.stack(pooled, dim=1).float().cpu()
    count = len(rows)
    clean_states = [state[:count] for state in states]
    variant_states = [state[count:] for state in states]
    return clean_states, variant_states, attention_bias, pooled_tensor


@torch.inference_mode()
def _encode_pooled(model, rows: list[dict], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    clean_chunks, variant_chunks = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        clean_states, variant_states, _, pooled = _encode_batch(model, batch)
        count = len(batch)
        clean_chunks.append(pooled[:count])
        variant_chunks.append(pooled[count:])
        del clean_states, variant_states, pooled
        print(f"encoded pooled activations {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return torch.cat(clean_chunks).numpy(), torch.cat(variant_chunks).numpy()


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = score[y == 1]
    negative = score[y == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    # Pairwise form handles ties explicitly and is tiny for this PoC.
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def _ridge_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha_multiplier: float,
) -> tuple[np.ndarray, float]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    train = (x_train - mean) / std
    test = (x_test - mean) / std
    target_mean = float(y_train.mean())
    target = y_train - target_mean
    scale = max(float(np.trace(train @ train.T)) / max(len(train), 1), 1e-8)
    alpha = alpha_multiplier * scale
    dual = np.linalg.solve(train @ train.T + alpha * np.eye(len(train)), target)
    weight = train.T @ dual
    score = test @ weight + target_mean
    return score, alpha


def _ridge_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_groups: np.ndarray | None = None,
    fixed_multiplier: float | None = None,
) -> dict[str, float]:
    multipliers = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv_auc = None
    if fixed_multiplier is None:
        if train_groups is None or len(np.unique(train_groups)) < 2:
            raise ValueError("task-grouped cross-validation needs at least two train groups")
        candidates = []
        for multiplier in multipliers:
            fold_auc = []
            for group in np.unique(train_groups):
                heldout = train_groups == group
                score, _ = _ridge_score(x_train[~heldout], y_train[~heldout], x_train[heldout], multiplier)
                fold_auc.append(_auc(y_train[heldout], score))
            candidates.append((float(np.mean(fold_auc)), multiplier))
        # Prefer stronger regularization on exact CV ties.
        cv_auc, fixed_multiplier = max(candidates)
    score, alpha = _ridge_score(x_train, y_train, x_test, fixed_multiplier)
    prediction = (score >= 0.5).astype(np.int64)
    return {
        "accuracy": float((prediction == y_test).mean()),
        "auc": _auc(y_test, score),
        "ridge_alpha": alpha,
        "ridge_alpha_multiplier": fixed_multiplier,
        "train_task_cv_auc": cv_auc,
    }


def _probe_metrics(
    clean: np.ndarray,
    variant: np.ndarray,
    rows: list[dict],
    train_tasks: set[str],
    test_tasks: set[str],
    shuffle_repeats: int,
    seed: int,
) -> dict:
    result = {}
    rng = np.random.default_rng(seed)
    for category in CATEGORIES:
        category_indices = np.asarray([index for index, row in enumerate(rows) if row["category"] == category])
        train_pairs = np.asarray([index for index in category_indices if rows[index]["base_task"] in train_tasks])
        test_pairs = np.asarray([index for index in category_indices if rows[index]["base_task"] in test_tasks])
        x_train = np.concatenate([clean[train_pairs], variant[train_pairs]], axis=0)
        x_test = np.concatenate([clean[test_pairs], variant[test_pairs]], axis=0)
        y_train = np.concatenate([np.zeros(len(train_pairs), dtype=np.int64), np.ones(len(train_pairs), dtype=np.int64)])
        y_test = np.concatenate([np.zeros(len(test_pairs), dtype=np.int64), np.ones(len(test_pairs), dtype=np.int64)])
        train_groups = np.concatenate(
            [
                np.asarray([rows[index]["base_task"] for index in train_pairs]),
                np.asarray([rows[index]["base_task"] for index in train_pairs]),
            ]
        )
        layers = []
        for layer in range(clean.shape[1]):
            score = _ridge_binary(
                x_train[:, layer],
                y_train,
                x_test[:, layer],
                y_test,
                train_groups=train_groups,
            )
            shuffled = []
            for _ in range(shuffle_repeats):
                shuffled_y = rng.permutation(y_train)
                shuffled.append(
                    _ridge_binary(
                        x_train[:, layer],
                        shuffled_y,
                        x_test[:, layer],
                        y_test,
                        fixed_multiplier=float(score["ridge_alpha_multiplier"]),
                    )["auc"]
                )
            layers.append(
                {
                    "layer": layer,
                    **score,
                    "shuffled_auc_mean": float(np.mean(shuffled)),
                    "shuffled_auc_std": float(np.std(shuffled, ddof=1)) if len(shuffled) > 1 else 0.0,
                }
            )
        result[category] = {
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
            "layers": layers,
        }
    return result


def _fit_one_basis(
    clean: np.ndarray, variant: np.ndarray, indices: np.ndarray, max_rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    mean = clean[indices].mean(axis=0)
    delta = variant[indices] - clean[indices]
    _, singular, right = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    rank = min(max_rank, right.shape[0])
    basis = right[:rank].T
    energy = np.square(singular)
    total = max(float(energy.sum()), 1e-12)
    explained = {str(k): float(energy[: min(k, len(energy))].sum() / total) for k in (1, 2, 4, 8, 16, 32) if k <= rank}
    return (
        mean.astype(np.float32),
        basis.astype(np.float32),
        singular[:rank].astype(np.float32),
        explained,
    )


def _fit_bases(
    clean: np.ndarray,
    variant: np.ndarray,
    rows: list[dict],
    train_tasks: set[str],
    max_rank: int,
) -> tuple[dict, dict]:
    modes = ("joint", *CATEGORIES)
    payload = {
        "format": "starvla_libero_plus_paired_subspace_v1",
        "space": "projected",
        "num_layers": int(clean.shape[1]),
        "hidden_dim": int(clean.shape[2]),
        "modes": {},
    }
    metrics = {}
    for mode in modes:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["base_task"] in train_tasks and (mode == "joint" or row["category"] == mode)
            ]
        )
        mode_payload, mode_metrics = {}, []
        for layer in range(clean.shape[1]):
            mean, basis, singular, explained = _fit_one_basis(clean[:, layer], variant[:, layer], indices, max_rank)
            mode_payload[str(layer)] = {
                "mean": torch.from_numpy(mean),
                "basis": torch.from_numpy(basis),
                "singular_values": torch.from_numpy(singular),
            }
            mode_metrics.append({"layer": layer, "explained_delta_energy": explained})
        payload["modes"][mode] = mode_payload
        metrics[mode] = {"train_pairs": len(indices), "layers": mode_metrics}
    return payload, metrics


def _random_payload(payload: dict, seed: int) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    output = {"modes": {}}
    for mode, layers in payload["modes"].items():
        output["modes"][mode] = {}
        for layer, entry in layers.items():
            dimension, rank = entry["basis"].shape
            random = torch.randn(dimension, rank, generator=generator)
            basis, _ = torch.linalg.qr(random, mode="reduced")
            output["modes"][mode][layer] = {
                "mean": entry["mean"],
                "basis": basis,
            }
    return output


def _project(states: list[torch.Tensor], payload: dict, mode: str, rank: int) -> list[torch.Tensor]:
    output = list(states)
    for layer_text, entry in payload["modes"][mode].items():
        layer = int(layer_text)
        hidden = output[layer]
        mean = entry["mean"].to(hidden.device, hidden.dtype)
        basis = entry["basis"][:, :rank].to(hidden.device, hidden.dtype)
        if basis.shape[1] != rank:
            raise ValueError(f"{mode}/layer {layer} supports only rank {basis.shape[1]}")
        centered = hidden - mean.view(1, 1, -1)
        output[layer] = hidden - (centered @ basis) @ basis.T
    return output


def _slice_bias(bias: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    return bias[start:stop]


@torch.inference_mode()
def _predict(model, states: list[torch.Tensor], bias: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Checkpoint modules load in FP32 while Qwen's projected memory is BF16.
    # Match the established PI diagnostic path and ordinary mixed-precision
    # rollout by autocasting action-head linear operations to BF16.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.action_model.predict_action(
            states,
            None,
            encoder_attention_mask=bias,
            z_conditioning=None,
            encoder_memory_keep=None,
        ).float()


def _add_sse(bucket: dict, name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    difference = (left - right).double()
    bucket[name] += float(difference.square().sum().item())
    bucket[f"{name}_count"] += int(difference.numel())


def _new_bucket() -> defaultdict[str, float]:
    return defaultdict(float)


def _finalize_action_metrics(accumulator: dict) -> dict:
    output = {}
    for category, category_data in accumulator.items():
        baseline = category_data["baseline"]
        baseline_sse = baseline["variant_to_clean"]
        result = {
            "pairs": int(baseline["pairs"]),
            "baseline_variant_to_clean_mse": baseline_sse / max(baseline["variant_to_clean_count"], 1),
            "baseline_relative_rmse": math.sqrt(baseline_sse / max(baseline["clean_energy"], 1e-12)),
            "interventions": {},
        }
        for key, values in category_data.items():
            if key == "baseline":
                continue
            variant_sse = values["variant_to_clean"]
            clean_sse = values["clean_damage"]
            result["interventions"][key] = {
                "variant_to_clean_mse": variant_sse / max(values["variant_to_clean_count"], 1),
                "recovery_fraction": 1.0 - variant_sse / max(baseline_sse, 1e-12),
                "clean_damage_mse": clean_sse / max(values["clean_damage_count"], 1),
                "clean_damage_over_baseline": clean_sse / max(baseline_sse, 1e-12),
            }
        output[category] = result
    return output


@torch.inference_mode()
def _action_sweep(
    model,
    rows: list[dict],
    test_tasks: set[str],
    payload: dict,
    random_payload: dict,
    ranks: list[int],
    batch_size: int,
    seed: int,
) -> dict:
    accumulator: dict[str, dict] = defaultdict(dict)
    total_done = 0
    total_heldout = sum(row["base_task"] in test_tasks for row in rows)
    for category in CATEGORIES:
        heldout = [row for row in rows if row["base_task"] in test_tasks and row["category"] == category]
        for start in range(0, len(heldout), batch_size):
            batch = heldout[start : start + batch_size]
            clean_states, variant_states, bias, _ = _encode_batch(model, batch)
            count = len(batch)
            clean_bias = _slice_bias(bias, 0, count)
            variant_bias = _slice_bias(bias, count, 2 * count)
            # Each intervention below uses the same batch shape and seed. The
            # flow sampler therefore starts from bit-identical diffusion noise.
            action_seed = seed + total_done
            clean_action = _predict(model, clean_states, clean_bias, action_seed)
            variant_action = _predict(model, variant_states, variant_bias, action_seed)

            baseline = accumulator[category].setdefault("baseline", _new_bucket())
            _add_sse(baseline, "variant_to_clean", variant_action, clean_action)
            baseline["clean_energy"] += float(clean_action.double().square().sum().item())
            baseline["pairs"] += count

            for mode in ("joint", "category"):
                for rank in ranks:
                    basis_mode = "joint" if mode == "joint" else category
                    for control, source in (("learned", payload), ("random", random_payload)):
                        erased_clean = _predict(
                            model,
                            _project(clean_states, source, basis_mode, rank),
                            clean_bias,
                            action_seed,
                        )
                        erased_variant = _predict(
                            model,
                            _project(variant_states, source, basis_mode, rank),
                            variant_bias,
                            action_seed,
                        )
                        key = f"{mode}/rank_{rank}/{control}"
                        bucket = accumulator[category].setdefault(key, _new_bucket())
                        _add_sse(bucket, "variant_to_clean", erased_variant, clean_action)
                        _add_sse(bucket, "clean_damage", erased_clean, clean_action)
            del clean_states, variant_states, clean_action, variant_action, bias
            total_done += count
            print(f"action sweep {total_done}/{total_heldout}", flush=True)
    return _finalize_action_metrics(accumulator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--test-tasks", type=int, default=3)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--ranks", default="1,4,8,16")
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--skip-actions", action="store_true")
    args = parser.parse_args()

    rows = _read_manifest(args.manifest)
    train_tasks, test_tasks = _task_split(rows, args.test_tasks, args.seed)
    ranks = sorted({int(value) for value in args.ranks.split(",") if value.strip()})
    if not ranks or min(ranks) < 1 or max(ranks) > args.max_rank:
        raise ValueError("--ranks must be positive and no larger than --max-rank")

    model = baseframework.from_pretrained(str(args.checkpoint), is_inference=False).cuda().eval()
    clean, variant = _encode_pooled(model, rows, args.batch_size)
    if clean.shape != variant.shape or clean.ndim != 3:
        raise RuntimeError(f"unexpected pooled activation shapes {clean.shape}, {variant.shape}")

    probes = _probe_metrics(
        clean,
        variant,
        rows,
        train_tasks,
        test_tasks,
        args.shuffle_repeats,
        args.seed,
    )
    payload, basis_metrics = _fit_bases(clean, variant, rows, train_tasks, args.max_rank)
    payload.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest": str(args.manifest.resolve()),
            "train_tasks": sorted(train_tasks),
            "test_tasks": sorted(test_tasks),
            "max_rank": args.max_rank,
        }
    )
    random_payload = _random_payload(payload, args.seed + 991)
    action_metrics = None
    if not args.skip_actions:
        action_metrics = _action_sweep(
            model,
            rows,
            test_tasks,
            payload,
            random_payload,
            ranks,
            args.batch_size,
            args.seed + 10000,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "pooled_activations.npz",
        clean=clean,
        variant=variant,
        pair_id=np.asarray([row["pair_id"] for row in rows]),
        base_task=np.asarray([row["base_task"] for row in rows]),
        category=np.asarray([row["category"] for row in rows]),
    )
    torch.save(payload, args.output_dir / "paired_subspaces.pt")
    report = {
        "format": "starvla_dfr_causal_poc_v1",
        "interpretation": (
            "Diagnostic only: action agreement with the clean policy is not expert correctness "
            "and this run does not train a DFR head."
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "pairs": len(rows),
        "activation_shape": list(clean.shape),
        "train_tasks": sorted(train_tasks),
        "test_tasks": sorted(test_tasks),
        "nuisance_probes": probes,
        "paired_difference_subspaces": basis_metrics,
        "action_interventions": action_metrics,
        "ranks": ranks,
        "seed": args.seed,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    del model, clean, variant
    gc.collect()
    torch.cuda.empty_cache()
    print(f"wrote PoC outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

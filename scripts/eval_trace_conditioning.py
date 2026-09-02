#!/usr/bin/env python3
"""Paired open-loop diagnostic for whether a GR00T DiT uses CoT traces.

The script evaluates fixed LIBERO-plus demonstration observations under six
trace conditions while resetting the PyTorch RNG before every flow-matching
loss and action sample.  Consequently, differences between conditions are due
to the conditioning sequence, not different diffusion noise.

This is deliberately an offline diagnostic.  Actions and errors are in the
dataset's normalized training space; no simulator or action unnormalizer is
involved.

Example:
    python scripts/eval_trace_conditioning.py \
      --ours-checkpoint playground/Checkpoints/...ours.../final_model/pytorch_model.pt \
      --det-checkpoint  playground/Checkpoints/...det.../final_model/pytorch_model.pt \
      --num-samples 750 --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402
from starVLA.model.framework.base_framework import baseframework  # noqa: E402

CONDITIONS = (
    "ours_teacher",
    "generated",
    "detector_teacher",
    "foreign_ours",
    "shuffled_ours",
    "no_trace",
)
REFERENCE_CONDITION = "ours_teacher"
TRACE_RE = re.compile(r"<\|trace\|>(.*?)<\|/trace\|>", re.DOTALL)
TRACE_ARRAY_RE = re.compile(r'("trace_2d"\s*:\s*)(\[\s*\[.*?\]\s*\])', re.DOTALL)


def _resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / "final_model" / "pytorch_model.pt",
        path / "pytorch_model.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve checkpoint from {path}; expected a .pt file or <run>/final_model/pytorch_model.pt"
    )


def _run_dir(checkpoint: Path) -> Path:
    # Both <run>/final_model/pytorch_model.pt and <run>/checkpoints/steps_*.pt.
    return checkpoint.parents[1]


def _load_run_config(checkpoint: Path, *, prefer_full: bool) -> Any:
    run_dir = _run_dir(checkpoint)
    full_config = run_dir / "config.full.yaml"
    config_path = full_config if prefer_full and full_config.exists() else run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return OmegaConf.load(config_path)


def _mapping_path(cfg: Any) -> Path:
    path = Path(str(cfg.datasets.vla_data.cot.mapping_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing CoT mapping: {path}")
    return path.resolve()


def _assistant_text(conversation: list[dict[str, str]]) -> str:
    return str(conversation[1]["value"])


def _human_text(conversation: list[dict[str, str]]) -> str:
    return str(conversation[0]["value"])


def _conversation(human: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"from": "human", "value": human},
        {"from": "gpt", "value": assistant},
    ]


def _valid_trace(text: str) -> bool:
    match = TRACE_RE.search(text or "")
    if not match:
        return False
    try:
        payload = json.loads(match.group(1))
        points = payload["trace_2d"]
        return len(points) > 0 and all(len(point) == 2 for point in points)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _shuffle_trace(text: str, seed: int) -> str:
    """Shuffle point order without changing syntax or the coordinate set."""
    match = TRACE_RE.search(text)
    if not match:
        return text
    array_match = TRACE_ARRAY_RE.search(match.group(1))
    if not array_match:
        return text
    try:
        payload = json.loads(match.group(1))
        points = list(payload["trace_2d"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return text
    if len(points) < 2:
        return text

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points)).tolist()
    if order == list(range(len(points))):
        order = [*range(1, len(points)), 0]
    shuffled_array = "[" + ", ".join(json.dumps(points[i]) for i in order) + "]"
    payload_text = match.group(1)
    replacement = payload_text[: array_match.start(2)] + shuffled_array + payload_text[array_match.end(2) :]
    return text[: match.start(1)] + replacement + text[match.end(1) :]


def _stable_priority(seed: int, trajectory_name: str, frame_index: int) -> int:
    token = f"{seed}:{trajectory_name}:{frame_index}".encode()
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big")


def _iter_mapping(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _eligible_trajectory_names(dataset: Any) -> set[str]:
    names: set[str] = set()
    for single in dataset.datasets:
        for trajectory_id in single.trajectory_ids:
            episode = int(trajectory_id)
            names.add(f"{single.dataset_name}/{episode // single.chunk_size}/{episode % single.chunk_size}")
    return names


def _sample_detector_rows(
    detector_mapping: Path,
    eligible_trajectories: set[str],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep the `count` detector rows with smallest stable hash priorities."""
    heap: list[tuple[int, int, dict[str, Any]]] = []
    serial = 0
    for entry in tqdm(_iter_mapping(detector_mapping), desc="scan detector mapping", unit="rows"):
        trajectory_name = str(entry["trajectory_name"])
        if trajectory_name not in eligible_trajectories:
            continue
        frame_index = int(entry["start_frame"])
        priority = _stable_priority(seed, trajectory_name, frame_index)
        item = (-priority, serial, entry)
        serial += 1
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda x: -x[0])]


def _attach_ours_rows(
    candidates: list[dict[str, Any]],
    ours_mapping: Path,
) -> list[dict[str, Any]]:
    wanted: dict[str, set[int]] = defaultdict(set)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for detector in candidates:
        key = (str(detector["trajectory_name"]), int(detector["start_frame"]))
        wanted[key[0]].add(key[1])
        by_key[key] = {"detector": detector}

    for ours in tqdm(_iter_mapping(ours_mapping), desc="scan ours mapping", unit="rows"):
        trajectory_name = str(ours["trajectory_name"])
        frames = wanted.get(trajectory_name)
        if not frames:
            continue
        start, end = int(ours["start_frame"]), int(ours["end_frame"])
        for frame_index in tuple(frames):
            if start <= frame_index <= end:
                by_key[(trajectory_name, frame_index)]["ours"] = ours
                frames.remove(frame_index)

    common = []
    for candidate in candidates:
        key = (str(candidate["trajectory_name"]), int(candidate["start_frame"]))
        row = by_key[key]
        if "ours" in row:
            common.append(row)
    return common


def _dataset_indices(dataset: Any, wanted_keys: set[tuple[str, int]]) -> dict[tuple[str, int], tuple[int, int]]:
    found: dict[tuple[str, int], tuple[int, int]] = {}
    wanted_by_traj: dict[str, set[int]] = defaultdict(set)
    for trajectory_name, frame_index in wanted_keys:
        wanted_by_traj[trajectory_name].add(frame_index)

    for dataset_index, single in enumerate(dataset.datasets):
        for sample_index, (trajectory_id, frame_index) in enumerate(single.all_steps):
            frame_index = int(frame_index)
            episode = int(trajectory_id)
            trajectory_name = f"{single.dataset_name}/{episode // single.chunk_size}/{episode % single.chunk_size}"
            if frame_index in wanted_by_traj.get(trajectory_name, ()):
                found[(trajectory_name, frame_index)] = (dataset_index, sample_index)
    return found


def _foreign_donors(rows: list[dict[str, Any]], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(rows)).tolist()
    ours_texts = [_assistant_text(row["ours_conversation"]) for row in rows]
    donors: list[int] = []
    for index, own_text in enumerate(ours_texts):
        valid = [j for j in permutation if j != index and ours_texts[j] != own_text]
        if not valid:
            raise RuntimeError("Could not find a distinct foreign trace donor")
        donors.append(valid[index % len(valid)])
    return donors


def _write_manifest(
    path: Path,
    dataset: Any,
    ours_mapping: Path,
    detector_mapping: Path,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = _eligible_trajectory_names(dataset)
    # Oversample because a small number of detector frames may not exist in ours.
    detector_rows = _sample_detector_rows(detector_mapping, eligible, max(num_samples * 2, num_samples + 128), seed)
    common = _attach_ours_rows(detector_rows, ours_mapping)
    if len(common) < num_samples:
        print(f"Warning: requested {num_samples} common-support samples, found {len(common)}")
    common = common[:num_samples]

    wanted = {(str(row["detector"]["trajectory_name"]), int(row["detector"]["start_frame"])) for row in common}
    indices = _dataset_indices(dataset, wanted)
    rows = []
    for row in common:
        detector, ours = row["detector"], row["ours"]
        key = (str(detector["trajectory_name"]), int(detector["start_frame"]))
        if key not in indices:
            continue
        dataset_index, sample_index = indices[key]
        rows.append(
            {
                "trajectory_name": key[0],
                "frame_index": key[1],
                "dataset_index": dataset_index,
                "sample_index": sample_index,
                "ours_conversation": ours["conversations"],
                "detector_conversation": detector["conversations"],
            }
        )

    donors = _foreign_donors(rows, seed + 17)
    for index, donor in enumerate(donors):
        rows[index]["foreign_donor"] = {
            "trajectory_name": rows[donor]["trajectory_name"],
            "frame_index": rows[donor]["frame_index"],
            "row_index": donor,
        }
        donor_text = _assistant_text(rows[donor]["ours_conversation"])
        rows[index]["foreign_ours_conversation"] = _conversation(
            _human_text(rows[index]["ours_conversation"]), donor_text
        )
        shuffled = _shuffle_trace(_assistant_text(rows[index]["ours_conversation"]), seed + index)
        rows[index]["shuffled_ours_conversation"] = _conversation(
            _human_text(rows[index]["ours_conversation"]), shuffled
        )

    payload = {
        "seed": seed,
        "num_samples": len(rows),
        "ours_mapping": str(ours_mapping),
        "detector_mapping": str(detector_mapping),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return rows


def _load_or_create_manifest(
    path: Path,
    dataset: Any,
    ours_mapping: Path,
    detector_mapping: Path,
    num_samples: int,
    seed: int,
    rebuild: bool,
) -> list[dict[str, Any]]:
    if path.exists() and not rebuild:
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        print(f"Reusing fixed manifest with {len(rows)} samples: {path}")
        return rows
    rows = _write_manifest(path, dataset, ours_mapping, detector_mapping, num_samples, seed)
    print(f"Wrote fixed manifest with {len(rows)} samples: {path}")
    return rows


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_torch_rng(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def _encode_context(model: Any, examples: list[dict[str, Any]], conversations: list | None) -> torch.Tensor:
    inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[example["image"] for example in examples],
        instructions=[example["lang"] for example in examples],
        cot_conversations=conversations,
    )
    # Labels affect only CE computation, not hidden states. Removing them avoids
    # allocating logits/loss for teacher-forced diagnostic conditions.
    inputs.pop("labels", None)
    device = next(model.parameters()).device
    with _autocast(device):
        outputs = model.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        context = outputs.hidden_states[-1]
    if model.readout_projector is not None:
        context = model.readout_projector(context)
    return context


def _condition_conversations(
    condition: str,
    rows: list[dict[str, Any]],
    generated: list[str] | None,
) -> list | None:
    if condition == "no_trace":
        return None
    if condition == "ours_teacher":
        return [row["ours_conversation"] for row in rows]
    if condition == "detector_teacher":
        return [row["detector_conversation"] for row in rows]
    if condition == "foreign_ours":
        return [row["foreign_ours_conversation"] for row in rows]
    if condition == "shuffled_ours":
        return [row["shuffled_ours_conversation"] for row in rows]
    if condition == "generated":
        if generated is None:
            raise RuntimeError("generated condition requested without generated traces")
        return [
            _conversation(_human_text(row["ours_conversation"]), text) for row, text in zip(rows, generated, strict=True)
        ]
    raise ValueError(f"Unknown condition: {condition}")


@torch.inference_mode()
def _evaluate_batch(
    model: Any,
    examples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    conditions: tuple[str, ...],
    action_seed: int,
    repeated_loss_samples: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str] | None]:
    device = next(model.parameters()).device
    generated = None
    if "generated" in conditions:
        generated = model._generate_cot(
            [example["image"] for example in examples],
            [example["lang"] for example in examples],
        )

    actions = torch.as_tensor(
        np.asarray([example["action"] for example in examples]),
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    targets = actions[:, -int(model.action_horizon) :, :]
    state = None
    if "state" in examples[0] and int(model.config.framework.action_model.state_dim) > 0:
        state = torch.as_tensor(
            np.asarray([example["state"] for example in examples]),
            device=device,
            dtype=targets.dtype,
        )

    results: dict[str, dict[str, np.ndarray]] = {}
    for condition in conditions:
        conversations = _condition_conversations(condition, rows, generated)
        context = _encode_context(model, examples, conversations)

        context_repeated = context.repeat(repeated_loss_samples, 1, 1)
        targets_repeated = targets.repeat(repeated_loss_samples, 1, 1)
        state_repeated = state.repeat(repeated_loss_samples, 1, 1) if state is not None else None

        # These resets are the core pairing guarantee: every condition receives
        # exactly the same flow noise, beta timesteps, and inference noise.
        _set_torch_rng(action_seed, device)
        with _autocast(device):
            loss_rows = model.action_model(
                context_repeated,
                targets_repeated,
                state_repeated,
                reduction="none",
            )
        loss_per_sample = loss_rows.reshape(repeated_loss_samples, len(examples)).mean(dim=0)

        _set_torch_rng(action_seed + 1, device)
        with _autocast(device):
            predictions = model.action_model.predict_action(context, state)

        prediction_error = ((predictions.float() - targets.float()) ** 2).mean(dim=(1, 2))
        first_error = torch.linalg.vector_norm(predictions[:, 0].float() - targets[:, 0].float(), dim=-1)
        results[condition] = {
            "flow_loss": loss_per_sample.float().cpu().numpy(),
            "prediction_mse": prediction_error.cpu().numpy(),
            "first_action_error_l2": first_error.cpu().numpy(),
            "predictions": predictions.float().cpu().numpy(),
        }
        del context, context_repeated, targets_repeated, loss_rows, predictions

    return results, generated


def _cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    unique = np.unique(clusters)
    point = float(values.mean())
    if len(unique) < 2:
        return {"mean": point, "ci95_low": math.nan, "ci95_high": math.nan}
    grouped = {cluster: values[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates[draw] = np.concatenate([grouped[cluster] for cluster in sampled]).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean": point, "ci95_low": float(low), "ci95_high": float(high)}


def _summarize(
    model_name: str,
    rows: list[dict[str, Any]],
    condition_arrays: dict[str, dict[str, np.ndarray]],
    generated_texts: list[str | None],
    thresholds: list[float],
    seed: int,
) -> dict[str, Any]:
    clusters = np.asarray([row["trajectory_name"] for row in rows])
    reference = condition_arrays[REFERENCE_CONDITION]
    summary: dict[str, Any] = {
        "model": model_name,
        "num_observations": len(rows),
        "num_trajectories": len(np.unique(clusters)),
        "reference_condition": REFERENCE_CONDITION,
        "conditions": {},
    }
    for condition, arrays in condition_arrays.items():
        pred_delta = arrays["predictions"] - reference["predictions"]
        action_l2 = np.linalg.norm(pred_delta, axis=-1).mean(axis=-1)
        first_l2 = np.linalg.norm(pred_delta[:, 0], axis=-1)
        condition_summary = {
            "flow_loss": _cluster_bootstrap_mean(arrays["flow_loss"], clusters, seed + 1),
            "prediction_mse": _cluster_bootstrap_mean(arrays["prediction_mse"], clusters, seed + 2),
            "first_action_error_l2": _cluster_bootstrap_mean(arrays["first_action_error_l2"], clusters, seed + 3),
            "delta_flow_loss_vs_reference": _cluster_bootstrap_mean(
                arrays["flow_loss"] - reference["flow_loss"], clusters, seed + 4
            ),
            "delta_prediction_mse_vs_reference": _cluster_bootstrap_mean(
                arrays["prediction_mse"] - reference["prediction_mse"], clusters, seed + 5
            ),
            "action_l2_vs_reference": _cluster_bootstrap_mean(action_l2, clusters, seed + 6),
            "first_action_l2_vs_reference": _cluster_bootstrap_mean(first_l2, clusters, seed + 7),
            "first_action_changed_fraction": {
                str(threshold): float(np.mean(first_l2 > threshold)) for threshold in thresholds
            },
        }
        if condition == "generated":
            condition_summary["valid_trace_fraction"] = float(
                np.mean([_valid_trace(text or "") for text in generated_texts])
            )
        summary["conditions"][condition] = condition_summary
    return summary


def _markdown_table(summary: dict[str, Any], threshold: float) -> str:
    lines = [
        f"# Trace-conditioning diagnostic: {summary['model']}",
        "",
        f"Observations: {summary['num_observations']} across {summary['num_trajectories']} trajectories. "
        f"Reference: `{summary['reference_condition']}`.",
        "",
        "| condition | flow loss | action MSE | Δ action MSE | action L2 vs ref | first-action L2 | changed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["conditions"].items():
        lines.append(
            "| {condition} | {loss:.6f} | {mse:.6f} | {delta:+.6f} | {l2:.6f} | {first:.6f} | {changed:.1%} |".format(
                condition=condition,
                loss=metrics["flow_loss"]["mean"],
                mse=metrics["prediction_mse"]["mean"],
                delta=metrics["delta_prediction_mse_vs_reference"]["mean"],
                l2=metrics["action_l2_vs_reference"]["mean"],
                first=metrics["first_action_l2_vs_reference"]["mean"],
                changed=metrics["first_action_changed_fraction"][str(threshold)],
            )
        )
    lines.extend(
        [
            "",
            f"`changed` means first-action L2 difference > {threshold:g} in normalized action space.",
            "Confidence intervals in summary.json use a trajectory-cluster bootstrap; they are NaN with one trajectory.",
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_checkpoint(
    checkpoint: Path,
    dataset: Any,
    rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_name = _run_dir(checkpoint).name
    model_output = output_dir / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading {model_name}: {checkpoint}")
    model = baseframework.from_pretrained(str(checkpoint), is_inference=False, skip_cot_resolver=True)
    if not hasattr(model, "_generate_cot"):
        raise TypeError(f"{type(model).__name__} does not support explicit CoT generation")
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model = model.to(device=args.device, dtype=dtype).eval()

    all_arrays: dict[str, dict[str, list[np.ndarray]]] = {condition: defaultdict(list) for condition in args.conditions}
    generated_texts: list[str | None] = []
    per_sample: list[dict[str, Any]] = []
    target_chunks: list[np.ndarray] = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc=model_name, unit="batch"):
        batch_rows = rows[start : start + args.batch_size]
        examples = [dataset.datasets[row["dataset_index"]][row["sample_index"]] for row in batch_rows]
        target_chunks.append(
            np.asarray([example["action"] for example in examples], dtype=np.float32)[:, -int(model.action_horizon) :, :]
        )
        batch_results, generated = _evaluate_batch(
            model,
            examples,
            batch_rows,
            args.conditions,
            action_seed=args.seed + 100_000 + start,
            repeated_loss_samples=args.loss_noise_samples,
        )
        generated_texts.extend(generated if generated is not None else [None] * len(batch_rows))
        reference_predictions = batch_results[REFERENCE_CONDITION]["predictions"]

        for condition, metrics in batch_results.items():
            for key, values in metrics.items():
                all_arrays[condition][key].append(values)

        for offset, row in enumerate(batch_rows):
            record: dict[str, Any] = {
                "trajectory_name": row["trajectory_name"],
                "frame_index": row["frame_index"],
                "generated_trace": generated[offset] if generated is not None else None,
                "conditions": {},
            }
            for condition, metrics in batch_results.items():
                pred_delta = metrics["predictions"][offset] - reference_predictions[offset]
                record["conditions"][condition] = {
                    "flow_loss": float(metrics["flow_loss"][offset]),
                    "prediction_mse": float(metrics["prediction_mse"][offset]),
                    "first_action_error_l2": float(metrics["first_action_error_l2"][offset]),
                    "action_l2_vs_reference": float(np.linalg.norm(pred_delta, axis=-1).mean()),
                    "first_action_l2_vs_reference": float(np.linalg.norm(pred_delta[0])),
                }
            per_sample.append(record)

    arrays = {
        condition: {key: np.concatenate(chunks, axis=0) for key, chunks in metrics.items()}
        for condition, metrics in all_arrays.items()
    }
    summary = _summarize(
        model_name,
        rows,
        arrays,
        generated_texts,
        args.change_thresholds,
        args.seed,
    )
    (model_output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    (model_output / "summary.md").write_text(_markdown_table(summary, args.change_thresholds[0]))
    with (model_output / "per_sample.jsonl").open("w") as handle:
        for record in per_sample:
            handle.write(json.dumps(record) + "\n")
    np.savez_compressed(
        model_output / "predictions_and_targets.npz",
        targets=np.concatenate(target_chunks, axis=0),
        **{f"predictions__{condition}": metrics["predictions"] for condition, metrics in arrays.items()},
    )
    print((model_output / "summary.md").read_text())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-checkpoint", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/trace_conditioning_diagnostic")
    parser.add_argument("--manifest", default=None, help="Defaults to <output-dir>/manifest.json")
    parser.add_argument("--num-samples", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--video-backend",
        choices=("decord", "torchcodec", "torchvision_av"),
        default="decord",
        help="Video decoder for demonstration observations (default: decord, available in the project env).",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--split",
        choices=("all", "train", "eval"),
        default="all",
        help=(
            "Dataset observations to sample. 'all' is recommended for this sensitivity diagnostic. "
            "The checkpoint config holds out only one trajectory, so 'eval' cannot provide 500-1000 samples."
        ),
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--loss-noise-samples", type=int, default=8)
    parser.add_argument("--change-thresholds", default="0.05,0.01,0.1")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.change_thresholds = [float(value) for value in args.change_thresholds.split(",")]
    args.conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    unknown = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown:
        parser.error(f"unknown conditions: {unknown}; choices are {CONDITIONS}")
    if REFERENCE_CONDITION not in args.conditions:
        parser.error(f"--conditions must include reference condition {REFERENCE_CONDITION!r}")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.loss_noise_samples <= 0:
        parser.error("sample, batch, and loss-noise counts must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    ours_checkpoint = _resolve_checkpoint(args.ours_checkpoint)
    det_checkpoint = _resolve_checkpoint(args.det_checkpoint)
    ours_cfg = _load_run_config(ours_checkpoint, prefer_full=True)
    det_cfg = _load_run_config(det_checkpoint, prefer_full=True)
    ours_mapping = _mapping_path(ours_cfg)
    detector_mapping = _mapping_path(det_cfg)

    # Dataset construction is shared by both checkpoints. 'all' bypasses the
    # train/eval splitter but remains deterministic because we index the
    # underlying single datasets directly from the saved manifest.
    dataset_mode = "diagnostic" if args.split == "all" else args.split
    data_cfg = ours_cfg.datasets.vla_data
    data_cfg.video_backend = args.video_backend
    data_cfg.video_reader_cache_size = max(int(data_cfg.get("video_reader_cache_size", 0)), 8)
    dataset = get_vla_dataset(data_cfg=data_cfg, mode=dataset_mode, seed=args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_dir / "manifest.json"
    rows = _load_or_create_manifest(
        manifest_path,
        dataset,
        ours_mapping,
        detector_mapping,
        args.num_samples,
        args.seed,
        args.rebuild_manifest,
    )
    if not rows:
        raise RuntimeError("Manifest contains no common-support observations")

    run_metadata = {
        "ours_checkpoint": str(ours_checkpoint),
        "det_checkpoint": str(det_checkpoint),
        "manifest": str(manifest_path),
        "split": args.split,
        "conditions": args.conditions,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "video_backend": args.video_backend,
        "loss_noise_samples": args.loss_noise_samples,
        "change_thresholds": args.change_thresholds,
        "normalized_action_space": True,
        "pairing": "torch RNG reset before each condition's loss and action sample",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    summaries = {}
    for checkpoint in (ours_checkpoint, det_checkpoint):
        summary = _evaluate_checkpoint(checkpoint, dataset, rows, output_dir, args)
        summaries[summary["model"]] = summary
    (output_dir / "all_summaries.json").write_text(json.dumps(summaries, indent=2, allow_nan=True))
    print(f"Done. Results: {output_dir}")


if __name__ == "__main__":
    main()

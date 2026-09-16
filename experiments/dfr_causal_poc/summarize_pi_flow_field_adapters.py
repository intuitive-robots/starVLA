#!/usr/bin/env python3
"""Merge the four PI clean-flow adapter validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = (
    "projection_lora",
    "xattn_qk_lora",
    "xattn_vout_lora",
    "xattn_head_gate",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for arm in ARMS:
        path = args.root / arm / "report.json"
        report = json.loads(path.read_text())
        metrics = report["validation"]["overall"]
        interval = report["validation"]["task_cluster_bootstrap"][
            "field_mse_improvement_95pct"
        ]
        rows.append(
            {
                "arm": arm,
                "parameters": report["trainable_parameters"],
                "field_mse_improvement": metrics["field_mse_improvement"],
                "field_mse_improvement_95pct": interval,
                "endpoint_mse_improvement": metrics["endpoint_mse_improvement"],
                "successful_endpoint_mse_improvement": metrics["successful_endpoint_mse_improvement"],
                "successful_flow_mse_improvement": metrics["successful_flow_mse_improvement"],
                "clean_drift_over_teacher_energy": metrics["clean_drift_over_teacher_energy"],
                "base_field_cosine": metrics["base_field_cosine"],
                "adapted_field_cosine": metrics["adapted_field_cosine"],
                "passes_all": report["screen"]["passes_all"],
            }
        )
    eligible = [row for row in rows if row["passes_all"]]
    selected = max(eligible, key=lambda row: row["field_mse_improvement"])["arm"] if eligible else None
    summary = {
        "format": "starvla_pi_flow_field_adapter_comparison_v1",
        "status": "complete",
        "scope": "validation-only mechanism screen; test split remains sealed",
        "rows": rows,
        "selected_for_heldout_test": selected,
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# PI flow-field adapter validation screen",
        "",
        "The frozen clean PI policy is the teacher. Clean and perturbed observations share the exact physical state, instruction, initial flow noise, Action-DiT state, and Euler time. Only the named adapter is trained. These are offline validation metrics, not rollout success.",
        "",
        "| Arm | Params | Pair-field improvement (task 95% CI) | Successful-flow improvement | Endpoint-to-clean improvement | Endpoint-to-successful improvement | Clean drift / teacher energy | Cosine before -> after | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['parameters']:,} | {100*row['field_mse_improvement']:+.2f}% "
            f"[{100*row['field_mse_improvement_95pct'][0]:+.2f}, {100*row['field_mse_improvement_95pct'][1]:+.2f}] | "
            f"{100*row['successful_flow_mse_improvement']:+.2f}% | "
            f"{100*row['endpoint_mse_improvement']:+.2f}% | {100*row['successful_endpoint_mse_improvement']:+.2f}% | "
            f"{100*row['clean_drift_over_teacher_energy']:.3f}% | "
            f"{row['base_field_cosine']:.4f} -> {row['adapted_field_cosine']:.4f} | "
            f"{'yes' if row['passes_all'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Selected for the sealed held-out test: **{selected or 'none'}**.",
            "",
            "Q/K adaptation tests token routing (attention logits). V/output adaptation tests transported content. The head gate tests whether reweighting existing cross-attention heads is sufficient. Projection LoRA changes the learned VLM-to-action map directly.",
        ]
    )
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

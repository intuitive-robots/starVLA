#!/usr/bin/env python3
"""Summarize the larger orthogonal/conditional spectral-gate experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = (
    "warm_final_balanced_v2",
    "static_orthogonal_gate_warm_final",
    "conditional_orthogonal_gate_warm_final",
    "static_random_gate_warm_final_v2",
)
APPEARANCE = (
    "Background Textures",
    "Light Conditions",
    "Sensor Noise",
    "Camera Viewpoints",
)
PHYSICAL_STATE = ("Robot Initial States", "Objects Layout")
ALL_CATEGORIES = (*APPEARANCE, *PHYSICAL_STATE, "Language Instructions")


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _plus_episodes(root: Path, arm: str, exact: int) -> dict:
    directory = root / arm / "closed_loop" / f"libero_plus_exact{exact}" / "logs" / "libero_10"
    episodes = {}
    for path in sorted(directory.glob("*_episodes.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            key = (item["task_id"], item["episode_idx"])
            value = {"success": bool(item["success"]), "category": item["category"]}
            if key in episodes and episodes[key] != value:
                raise ValueError(f"conflicting duplicate episode {key} in {directory}")
            episodes[key] = value
    return episodes


def _clean_episodes(root: Path, arm: str, trials: int) -> dict:
    directory = root / arm / "closed_loop" / f"libero_clean_{trials}x10" / "logs" / "libero_10"
    episodes = {}
    for path in sorted(directory.glob("*_to_*.json")):
        for item in json.loads(path.read_text()).get("episodes", []):
            key = (item["task_id"], item["episode_idx"])
            value = bool(item["success"])
            if key in episodes and episodes[key] != value:
                raise ValueError(f"conflicting duplicate episode {key} in {directory}")
            episodes[key] = value
    return episodes


def _mcnemar(arm: dict, control: dict, categories=None) -> dict:
    common = sorted(set(arm) & set(control))
    if categories is not None:
        common = [key for key in common if arm[key]["category"] in categories]
    gained = sum(arm[key]["success"] and not control[key]["success"] for key in common)
    lost = sum(control[key]["success"] and not arm[key]["success"] for key in common)
    discordant = gained + lost
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "matched_episodes": len(common),
        "gate_only_successes": gained,
        "control_only_successes": lost,
        "mcnemar_exact_two_sided_p": p_value,
    }


def _rollout_comparison(arm_plus, control_plus, arm_clean, control_clean) -> dict:
    arm_clean_wrapped = {key: {"success": value} for key, value in arm_clean.items()}
    control_clean_wrapped = {key: {"success": value} for key, value in control_clean.items()}
    return {
        "plus_overall": _mcnemar(arm_plus, control_plus),
        "appearance_invariance": _mcnemar(arm_plus, control_plus, APPEARANCE),
        "physical_state_robustness": _mcnemar(arm_plus, control_plus, PHYSICAL_STATE),
        "by_category": {
            category: _mcnemar(arm_plus, control_plus, (category,))
            for category in ALL_CATEGORIES
        },
        "clean": _mcnemar(arm_clean_wrapped, control_clean_wrapped),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(__file__).parent / "outputs" / "action_refit_v1",
    )
    parser.add_argument("--plus-exact", type=int, default=140)
    parser.add_argument("--clean-trials", type=int, default=3)
    args = parser.parse_args()

    baseline_plus = _plus_episodes(args.baseline_root, "baseline", args.plus_exact)
    baseline_clean = _clean_episodes(args.baseline_root, "baseline", args.clean_trials)
    plus, clean, rows = {}, {}, []
    for arm in ARMS:
        plus[arm] = _plus_episodes(args.root, arm, args.plus_exact)
        clean[arm] = _clean_episodes(args.root, arm, args.clean_trials)
        plus_report = _read(
            args.root / arm / "closed_loop" / f"libero_plus_exact{args.plus_exact}"
            / "libero_10" / "overall_results.json"
        )
        rows.append(
            {
                "arm": arm,
                "offline": _read(args.root / arm / "report.json"),
                "libero_plus": plus_report,
                "libero_clean": _read(
                    args.root / arm / "closed_loop" / f"libero_clean_{args.clean_trials}x10"
                    / "overall_results.json"
                ),
                "versus_source_baseline": _rollout_comparison(
                    plus[arm], baseline_plus, clean[arm], baseline_clean
                ),
            }
        )

    primary = "conditional_orthogonal_gate_warm_final"
    controls = (
        "warm_final_balanced_v2",
        "static_orthogonal_gate_warm_final",
        "static_random_gate_warm_final_v2",
    )
    summary = {
        "format": "starvla_spectral_gate_v2_summary",
        "formal_targets": {
            "appearance_invariance": list(APPEARANCE),
            "physical_state_success_not_action_identity": list(PHYSICAL_STATE),
        },
        "source_baseline": {
            "plus_successes": sum(item["success"] for item in baseline_plus.values()),
            "plus_count": len(baseline_plus),
            "clean_successes": sum(baseline_clean.values()),
            "clean_count": len(baseline_clean),
        },
        "arms": rows,
        "primary_control_comparisons": {
            f"{primary}_vs_{control}": _rollout_comparison(
                plus[primary], plus[control], clean[primary], clean[control]
            )
            for control in controls
        },
        "static_control_comparisons": {
            f"static_orthogonal_gate_warm_final_vs_{control}": _rollout_comparison(
                plus["static_orthogonal_gate_warm_final"],
                plus[control],
                clean["static_orthogonal_gate_warm_final"],
                clean[control],
            )
            for control in (
                "warm_final_balanced_v2",
                "static_random_gate_warm_final_v2",
            )
        },
        "decision_rule": (
            "Promising only if the orthogonal learned-subspace arm improves matched "
            "appearance-shift rollouts over warm-only and random-subspace controls, "
            "retains clean success, and does not degrade physical-state robustness. "
            "Conditional gating is useful only if it also beats the static learned gate."
        ),
    }
    destination = args.root / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

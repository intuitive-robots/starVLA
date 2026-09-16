#!/usr/bin/env python3
"""Summarize alignment screens and matched closed-loop evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = (
    "warm_final_balanced",
    "learned_gate_all",
    "learned_gate_all_warm_final",
    "random_gate_all_warm_final",
)
TARGET_CATEGORIES = ("Background Textures", "Light Conditions", "Sensor Noise")


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


def _mcnemar(arm: dict, baseline: dict, categories=None) -> dict:
    common = sorted(set(arm) & set(baseline))
    if categories is not None:
        common = [key for key in common if arm[key]["category"] in categories]
    gained = sum(arm[key]["success"] and not baseline[key]["success"] for key in common)
    lost = sum(baseline[key]["success"] and not arm[key]["success"] for key in common)
    discordant = gained + lost
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
        p = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p = 1.0
    return {
        "matched_episodes": len(common),
        "alignment_only_successes": gained,
        "baseline_only_successes": lost,
        "mcnemar_exact_two_sided_p": p,
    }


def _compare_rollouts(arm_plus, control_plus, arm_clean, control_clean) -> dict:
    wrapped_arm = {key: {"success": value} for key, value in arm_clean.items()}
    wrapped_control = {key: {"success": value} for key, value in control_clean.items()}
    return {
        "plus_overall": _mcnemar(arm_plus, control_plus),
        "plus_target_categories": _mcnemar(
            arm_plus, control_plus, TARGET_CATEGORIES
        ),
        "plus_by_category": {
            name: _mcnemar(arm_plus, control_plus, (name,))
            for name in TARGET_CATEGORIES
        },
        "clean": _mcnemar(wrapped_arm, wrapped_control),
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
    rows, comparisons = [], {}
    plus_by_arm, clean_by_arm = {}, {}
    for arm in ARMS:
        report = _read(args.root / arm / "report.json")
        plus_report = _read(args.root / arm / "closed_loop" / f"libero_plus_exact{args.plus_exact}" / "libero_10" / "overall_results.json")
        clean_report = _read(args.root / arm / "closed_loop" / f"libero_clean_{args.clean_trials}x10" / "overall_results.json")
        row = {"arm": arm, "offline": report}
        if plus_report is not None:
            row["libero_plus"] = {
                "overall": plus_report["overall"],
                **{name: plus_report.get(name) for name in TARGET_CATEGORIES},
            }
            episodes = _plus_episodes(args.root, arm, args.plus_exact)
            plus_by_arm[arm] = episodes
            comparisons[arm] = {
                "plus_overall": _mcnemar(episodes, baseline_plus),
                "plus_target_categories": _mcnemar(episodes, baseline_plus, TARGET_CATEGORIES),
                "plus_by_category": {
                    name: _mcnemar(episodes, baseline_plus, (name,))
                    for name in TARGET_CATEGORIES
                },
            }
        if clean_report is not None:
            row["libero_clean"] = clean_report
            clean = _clean_episodes(args.root, arm, args.clean_trials)
            clean_by_arm[arm] = clean
            wrapped = {key: {"success": value} for key, value in clean.items()}
            wrapped_base = {key: {"success": value} for key, value in baseline_clean.items()}
            comparisons.setdefault(arm, {})["clean"] = _mcnemar(wrapped, wrapped_base)
        rows.append(row)

    primary = "learned_gate_all_warm_final"
    primary_control_comparisons = {
        f"{primary}_vs_{control}": _compare_rollouts(
            plus_by_arm[primary],
            plus_by_arm[control],
            clean_by_arm[primary],
            clean_by_arm[control],
        )
        for control in ("warm_final_balanced", "random_gate_all_warm_final")
    }

    summary = {
        "format": "starvla_pi_action_alignment_summary_v1",
        "baseline_root": str(args.baseline_root.resolve()),
        "baseline_plus_successes": sum(x["success"] for x in baseline_plus.values()),
        "baseline_plus_count": len(baseline_plus),
        "baseline_clean_successes": sum(baseline_clean.values()),
        "baseline_clean_count": len(baseline_clean),
        "arms": rows,
        "matched_comparisons_to_baseline": comparisons,
        "primary_control_comparisons": primary_control_comparisons,
        "decision_rule": (
            "Promising only if the learned-basis arm beats both warm-only and equal-rank "
            "random-basis controls on matched target-category rollouts without losing clean success."
        ),
    }
    destination = args.root / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

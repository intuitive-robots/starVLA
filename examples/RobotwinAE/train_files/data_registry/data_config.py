"""RoboTwin data-selection experiment (AE_Data_SelectioN_Robotic, 2026-10-09): training mixtures.

Additive registry: no data configs of its own, it reuses the RoboTwin `robotwin50` config
(examples/Robotwin) and only defines mixtures. Use with
    datasets.vla_data.data_root_dir = /e/scratch/m3/blank4/ae_jupiter/robotwin
    datasets.vla_data.use_keep_ranges = true
(balance_* keys are not read by get_vla_dataset: episodes are drawn uniformly within a view, so
selections are split into SLOT views with at most one window per episode, tools/make_robotwin_arm_views.py)

Per target task T and arm A:
  rtae_<T>_target   the base's 50 randomized demos of T only
  rtae_<T>_all      target + the whole pool (pool_clean + pool_rand_s7), no selection
  rtae_<T>_<A>      target + the arm's selected windows (views/sel/<T>/<A>__<pool view>__w<j>, keep_ranges)
Target weight 1.0; the arm's views share 1.0 in proportion to their frames (kept frames for
selection views) -> 50 % target / 50 % added data per step, uniform over the added frames.
`rtae_selftest_<T>_<A>` are the same for the synthetic-feature plumbing test.
Mixtures are derived from the views on disk at import; missing directories just give fewer entries.
"""
import json
from pathlib import Path

ROOT = Path("/e/scratch/m3/blank4/ae_jupiter/robotwin")
TAG = "robotwin50"
TARGETS = ("lift_pot", "place_shoe", "pick_diverse_bottles", "place_object_basket", "stack_bowls_two")
POOL = ("pool_clean", "pool_rand_s7")

ROBOT_TYPE_CONFIG_MAP = {}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
DATASET_NAMED_MIXTURES = {}


def _frames(view: Path) -> int:
    manifest = view / "meta/subset_manifest.json"
    m = json.loads(manifest.read_text())
    return int(m["kept_frames"] if "kept_frames" in m else m["total_frames"])


def _with_added(target: str, added: list[Path]) -> list:
    frames = [_frames(v) for v in added]
    total = float(sum(frames))
    return [(f"views/target_{target}_rand", 1.0, TAG)] + [
        (str(v.relative_to(ROOT)), f / total, TAG) for v, f in zip(added, frames)]


def _register() -> None:
    views = ROOT / "views"
    for target in TARGETS:
        if not (views / f"target_{target}_rand/meta/subset_manifest.json").is_file():
            continue
        DATASET_NAMED_MIXTURES[f"rtae_{target}_target"] = [(f"views/target_{target}_rand", 1.0, TAG)]
        pool = [views / p for p in POOL if (views / p / "meta/subset_manifest.json").is_file()]
        if len(pool) == len(POOL):
            DATASET_NAMED_MIXTURES[f"rtae_{target}_all"] = _with_added(target, pool)
        for prefix, sel_root in (("rtae", views / "sel"), ("rtae_selftest", ROOT / "selftest_sel_views")):
            tdir = sel_root / target
            if not tdir.is_dir():
                continue
            arms = sorted({d.name.split("__")[0] for d in tdir.iterdir() if "__" in d.name})
            for arm in arms:
                added = sorted(d for d in tdir.iterdir() if d.name.startswith(f"{arm}__")
                               and (d / "meta/subset_manifest.json").is_file())
                if added:
                    DATASET_NAMED_MIXTURES[f"{prefix}_{target}_{arm}"] = _with_added(target, added)


try:
    _register()
except Exception as exc:  # never break the shared registry import
    print(f"[RobotwinAE registry] skipped: {type(exc).__name__}: {exc}")

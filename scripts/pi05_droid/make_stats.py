"""openpi norm_stats.json (pi05_droid_jointpos) -> the two starVLA statistics files (2026-10-11).

1. <out_dir>/stats_override_openpi_droid_jointpos.json: LeRobot-style per-column statistics for
   datasets.vla_data.stats_override_path (gr00t loader). Columns: observation.joint_position (7),
   observation.gripper_position (1), action.joint_position (7, joint DELTAS a_t - s_0 as in openpi), action.gripper_position (1).
2. <ckpt_dir>/dataset_statistics.json: the starVLA run format {tag: {action: {...,mask}, state: {...}, ...}} with
   flat 8-d arrays, so the converted checkpoint can be served directly.
openpi has no min/max in its quantile stats; min/max are set to q01/q99 (only min_max mode would read them).
usage: python3 -I make_stats.py <norm_stats.json> <out_dir> <ckpt_dir> [tag]
"""
import json
import os
import sys

src, out_dir, ckpt_dir = sys.argv[1], sys.argv[2], sys.argv[3]
tag = sys.argv[4] if len(sys.argv) > 4 else "franka"
ns = json.load(open(src))["norm_stats"]


def take(stats, a, b):
    out = {k: [float(x) for x in stats[k][a:b]] for k in ("mean", "std", "q01", "q99")}
    out["min"], out["max"] = list(out["q01"]), list(out["q99"])
    return out


st, ac = ns["state"], ns["actions"]
override = {
    "observation.joint_position": take(st, 0, 7),
    "observation.gripper_position": take(st, 7, 8),
    "action.joint_position": take(ac, 0, 7),
    "action.gripper_position": take(ac, 7, 8),
}
os.makedirs(out_dir, exist_ok=True)
p1 = os.path.join(out_dir, "stats_override_openpi_droid_jointpos.json")
json.dump({"source": src, "note": "openpi pi05_droid_jointpos quantile stats; actions = joint deltas to s_0 + abs gripper",
           "statistics": override}, open(p1, "w"), indent=1)

flat_s = take(st, 0, 8)
flat_a = take(ac, 0, 8)
flat_a["mask"] = [True] * 8
ds = {tag: {"action": flat_a, "state": flat_s, "num_transitions": 0, "num_trajectories": 0,
            "_source": src, "_action_space": "joints relative to the current state (a_t - s_0) + absolute gripper"}}
os.makedirs(ckpt_dir, exist_ok=True)
p2 = os.path.join(ckpt_dir, "dataset_statistics.json")
json.dump(ds, open(p2, "w"), indent=1)
print("wrote", p1, p2)

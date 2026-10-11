#!/usr/bin/env python3
"""Compare render_skip_equivalence.py outputs: compare_render_skip.py A.npz B.npz [--label X]."""
import sys
import numpy as np


def main():
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    res = {}
    same_len = a["joints"].shape == b["joints"].shape
    res["n_actions"] = (len(a["joints"]), len(b["joints"]))
    res["names_equal"] = bool(np.array_equal(a["names"], b["names"]))
    if same_len and res["names_equal"]:
        res["joints_maxabs"] = float(np.abs(a["joints"] - b["joints"]).max())
        res["poses_maxabs"] = float(np.abs(a["poses"] - b["poses"]).max())
        res["joints_bitequal"] = bool(np.array_equal(a["joints"], b["joints"]))
        res["poses_bitequal"] = bool(np.array_equal(a["poses"], b["poses"]))
    res["success_equal"] = bool(np.array_equal(a["success"], b["success"]))
    res["final_success"] = (bool(a["success"][-1]), bool(b["success"][-1]))
    if a["obs_imgs"].shape == b["obs_imgs"].shape:
        d = np.abs(a["obs_imgs"].astype(np.int16) - b["obs_imgs"].astype(np.int16))
        res["obs_imgs_bitequal"] = bool((d == 0).all())
        res["obs_imgs_maxabs"] = int(d.max())
        res["obs_imgs_frac_px_diff"] = float((d.max(-1) > 0).mean())
        res["obs_state_bitequal"] = bool(np.array_equal(a["obs_state"], b["obs_state"]))
    res["crazy"] = (bool(a["crazy"]), bool(b["crazy"]))
    res["skip"] = (bool(a["skip_flag"]), bool(b["skip_flag"]))
    res["mean_dt"] = (round(float(a["dt"].mean()), 4), round(float(b["dt"].mean()), 4))
    print(f"[COMPARE {label}] " + " ".join(f"{k}={v}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()

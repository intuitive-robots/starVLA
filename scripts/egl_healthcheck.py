#!/usr/bin/env python3
"""Is this GPU's EGL stack fit to render, right now?

Every eval failure we traced today was a GPU whose EGL state had gone bad: it aborts inside
mjr_readPixels, or (worse) returns black frames that never raise and just score as failed
rollouts. Both are per-GPU and transient -- one GPU of a node failing every unit while its
three siblings finish everything. Finding out after a unit dies costs 5-10 minutes; finding
out here costs about five seconds, so a launcher can skip the bad device and keep its
concurrency instead of throttling everything to be safe.

    python scripts/egl_healthcheck.py --device 0        # exit 0 healthy, 1 bad
    python scripts/egl_healthcheck.py --device 0 --contexts 8

--contexts renders from several contexts in one process, which is the state that actually
breaks: a second renderer on a busy GPU is what returns black frames.
"""
import argparse
import os
import sys


def check(device: int, contexts: int, size: int) -> tuple[bool, str]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device)
    os.environ.pop("DISPLAY", None)  # EGL is sensitive to it even headless
    import numpy as np
    import mujoco

    # A lit red box on a dark ground: a correct render is bright and colourful, a broken one
    # is uniformly black. Checking the mean alone would pass a grey framebuffer, so we also
    # require some spread.
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="%d" offheight="%d"/></visual>'
        '<worldbody><light pos="0 0 3" dir="0 0 -1"/>'
        '<geom type="box" size=".3 .3 .3" rgba="1 .2 .2 1"/>'
        '</worldbody></mujoco>' % (size, size)
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    frames = []
    renderers = []
    try:
        for _ in range(contexts):
            r = mujoco.Renderer(model, height=size, width=size)
            renderers.append(r)
            r.update_scene(data)
            frames.append(np.asarray(r.render(), dtype=np.float32))
    except Exception as exc:  # EGL/driver failures surface here
        return False, f"render raised {type(exc).__name__}: {exc}"
    finally:
        for r in renderers:
            try:
                r.close()
            except Exception:
                pass

    for i, frame in enumerate(frames):
        if frame.mean() < 1.0:
            return False, f"context {i}: black frame (mean={frame.mean():.3f})"
        if frame.std() < 1.0:
            return False, f"context {i}: flat frame (std={frame.std():.3f})"
    means = [f"{f.mean():.1f}" for f in frames]
    return True, f"{contexts} context(s) OK, means {' '.join(means)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="MUJOCO_EGL_DEVICE_ID (global index)")
    ap.add_argument("--contexts", type=int, default=4)
    ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args()
    ok, detail = check(a.device, a.contexts, a.size)
    print(f"[egl-health] gpu{a.device}: {'OK' if ok else 'BAD'} -- {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

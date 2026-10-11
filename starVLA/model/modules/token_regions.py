"""Token-region helpers on the merged 8x8 image-token grid (E4/E5 object bias; A5 patching).

Shared by the policy server (deployment/model_server/policy_wrapper.py, ``STARVLA_OBJ_BIAS``)
and the offline analyses (papers/BiBrain_ICLR_2027/analysis/object_groups.py re-exports these),
so both use one definition. Input: per-token object area fractions ``[8, 8]`` in the policy's
image orientation (examples/LIBERO-plus/eval_files/online_obj_mask.py, capture_masks.py).
Pure numpy.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

GRID = 8
TAU = 0.25  # pre-registered 2026-09-28 10:53: a token belongs to the object if its object fraction >= 0.25


def token_sets(frac, tau: float = TAU, fallback_top1: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Flat row-major token indices of the object group and of the rest of the camera.

    Tokens with object fraction >= ``tau`` form G_obj. If none qualifies but the object is
    visible, the single highest-fraction token is used (``fallback_top1``). G_obj is empty when
    the object is not visible at all.
    """
    f = np.asarray(frac, dtype=np.float64).reshape(-1)
    obj = np.nonzero(f >= tau)[0]
    if not len(obj) and fallback_top1 and f.max() > 0:
        obj = np.array([int(np.argmax(f))])
    non = np.setdiff1d(np.arange(f.size), obj)
    return obj.astype(np.int64), non.astype(np.int64)


def random_region(obj: Sequence[int], rng: np.random.Generator, grid: int = GRID,
                  max_overlap: float = 0.0, tries: int = 200) -> np.ndarray:
    """Same-area control region: the object token shape translated to a random grid offset.

    Keeps the region's shape and size and avoids the object tokens (overlap fraction <=
    ``max_overlap``). Falls back to a random set of the same number of non-object tokens if no
    translation fits.
    """
    obj = np.asarray(obj, dtype=np.int64)
    if not len(obj):
        return obj
    r, c = obj // grid, obj % grid
    r0, c0 = r - r.min(), c - c.min()
    h, w = int(r0.max()) + 1, int(c0.max()) + 1
    objset = set(obj.tolist())
    for _ in range(tries):
        dr = int(rng.integers(0, grid - h + 1))
        dc = int(rng.integers(0, grid - w + 1))
        cand = (r0 + dr) * grid + (c0 + dc)
        if len(objset.intersection(cand.tolist())) <= max_overlap * len(obj):
            return np.sort(cand)
    pool = np.setdiff1d(np.arange(grid * grid), obj)
    return np.sort(rng.choice(pool, size=min(len(obj), len(pool)), replace=False))

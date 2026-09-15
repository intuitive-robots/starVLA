"""Opt-in, pixel/RNG-equivalent acceleration for the pinned LIBERO-Plus SIF.

Do not replace the benchmark's corruption with a true swap: its NumPy tuple
assignment aliases array views and actually copies the neighbour into the
current pixel. We intentionally preserve that behavior and the MT19937 draws.
"""
import hashlib
import inspect

import numpy as np
from numba import njit

SOURCE_SHA256 = 'ea1b44eb3f87b145cab3e70d1b196db29b444a8914e72d5e28814561e52c5a25'
PARAMETERS = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2),
              (1.1, 3, 2), (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2),
              (2.2, 5, 1), (2.5, 5, 1)]


@njit(cache=True)
def _copy_pixels(x, delta, iterations, offsets):
    index = 0
    for _ in range(iterations):
        for h in range(x.shape[0] - delta, delta, -1):
            for w in range(x.shape[1] - delta, delta, -1):
                dx, dy = offsets[index, 0], offsets[index, 1]
                for channel in range(x.shape[2]):
                    x[h, w, channel] = x[h + dy, w + dx, channel]
                index += 1
    return x


def build_fast_glass_blur(original):
    source_hash = hashlib.sha256(inspect.getsource(original).strip().encode()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise RuntimeError('Fast glass blur requires the validated pinned LIBERO-Plus implementation')
    gaussian = original.__globals__['gaussian']

    def glass_blur(x, severity=1):
        sigma, delta, iterations = PARAMETERS[severity - 1]
        x = np.uint8(gaussian(np.array(x) / 255., sigma=sigma, channel_axis=-1) * 255)
        if x.ndim != 3:
            raise ValueError('Fast glass blur requires an HWC image')
        count = iterations * max(0, x.shape[0] - 2 * delta) * max(0, x.shape[1] - 2 * delta)
        # Use NumPy's existing generator, not Numba's separate RNG. Drawing in
        # bulk preserves the stream consumed by repeated size=(2,) calls.
        offsets = np.random.randint(-delta, delta, size=(count, 2))
        x = _copy_pixels(x, delta, iterations, offsets)
        return np.clip(gaussian(x / 255., sigma=sigma, channel_axis=-1), 0, 1) * 255

    return glass_blur


def install():
    from libero.libero.envs import env_wrapper
    env_wrapper.glass_blur = build_fast_glass_blur(env_wrapper.glass_blur)

"""Time PolicyServerWrapper.predict_action (exactly what the eval policy server runs per batch) on a GPU with no
simulators rendering, for several batch sizes. Inputs match the LIBERO client: two 224x224 uint8 camera images and an
instruction per example, DDIM with 10 steps. Usage: python time_inference.py <ckpt> [reps]"""

import sys
import time

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper

ckpt = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
policy = PolicyServerWrapper(ckpt, use_bf16=True, is_inference=True)
rng = np.random.default_rng(0)


def batch(n):
    return [{"image": [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)],
             "lang": "pick up the black bowl between the plate and the ramekin and place it on the plate"}
            for _ in range(n)]


kw = dict(do_sample=False, use_ddim=True, num_ddim_steps=10)
for _ in range(3):
    policy.predict_action(examples=batch(4), **kw)
print(f"GPU {torch.cuda.get_device_name()}  ckpt {ckpt}")
for n in (1, 2, 4, 6, 8, 16):
    ex = batch(n)
    t = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.predict_action(examples=ex, **kw)
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    print(f"batch {n:2d}: median {np.median(t) * 1000:6.1f} ms  p10 {np.percentile(t, 10) * 1000:6.1f}  p90 {np.percentile(t, 90) * 1000:6.1f}")

"""openpi PyTorch checkpoint (examples/convert_jax_model_to_pytorch.py output) -> starVLA PI05 safetensors (2026-10-11).

Same layout as /e/scratch/m3/blank4/hf_pi_model_starvla/pi05_base_starvla/bfloat16/model.safetensors: the PI05
framework's own state_dict (keys vlm.model.*, action_head.*), bf16 with the PI05.to_precision fp32 exceptions.
The openpi -> starVLA key mapping is PI0._normalize_state_dict (the loader every starVLA PI run uses).

usage: python convert_to_starvla.py --code <starVLA code dir> --yaml <PI05 yaml> --src <openpi model.safetensors>
         --out <out model.safetensors> [--ref <reference starVLA safetensors>] [--compare-values]
--ref: assert identical key set / shapes / dtypes; --compare-values: also report max |diff| per tensor
(used to check the pipeline by converting pi05_base and comparing with the published pi05_base_starvla).
"""
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--code", required=True)
p.add_argument("--yaml", required=True)
p.add_argument("--src", required=True)
p.add_argument("--out", default="")
p.add_argument("--ref", default="")
p.add_argument("--compare-values", action="store_true")
a = p.parse_args()
sys.path.insert(0, a.code)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import starVLA  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

print("starVLA from", starVLA.__file__, flush=True)
cfg = OmegaConf.load(a.yaml)
cfg.trainer.pretrained_checkpoint = None
model = build_framework(cfg)
assert type(model).__name__ == "PI05", type(model).__name__
current = model.state_dict()
src = model._normalize_state_dict(model._load_checkpoint_state_dict(a.src))
missing = [k for k in current if k not in src]
unexpected = [k for k in src if k not in current]
shape_bad = [(k, tuple(src[k].shape), tuple(current[k].shape)) for k in current if k in src and src[k].shape != current[k].shape]
print(f"src tensors {len(src)} model tensors {len(current)} missing {len(missing)} unexpected {len(unexpected)} "
      f"shape_mismatch {len(shape_bad)}", flush=True)
print("missing:", missing[:20]); print("unexpected:", unexpected[:20]); print("shape mismatch:", shape_bad[:20])
if shape_bad or [k for k in missing if not model._is_expected_openpi_missing_key(k)]:
    raise SystemExit("FAIL: checkpoint does not cover the PI05 model")
res = model.load_state_dict({k: v for k, v in src.items() if k in current}, strict=False)
assert not res.unexpected_keys
model.to_precision("bfloat16")
sd = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
dt = {}
for v in sd.values():
    dt[str(v.dtype)] = dt.get(str(v.dtype), 0) + 1
print("dtypes", dt, flush=True)

if a.ref:
    with safe_open(a.ref, "pt") as f:
        rk = set(f.keys())
        mk = set(sd)
        print(f"ref keys {len(rk)} ours {len(mk)} only_ref {sorted(rk - mk)[:10]} only_ours {sorted(mk - rk)[:10]}")
        bad, worst = [], []
        for k in sorted(rk & mk):
            r = f.get_tensor(k)
            if r.shape != sd[k].shape or r.dtype != sd[k].dtype:
                bad.append((k, tuple(r.shape), str(r.dtype), tuple(sd[k].shape), str(sd[k].dtype)))
            elif a.compare_values:
                d = (r.float() - sd[k].float()).abs().max().item()
                worst.append((d, k))
        print("shape/dtype mismatches vs ref:", len(bad), bad[:10])
        if a.compare_values:
            worst.sort(reverse=True)
            n0 = sum(1 for d, _ in worst if d == 0.0)
            print(f"VALUES vs ref: {n0}/{len(worst)} tensors bit-identical; largest |diff|: {worst[:5]}")
        if rk != mk or bad:
            raise SystemExit("FAIL: layout differs from the reference")
        print("LAYOUT OK: same keys, shapes and dtypes as", a.ref)

if a.out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_file(sd, a.out, metadata={"format": "pt", "source": a.src, "converter": os.path.abspath(__file__)})
    print("saved", a.out, os.path.getsize(a.out) / 1e9, "GB", flush=True)

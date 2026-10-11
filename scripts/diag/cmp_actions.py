"""Compare predicted action chunks: LIT stage-2 TRACE vs CONTROL.

Both trained healthily (control action_dit_loss min 0.0142 vs trace 0.0301 final; open-loop
MSE 0.0223 vs 0.0184) and both load cleanly with identical key counts and byte-identical
dataset_statistics.json -- yet closed-loop is 75.2% vs 0.0%, with 0/20 on EVERY task of all
four suites. Uniform zero with no variance means the control emits actions the simulator
cannot use. This inspects the actions directly.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/e/project1/m3/blank4/code/starVLA")
from PIL import Image
from starVLA.model.framework.base_framework import baseframework

CK = {
 "TRACE  (75.2%)": "playground/Checkpoints/ervla_lit_stage2_gr00t_b32_s42/checkpoints/steps_25000_pytorch_model.pt",
 "CONTROL (0.0%)": "playground/Checkpoints/ervla_lit_stage2ctrl_gr00t_b32_s42/checkpoints/steps_25000_pytorch_model.pt",
}
rng = np.random.default_rng(0)
img = [Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(2)]
state = rng.uniform(-1, 1, (1, 8)).astype(np.float32)
ex = [{"image": img, "lang": "pick up the black bowl and place it on the plate", "state": state}]

for name, p in CK.items():
    try:
        m = baseframework.from_pretrained(p, is_inference=True).to(torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            out = m.predict_action(examples=ex)
        a = np.asarray(out["normalized_actions"], dtype=np.float64)
        print(f"\n=== {name} ===", flush=True)
        print(f"  shape={a.shape}  NaN={int(np.isnan(a).sum())}  Inf={int(np.isinf(a).sum())}")
        if not np.all(np.isnan(a)):
            print(f"  min={np.nanmin(a):+.4f} max={np.nanmax(a):+.4f} mean={np.nanmean(a):+.4f} std={np.nanstd(a):.4f}")
            print(f"  saturated |a|>1 : {np.mean(np.abs(a) > 1.0):.3f}")
            b = a.reshape(a.shape[0], -1, a.shape[-1])
            print(f"  std across chunk timesteps (0 => constant): {np.nanstd(b, axis=1)[0].round(4)}")
            print(f"  first timestep  : {b[0,0].round(4)}")
            print(f"  last  timestep  : {b[0,-1].round(4)}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        print(f"\n=== {name} === FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc()

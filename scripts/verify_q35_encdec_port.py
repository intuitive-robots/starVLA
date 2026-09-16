"""Does starVLA's Qwen3.5 encoder-only interface reproduce the reference encoder?

Runs the SAME inputs through (a) the reference Qwen35EncDecForConditionalGeneration built
by train_downstream, and (b) starVLA's _QWen3_5_EncDec_Interface, and compares the encoder
output tensor. Also checks the per-layer hidden states the action head consumes.
"""
import sys, torch
sys.path.insert(0, "/e/project1/m3/blank4/code/starVLA")
sys.path.insert(0, "/e/project1/m3/blank4/code/train_downstream")

CKPT = "/e/project1/m3/blank4/code/train_downstream/train/outputs/q35_enc_dec_v5_tb18432/2026-09-15_00-20-14_job1776705/checkpoints/checkpoint-28786"
BASE = "/e/scratch/m3/misc_data/nogga2/weights/Qwen3.5-2B"

from omegaconf import OmegaConf
cfg = OmegaConf.create({"datasets": {"vla_data": {}}, "framework": {"qwenvl": {
    "base_vlm": BASE, "enc_dec": True, "encdec_ckpt": CKPT, "skip_decoder": True,
    "collect_encoder_layers": True, "attn_implementation": "sdpa", "encdec_impl": "qwen35",
}}})

from starVLA.model.modules.vlm import get_vlm_model
iface = get_vlm_model(cfg).cuda().eval()
print("interface:", type(iface).__name__, "| encoder layers:", iface.num_encoder_feature_layers)

# Realistic inputs: two camera views per sample, built exactly as training does.
from PIL import Image
import numpy as np
rng = np.random.default_rng(0)
imgs = [[Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(2)]
        for _ in range(2)]
instr = ["put the cup on the plate [STATE] 12 44 [ACTION]", "open the drawer [STATE] 9 7 [ACTION]"]
inputs = iface.build_qwenvl_inputs(imgs, instr)
inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
input_ids, attn = inputs["input_ids"], inputs["attention_mask"]

with torch.no_grad():
    out = iface(**inputs)
    ours = out.hidden_states[-1].float()
    n_layers = len(out.hidden_states)

    # reference path: same preamble, same _encode, run directly on the built model
    m = iface.model.model
    embeds = m.get_input_embeddings()(input_ids)
    img = m.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True).pooler_output
    img = torch.cat(img, 0).to(embeds.device, embeds.dtype)
    image_mask, _ = m.get_placeholder_mask(input_ids, inputs_embeds=embeds, image_features=img)
    embeds = embeds.masked_scatter(image_mask, img)
    pos = m.compute_3d_position_ids(input_ids=input_ids, inputs_embeds=embeds,
                                    image_grid_thw=inputs["image_grid_thw"], video_grid_thw=None,
                                    attention_mask=attn, past_key_values=None,
                                    mm_token_type_ids=inputs.get("mm_token_type_ids"))
    ref = iface.model._encode(embeds, attn.bool(), pos).float()

print("hidden_states returned:", n_layers)
print("ours", tuple(ours.shape), "ref", tuple(ref.shape))
d = (ours - ref).abs()
print(f"max abs diff {d.max().item():.3e}   mean {d.mean().item():.3e}")
print("EXACT MATCH:", torch.equal(ours, ref))
print("mask:", tuple(iface._last_encoder_attention_mask.shape),
      "valid tokens:", iface._last_encoder_attention_mask.sum().item())
# the trained encoder must not be a no-op relative to the raw embeddings
rel = (ref - embeds.float()).abs().mean() / embeds.float().abs().mean()
print(f"encoder output vs input embeds, relative change: {rel.item():.3f}")
print("bidir_gates absmean:", iface.model.bidir_gates.abs().mean().item(),
      "| enc_scale:", float(iface.model.enc_scale))

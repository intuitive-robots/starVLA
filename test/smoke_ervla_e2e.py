"""One-batch GPU smoke for CoT dropout + raw masked encoder-to-DiT conditioning."""

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T


CONFIG = "examples/LIBERO/train_files/ervla_c_action.yaml"


def main():
    cfg = OmegaConf.load(CONFIG)
    model = Qwen_GR00T(config=cfg).cuda().train()
    assert model.readout_projector is None

    prompt = cfg.datasets.vla_data.CoT_prompt
    conversation = [
        {"from": "human", "value": prompt},
        {
            "from": "gpt",
            "value": (
                "<subtask>reach for the bowl</subtask>"
                "<movement>move left and down</movement>"
                "<trajectory>[(0.55,0.40),(0.50,0.45),(0.45,0.50)]</trajectory>"
            ),
        },
    ]
    examples = [
        {
            "image": [Image.new("RGB", (256, 256), (40, 90, 140))],
            "lang": "put the black bowl in the drawer",
            "action": np.zeros((16, 7), dtype=np.float32),
            "cot_conversation": conversation,
            "cot_available": True,
            "cot_mode": "cot",
        },
        {
            "image": [Image.new("RGB", (256, 256), (100, 60, 20))],
            "lang": "turn on the stove",
            "action": np.zeros((16, 7), dtype=np.float32),
            "cot_conversation": None,
            "cot_available": True,
            "cot_mode": "no_cot",
        },
    ]

    with torch.no_grad():
        output = model(examples)

    mask = model.qwen_vl_interface._last_encoder_attention_mask
    assert mask.ndim == 2 and mask.shape[0] == 2
    assert mask.dtype == torch.bool
    assert torch.isfinite(output["action_loss"])
    assert torch.isfinite(output["cot_loss"])
    assert output["cot_coverage"] == 1.0
    assert output["cot_keep_rate"] == 0.5
    model.eval()
    rollout = model.predict_action([
        {"image": row["image"], "lang": row["lang"]} for row in examples
    ])["normalized_actions"]
    assert rollout.shape == (2, 16, 7)
    assert np.isfinite(rollout).all()
    print(
        "PASS",
        f"action={output['action_loss'].item():.4f}",
        f"cot={output['cot_loss'].item():.4f}",
        f"encoder_mask={tuple(mask.shape)}",
        f"valid={mask.sum(1).tolist()}",
        "modes=cot/no_cot",
        f"rollout={rollout.shape}",
    )


if __name__ == "__main__":
    main()

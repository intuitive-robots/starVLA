# Causal Qwen3-VL with a mask fine-tuned decoder tail.
#
# Why this exists
# ---------------
# Every Tier-3 arm carrying dense mask supervision was an encoder--decoder, so "does mask
# supervision help a policy" was only ever asked of one architecture. This is the missing
# decoder-only arm: the same dense role-mask objective, applied to the causal stack.
#
# What the tail is
# ----------------
# encdec-vlm's phase0c fine-tunes blocks 0..L of the causal text stack against the
# role-mask loss with the instruction supplied through the read-out query, reading hidden
# states at layer L (=11 of 28). The layer matters: read at the FINAL layer instead, the
# same recipe scores at chance, because a causal LM's last layers specialise toward
# next-token prediction. The blob stores only those blocks, keyed by offset within the
# tail; everything else is untouched base weights.
#
# A caveat worth stating plainly
# ------------------------------
# The mask loss shaped the representation AT LAYER 11, but the VLM4A contract hands the
# action expert `hidden_states[-1]`. The fine-tuned blocks sit below that, so they still
# influence what the policy reads, but the policy does not read the site the supervision
# targeted. Reading layer 11 here instead would change what every other arm is compared
# against, so we keep the contract identical across arms and note the asymmetry.

import json
from typing import Optional

import torch

from starVLA.training.trainer_utils import initialize_overwatch

from .QWen3 import _QWen3_VL_Interface

logger = initialize_overwatch(__name__)


class _QWen3_CausalMaskFT_Interface(_QWen3_VL_Interface):
    """Causal Qwen3-VL + phase0c decoder tail. Config: framework.qwenvl.causal_maskft."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(config, **kwargs)
        blob_path = config.framework.qwenvl.get("causal_maskft")
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        tail, cfg = blob["decoder_tail"], blob["cfg"]

        blocks = self.model.model.language_model.layers
        L = cfg["layer"] if cfg["layer"] >= 0 else len(blocks)
        n = int(cfg["unfreeze"])
        sel = blocks[max(0, L - n):L] if cfg.get("lower") else blocks[-n:]

        per_block: dict = {}
        for k, v in tail.items():
            i, rest = k.split(".", 1)
            per_block.setdefault(int(i), {})[rest] = v
        if len(per_block) != len(sel):
            raise RuntimeError(
                f"tail has {len(per_block)} blocks but target slice has {len(sel)}; "
                f"cfg={cfg}")
        for i, sd in per_block.items():
            sel[i].load_state_dict(sd)
        logger.info(f"applied CAUSAL mask-FT tail from {blob_path} "
                    f"({len(tail)} tensors into blocks "
                    f"[{max(0, L - n)}:{L}], cfg={json.dumps(cfg)})")

        # Vision tower, when the tail was trained with --unfreeze-vit. Tails predating that
        # flag carry no "vision_tail" and skip this entirely.
        vis_tail = blob.get("vision_tail", {})
        n_vit = int(cfg.get("unfreeze_vit", 0) or 0)
        if vis_tail and n_vit > 0:
            visual = (getattr(self.model.model, "visual", None)
                      or getattr(self.model, "visual", None))
            if visual is None:
                raise RuntimeError("tail carries a vision_tail but no vision tower found")
            vstack = getattr(visual, "blocks", None) or getattr(visual, "layers", None)
            if vstack is None:
                raise RuntimeError(f"vision tower has no .blocks/.layers: {type(visual)}")
            vblocks = vstack[-n_vit:]
            per_v: dict = {}
            for k, v in vis_tail.items():
                i, rest = k.split(".", 1)
                per_v.setdefault(int(i), {})[rest] = v
            if len(per_v) != len(vblocks):
                raise RuntimeError(f"vision tail has {len(per_v)} blocks but slice has "
                                   f"{len(vblocks)}; cfg={cfg}")
            for i, sd in per_v.items():
                vblocks[i].load_state_dict(sd)
            logger.info(f"applied CAUSAL mask-FT VISION tail: last {n_vit} tower blocks")
        elif vis_tail or n_vit:
            raise RuntimeError(f"inconsistent vision tail: {len(vis_tail)} tensors vs "
                               f"unfreeze_vit={n_vit}")

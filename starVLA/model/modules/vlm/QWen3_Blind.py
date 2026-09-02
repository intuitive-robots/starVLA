# Blind Qwen3-VL backbone: identical in every way except that it cannot see.
#
# Why this exists
# ---------------
# Tier-2 (encdec-vlm results.md, P3) found that an action head fed ONLY proprioception
# matches the best VLM backbone at 25 demos/task and beats the off-the-shelf Qwen3-VL at
# every demo count, winning the gripper dimension outright. In LIBERO the action at time t
# is close to determined by joint state plus task identity: fixed camera, fixed per-task
# layout, near-identical scripted demos, short horizons. Vision has little marginal value.
#
# That makes every Tier-3 comparison suspect in the same way. If a policy with no image
# information reaches comparable rollout success, then rollout success on this suite cannot
# discriminate between representations either, and the base/encdec/maskft numbers say
# nothing about the representations they were meant to test. This arm measures that ceiling
# directly, and it is the one number that decides whether the others are interpretable.
#
# How the blinding works
# ----------------------
# `pixel_values` is zeroed AFTER the processor has run. Patch count, token count, sequence
# length, attention masks, positions and FLOPs are all bit-identical to the sighted arms --
# the only thing removed is the information in the pixels. Dropping the images instead
# would change the sequence length and confound the comparison with a different prompt.
#
# It stays a constant (not noise): noise injects entropy the model could latch onto as a
# spurious per-step signal, whereas a constant is unambiguously information-free.

from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.training.trainer_utils import initialize_overwatch

from .QWen3 import _QWen3_VL_Interface

logger = initialize_overwatch(__name__)


class _QWen3_Blind_Interface(_QWen3_VL_Interface):
    """Causal Qwen3-VL with the image content zeroed. Config: framework.qwenvl.blind=true."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(config, **kwargs)
        self._blind_warned = False
        logger.info("BLIND backbone: pixel_values will be zeroed on every forward. "
                    "This arm exists to measure how much of LIBERO rollout success is "
                    "reachable with no image information at all.")

    def _blind(self, kwargs):
        pv = kwargs.get("pixel_values")
        if pv is None:
            if not self._blind_warned:
                logger.warning("BLIND: no pixel_values in this batch -- nothing to zero. "
                               "If that happens every step the ablation is not doing "
                               "anything and the arm is invalid.")
                self._blind_warned = True
            return kwargs
        kwargs["pixel_values"] = torch.zeros_like(pv)
        return kwargs

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        return super().forward(**self._blind(kwargs))

    def generate(self, **kwargs):
        return super().generate(**self._blind(kwargs))

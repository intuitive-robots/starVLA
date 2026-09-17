"""QwenPI_v4: Qwen VLM with a fused self/cross-attention layer-wise DiT.

QwenPI_v4 keeps the QwenPI_v3 VLM input, layer selection, projectors, flow
matching loss, sampler, and VLM-token masking.  Its action head is separate:
every DiT block uses action/state tokens as queries and the concatenation of
action/state and VLM tokens as keys/values, followed by an FFN.
"""

from dataclasses import dataclass, field
from typing import Optional

from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenPI_v3 import (
    QwenPI_v3DefaultConfig,
    Qwen_PI_v3,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader_v4 import (
    LayerwiseFlowmatchingActionHeadV4,
    get_action_model_v4,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenPI_v4DefaultConfig(QwenPI_v3DefaultConfig):
    """Defaults for the fixed-topology QwenPI_v4 action head."""

    name: str = "QwenPI_v4"
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM_v4",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v4")
class Qwen_PI_v4(Qwen_PI_v3):
    """Qwen VLM + independent fused self/cross-attention DiT action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # Keep every VLM/data/auxiliary/inference field in lockstep with v3.
        # QwenPI_v3 calls the polymorphic _build_action_model hook below, so no
        # legacy head is constructed transiently.
        merged = merge_framework_config(QwenPI_v4DefaultConfig, config)
        shared_z = dict(merged.framework.get("shared_z", {}) or {})
        if bool(shared_z.get("enabled", False)):
            raise ValueError("QwenPI_v4 shared-z conditioning is not implemented")
        super().__init__(config=merged, **kwargs)

    def _build_action_model(self) -> LayerwiseFlowmatchingActionHeadV4:
        return get_action_model_v4(config=self.config)

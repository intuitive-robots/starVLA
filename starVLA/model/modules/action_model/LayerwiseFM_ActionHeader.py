# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Jinhui YE/ HKUST] in [2026].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].


from dataclasses import dataclass, field
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}  # default for qwen2.5-vl


class LayerwiseFlowmatchingActionHead(nn.Module):
    """
    Layer-wise cross-attention DiT action head.

    NOTE on configuration boundary:
        This module is intentionally decoupled from any specific VLM backbone.
        It ONLY reads from ``global_config.framework.action_model`` (and its
        ``diffusion_model_cfg`` sub-tree).  The framework (e.g. ``Qwen_PI`` /
        ``Qwen_PI_v3``) is responsible for populating ``diffusion_model_cfg``
        with the correct DiT shape **before** calling ``get_action_model``:

            diffusion_model_cfg.num_layers           = <DiT depth>
            diffusion_model_cfg.input_embedding_dim  = <DiT internal hidden>
            diffusion_model_cfg.cross_attention_dim  = <DiT internal hidden>
            diffusion_model_cfg.num_attention_heads  = input_embedding_dim // attention_head_dim

        The head no longer reads ``framework.qwenvl.*`` directly.  See
        ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for details.
    """

    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # ------------------------------------------------------------------
        # Pure consumer: trust diffusion_model_cfg.  Apply DiTConfig only as
        # a fallback for keys the framework forgot to set.  This keeps the
        # head free of any VLM-specific knowledge.
        # ------------------------------------------------------------------
        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v

        # Build a plain dict view and drop framework-side hints that are NOT
        # DiT constructor kwargs (e.g. `action_dit_hidden_dim`).  This keeps the
        # head agnostic of any framework convention while still being safe
        # against accidentally-leaked hint keys.
        _DIT_NON_KWARGS = {"action_dit_hidden_dim"}
        diffusion_model_cfg_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = DiT(**diffusion_model_cfg_kwargs)  # TODO: ideally copy LLM init from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        # `action_horizon` is the canonical chunk length.  Legacy YAMLs are
        # normalised by share_tools.apply_config_compat upstream, so this
        # head never reads `future_action_window_size`.
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps
        self.layerwise_attention_layout = str(
            action_config.get("layerwise_attention_layout", "legacy_all_cross")
        ).lower()
        if self.layerwise_attention_layout not in {"legacy_all_cross", "alternating"}:
            raise ValueError(
                "layerwise_attention_layout must be 'legacy_all_cross' or 'alternating', "
                f"got {self.layerwise_attention_layout!r}"
            )

        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                output_dim=self.input_embedding_dim,
            )
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        num_future_tokens = int(action_config.num_target_vision_tokens)
        if num_future_tokens < 0:
            raise ValueError("num_target_vision_tokens must be non-negative")
        # Avoid a zero-sized parameter: it is unnecessary and can upset
        # parameter partitioning/checkpointing in distributed training.
        self.future_tokens = (
            nn.Embedding(num_future_tokens, self.input_embedding_dim)
            if num_future_tokens > 0
            else None
        )
        if self.future_tokens is not None:
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

        # Optional inference-only successful-pair residual readout.  It is
        # loaded lazily after checkpoint restoration, leaving existing
        # checkpoint/state-dict behavior unchanged when the environment
        # variable is unset.
        self._flow_residual_readout_path = None
        self._flow_residual_readout = None
        self._flow_residual_mean = None
        self._flow_residual_scale = None
        self._flow_residual_strength = None
        self._flow_residual_norm_cap_ratio = None

    def _maybe_load_flow_residual_readout(self, device: torch.device) -> None:
        requested = os.environ.get("STARVLA_FLOW_RESIDUAL_READOUT", "").strip()
        if not requested:
            return
        resolved = str(Path(requested).expanduser().resolve())
        if self._flow_residual_readout_path == resolved:
            return
        if self._flow_residual_readout_path is not None:
            raise RuntimeError(
                "Changing STARVLA_FLOW_RESIDUAL_READOUT after inference started is unsupported"
            )
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if payload.get("format") != "starvla_successful_pair_linear_readout_v1":
            raise ValueError(f"unsupported flow residual readout: {payload.get('format')!r}")
        if payload.get("locus") != "pi_late" or payload.get("arm") != "paired_balanced":
            raise ValueError(
                "closed-loop flow correction requires the paired_balanced pi_late readout"
            )
        mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32).reshape(1, 1, -1)
        scale = torch.as_tensor(payload["feature_scale"], dtype=torch.float32).reshape(1, 1, -1)
        weight = torch.as_tensor(payload["state_dict"]["weight"], dtype=torch.float32)
        bias = torch.as_tensor(payload["state_dict"]["bias"], dtype=torch.float32)
        if weight.shape != (self.action_dim, mean.shape[-1]) or bias.shape != (self.action_dim,):
            raise ValueError(
                f"flow residual readout shape mismatch: weight={tuple(weight.shape)}, "
                f"bias={tuple(bias.shape)}, expected=({self.action_dim}, {mean.shape[-1]})"
            )
        adapter = nn.Linear(mean.shape[-1], self.action_dim)
        adapter.load_state_dict({"weight": weight, "bias": bias})
        adapter.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
        strength = float(os.environ.get("STARVLA_FLOW_RESIDUAL_STRENGTH", "1.0"))
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"invalid STARVLA_FLOW_RESIDUAL_STRENGTH={strength!r}")
        cap_value = os.environ.get("STARVLA_FLOW_RESIDUAL_NORM_CAP_RATIO", "").strip()
        norm_cap_ratio = None if not cap_value else float(cap_value)
        if norm_cap_ratio is not None and (
            not math.isfinite(norm_cap_ratio) or norm_cap_ratio <= 0.0
        ):
            raise ValueError(
                "STARVLA_FLOW_RESIDUAL_NORM_CAP_RATIO must be finite and positive, "
                f"got {norm_cap_ratio!r}"
            )
        self._flow_residual_readout = adapter
        self._flow_residual_mean = mean.to(device=device)
        self._flow_residual_scale = scale.to(device=device)
        self._flow_residual_readout_path = resolved
        self._flow_residual_strength = strength
        self._flow_residual_norm_cap_ratio = norm_cap_ratio
        print(
            "Loaded paired PI pi_late flow residual readout "
            f"from {resolved} (base={payload.get('checkpoint')}, strength={strength}, "
            f"norm_cap_ratio={norm_cap_ratio})",
            flush=True,
        )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return self.config.noise_s * (1 - sample)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _assemble_action_tokens(self, action_features, state_features=None):
        token_groups = []
        if state_features is not None:
            token_groups.append(state_features)
        if self.future_tokens is not None:
            token_groups.append(
                self.future_tokens.weight.unsqueeze(0).expand(action_features.shape[0], -1, -1)
            )
        token_groups.append(action_features)
        return action_features if len(token_groups) == 1 else torch.cat(token_groups, dim=1)

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        return_clean_actions: bool = False,
        z_conditioning: torch.Tensor = None,
        encoder_memory_keep: torch.Tensor = None,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, D_action)
        """
        device = actions.device
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = self._assemble_action_tokens(action_features, state_features)

        # Route through DiT.forward so the configured cross/self-attention
        # interleaving is honored. The former hand-written loop passed VLM
        # states to every block, silently turning intended self-attention
        # blocks into cross-attention blocks.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
            force_layerwise_all_cross=self.layerwise_attention_layout == "legacy_all_cross",
            extra_conditioning=z_conditioning,
            cross_attention_row_mask=encoder_memory_keep,
        )

        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        if not return_clean_actions:
            return loss

        # Linear conditional-flow path:
        #   x_t = (1-t) * noise + t * action,  v = action - noise
        # hence action = x_t + (1-t) * v.  Exposing this differentiable clean-action
        # estimate lets framework-side objectives regularise multi-horizon motion without
        # duplicating or reaching into the DiT implementation.
        clean_actions = noisy_trajectory + (1 - t) * pred_actions
        return loss, clean_actions

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        z_conditioning: torch.Tensor = None,
        encoder_memory_keep: torch.Tensor = None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        self._maybe_load_flow_residual_readout(device)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = self._assemble_action_tokens(action_features, state_features)

            model_kwargs = dict(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_list,
                timestep=timesteps_tensor,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
                force_layerwise_all_cross=self.layerwise_attention_layout == "legacy_all_cross",
                extra_conditioning=z_conditioning,
                cross_attention_row_mask=encoder_memory_keep,
            )
            if self._flow_residual_readout is None:
                model_output = self.model(**model_kwargs)
                late_hidden = None
            else:
                model_output, hidden_states = self.model(
                    **model_kwargs, return_all_hidden_states=True
                )
                late_hidden = hidden_states[-1][:, -self.action_horizon :]
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            if late_hidden is not None:
                standardized = (
                    late_hidden.float() - self._flow_residual_mean
                ) / self._flow_residual_scale
                residual = self._flow_residual_readout(standardized)
                if self._flow_residual_norm_cap_ratio is not None:
                    base_norm = pred_velocity.float().norm(dim=-1, keepdim=True)
                    residual_norm = residual.norm(dim=-1, keepdim=True)
                    maximum = self._flow_residual_norm_cap_ratio * base_norm
                    residual = residual * torch.clamp(
                        maximum / residual_norm.clamp_min(1e-12), max=1.0
                    )
                pred_velocity = pred_velocity + self._flow_residual_strength * residual.to(
                    pred_velocity.dtype
                )

            # Euler integration
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(global_config=config)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass

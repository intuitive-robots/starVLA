"""Shared-z bottleneck: pooler, auxiliary heads, losses.

Lifted verbatim out of ``QwenPI_v3`` so both action-head frameworks can use it. The
bottleneck pools the encoder into a single ``dim``-d vector z, supervises it against
structured targets (ground relations, trajectory, phase, ...), and hands z to the DiT as
``extra_conditioning``. The legacy ``memory_dropout_rate: 1`` path drops the entire
cross-attention result, so z directly conditions only self-attention blocks in an
alternating DiT. ``cross_memory_tokens`` fixes that by decoding the same z into
always-visible cross-attention memory tokens while masking only encoder/readout tokens.

Nothing here is PI-specific. The mixin needs its host to provide:

* ``self.config``                  -- the framework config
* ``self.qwen_vl_interface``       -- for encoding future frames (temporal target)
* ``self.training`` / ``self.is_inference``

``shared_z.decoder_memory`` stays in QwenPI_v3: it patches the enc-dec decoder's memory
hook, which only exists there.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from starVLA.dataloader.cot_resolver import extract_structured_cot_targets


class SharedZPooler(nn.Module):
    """One-way learned-query pooler: queries read encoder tokens, never vice versa."""

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        num_queries: int = 4,
        query_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if num_queries < 1 or z_dim < 1 or query_dim % num_heads:
            raise ValueError("invalid shared-z query dimensions")
        self.input_norm = nn.LayerNorm(input_dim)
        self.kv_proj = nn.Linear(input_dim, query_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            query_dim, num_heads=num_heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.Linear(num_queries * query_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or valid.shape != hidden.shape[:2]:
            raise ValueError(
                f"shared-z hidden/mask mismatch: {tuple(hidden.shape)} vs {tuple(valid.shape)}"
            )
        memory = self.kv_proj(self.input_norm(hidden))
        query = self.queries.to(memory.dtype)[None].expand(hidden.shape[0], -1, -1)
        pooled, _ = self.attention(
            query, memory, memory, key_padding_mask=~valid.bool(), need_weights=False
        )
        return self.output(pooled.flatten(1))


class SharedZRegressionHead(nn.Module):
    def __init__(self, z_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim), nn.Linear(z_dim, z_dim), nn.SiLU(),
            nn.Linear(z_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SharedZMixin:
    """Config parsing, module construction and losses for the shared-z bottleneck."""

    def _configure_shared_z(self, config, diffusion_model_cfg: dict) -> dict:
        """Parse framework.shared_z and size the DiT's extra conditioning. Returns the cfg."""
        shared_z_cfg = dict(config.framework.get("shared_z", {}) or {})
        self.shared_z_enabled = bool(shared_z_cfg.get("enabled", False))
        self.shared_z_dim = int(shared_z_cfg.get("dim", 128))
        self.shared_z_memory_dropout_rate = float(
            shared_z_cfg.get("memory_dropout_rate", 0.15)
        )
        if not 0.0 <= self.shared_z_memory_dropout_rate <= 1.0:
            raise ValueError("framework.shared_z.memory_dropout_rate must be in [0,1]")
        self.shared_z_shuffle_targets = bool(shared_z_cfg.get("shuffle_targets", False))
        self.shared_z_distribution_weight = float(
            shared_z_cfg.get("distribution_weight", 0.01)
        )
        self.shared_z_separation_weight = float(
            shared_z_cfg.get("separation_weight", 0.1)
        )
        self.shared_z_separation_margin = float(
            shared_z_cfg.get("separation_margin", 0.5)
        )
        self.shared_z_target_specs = {
            str(name): dict(spec)
            for name, spec in dict(shared_z_cfg.get("targets", {}) or {}).items()
        }
        temporal_cfg = dict(shared_z_cfg.get("temporal", {}) or {})
        self.shared_z_temporal_enabled = bool(temporal_cfg.get("enabled", False))
        self.shared_z_future_offset = int(temporal_cfg.get("future_offset", 8))
        self.shared_z_temporal_weight = float(temporal_cfg.get("weight", 1.0))
        self.shared_z_decoder_memory = bool(shared_z_cfg.get("decoder_memory", False))
        self.shared_z_cross_memory_tokens = int(
            shared_z_cfg.get("cross_memory_tokens", 0)
        )
        if self.shared_z_cross_memory_tokens < 0:
            raise ValueError("framework.shared_z.cross_memory_tokens must be >= 0")
        if self.shared_z_cross_memory_tokens and not self.shared_z_enabled:
            raise ValueError(
                "framework.shared_z.cross_memory_tokens requires shared_z.enabled=true"
            )
        diffusion_model_cfg["extra_conditioning_dim"] = (
            self.shared_z_dim if self.shared_z_enabled else 0
        )
        return shared_z_cfg

    def _build_shared_z_modules(self, shared_z_cfg: dict, llm_hidden_size: int) -> None:
        """Pooler, per-target heads, and the temporal predictor/decoder pair."""
        self.shared_z_pooler = None
        self.shared_z_heads = nn.ModuleDict()
        self.shared_z_future_predictor = None
        self.shared_z_difference_decoder = None
        self.shared_z_decoder_up = None
        self.shared_z_memory_projector = None
        self._shared_z_from_decoder = None
        if not self.shared_z_enabled:
            return
        self.shared_z_pooler = SharedZPooler(
            input_dim=llm_hidden_size,
            z_dim=self.shared_z_dim,
            num_queries=int(shared_z_cfg.get("num_queries", 4)),
            query_dim=int(shared_z_cfg.get("query_dim", 512)),
            num_heads=int(shared_z_cfg.get("num_heads", 8)),
        )
        if self.shared_z_cross_memory_tokens:
            cross_attention_dim = int(
                self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
            )
            self.shared_z_memory_projector = nn.Sequential(
                nn.LayerNorm(self.shared_z_dim),
                nn.Linear(
                    self.shared_z_dim,
                    self.shared_z_cross_memory_tokens * cross_attention_dim,
                ),
            )
        self.shared_z_heads = nn.ModuleDict({
            name: SharedZRegressionHead(self.shared_z_dim, int(spec["dim"]))
            for name, spec in self.shared_z_target_specs.items()
            if str(spec.get("loss", "smooth_l1")) != "cross_entropy"
        })
        for name, spec in self.shared_z_target_specs.items():
            if str(spec.get("loss", "smooth_l1")) == "cross_entropy":
                self.shared_z_heads[name] = SharedZRegressionHead(
                    self.shared_z_dim, int(spec["dim"])
                )
        if self.shared_z_temporal_enabled:
            action_flat_dim = self.shared_z_future_offset * int(
                self.config.framework.action_model.action_dim
            )
            self.shared_z_future_predictor = SharedZRegressionHead(
                self.shared_z_dim + action_flat_dim, self.shared_z_dim
            )
            self.shared_z_difference_decoder = SharedZRegressionHead(
                self.shared_z_dim, action_flat_dim
            )

    def _encode_future_shared_z(
        self,
        future_images: List,
        instructions: List[str],
    ) -> torch.Tensor:
        """Encode target frames without exposing pixels or gradients to the policy path."""
        if not self.shared_z_enabled or self.shared_z_pooler is None:
            raise RuntimeError("future shared-z requested while shared_z is disabled")
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=future_images, instructions=instructions,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            hidden = outputs.hidden_states[-1]
            valid = getattr(self.qwen_vl_interface, "_last_encoder_attention_mask", None)
            if valid is None:
                valid = inputs.get("attention_mask")
            if valid is None:
                valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
            valid = valid[:, :hidden.shape[1]].to(device=hidden.device, dtype=torch.bool)
            target_z = self.shared_z_pooler(hidden, valid)
        return target_z.detach()

    @staticmethod
    def _gather_shared_z(z: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return z
        from torch.distributed.nn.functional import all_gather
        return torch.cat(tuple(all_gather(z)), dim=0)

    def _shared_z_distribution_loss(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Weak-SIGReg-style random-projection moment matching on the global batch."""
        gathered = self._gather_shared_z(z.float())
        centered = gathered - gathered.mean(dim=0, keepdim=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(271828)
        directions = torch.randn(
            64, self.shared_z_dim, generator=generator, dtype=torch.float32
        ).to(device=z.device)
        directions = F.normalize(directions, dim=-1)
        projected = centered @ directions.t()
        mean = projected.mean(dim=0)
        variance = projected.var(dim=0, unbiased=False)
        loss = mean.square().mean() + (variance - 1.0).square().mean()
        singular = torch.linalg.svdvals(centered.detach())
        probabilities = singular.square() / singular.square().sum().clamp_min(1.0e-12)
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        )
        return loss, {
            "shared_z/distribution_loss": loss.detach(),
            "shared_z/std_mean": centered.std(dim=0, unbiased=False).mean().detach(),
            "shared_z/effective_rank": effective_rank.detach(),
        }

    def _shared_z_supervision(
        self,
        z: torch.Tensor,
        examples: List[dict],
        conversations: List[list | None],
        actions_target: torch.Tensor,
        future_z: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parsed = [
            example.get("cot_structured_targets")
            or extract_structured_cot_targets(conversation)
            for example, conversation in zip(examples, conversations)
        ]
        if self.shared_z_shuffle_targets and len(parsed) > 1:
            parsed = parsed[1:] + parsed[:1]
            actions_target = actions_target.roll(shifts=-1, dims=0)
            if future_z is not None:
                future_z = future_z.roll(shifts=-1, dims=0)
            shuffle_fraction = 1.0
        else:
            shuffle_fraction = 0.0

        losses, weights = [], []
        metrics: dict[str, torch.Tensor] = {}
        metrics["shared_z/shuffle_fraction"] = torch.tensor(
            shuffle_fraction, device=z.device
        )
        for name, spec in self.shared_z_target_specs.items():
            dim = int(spec["dim"])
            weight = float(spec.get("weight", 1.0))
            prediction = self.shared_z_heads[name](z).float()
            present = []
            values = []
            for target in parsed:
                value = target.get(name)
                if name == "ground_relation" and value is None:
                    point, box = target.get("target_point"), target.get("object_box")
                    if point is not None and box is not None:
                        value = [point[0] - 0.5 * (box[0] + box[2]),
                                 point[1] - 0.5 * (box[1] + box[3])]
                keep = value is not None and (
                    isinstance(value, (int, np.integer)) or len(value) == dim
                )
                present.append(keep)
                if keep:
                    values.append(value)
            if any(present):
                select = torch.tensor(present, device=z.device, dtype=torch.bool)
                if str(spec.get("loss", "smooth_l1")) == "cross_entropy":
                    target_tensor = torch.tensor(values, device=z.device, dtype=torch.long)
                    head_loss = F.cross_entropy(prediction[select], target_tensor)
                else:
                    target_tensor = torch.tensor(values, device=z.device, dtype=torch.float32)
                    head_loss = F.smooth_l1_loss(prediction[select], target_tensor)
            else:
                head_loss = prediction.sum() * 0.0
            losses.append(head_loss)
            # Keep the zero-valued head in the graph on every rank (DeepSpeed
            # requires matching parameter participation), but do not let a target
            # absent from the whole local batch dilute the active-loss denominator.
            weights.append(weight if any(present) else 0.0)
            metrics[f"shared_z/{name}_loss"] = head_loss.detach()
            metrics[f"shared_z/{name}_coverage"] = torch.tensor(
                sum(present) / max(len(present), 1), device=z.device
            )

        distribution_loss, distribution_metrics = self._shared_z_distribution_loss(z)
        metrics.update(distribution_metrics)
        losses.append(distribution_loss)
        weights.append(self.shared_z_distribution_weight)

        separation_terms = []
        for left in range(len(examples)):
            left_point = parsed[left].get("target_point")
            if left_point is None:
                continue
            for right in range(left + 1, len(examples)):
                if examples[left].get("lang") != examples[right].get("lang"):
                    continue
                right_point = parsed[right].get("target_point")
                if right_point is None:
                    continue
                point_delta = torch.tensor(left_point, device=z.device) - torch.tensor(
                    right_point, device=z.device
                )
                if point_delta.float().norm() < 0.05:
                    continue
                latent_distance = (z[left].float() - z[right].float()).norm() / (
                    self.shared_z_dim ** 0.5
                )
                separation_terms.append(
                    F.relu(self.shared_z_separation_margin - latent_distance)
                )
        separation_loss = (
            torch.stack(separation_terms).mean() if separation_terms else z.sum() * 0.0
        )
        losses.append(separation_loss)
        weights.append(self.shared_z_separation_weight if separation_terms else 0.0)
        metrics["shared_z/separation_loss"] = separation_loss.detach()
        metrics["shared_z/separation_pairs"] = torch.tensor(
            len(separation_terms), device=z.device, dtype=torch.float32
        )

        if self.shared_z_temporal_enabled:
            action_prefix = actions_target[:, :self.shared_z_future_offset].float().flatten(1)
            if future_z is not None:
                predicted_future = self.shared_z_future_predictor(
                    torch.cat([z.float(), action_prefix], dim=-1)
                )
                transition_loss = 1.0 - F.cosine_similarity(
                    predicted_future, future_z.float(), dim=-1
                ).mean()
                decoded_action = self.shared_z_difference_decoder(
                    future_z.float() - z.float()
                )
                difference_loss = F.smooth_l1_loss(decoded_action, action_prefix)
            else:
                transition_loss = sum(
                    parameter.sum() * 0.0
                    for parameter in self.shared_z_future_predictor.parameters()
                )
                difference_loss = sum(
                    parameter.sum() * 0.0
                    for parameter in self.shared_z_difference_decoder.parameters()
                )
            temporal_loss = 0.5 * (transition_loss + difference_loss)
            losses.append(temporal_loss)
            weights.append(self.shared_z_temporal_weight if future_z is not None else 0.0)
            metrics["shared_z/transition_loss"] = transition_loss.detach()
            metrics["shared_z/difference_action_loss"] = difference_loss.detach()

        positive_weight = sum(weight for weight in weights if weight > 0.0)
        if positive_weight <= 0.0:
            raise ValueError("shared-z loss weights must sum to a positive value")
        total = sum(weight * loss for weight, loss in zip(weights, losses)) / positive_weight
        metrics["shared_z/loss"] = total.detach()
        return total, metrics

    def _shared_z_memory_keep(self, batch: int, device: torch.device) -> torch.Tensor:
        rate = self.shared_z_memory_dropout_rate
        if self.training:
            return torch.rand(batch, device=device) >= rate
        if rate >= 1.0:
            return torch.zeros(batch, device=device, dtype=torch.bool)
        return torch.ones(batch, device=device, dtype=torch.bool)

    def _augment_shared_z_cross_memory(
        self,
        encoder_memory: torch.Tensor,
        encoder_valid: torch.Tensor,
        z: torch.Tensor,
        encoder_memory_keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend always-visible z tokens and mask only optional encoder memory.

        The legacy memory-dropout path masks the output of every cross-attention block.
        That also removes attention to z-conditioned queries.  With cross-memory tokens,
        z remains available to every cross block while dropout affects only the original
        encoder/readout tokens.
        """
        if self.shared_z_memory_projector is None:
            raise RuntimeError("shared-z cross-memory tokens are not configured")
        if encoder_memory.ndim != 3 or encoder_valid.ndim != 2:
            raise ValueError("encoder_memory must be [B,L,D] and encoder_valid [B,L]")
        if encoder_memory.shape[:2] != encoder_valid.shape:
            raise ValueError(
                "encoder memory/mask mismatch: "
                f"{tuple(encoder_memory.shape)} vs {tuple(encoder_valid.shape)}"
            )
        batch, _, width = encoder_memory.shape
        if z.shape != (batch, self.shared_z_dim):
            raise ValueError(
                f"shared z must be {(batch, self.shared_z_dim)}, got {tuple(z.shape)}"
            )
        if encoder_memory_keep.shape != (batch,):
            raise ValueError(
                f"encoder_memory_keep must be {(batch,)}, got {tuple(encoder_memory_keep.shape)}"
            )

        z_memory = self.shared_z_memory_projector(z).reshape(
            batch, self.shared_z_cross_memory_tokens, width
        )
        z_memory = z_memory.to(device=encoder_memory.device, dtype=encoder_memory.dtype)
        z_valid = torch.ones(
            batch,
            self.shared_z_cross_memory_tokens,
            device=encoder_valid.device,
            dtype=torch.bool,
        )
        kept_encoder_valid = encoder_valid.bool() & encoder_memory_keep.bool()[:, None]
        return (
            torch.cat([z_memory, encoder_memory], dim=1),
            torch.cat([z_valid, kept_encoder_valid], dim=1),
        )

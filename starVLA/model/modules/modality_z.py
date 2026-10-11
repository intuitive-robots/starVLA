"""Modality-preserving z-only interface for action conditioning.

The interface pools *final, contextualized* VLM tokens independently for the external
camera, wrist camera, language, and quantized-state spans.  The action expert receives only
the resulting fixed-size latent sequence; it never sees the original VLM memory.

Each pooler is deliberately independent.  In particular, the external and wrist cameras do
not share weights: their viewpoints and control roles differ.  Cross-modal fusion happens in
the VLM before pooling and in the action expert after concatenation, not in an extra interface
transformer that could repair a causal backbone after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


class _PoolerLayer(nn.Module):
    """Pre-norm query self-attention, masked cross-attention, and FFN."""

    def __init__(
        self,
        memory_dim: int,
        latent_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(
            latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_query_norm = nn.LayerNorm(latent_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.cross_attn = nn.MultiheadAttention(
            latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=memory_dim,
            vdim=memory_dim,
        )
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_norm(queries)
        update, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + update

        q = self.cross_query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        update, _ = self.cross_attn(
            q,
            normalized_memory,
            normalized_memory,
            key_padding_mask=~memory_valid.bool(),
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ffn(self.ffn_norm(queries))


class ModalityQueryPooler(nn.Module):
    """Compress one contextualized modality span into learned latent tokens."""

    def __init__(
        self,
        memory_dim: int,
        output_dim: int,
        num_tokens: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_tokens < 1 or num_layers < 1:
            raise ValueError("modality-z token and layer counts must be positive")
        if latent_dim % num_heads:
            raise ValueError("modality-z latent_dim must be divisible by num_heads")
        self.queries = nn.Parameter(torch.empty(1, num_tokens, latent_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList(
            _PoolerLayer(memory_dim, latent_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.output = nn.Linear(latent_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, memory: torch.Tensor, memory_valid: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3 or memory_valid.shape != memory.shape[:2]:
            raise ValueError(
                "modality pooler memory/mask mismatch: "
                f"{tuple(memory.shape)} vs {tuple(memory_valid.shape)}"
            )
        if not bool(memory_valid.any(dim=1).all()):
            raise RuntimeError("every row must contain at least one token for each modality")
        queries = self.queries.expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory, memory_valid)
        return self.output_norm(self.output(self.latent_norm(queries)))


def _run_fp32(module: nn.Module, memory: torch.Tensor, memory_valid: torch.Tensor) -> torch.Tensor:
    """Run a pooler in fp32 and return the result in the memory's dtype.

    Under DeepSpeed bf16 the pooler weights are bf16 and the forward runs inside a
    bf16 autocast. The LayerNorm and attention on raw VLM hidden states then
    overflowed to NaN mid-training on every modality-z run (C28/C29). Casting the
    parameters with functional_call keeps gradients flowing to the bf16 weights.
    """
    params = {k: v.float() for k, v in module.named_parameters()}
    params.update({k: v.float() for k, v in module.named_buffers()})
    with torch.autocast(device_type=memory.device.type, enabled=False):
        out = torch.func.functional_call(module, params, (memory.float(), memory_valid))
    return out.to(memory.dtype)


@dataclass(frozen=True)
class ModalityMasks:
    external: torch.Tensor
    wrist: torch.Tensor
    language: torch.Tensor
    state: torch.Tensor


class ModalityZInterface(nn.Module):
    """Independent external/wrist/language/state poolers with no post-pool fusion."""

    ORDER = ("external", "wrist", "language", "state")

    def __init__(self, input_dim: int, config: dict) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(config.get("latent_dim", 768))
        self.num_layers = int(config.get("num_layers", 2))
        self.num_heads = int(config.get("num_heads", 8))
        self.ffn_dim = int(config.get("ffn_dim", 4 * self.latent_dim))
        self.dropout = float(config.get("dropout", 0.0))
        self.token_counts: Dict[str, int] = {
            "external": int(config.get("external_tokens", 8)),
            "wrist": int(config.get("wrist_tokens", 8)),
            "language": int(config.get("language_tokens", 4)),
            "state": int(config.get("state_tokens", 4)),
        }
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("framework.modality_z.dropout must be in [0,1)")

        def build(name: str) -> ModalityQueryPooler:
            return ModalityQueryPooler(
                memory_dim=self.input_dim,
                output_dim=self.input_dim,
                num_tokens=self.token_counts[name],
                latent_dim=self.latent_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
            )

        # Do not share camera weights: agent-view and wrist-view features have different
        # geometry and control semantics.
        self.external_pooler = build("external")
        self.wrist_pooler = build("wrist")
        self.language_pooler = build("language")
        self.state_pooler = build("state")
        self.num_output_tokens = sum(self.token_counts.values())
        self._logged_contract = False

    @staticmethod
    def _image_span_masks(
        input_ids: torch.Tensor,
        valid: torch.Tensor,
        image_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        external = torch.zeros_like(valid, dtype=torch.bool)
        wrist = torch.zeros_like(valid, dtype=torch.bool)
        image_positions = input_ids.eq(int(image_token_id)) & valid
        for row in range(input_ids.shape[0]):
            positions = torch.where(image_positions[row])[0]
            if positions.numel() == 0:
                raise RuntimeError(f"modality-z row {row}: no image tokens found")
            split_after = torch.where(positions[1:] != positions[:-1] + 1)[0] + 1
            spans = torch.tensor_split(positions, split_after.cpu().tolist())
            if len(spans) != 2:
                raise RuntimeError(
                    f"modality-z row {row}: expected exactly two image spans "
                    f"(external, wrist), found {len(spans)}"
                )
            external[row, spans[0]] = True
            wrist[row, spans[1]] = True
        return external, wrist

    @classmethod
    def build_masks(
        cls,
        input_ids: torch.Tensor,
        valid: torch.Tensor,
        state_mask: torch.Tensor,
        image_token_id: int,
        expected_state_tokens: int,
    ) -> ModalityMasks:
        if input_ids.shape != valid.shape or state_mask.shape != valid.shape:
            raise ValueError("modality-z input_ids, valid mask, and state mask must match")
        valid = valid.bool()
        state = state_mask.bool() & valid
        counts = state.sum(dim=1)
        if not bool(counts.eq(expected_state_tokens).all()):
            raise RuntimeError(
                f"modality-z expected {expected_state_tokens} state-ID tokens per row, "
                f"got {counts.tolist()}"
            )
        external, wrist = cls._image_span_masks(input_ids, valid, image_token_id)
        language = valid & ~external & ~wrist & ~state
        if not bool(language.any(dim=1).all()):
            raise RuntimeError("modality-z found an empty language span")
        return ModalityMasks(external, wrist, language, state)

    @staticmethod
    def _pack(hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = mask.sum(dim=1)
        width = int(counts.max().item())
        packed = hidden.new_zeros(hidden.shape[0], width, hidden.shape[-1])
        packed_valid = torch.zeros(
            hidden.shape[0], width, device=hidden.device, dtype=torch.bool
        )
        for row in range(hidden.shape[0]):
            selected = hidden[row, mask[row]]
            packed[row, : selected.shape[0]] = selected
            packed_valid[row, : selected.shape[0]] = True
        return packed, packed_valid

    def forward(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        input_ids: torch.Tensor,
        state_mask: torch.Tensor,
        image_token_id: int,
        expected_state_tokens: int,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"modality-z hidden must be [B,L,H], got {tuple(hidden.shape)}")
        length = hidden.shape[1]
        input_ids = input_ids[:, :length].to(device=hidden.device)
        valid = valid[:, :length].to(device=hidden.device, dtype=torch.bool)
        state_mask = state_mask[:, :length].to(device=hidden.device, dtype=torch.bool)
        if input_ids.shape != hidden.shape[:2] or valid.shape != hidden.shape[:2]:
            raise RuntimeError("modality-z metadata does not align with VLM hidden states")

        masks = self.build_masks(
            input_ids, valid, state_mask, image_token_id, expected_state_tokens
        )
        outputs = []
        observed = {}
        for name in self.ORDER:
            memory, memory_valid = self._pack(hidden, getattr(masks, name))
            observed[name] = memory_valid.sum(dim=1).tolist()
            outputs.append(_run_fp32(getattr(self, f"{name}_pooler"), memory, memory_valid))
        z = torch.cat(outputs, dim=1)
        if z.shape[1:] != (self.num_output_tokens, self.input_dim):
            raise RuntimeError(
                "modality-z output contract failed: "
                f"expected (*,{self.num_output_tokens},{self.input_dim}), got {tuple(z.shape)}"
            )
        if not self._logged_contract:
            self._logged_contract = True
            print(
                "__MODALITY_Z__ "
                f"tokens={self.token_counts} input_counts={observed} "
                f"latent_dim={self.latent_dim} layers={self.num_layers} "
                f"output_shape={tuple(z.shape)} full_memory_visible=false",
                flush=True,
            )
        return z

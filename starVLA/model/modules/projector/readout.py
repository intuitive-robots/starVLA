"""
Q-Former style readout projector for VLA action conditioning.

Compresses the full VLM token sequence [B, L, H] into a fixed set of
learned readout tokens [B, num_tokens, H] via stacked cross-attending layers,
before passing to the DiT action expert.

Each layer performs:
    1. Self-attention  among readout queries  (queries integrate each other)
    2. Cross-attention queries → VLM hidden states  (pull in task/scene info)
    3. FFN

By cross-attending to the full VLM sequence at every layer, the readout tokens
progressively refine a compressed summary of the scene and instruction,
analogous to BLIP-2's Q-Former but used for action conditioning rather than
image-text alignment.

Benefits over raw full-sequence DiT cross-attention:
  - Fixed context length O(num_tokens) regardless of image resolution / CoT length
  - Learned compression: queries specialise to task-relevant features
  - Multiple refinement passes before the DiT sees the context

Config (framework.readout_tokens):
    enabled:    false         # set true to activate
    num_tokens: 32            # number of output readout queries (recommend 32)
    num_layers: 2             # number of Q-Former layers     (recommend 2)
    num_heads:  8             # MHA heads                     (recommend 8)
    ffn_dim:    null          # FFN hidden width; null → 2 × hidden_dim
    dropout:    0.0
"""

import torch
import torch.nn as nn
from typing import Optional


class _ReadoutLayer(nn.Module):
    """Single Q-Former block: self-attn → cross-attn → FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        # ── self-attention among query tokens ────────────────────────────────
        self.self_attn  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_sa    = nn.LayerNorm(hidden_dim)

        # ── cross-attention: queries → VLM hidden states ─────────────────────
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_ca    = nn.LayerNorm(hidden_dim)

        # ── FFN ───────────────────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, vl_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries:  [B, num_tokens, H]
            vl_embs:  [B, L, H]
        Returns:
            [B, num_tokens, H]
        """
        # Self-attention (pre-norm)
        q = self.norm_sa(queries)
        sa_out, _ = self.self_attn(q, q, q)
        queries = queries + sa_out

        # Cross-attention to VLM hidden states (pre-norm)
        q = self.norm_ca(queries)
        ca_out, _ = self.cross_attn(q, vl_embs, vl_embs)
        queries = queries + ca_out

        # FFN (pre-norm)
        queries = queries + self.ffn(self.norm_ff(queries))

        return queries


class ReadoutProjector(nn.Module):
    """
    Stacked Q-Former that compresses VLM hidden states into fixed readout tokens.

    Args:
        hidden_dim:   VLM hidden size (= DiT cross_attention_dim)
        num_tokens:   Number of readout query tokens  (default 32)
        num_layers:   Number of Q-Former layers       (default 2)
        num_heads:    MHA heads                        (default 8)
        ffn_dim:      FFN width; None → 2 × hidden_dim
        dropout:      Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 2

        self.queries = nn.Parameter(torch.empty(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.layers = nn.ModuleList([
            _ReadoutLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, vl_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vl_embs: [B, L, H]  — full VLM last-layer hidden states
        Returns:
            [B, num_tokens, H]  — compressed readout representations
        """
        queries = self.queries.expand(vl_embs.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, vl_embs)
        return self.norm_out(queries)

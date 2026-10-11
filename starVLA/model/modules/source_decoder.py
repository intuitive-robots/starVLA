"""Fusion forcing at the source tokens (2026-10-11): source decoder and head-input dropout.

Probes (fusion_probe*.py, 2026-10-11) showed that masked training fuses the encoder only partly into the image tokens,
and that the action head often reads state / task from the separate text tokens instead. Two training-only devices
push the information into the image tokens and make the head use it. Neither changes inference.

framework.source_decoder (default off)
  A small decoder outside the encoder sequence. One learned query per target cross-attends to the encoder's OUTPUT
  hidden states (the tokens the action head reads) restricted to the IMAGE tokens, and regresses the target:
    state  the row's true state in model space (examples[i]["state"]), even when the state tokens were masked
    task   the mean input embedding of the row's raw instruction (detached), even when the instruction was masked
  A query never reads its own modality, so the only route to the target is through the image tokens: copied via
  attention when the source is visible to the encoder, inferred from pixels when it was masked. Rows whose target was
  masked weigh ``weight_masked`` (default 1), the others ``weight_visible`` (default 0.25). The decoder is weak on
  purpose (one attention head, ``dim`` 256, a linear readout) so the information has to be explicit in the tokens.
  The loss joins the structured_aux slot (trainer.loss_scale.structured_aux), added to any fusion-register loss.

framework.head_input_dropout (default off)
  In a ``rate`` fraction of training rows the action head may cross-attend only to the image tokens, so the action
  loss itself has to find state and task in them (the encoder still sees everything).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _SourceQuery(nn.Module):
    """One learned query, single-head attention over the allowed source tokens, linear readout."""

    def __init__(self, hidden: int, dim: int, out: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.k = nn.Linear(hidden, dim)
        self.v = nn.Linear(hidden, dim)
        self.q = nn.Parameter(torch.randn(dim) / dim ** 0.5)
        self.out = nn.Linear(dim, out)
        self.scale = dim ** -0.5

    def forward(self, hidden: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden)
        logits = (self.k(h) @ self.q) * self.scale  # B, L
        logits = logits.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(logits, dim=-1).nan_to_num(0.0)
        return self.out(torch.einsum("bl,bld->bd", attn, self.v(h)))


class SourceDecoderMixin:
    def _configure_source_decoder(self, config) -> None:
        cfg = dict(config.framework.get("source_decoder", {}) or {})
        self.source_decoder_enabled = bool(cfg.get("enabled", False))
        self.source_targets = [str(t) for t in cfg.get("targets", ["state", "task"])]
        self.source_dim = int(cfg.get("dim", 256))
        self.source_state_dim = int(cfg.get("state_dim", 0))
        self.source_weight_masked = float(cfg.get("weight_masked", 1.0))
        self.source_weight_visible = float(cfg.get("weight_visible", 0.25))
        self.source_weight = float(cfg.get("weight", 1.0))
        if self.source_decoder_enabled:
            bad = [t for t in self.source_targets if t not in ("state", "task")]
            if bad:
                raise ValueError(f"framework.source_decoder.targets: unknown {bad}")
            if "state" in self.source_targets and self.source_state_dim <= 0:
                raise ValueError("framework.source_decoder.state_dim must be the model-space state size")
            logger.info("source decoder ON: targets=%s dim=%d weights masked/visible %.2f/%.2f",
                        self.source_targets, self.source_dim, self.source_weight_masked, self.source_weight_visible)
        hd = dict(config.framework.get("head_input_dropout", {}) or {})
        self.head_input_dropout_rate = float(hd.get("rate", 0.0)) if bool(hd.get("enabled", False)) else 0.0
        if not 0.0 <= self.head_input_dropout_rate <= 1.0:
            raise ValueError("framework.head_input_dropout.rate must be in [0, 1]")
        if self.head_input_dropout_rate > 0:
            logger.info("head-input dropout ON: rate=%.2f (head reads image tokens only)", self.head_input_dropout_rate)

    def _build_source_decoder(self, hidden_size: int) -> None:
        if not getattr(self, "source_decoder_enabled", False):
            return
        out = {"state": self.source_state_dim, "task": int(hidden_size)}
        self.source_decoder = nn.ModuleDict({t: _SourceQuery(int(hidden_size), self.source_dim, out[t])
                                             for t in self.source_targets})

    # ── helpers ──────────────────────────────────────────────────────────────
    def _source_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.qwen_vl_interface.model.config
        return input_ids == int(getattr(cfg, "image_token_id", 151655))

    def _instruction_embedding(self, raw_instructions: Sequence[str], device) -> torch.Tensor:
        tok = self.qwen_vl_interface.processor.tokenizer
        emb = self.qwen_vl_interface.model.get_input_embeddings()
        rows = []
        for text in raw_instructions:
            ids = tok(str(text), add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
            with torch.no_grad():
                rows.append(emb(torch.as_tensor(ids, device=device)).float().mean(0))
        return torch.stack(rows)

    # ── training ─────────────────────────────────────────────────────────────
    def _head_input_dropout(self, qwen_inputs: dict, valid: torch.Tensor) -> torch.Tensor:
        rate = getattr(self, "head_input_dropout_rate", 0.0)
        self._last_head_input_drop = None
        if rate <= 0 or not self.training:
            return valid
        if getattr(self, "readout_projector", None) is not None or getattr(self, "modality_z_interface", None) is not None:
            raise RuntimeError("head_input_dropout needs the head to read the raw encoder tokens")
        rows = torch.rand(valid.shape[0], device=valid.device) < rate
        img = self._source_image_mask(qwen_inputs["input_ids"])[:, : valid.shape[1]].to(valid.device)
        keep_rows = rows & img.any(dim=1)
        out = valid.clone().bool()
        out[keep_rows] = out[keep_rows] & img[keep_rows]
        self._last_head_input_drop = float(keep_rows.float().mean())
        return out

    def _source_decoder_loss(self, qwen_inputs: dict, last_hidden: torch.Tensor, examples: List[dict],
                             state_keep: Optional[Sequence[bool]], raw_instructions: Sequence[str]):
        if not getattr(self, "source_decoder_enabled", False) or not self.training:
            return None, {}
        device = last_hidden.device
        B = last_hidden.shape[0]
        allowed = self._source_image_mask(qwen_inputs["input_ids"])[:, : last_hidden.shape[1]].to(device)
        hidden = last_hidden.float()
        losses, metrics = [], {}
        for t in self.source_targets:
            if t == "state":
                if "state" not in examples[0]:
                    continue
                y = torch.as_tensor(np.stack([np.asarray(ex["state"], np.float32).reshape(-1) for ex in examples]),
                                    device=device)
                if y.shape[1] != self.source_state_dim:
                    raise RuntimeError(f"source_decoder.state_dim {self.source_state_dim} != state size {y.shape[1]}")
                masked = (torch.as_tensor(~np.asarray(state_keep, dtype=bool), device=device)
                          if state_keep is not None else torch.zeros(B, dtype=torch.bool, device=device))
            else:
                y = self._instruction_embedding(raw_instructions, device)
                im = getattr(self, "_instr_mask", None)
                masked = (im.to(device).any(dim=1) if im is not None and im.shape[0] == B
                          else torch.zeros(B, dtype=torch.bool, device=device))
            head = self.source_decoder[t]
            pred = head(hidden.to(head.k.weight.dtype), allowed).float()
            mu = y.mean(dim=0, keepdim=True).detach()
            sd = y.std(dim=0, keepdim=True).clamp_min(1e-4).detach()
            err = ((pred - (y - mu) / sd) ** 2).mean(dim=1)  # per row, ~1 for predicting the batch mean
            w = torch.where(masked, torch.full_like(err, self.source_weight_masked),
                            torch.full_like(err, self.source_weight_visible))
            losses.append((w * err).sum() / w.sum().clamp_min(1e-6))
            if bool(masked.any()):
                metrics[f"source_dec/{t}_r2_masked"] = (1.0 - err[masked].mean()).detach()
            if bool((~masked).any()):
                metrics[f"source_dec/{t}_r2_visible"] = (1.0 - err[~masked].mean()).detach()
        if not losses:
            return None, metrics
        return self.source_weight * torch.stack(losses).mean(), metrics

    def _source_metrics(self) -> dict:
        drop = getattr(self, "_last_head_input_drop", None)
        return {} if drop is None else {"head_input_drop_rate": drop}

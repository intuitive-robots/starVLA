"""Tied two-pass encoder trace loop for GR00T action conditioning.

Pass 1 predicts five normalized image waypoints from typed trace slots.  Quantized
coordinate embeddings replace those slot embeddings for a second pass through the same
bidirectional encoder, and GR00T consumes the complete pass-2 hidden sequence.  There is
no learned readout compressor and no continuous bypass from pass 1 to the action head.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.dataloader.cot_resolver import extract_structured_cot_targets
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY


class TraceLoopModules(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_points: int,
        coordinate_dim: int,
        num_frequencies: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.coordinate_dim = int(coordinate_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.coordinate_dim),
        )
        feature_dim = self.coordinate_dim * (1 + 2 * int(num_frequencies))
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(feature_dim, int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden_dim), hidden_size),
        )
        self.waypoint_embedding = nn.Parameter(
            torch.randn(self.num_points, hidden_size) * 0.02
        )
        frequencies = torch.pi * (2.0 ** torch.arange(int(num_frequencies)).float())
        self.register_buffer("frequencies", frequencies, persistent=False)

    def predict(self, slot_hidden: torch.Tensor) -> torch.Tensor:
        return self.predictor(slot_hidden).sigmoid()

    def embed(self, coordinates: torch.Tensor) -> torch.Tensor:
        angles = coordinates[..., None] * self.frequencies
        features = torch.cat(
            [
                coordinates,
                angles.sin().flatten(-2),
                angles.cos().flatten(-2),
            ],
            dim=-1,
        )
        return self.coordinate_mlp(features) + self.waypoint_embedding[None]


@FRAMEWORK_REGISTRY.register("QwenGR00T_TraceLoop")
class QwenGR00TTraceLoop(Qwen_GR00T):
    """QwenGR00T with an optional tied second encoder pass through predicted trace slots."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        cfg = dict(self.config.framework.get("trace_loop", {}) or {})
        if not bool(cfg.get("enabled", False)):
            raise ValueError("QwenGR00T_TraceLoop requires framework.trace_loop.enabled=true")
        if self.readout_projector is not None:
            raise ValueError("trace loop requires readout_tokens.enabled=false")
        if self.shared_z_enabled:
            raise ValueError("the first trace-loop mechanism arm must use shared_z.enabled=false")

        self.trace_num_points = int(cfg.get("num_points", 5))
        self.trace_coordinate_dim = int(cfg.get("coordinate_dim", 2))
        self.trace_num_passes = int(cfg.get("num_passes", 2))
        self.trace_pass2_source = str(cfg.get("pass2_source", "predicted"))
        self.trace_action_context = str(cfg.get("action_context", "all_pass2"))
        self.trace_detach_pass2_coordinates = bool(
            cfg.get("detach_pass2_coordinates", True)
        )
        self.trace_quantization_bins = int(cfg.get("quantization_bins", 1001))
        self.trace_target_field = str(cfg.get("target_field", "trajectory2d"))
        self.trace_slot_token = str(cfg.get("slot_token", "<|fim_middle|>"))
        if self.trace_num_passes not in {1, 2}:
            raise ValueError("trace_loop.num_passes must be 1 or 2")
        if self.trace_action_context not in {"all_pass1", "all_pass2"}:
            raise ValueError("trace_loop.action_context must be all_pass1 or all_pass2")
        expected_context = "all_pass1" if self.trace_num_passes == 1 else "all_pass2"
        if self.trace_action_context != expected_context:
            raise ValueError(
                f"num_passes={self.trace_num_passes} requires action_context={expected_context}"
            )
        allowed_sources = {"none"} if self.trace_num_passes == 1 else {
            "predicted", "null", "oracle"
        }
        if self.trace_pass2_source not in allowed_sources:
            raise ValueError(
                f"invalid pass2_source={self.trace_pass2_source!r} for "
                f"num_passes={self.trace_num_passes}"
            )
        if self.trace_quantization_bins < 2:
            raise ValueError("trace_loop.quantization_bins must be at least 2")

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        encoded_slot = tokenizer.encode(self.trace_slot_token, add_special_tokens=False)
        if len(encoded_slot) != 1:
            raise ValueError(
                f"trace slot token must encode to one id, got {encoded_slot} for "
                f"{self.trace_slot_token!r}"
            )
        self.trace_slot_token_id = int(encoded_slot[0])
        emb_cfg = dict(cfg.get("coordinate_embedding", {}) or {})
        self.trace_loop = TraceLoopModules(
            hidden_size=int(self.qwen_vl_interface.model.config.hidden_size),
            num_points=self.trace_num_points,
            coordinate_dim=self.trace_coordinate_dim,
            num_frequencies=int(emb_cfg.get("num_frequencies", 16)),
            mlp_hidden_dim=int(emb_cfg.get("hidden_dim", 512)),
        )
        self._last_trace_prediction = None
        self._last_trace_coverage = None
        self._last_trace_pass_delta = None

    def _append_trace_slots(self, instruction: str) -> str:
        slots = " ".join([self.trace_slot_token] * self.trace_num_points)
        return f"{instruction}\nPlanned image trajectory slots: {slots}"

    def _trace_targets(self, examples: List[dict]) -> tuple[list, list[bool]]:
        expected = self.trace_num_points * self.trace_coordinate_dim
        targets, present = [], []
        for example in examples:
            parsed = example.get("cot_structured_targets")
            if not parsed:
                parsed = extract_structured_cot_targets(example.get("cot_conversation"))
            value = parsed.get(self.trace_target_field) if parsed else None
            valid = value is not None and len(value) == expected
            present.append(valid)
            targets.append(value if valid else [0.5] * expected)
        return targets, present

    def _slot_hidden(
        self, hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_mask = input_ids[:, : hidden.shape[1]].eq(self.trace_slot_token_id)
        counts = slot_mask.sum(dim=1)
        if not bool(counts.eq(self.trace_num_points).all()):
            raise RuntimeError(
                f"trace prompt must contain {self.trace_num_points} slots per row; "
                f"found {counts.tolist()}"
            )
        slot_hidden = hidden[slot_mask].reshape(
            hidden.shape[0], self.trace_num_points, hidden.shape[-1]
        )
        return slot_hidden, slot_mask

    @contextmanager
    def _coordinate_embedding_override(
        self, slot_mask: torch.Tensor, coordinate_embeddings: torch.Tensor
    ):
        text_model = self.qwen_vl_interface._text_model()
        embed_tokens = getattr(text_model, "embed_tokens", None)
        if embed_tokens is None:
            raise RuntimeError("trace loop could not locate encoder embed_tokens")

        def replace_slots(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 3:
                return output
            if output.shape[:2] != slot_mask.shape:
                return output
            replacement = coordinate_embeddings.to(
                device=output.device, dtype=output.dtype
            )
            updated = output.clone()
            updated[slot_mask] = replacement.reshape(-1, replacement.shape[-1])
            return updated

        handle = embed_tokens.register_forward_hook(replace_slots)
        try:
            yield
        finally:
            handle.remove()

    def _wrap_encoder_for_trace(self, examples: List[dict]):
        original_forward = self.qwen_vl_interface.forward
        raw_targets, target_present = self._trace_targets(examples)

        def traced_forward(*args, **kwargs):
            pass1 = original_forward(*args, **kwargs)
            hidden1 = pass1.hidden_states[-1]
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise RuntimeError("trace loop requires input_ids")
            slot_hidden, slot_mask = self._slot_hidden(hidden1, input_ids)
            prediction = self.trace_loop.predict(slot_hidden).float()
            self._last_trace_prediction = prediction
            self._last_trace_coverage = prediction.new_tensor(
                sum(target_present) / max(len(target_present), 1)
            )
            if self.trace_num_passes == 1:
                self._last_trace_pass_delta = prediction.new_zeros(())
                return pass1

            if self.trace_pass2_source == "predicted":
                coordinates = prediction.detach() if self.trace_detach_pass2_coordinates else prediction
            elif self.trace_pass2_source == "null":
                coordinates = torch.full_like(prediction, 0.5)
            else:
                target = torch.tensor(
                    raw_targets,
                    device=prediction.device,
                    dtype=prediction.dtype,
                ).reshape_as(prediction)
                present = torch.tensor(
                    target_present, device=prediction.device, dtype=torch.bool
                )[:, None, None]
                coordinates = torch.where(present, target, prediction.detach())
            bins = self.trace_quantization_bins - 1
            coordinates = (coordinates.clamp(0, 1) * bins).round() / bins
            coordinate_embeddings = self.trace_loop.embed(coordinates)
            with self._coordinate_embedding_override(slot_mask, coordinate_embeddings):
                pass2 = original_forward(*args, **kwargs)
            self._last_trace_pass_delta = (
                pass2.hidden_states[-1].detach().float()
                - hidden1.detach().float()
            ).abs().mean()
            return pass2

        return original_forward, traced_forward, raw_targets, target_present

    def forward(self, examples: List[dict] = None, **kwargs):
        traced_examples = [
            {**example, "lang": self._append_trace_slots(example["lang"])}
            for example in examples
        ]
        original, traced, raw_targets, present = self._wrap_encoder_for_trace(
            traced_examples
        )
        object.__setattr__(self.qwen_vl_interface, "forward", traced)
        try:
            result = super().forward(examples=traced_examples, **kwargs)
        finally:
            object.__setattr__(self.qwen_vl_interface, "forward", original)

        prediction = self._last_trace_prediction
        if prediction is None:
            raise RuntimeError("trace predictor did not run")
        present_mask = torch.tensor(present, device=prediction.device, dtype=torch.bool)
        target = torch.tensor(
            raw_targets, device=prediction.device, dtype=prediction.dtype
        ).reshape_as(prediction)
        if bool(present_mask.any()):
            trace_loss = F.smooth_l1_loss(
                prediction[present_mask], target[present_mask]
            )
        else:
            trace_loss = prediction.sum() * 0.0
        if result.get("structured_aux_loss") is not None:
            raise RuntimeError("trace loop cannot share the structured_aux trainer slot")
        result["structured_aux_loss"] = trace_loss
        result["structured_aux/trace_loss"] = trace_loss.detach()
        result["structured_aux/trace_coverage"] = self._last_trace_coverage.detach()
        result["structured_aux/trace_prediction_std"] = prediction.detach().std(
            unbiased=False
        )
        result["structured_aux/trace_pass_delta"] = (
            self._last_trace_pass_delta.detach()
        )
        return result

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        traced_examples = [
            {**example, "lang": self._append_trace_slots(example["lang"])}
            for example in examples
        ]
        original, traced, _, _ = self._wrap_encoder_for_trace(traced_examples)
        object.__setattr__(self.qwen_vl_interface, "forward", traced)
        try:
            return super().predict_action(traced_examples, **kwargs)
        finally:
            object.__setattr__(self.qwen_vl_interface, "forward", original)

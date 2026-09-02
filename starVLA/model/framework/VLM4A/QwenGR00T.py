# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.cot_augmentation import augment_cot_batch
from starVLA.dataloader.cot_resolver import (
    assert_cot_prompt_consistent,
    build_cot_resolver,
    extract_structured_cot_targets,
)
from starVLA.model.modules.projector.readout import ReadoutProjector
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # Optional ERVLA candidate-action supervision. The DiT remains the executed
    # policy; these training-only queries are removed from DiT conditioning.
    choice_policy: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "location": "encoder",
            "num_choices": 5,
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


class ERVLAChoiceHead(nn.Module):
    """Candidate chunk and error-score heads over T action queries + one score query."""

    def __init__(self, hidden_size: int, action_horizon: int, action_dim: int,
                 num_choices: int = 5) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.num_choices = int(num_choices)
        if self.action_horizon <= 0 or self.action_dim <= 0 or self.num_choices <= 0:
            raise ValueError("choice head dimensions must all be positive")
        self.action_head = nn.Linear(hidden_size, self.num_choices * self.action_dim)
        self.score_head = nn.Linear(hidden_size, self.num_choices)

    def forward(self, query_hidden: torch.Tensor, target: torch.Tensor,
                time_mask: torch.Tensor | None = None) -> dict:
        expected = self.action_horizon + 1
        if query_hidden.ndim != 3 or query_hidden.shape[1] != expected:
            raise RuntimeError(
                f"choice queries must be [B,{expected},H], got {tuple(query_hidden.shape)}"
            )
        if target.shape[1:] != (self.action_horizon, self.action_dim):
            raise RuntimeError(
                "choice target shape mismatch: "
                f"expected [B,{self.action_horizon},{self.action_dim}], got {tuple(target.shape)}"
            )
        batch = target.shape[0]
        candidates = self.action_head(query_hidden[:, :self.action_horizon]).view(
            batch, self.action_horizon, self.num_choices, self.action_dim
        ).permute(0, 2, 1, 3).contiguous()  # [B,N,T,D]
        scores = self.score_head(query_hidden[:, -1])  # [B,N], predicts candidate MAE

        if time_mask is None:
            time_mask = torch.ones(
                batch, self.action_horizon, dtype=torch.bool, device=target.device
            )
        else:
            time_mask = time_mask.to(device=target.device, dtype=torch.bool)
        if time_mask.shape != (batch, self.action_horizon):
            raise RuntimeError(
                f"choice time mask must be [B,{self.action_horizon}], got {tuple(time_mask.shape)}"
            )
        weight = time_mask[:, None, :, None].float()
        denom = time_mask.sum(dim=1, keepdim=True).clamp_min(1).float() * self.action_dim
        candidate_errors = (
            (candidates.float() - target[:, None].float()).abs() * weight
        ).sum(dim=(2, 3)) / denom  # [B,N]
        best_error, winner = candidate_errors.min(dim=1)
        choice_loss = best_error.mean()
        # ERVLA Eq. (6): average over the batch of the squared L2 norm
        # across N candidates (sum over N, not an elementwise mean).
        score_loss = F.mse_loss(
            scores.float(), candidate_errors.detach(), reduction="sum"
        ) / batch

        if self.num_choices > 1:
            pairwise = (
                candidates[:, :, None].float() - candidates[:, None, :].float()
            ).abs().mean(dim=(3, 4))
            off_diagonal = ~torch.eye(
                self.num_choices, dtype=torch.bool, device=pairwise.device
            )
            diversity = pairwise[:, off_diagonal].mean()
        else:
            diversity = candidates.new_zeros((), dtype=torch.float32)
        histogram = torch.bincount(winner, minlength=self.num_choices).float() / max(batch, 1)
        return {
            "choice_loss": choice_loss,
            "score_loss": score_loss,
            "choice_min_error": best_error.detach().mean(),
            "choice_score_mae": (scores.float() - candidate_errors.detach()).abs().mean().detach(),
            "choice_diversity": diversity.detach(),
            "choice_winner_histogram": histogram.detach(),
        }


class StructuredEncoderRegressionHead(nn.Module):
    """Learned masked pooling plus a small regression MLP for one CoT field."""

    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)
        bottleneck = max(128, hidden_size // 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, output_dim),
        )

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(hidden)
        scores = torch.einsum("blh,h->bl", hidden, self.query) / hidden.shape[-1] ** 0.5
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        pooled = torch.einsum("bl,blh->bh", scores.softmax(dim=-1), hidden)
        return self.mlp(pooled)


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.is_inference = bool(kwargs.get("is_inference", False))

        choice_cfg = self.config.framework.get("choice_policy", {})
        if bool(choice_cfg.get("enabled", False)):
            if not bool(self.config.framework.qwenvl.get("enc_dec", False)):
                raise ValueError("ERVLA choice queries currently require qwenvl.enc_dec=true")
            num_action_queries = int(
                choice_cfg.get("num_action_queries", self.action_horizon)
            )
            if num_action_queries != self.action_horizon:
                raise ValueError(
                    "choice_policy.num_action_queries must match action_horizon: "
                    f"{num_action_queries} vs {self.action_horizon}"
                )
            self.choice_head = ERVLAChoiceHead(
                hidden_size=int(self.qwen_vl_interface.model.config.hidden_size),
                action_horizon=self.action_horizon,
                action_dim=int(self.config.framework.action_model.action_dim),
                num_choices=int(choice_cfg.get("num_choices", 5)),
            )
        else:
            self.choice_head = None

        structured_cfg = dict(self.config.framework.get("structured_aux", {}) or {})
        self.structured_aux_enabled = bool(structured_cfg.get("enabled", False))
        self.structured_aux_specs = {
            str(name): dict(spec)
            for name, spec in dict(structured_cfg.get("targets", {}) or {}).items()
        } if self.structured_aux_enabled else {}
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.structured_aux_heads = nn.ModuleDict({
            name: StructuredEncoderRegressionHead(hidden_size, int(spec["dim"]))
            for name, spec in self.structured_aux_specs.items()
        })

        cot_cfg = self.config.datasets.vla_data.get("cot", None)
        cot_source = cot_cfg.get("source", "none") if cot_cfg is not None else "none"
        self.cot_dropout_enabled = bool(
            cot_cfg.get("dropout_enabled", True) if cot_cfg is not None else False
        ) and cot_source in {"mapping", "sparc_sqlite"}
        self.cot_dropout_rate = float(
            cot_cfg.get("dropout_rate", 0.5) if cot_cfg is not None else 0.0
        )
        self.cot_text_supervision = bool(
            cot_cfg.get("text_supervision", True) if cot_cfg is not None else False
        )

        # Step-based CoT resolver — NullCoTResolver when cot.source is absent/none.
        # Training LeRobot datasets resolve CoT before worker-side transforms, so do not
        # parse and retain a duplicate ~273k-line mapping in every GPU process. Keep lazy
        # fallback construction for direct/manual examples, and build eagerly at inference
        # where there is no training dataloader.
        self.cot_resolver = None
        if self.is_inference:
            self.cot_resolver = build_cot_resolver(self.config, is_inference=True)
            assert_cot_prompt_consistent(self.cot_resolver, self.config)

        # GR00T N1.5-style readout projector (optional)
        readout_cfg = getattr(self.config.framework, "readout_tokens", None)
        _readout_enabled = bool(
            readout_cfg.get("enabled", False) if hasattr(readout_cfg, "get")
            else getattr(readout_cfg, "enabled", False)
        ) if readout_cfg is not None else False

        if _readout_enabled:
            _g = lambda k, d: int(readout_cfg.get(k, d) if hasattr(readout_cfg, "get") else getattr(readout_cfg, k, d))
            _num_tokens = _g("num_tokens", 32)
            _num_layers = _g("num_layers",  2)
            _num_heads  = _g("num_heads",   8)
            _ffn_dim    = _g("ffn_dim",     0) or None
            _dropout    = float(readout_cfg.get("dropout", 0.0) if hasattr(readout_cfg, "get") else getattr(readout_cfg, "dropout", 0.0))
            vlm_hidden  = self.qwen_vl_interface.model.config.hidden_size
            self.readout_projector = ReadoutProjector(
                vlm_hidden, _num_tokens, _num_layers, _num_heads, _ffn_dim, _dropout
            )
        else:
            self.readout_projector = None

    def _dit_valid_mask(self, qwen_inputs: dict, last_hidden: torch.Tensor) -> torch.Tensor:
        """Return the exact valid-token mask corresponding to ``last_hidden``."""
        if self.readout_projector is not None:
            return torch.ones(
                (last_hidden.shape[0], self.readout_projector.queries.shape[1]),
                device=last_hidden.device,
                dtype=torch.bool,
            )

        valid = getattr(self.qwen_vl_interface, "_last_encoder_attention_mask", None)
        if valid is None:
            valid = qwen_inputs.get("attention_mask")
        if valid is None:
            valid = torch.ones(last_hidden.shape[:2], device=last_hidden.device, dtype=torch.bool)
        valid = valid[:, : last_hidden.shape[1]].to(device=last_hidden.device, dtype=torch.bool)
        if valid.shape != last_hidden.shape[:2]:
            raise RuntimeError(
                "DiT encoder mask/hidden-state mismatch: "
                f"{tuple(valid.shape)} vs {tuple(last_hidden.shape[:2])}"
            )
        return valid

    @staticmethod
    def _dit_attention_bias(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert a [B,L] keep mask to the additive mask expected by Diffusers attention."""
        bias = torch.zeros(valid.shape, device=valid.device, dtype=dtype)
        bias.masked_fill_(~valid, -10_000.0)
        return bias[:, None, :]

    def _structured_aux_forward(
        self,
        examples: List[dict],
        conversations: List[list | None],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if not getattr(self, "structured_aux_enabled", False):
            return None, {}
        layer_states = self.qwen_vl_interface._structured_layer_out
        encoder_valid = self.qwen_vl_interface._last_encoder_attention_mask.bool()
        losses, metrics = [], {}
        for name, spec in self.structured_aux_specs.items():
            layer = int(spec["layer"])
            dim = int(spec["dim"])
            if layer not in layer_states:
                raise RuntimeError(f"structured auxiliary layer {layer} was not captured")
            hidden = layer_states[layer]
            valid_tokens = encoder_valid[:, : hidden.shape[1]]
            if valid_tokens.shape != hidden.shape[:2]:
                raise RuntimeError(
                    f"structured auxiliary mask mismatch at layer {layer}: "
                    f"{tuple(valid_tokens.shape)} vs {tuple(hidden.shape[:2])}"
                )
            parsed = [
                ex.get("cot_structured_targets")
                or extract_structured_cot_targets(conversation)
                for ex, conversation in zip(examples, conversations)
            ]
            present = [name in target and len(target[name]) == dim for target in parsed]
            prediction = self.structured_aux_heads[name](hidden, valid_tokens).float()
            if any(present):
                indices = torch.tensor(present, device=hidden.device, dtype=torch.bool)
                target = torch.tensor(
                    [target[name] for target, keep in zip(parsed, present) if keep],
                    device=hidden.device,
                    dtype=torch.float32,
                )
                head_loss = F.smooth_l1_loss(prediction[indices], target)
                losses.append(head_loss)
                metrics[f"structured_aux/{name}_loss"] = head_loss.detach()
            else:
                # DeepSpeed requires identical parameter participation on every rank.
                # A field such as target_box can legitimately be absent from one local
                # batch while present on another, so retain this head with a zero term.
                losses.append(prediction.sum() * 0.0)
            metrics[f"structured_aux/{name}_coverage"] = torch.tensor(
                sum(present) / max(len(present), 1), device=hidden.device
            )
        if not losses:
            # Keep every auxiliary head in the distributed graph even for a pathological
            # unmapped batch; ordinary LIBERO batches have near-complete trajectories.
            zero = sum((p.sum() * 0.0 for p in self.structured_aux_heads.parameters()))
            return zero, metrics
        return torch.stack(losses).mean(), metrics

    @staticmethod
    def _apply_cot_graph_guard(
        cot_loss: torch.Tensor | None,
        examples: List[dict],
        conversations: List[list | None],
    ) -> torch.Tensor | None:
        """Keep the decoder graph but remove supervision from a synthetic guard row."""
        if cot_loss is None:
            return None
        guard_only = all(
            conversation is None or bool(example.get("cot_graph_guard", False))
            for example, conversation in zip(examples, conversations)
        )
        return cot_loss * 0.0 if guard_only else cot_loss

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        action_time_mask = (
            [example["action_time_mask"] for example in examples]
            if "action_time_mask" in examples[0]
            else None
        )

        # Resolve per-sample CoT texts from the step-based mapping.
        # Returns None per sample when no annotation covers that trajectory/frame (holdout).
        cot_from_workers = all("cot_conversation" in ex for ex in examples)
        if not cot_from_workers and self.cot_resolver is None:
            self.cot_resolver = build_cot_resolver(self.config, is_inference=False)
            assert_cot_prompt_consistent(self.cot_resolver, self.config)
        cot_conversations = [
            ex.get("cot_conversation") if cot_from_workers else self.cot_resolver.resolve(
                ex.get("trajectory_name", ""), ex.get("frame_index", 0))
            for ex in examples
        ]
        cot_available = [
            bool(ex.get("cot_available", c is not None)) if cot_from_workers else c is not None
            for ex, c in zip(examples, cot_conversations)
        ]
        if self.cot_dropout_enabled:
            if cot_from_workers:
                cot_modes = [ex.get("cot_mode", "cot" if c is not None else "no_cot")
                             for ex, c in zip(examples, cot_conversations)]
            else:
                # Direct/manual batches have no workers. Preserve the same semantics as the
                # worker path; normal training never enters this fallback.
                for i, conversation in enumerate(cot_conversations):
                    if (self.training and conversation is not None
                            and np.random.random() < self.cot_dropout_rate):
                        cot_conversations[i] = None
                cot_modes = ["cot" if c is not None else "no_cot" for c in cot_conversations]
        else:
            cot_modes = None
        text_supervision = getattr(self, "cot_text_supervision", True)
        has_cot = text_supervision and any(c is not None for c in cot_conversations)
        if self.training and self.cot_dropout_enabled and text_supervision and not has_cot:
            raise RuntimeError(
                "CoT dropout removed every target from this local training batch. "
                "The worker-side collator must retain at least one target per rank."
            )
        cot_coverage = sum(cot_available) / max(len(cot_available), 1)
        available_count = sum(cot_available)
        cot_keep_rate = (
            sum(c is not None for c in cot_conversations) / available_count
            if available_count else 0.0
        )

        # Coordinate-bearing CoT and image geometry must be augmented jointly. This is the
        # first point in the pipeline where decoded images and resolver-produced target text
        # coexist, so a sampled crop can be applied once and propagated to every <box>,
        # <point>, and <trajectory>. eval()/rollout is intentionally unchanged.
        augmentation = str(self.config.datasets.vla_data.get("augmentation", "none")).lower()
        if (self.training and has_cot and not cot_from_workers
                and augmentation in {"photometric", "crop_photometric"}):
            batch_images, cot_conversations = augment_cot_batch(
                batch_images, cot_conversations, mode=augmentation)

        # Step 1: QWenVL input format
        text_conversations = cot_conversations if has_cot else None
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            cot_conversations=text_conversations,
            cot_modes=cot_modes if has_cot else None,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                _run_choice_queries=self.choice_head is not None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # CoT CE loss from the VLM language head (only present when labels were attached).
        cot_loss = qwenvl_outputs.loss if (has_cot and qwenvl_outputs.loss is not None) else None
        cot_loss = self._apply_cot_graph_guard(cot_loss, examples, cot_conversations)
        structured_aux_loss, structured_aux_metrics = self._structured_aux_forward(
            examples, cot_conversations
        )

        # Readout projection (optional) — compress VLM tokens before DiT
        if self.readout_projector is not None:
            dit_context = self.readout_projector(last_hidden)  # [B, num_tokens, H]
        else:
            dit_context = last_hidden                          # [B, L, H]
        dit_valid_mask = self._dit_valid_mask(qwen_inputs, last_hidden)
        dit_attention_bias = self._dit_attention_bias(dit_valid_mask, dit_context.dtype)

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=dit_context.device, dtype=dit_context.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            choice_output = None
            if self.choice_head is not None:
                choice_hidden = getattr(
                    self.qwen_vl_interface, "_last_choice_hidden", None
                )
                if choice_hidden is None:
                    raise RuntimeError("choice policy enabled but VLM returned no choice-query states")
                time_mask_tensor = (
                    torch.tensor(
                        np.array(action_time_mask), device=dit_context.device, dtype=torch.bool
                    )[:, -self.action_horizon:]
                    if action_time_mask is not None else None
                )
                choice_output = self.choice_head(
                    choice_hidden, actions_target, time_mask=time_mask_tensor
                )

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            dit_context_repeated    = dit_context.repeat(repeated_diffusion_steps, 1, 1)
            dit_attention_bias_repeated = dit_attention_bias.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=dit_context.device, dtype=dit_context.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                dit_context_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=dit_attention_bias_repeated,
            )  # (B, chunk_len, action_dim)

        result = {
            "action_loss": action_loss,
            "cot_coverage": cot_coverage,
            "cot_keep_rate": cot_keep_rate,
        }
        if cot_loss is not None:
            result["cot_loss"] = cot_loss
        if structured_aux_loss is not None:
            result["structured_aux_loss"] = structured_aux_loss
            result.update(structured_aux_metrics)
        if choice_output is not None:
            result.update(choice_output)
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        cot_modes = ["no_cot"] * len(instructions) if self.cot_dropout_enabled else None
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            cot_modes=cot_modes,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        if self.readout_projector is not None:
            dit_context = self.readout_projector(last_hidden)  # [B, num_tokens, H]
        else:
            dit_context = last_hidden                          # [B, L, H]
        dit_valid_mask = self._dit_valid_mask(qwen_inputs, last_hidden)
        dit_attention_bias = self._dit_attention_bias(dit_valid_mask, dit_context.dtype)

        state = (
            torch.from_numpy(np.array(state)).to(dit_context.device, dtype=dit_context.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                dit_context,
                state,
                encoder_attention_mask=dit_attention_bias,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    action_dim = cfg.framework.action_model.action_dim
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, action_dim)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")

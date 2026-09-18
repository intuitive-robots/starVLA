# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenPI_v3 Framework
A Qwen2.5-VL / Qwen3-VL + layer-wise cross-DiT flow-matching action head.

Released checkpoint
─────────────────────────────
- Qwen3-VL-4B + Bridge V2 + RT-1 (OXE) co-training, 69.8% avg success on
  SimplerEnv WidowX:
  https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1

Key improvements over QwenPI
─────────────────────────────
1. **Compressed Action DiT via per-layer projectors**
   Each of the N VLM hidden-state layers is passed through a dedicated
   LayerNorm + Linear projector (`project_layers`) that maps the VLM hidden
   dimension (e.g. 2560) down to a smaller Action DiT latent dimension
   (e.g. 1024, controlled by `action_dit_hidden_dim`).  This reduces the
   action head parameter count by ~(vl_hidden / dit_hidden)² while keeping
   the full layer-wise cross-attention structure.

2. **Discretised-state language injection** (`add_discretized_state_to_instruction`)
   Proprioceptive state is quantised into 256 bins and appended to the
   language instruction as plain tokens (``[STATE] <bins> [ACTION]``),
   following the π₀.5 design.  This lets the VLM attend to state without
   any extra encoder module.

Together these two features bring QwenPI_v3 close to all the core
capabilities of π₀.5 within a single open-weight VLM framework.

Parameter breakdown (Qwen3-VL-4B + action_dit_hidden_dim=1024)
═══════════════════════════════════════════════════════════════
  Module                               Params        %
  ───────────────────────────────────────────────────
  qwen_vl_interface         4,437,815,808   87.5%
  action_model                538,678,305   10.6%
  project_layers               94,593,024    1.9%
  ───────────────────────────────────────────────────
  TOTAL                     5,071,087,137  100.0%
═══════════════════════════════════════════════════════════════
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
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
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.shared_z import (  # noqa: F401  (re-exported for callers)
    SharedZMixin,
    SharedZPooler,
    SharedZRegressionHead,
)
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.framework.VLM4A.QwenGR00T import StructuredEncoderRegressionHead
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100



####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI_v3
#  - Same shape as QwenPIDefaultConfig (see QwenPI.py) but introduces the
#    optional `action_dit_hidden_dim` knob inside diffusion_model_cfg.
#  - Setting action_dit_hidden_dim to a value smaller than the VLM hidden
#    size lets the Action DiT run at a "compressed" latent dim while
#    `Qwen_PI_v3.project_layers` does the LayerNorm+Linear compression of
#    each VL hidden state to that dim.
#  - Leaving it None (or omitting it from YAML) reproduces the QwenPI
#    behaviour: DiT hidden = VLM hidden, projection becomes nn.Identity().
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPI_v3DefaultConfig:
    """QwenPI_v3 framework default parameters.

    See ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for the
    relationship between vl_hidden_dim, action_dit_hidden_dim and
    cross_attention_dim.
    """

    name: str = "QwenPI_v3"

    # Probability of omitting the complete discretised proprioceptive-state
    # suffix for an individual training example. Evaluation/inference always
    # retains state. This is separate from the Action DiT's feature dropout.
    state_dropout_rate: float = 0.0

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 36,  # auto-overridden at runtime from the loaded VLM
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            # Old PI checkpoints were trained by a hand-written loop that sent
            # encoder memory to every DiT block. New runs must opt explicitly
            # into the corrected alternating cross/self-attention topology so
            # loading an old checkpoint cannot silently change its forward.
            "layerwise_attention_layout": "legacy_all_cross",
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                # When set (e.g. 1024), DiT internal hidden = action_dit_hidden_dim
                # and Qwen_PI_v3.project_layers compress VL hidden to this dim.
                # When None, DiT internal hidden = vl_hidden_dim (== QwenPI behaviour).
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v3")
class Qwen_PI_v3(SharedZMixin, baseframework):
    """
    Multimodal vision-language-action model (QwenPI_v3 variant).

    Architecture
    ────────────
    - Qwen2.5-VL / Qwen3-VL backbone for fused language / vision token embeddings.
    - Per-layer projectors (``project_layers``): one LayerNorm + Linear per VLM
      layer that compresses VLM hidden states from ``vl_hidden_dim`` down to
      ``action_dit_hidden_dim`` before feeding the Action DiT.
    - Layer-wise cross-DiT flow-matching action head that attends to every
      selected VLM layer in parallel.

    Focus: predict a future action chunk conditioned on multi-view images
    and a natural-language instruction (with optional discretised state prefix).
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
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(QwenPI_v3DefaultConfig, config)
        self.state_dropout_rate = float(self.config.framework.get("state_dropout_rate", 0.0))
        if not 0.0 <= self.state_dropout_rate <= 1.0:
            raise ValueError(
                "framework.state_dropout_rate must be between 0 and 1, "
                f"got {self.state_dropout_rate}"
            )
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        encoder_mlm_cfg = dict(
            self.config.framework.qwenvl.get("encoder_mlm", {}) or {}
        )
        self.encoder_mlm_enabled = bool(encoder_mlm_cfg.get("enabled", False))
        # On a sampled fraction of training rows, make every PI cross-attention
        # block consume only the fixed masked-reasoning slots. This is distinct
        # from label dropout: cloze targets remain supervised on every row, and
        # rollout/evaluation always retains the full encoder memory.
        self.encoder_mlm_action_slot_dropout_rate = float(
            encoder_mlm_cfg.get("action_slot_dropout_rate", 0.0)
        )
        if not 0.0 <= self.encoder_mlm_action_slot_dropout_rate <= 1.0:
            raise ValueError(
                "framework.qwenvl.encoder_mlm.action_slot_dropout_rate must be in [0,1]"
            )
        if self.encoder_mlm_action_slot_dropout_rate > 0.0 and not self.encoder_mlm_enabled:
            raise ValueError(
                "action_slot_dropout_rate requires framework.qwenvl.encoder_mlm.enabled=true"
            )

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors
        # (embedding output + every layer's output), and we keep the last
        # `num_hidden_layers` of them for layer-wise cross-attn — so the DiT
        # depth and project_layers count must match `num_hidden_layers` exactly.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        # Full-duplicate enc-dec exposes all source-depth encoder states (28 for
        # Qwen3-VL-2B), while a parameter-matched layer split exposes only its encoder
        # prefix (14 in the 14/14 experiment).  The HF config still reports the original
        # source depth, so ask the interface for the actual feature count when available.
        num_vl_layers = int(getattr(
            self.qwen_vl_interface,
            "num_encoder_feature_layers",
            text_cfg.num_hidden_layers,
        ))
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Resolve the Action DiT hidden dim BEFORE building the action head,
        # so that LayerwiseFlowmatchingActionHead constructs DiT at the right size.
        # If the user did not specify it, fall back to the LLM hidden size
        # (i.e. behave like QwenPI: project_layers becomes nn.Identity()).
        #
        # NOTE: `action_dit_hidden_dim` is a framework-side hint only — it is
        # NOT a DiT constructor kwarg, so we keep it out of diffusion_model_cfg
        # and instead pass it through `populate_layerwise_dit_cfg`, which writes
        # the canonical DiT-shape fields (input_embedding_dim, cross_attention_dim,
        # num_attention_heads).
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        shared_z_cfg = self._configure_shared_z(self.config, diffusion_model_cfg)
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        # Push the resolved DiT shape into diffusion_model_cfg.  The action head
        # is intentionally agnostic of qwenvl.* — it only consumes this dict.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = self._build_action_model()
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        # Layer-wise projector: map each selected VL hidden to Action DiT hidden space.
        # This explicitly decouples VL representation size from action DiT latent size.
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        self._build_shared_z_modules(shared_z_cfg, llm_hidden_size)
        if self.shared_z_enabled:
            if self.shared_z_decoder_memory:
                if bool(self.config.framework.qwenvl.get("skip_decoder", True)):
                    raise ValueError("shared_z.decoder_memory requires qwenvl.skip_decoder=false")
                if bool(self.config.framework.qwenvl.get("layerwise_decoder_cross_attention", False)):
                    raise ValueError("shared-z decoder memory must not use layerwise raw-H memory")
                self.shared_z_decoder_up = nn.Linear(self.shared_z_dim, llm_hidden_size)
                lm = self.qwen_vl_interface._text_model()

                def _z_only_decoder_memory(memory, valid, visual):
                    z_value = self.shared_z_pooler(memory, valid.bool())
                    self._shared_z_from_decoder = z_value
                    projected = self.shared_z_decoder_up(z_value).to(memory.dtype)
                    return projected[:, None, :].expand_as(memory)

                lm._decoder_encoder_intervention = _z_only_decoder_memory
                logger.info("shared-z decoder memory enabled: decoder K/V contains only z")

            # Preserve the existing timestep columns while making z initially a small,
            # nonzero perturbation. The effective neutral scale remains zero.
            for block in self.action_model.model.transformer_blocks:
                linear = getattr(getattr(block, "norm1", None), "linear", None)
                if linear is not None:
                    nn.init.normal_(linear.weight[:, -self.shared_z_dim:], std=1.0e-3)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.is_inference = bool(kwargs.get("is_inference", False))

        # Optional soft anchor to the *initial* pretrained encoder.  Keep the
        # teacher outside nn.Module registration: it is fixed reference state,
        # must not enter AdamW/DeepSpeed, and must not bloat policy checkpoints.
        anchor_cfg = dict(self.config.framework.get("representation_anchor", {}) or {})
        self.representation_anchor_enabled = bool(anchor_cfg.get("enabled", False))
        self.representation_anchor_layers = [
            int(index) for index in anchor_cfg.get("layer_indices", [2, 10, 18, 26])
        ]
        object.__setattr__(self, "_representation_anchor_teacher", None)
        object.__setattr__(self, "_representation_anchor_teacher_device", None)
        if self.representation_anchor_enabled and not self.is_inference:
            teacher = get_vlm_model(config=self.config)
            teacher.requires_grad_(False)
            teacher.eval()
            object.__setattr__(self, "_representation_anchor_teacher", teacher)

        dynamics_cfg = dict(self.config.framework.get("tied_dynamics", {}) or {})
        self.tied_dynamics_enabled = bool(dynamics_cfg.get("enabled", False))
        self.tied_dynamics_horizons = [
            int(horizon) for horizon in dynamics_cfg.get("horizons", [4, 8, 16])
        ]
        self.tied_dynamics_position_dims = int(dynamics_cfg.get("position_dims", 3))
        self.tied_dynamics_beta = float(dynamics_cfg.get("smooth_l1_beta", 0.1))
        if self.tied_dynamics_enabled:
            if not self.tied_dynamics_horizons:
                raise ValueError("framework.tied_dynamics.horizons must not be empty")
            if min(self.tied_dynamics_horizons) < 1 or max(self.tied_dynamics_horizons) > self.action_horizon:
                raise ValueError(
                    "framework.tied_dynamics.horizons must lie in [1, action_horizon], "
                    f"got {self.tied_dynamics_horizons} for horizon {self.action_horizon}"
                )
            if not 1 <= self.tied_dynamics_position_dims <= int(self.config.framework.action_model.action_dim):
                raise ValueError("framework.tied_dynamics.position_dims is outside the action dimension")
            if self.tied_dynamics_beta <= 0.0:
                raise ValueError("framework.tied_dynamics.smooth_l1_beta must be positive")

        # Keep structured representation supervision on raw encoder states. PI's
        # per-depth projectors are part of the action expert and must not become a
        # shortcut or silently change N's 2048-D encoder-probe objective.
        structured_cfg = dict(self.config.framework.get("structured_aux", {}) or {})
        self.structured_aux_enabled = bool(structured_cfg.get("enabled", False))
        self.structured_aux_specs = {
            str(name): dict(spec)
            for name, spec in dict(structured_cfg.get("targets", {}) or {}).items()
        } if self.structured_aux_enabled else {}
        self.structured_aux_heads = nn.ModuleDict({
            name: StructuredEncoderRegressionHead(llm_hidden_size, int(spec["dim"]))
            for name, spec in self.structured_aux_specs.items()
        })

        numeric_cfg = dict(self.config.framework.get("projected_numeric_aux", {}) or {})
        self.projected_numeric_aux = None
        self._numeric_aux_steps = 0
        self.numeric_aux_grad_interval = int(numeric_cfg.get("gradient_interval", 100))
        if numeric_cfg.get("enabled", False):
            if self.structured_aux_enabled or self.shared_z_enabled or not self.encoder_mlm_enabled:
                raise ValueError("Projected numeric heads require MLM slots and no other auxiliary heads")
            if self.qwen_vl_interface.encoder_mlm_loss_enabled:
                raise ValueError("Projected numeric experiment must disable token CE")
            from starVLA.model.modules.projected_numeric_aux import ProjectedNumericAux
            self.projected_numeric_aux = ProjectedNumericAux(
                int(action_dit_hidden_dim or llm_hidden_size), self.qwen_vl_interface.encoder_mlm_fields,
                numeric_cfg.get("layers", [12, 16, 20]), numeric_cfg.get("beta", 0.1))

        alignment_cfg = dict(self.config.framework.get("decoder_latent_alignment", {}) or {})
        self.decoder_latent_alignment = None
        self._latent_alignment_steps = 0
        self._latent_alignment_audited = False
        self.latent_alignment_gradient_interval = int(alignment_cfg.get("gradient_interval", 100))
        object.__setattr__(self, "_decoder_latent_teacher", None)
        if alignment_cfg.get("enabled", False):
            if (self.projected_numeric_aux is not None or self.structured_aux_enabled
                    or self.shared_z_enabled or not self.encoder_mlm_enabled
                    or self.qwen_vl_interface.encoder_mlm_loss_enabled):
                raise ValueError("Decoder alignment requires mask slots, no token CE/other auxiliaries")
            teacher_layers = alignment_cfg.get("teacher_layers", None)
            if teacher_layers is not None and list(teacher_layers) != list(alignment_cfg.get("layers", [])):
                raise ValueError("Intermediate alignment requires equal student/teacher layer indices")
            from starVLA.model.modules.decoder_latent_alignment import (
                DecoderLatentAlignment, FrozenDecoderTeacher)
            with torch.random.fork_rng(devices=[]):
                self.decoder_latent_alignment = DecoderLatentAlignment(
                    int(action_dit_hidden_dim or llm_hidden_size), int(alignment_cfg.get("teacher_dim", 2048)),
                    self.qwen_vl_interface.encoder_mlm_fields, alignment_cfg.get("layers", [4, 8, 12]))
            if not self.is_inference:
                object.__setattr__(self, "_decoder_latent_teacher", FrozenDecoderTeacher(
                    alignment_cfg["teacher_checkpoint"], alignment_cfg.get("teacher_layer", 23),
                    alignment_cfg.get("teacher_micro_batch_size", 4),
                    layers=alignment_cfg.get("teacher_layers", None),
                    benchmark_batches=alignment_cfg.get("benchmark_batches", False)))

        # Match QwenGR00T's CoT execution semantics. Full training resolves mappings and
        # performs dropout in dataloader workers; the lazy resolver is only a fallback for
        # direct/manual batches. Keeping this inside PI is essential for a controlled
        # readout comparison: otherwise changing framework.name silently removes the
        # frozen-decoder reasoning objective.
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
        self.cot_resolver = None
        if self.is_inference:
            self.cot_resolver = build_cot_resolver(self.config, is_inference=True)
            assert_cot_prompt_consistent(self.cot_resolver, self.config)

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project layer-wise VL hidden states to the hidden space expected by Action DiT."""
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        # Diagnostic-only causal intervention.  A probe fitted in PI's native
        # raw (typically 2048-D) or projected (typically 1024-D) state space
        # stores one centered basis per directly cross-attended layer.  Keeping
        # this behind an environment variable makes an ordinary checkpoint
        # bit-for-bit unchanged while allowing the batched rollout stack to
        # remove the same fitted subspace at either side of PI's projectors.
        basis_path = os.environ.get("STARVLA_PI_COT_SUBSPACE_PATH", "").strip()
        payload = None
        rank = 0
        if basis_path:
            rank = int(os.environ.get("STARVLA_PI_COT_SUBSPACE_RANK", "0"))
            if rank <= 0:
                raise ValueError(
                    "STARVLA_PI_COT_SUBSPACE_RANK must be positive when "
                    "STARVLA_PI_COT_SUBSPACE_PATH is set"
                )
            cache_key = (basis_path, rank)
            if getattr(self, "_cot_subspace_cache_key", None) != cache_key:
                payload = torch.load(basis_path, map_location="cpu", weights_only=True)
                if int(payload["num_layers"]) != len(vl_embs_list):
                    raise ValueError(
                        f"CoT subspace has {payload['num_layers']} layers, model has {len(vl_embs_list)}"
                    )
                self._cot_subspace_payload = payload
                self._cot_subspace_cache_key = cache_key
                logger.info(
                    "PI diagnostic: removing rank-%d %s-space CoT subspace from %s",
                    rank,
                    payload.get("space", "projected"),
                    basis_path,
                )
            payload = self._cot_subspace_payload

        def remove_subspace(states: List[torch.Tensor]) -> List[torch.Tensor]:
            intervened = list(states)
            for layer_text, entry in payload["layers"].items():
                layer = int(layer_text)
                hidden = intervened[layer]
                mean = entry["mean"].to(device=hidden.device, dtype=hidden.dtype)
                basis = entry["basis"][:, :rank].to(device=hidden.device, dtype=hidden.dtype)
                if basis.shape[1] < rank:
                    raise ValueError(
                        f"layer {layer} CoT basis has rank {basis.shape[1]}, requested {rank}"
                    )
                centered = hidden - mean.view(1, 1, -1)
                intervened[layer] = hidden - torch.matmul(
                    torch.matmul(centered, basis), basis.transpose(0, 1)
                )
            return intervened

        space = payload.get("space", "projected") if payload is not None else None
        if space == "raw":
            vl_embs_list = remove_subspace(vl_embs_list)
        elif space not in {None, "projected"}:
            raise ValueError(f"unsupported PI CoT subspace space: {space!r}")

        projected = [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]
        if space == "projected":
            projected = remove_subspace(projected)

        # Optional post-hoc rank-r update fitted through PI's complete action
        # sampler.  The sidecar implements exactly
        #   Linear(LayerNorm(h)) + B(A(LayerNorm(h))) / rank
        # on the selected layerwise projectors.  It is deliberately opt-in so
        # ordinary checkpoints and training are unchanged.
        projection_lora_path = os.environ.get(
            "STARVLA_PI_PROJECTION_LORA_PATH", ""
        ).strip()
        if projection_lora_path:
            sample = projected[0]
            cache_key = (
                projection_lora_path,
                str(sample.device),
                str(sample.dtype),
            )
            if getattr(self, "_projection_lora_cache_key", None) != cache_key:
                payload = torch.load(
                    projection_lora_path, map_location="cpu", weights_only=True
                )
                if payload.get("format") != "starvla_pi_unrolled_endpoint_adapter_v1":
                    raise ValueError(
                        "STARVLA_PI_PROJECTION_LORA_PATH is not an unrolled PI adapter"
                    )
                rank = int(payload["rank"])
                state = payload["state_dict"]
                updates = {}
                for layer in payload["cross_layers"]:
                    prefix = f"updates.projection_{int(layer)}"
                    updates[int(layer)] = (
                        state[f"{prefix}.down.weight"].to(
                            device=sample.device, dtype=sample.dtype
                        ),
                        state[f"{prefix}.up.weight"].to(
                            device=sample.device, dtype=sample.dtype
                        ),
                    )
                self._projection_lora_updates = updates
                self._projection_lora_rank = rank
                self._projection_lora_cache_key = cache_key
                logger.info(
                    "PI projection LoRA: loading rank-%d sampler-aware adapter from %s",
                    rank,
                    projection_lora_path,
                )
            adapted = list(projected)
            for layer, (down, up) in self._projection_lora_updates.items():
                projector = self.project_layers[layer]
                projector_input = (
                    projector[0](vl_embs_list[layer])
                    if isinstance(projector, nn.Sequential)
                    else vl_embs_list[layer]
                )
                adapted[layer] = adapted[layer] + F.linear(
                    F.linear(projector_input, down), up
                ) / self._projection_lora_rank
            projected = adapted

        # Optional post-hoc alignment gate. Unlike the binary CoT diagnostic
        # above, this payload stores a learned suppression coefficient for
        # every basis vector. Coefficients can be zero, making the intervention
        # exactly identity at initialization. The sidecar is deliberately not
        # part of ordinary checkpoints and is activated only for explicit
        # alignment evaluations.
        alignment_path = os.environ.get("STARVLA_PI_ALIGNMENT_PATH", "").strip()
        if alignment_path:
            if getattr(self, "_alignment_gate_cache_key", None) != alignment_path:
                alignment = torch.load(
                    alignment_path, map_location="cpu", weights_only=True
                )
                if alignment.get("space") != "projected":
                    raise ValueError("PI alignment gate currently requires projected space")
                if int(alignment["num_layers"]) != len(projected):
                    raise ValueError(
                        f"alignment gate has {alignment['num_layers']} layers, "
                        f"model has {len(projected)}"
                    )
                self._alignment_gate_payload = alignment
                self._alignment_gate_cache_key = alignment_path
                logger.info("PI alignment: loading gated subspace from %s", alignment_path)
            alignment = self._alignment_gate_payload
            aligned = list(projected)
            for layer_text, entry in alignment["layers"].items():
                layer = int(layer_text)
                hidden = aligned[layer]
                means = entry["means"].to(device=hidden.device, dtype=hidden.dtype)
                basis = entry["basis"].to(device=hidden.device, dtype=hidden.dtype)
                alpha = entry["alpha"].to(device=hidden.device, dtype=hidden.dtype)
                if means.ndim != 2 or basis.ndim != 2 or alpha.ndim != 1:
                    raise ValueError(f"invalid alignment tensors at layer {layer}")
                if means.shape != basis.transpose(0, 1).shape:
                    raise ValueError(f"alignment mean/basis mismatch at layer {layer}")
                if basis.shape[1] != alpha.shape[0] or basis.shape[0] != hidden.shape[-1]:
                    raise ValueError(f"alignment basis/alpha mismatch at layer {layer}")
                centered = hidden.unsqueeze(-2) - means.view(1, 1, *means.shape)
                coefficient = (centered * basis.transpose(0, 1)).sum(dim=-1)
                if bool(alignment.get("conditional", False)):
                    condition_weight = entry["condition_weight"].to(
                        device=hidden.device, dtype=hidden.dtype
                    )
                    condition_bias = entry["condition_bias"].to(
                        device=hidden.device, dtype=hidden.dtype
                    )
                    if condition_weight.shape != alpha.shape or condition_bias.shape != alpha.shape:
                        raise ValueError(f"alignment condition/alpha mismatch at layer {layer}")
                    evidence = coefficient.abs() / (1.0 + coefficient.abs())
                    alpha = alpha.view(1, 1, -1) * torch.sigmoid(
                        evidence * condition_weight.view(1, 1, -1)
                        + condition_bias.view(1, 1, -1)
                    )
                correction = (
                    coefficient.mul(alpha).unsqueeze(-1)
                    * basis.transpose(0, 1).view(1, 1, *means.shape)
                ).sum(dim=-2)
                aligned[layer] = hidden - correction
            projected = aligned
        if os.environ.get("STARVLA_MLM_NUMERIC_PATH", ""):
            from starVLA.model.modules.mlm_numeric_intervention import maybe_erase_numeric_slots
            projected = maybe_erase_numeric_slots(self, projected)
        return projected

    def _build_action_model(self):
        """Construct the action head; subclasses may replace only its topology."""
        return get_action_model(config=self.config)

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
        cot_conversations: List[list | None] | None = None,
        cot_modes: List[str] | None = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return projected states, CoT CE, mask bias, anchor loss, and shared z."""
        has_cot = bool(cot_conversations) and any(c is not None for c in cot_conversations)
        self._shared_z_from_decoder = None
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            cot_conversations=cot_conversations if has_cot else None,
            cot_modes=cot_modes,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            raw_vl_embs_list = list(qwenvl_outputs.hidden_states[-self.num_action_dit_layers:])
            vl_embs_list = self._project_vl_hidden_for_action(raw_vl_embs_list)
        last_hidden = vl_embs_list[-1]
        valid = getattr(self.qwen_vl_interface, "_last_encoder_attention_mask", None)
        if valid is None:
            valid = qwen_inputs.get("attention_mask")
        if valid is None:
            valid = torch.ones(
                last_hidden.shape[:2], device=last_hidden.device, dtype=torch.bool
            )
        valid = valid[:, : last_hidden.shape[1]].to(
            device=last_hidden.device, dtype=torch.bool
        )
        if valid.shape != last_hidden.shape[:2]:
            raise RuntimeError(
                "PI encoder mask/hidden-state mismatch: "
                f"{tuple(valid.shape)} vs {tuple(last_hidden.shape[:2])}"
            )
        action_valid = self._action_encoder_valid_mask(valid)
        attention_bias = self._dit_attention_bias(action_valid, last_hidden.dtype)
        cot_loss = (
            qwenvl_outputs.loss
            if has_cot and qwenvl_outputs.loss is not None
            else None
        )
        anchor_loss = self._representation_anchor_forward(
            qwen_inputs, raw_vl_embs_list, valid
        )
        shared_z = None
        if self.shared_z_enabled:
            shared_z = self._shared_z_from_decoder
            if shared_z is None:
                shared_z = self.shared_z_pooler(raw_vl_embs_list[-1], valid)
        return vl_embs_list, cot_loss, attention_bias, anchor_loss, shared_z






    def _representation_anchor_forward(
        self,
        qwen_inputs: dict,
        student_states: List[torch.Tensor],
        valid_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Cosine-anchor selected encoder layers to their pretrained initialization."""
        if not self.representation_anchor_enabled or self.is_inference or not self.training:
            return None
        teacher = self._representation_anchor_teacher
        if teacher is None:
            raise RuntimeError("representation anchor is enabled but its frozen teacher is missing")
        device = student_states[-1].device
        if self._representation_anchor_teacher_device != device:
            teacher.to(device)
            object.__setattr__(self, "_representation_anchor_teacher_device", device)
        teacher.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_outputs = teacher(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            teacher_states = list(teacher_outputs.hidden_states[-self.num_action_dit_layers:])
        losses = []
        for requested_index in self.representation_anchor_layers:
            index = requested_index % len(student_states)
            student = student_states[index]
            reference = teacher_states[index].detach()
            if student.shape != reference.shape:
                raise RuntimeError(
                    f"representation-anchor shape mismatch at layer {requested_index}: "
                    f"{tuple(student.shape)} vs {tuple(reference.shape)}"
                )
            mask = valid_tokens[:, : student.shape[1]]
            token_loss = 1.0 - F.cosine_similarity(student.float(), reference.float(), dim=-1)
            losses.append(token_loss.masked_select(mask).mean())
        if not losses:
            raise ValueError("framework.representation_anchor.layer_indices must not be empty")
        return torch.stack(losses).mean()

    def _tied_dynamics_forward(
        self,
        clean_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Tie action-head updates to multi-horizon cumulative delta-EEF motion.

        Both tensors remain in the dataset's normalized delta-action coordinates.
        Comparing cumulative prediction and target cancels the affine offset and
        avoids introducing an incorrect camera/world-frame transform.
        """
        dims = self.tied_dynamics_position_dims
        predicted_path = clean_actions[..., :dims].float().cumsum(dim=1)
        target_path = target_actions[..., :dims].float().cumsum(dim=1)
        indices = torch.tensor(
            [horizon - 1 for horizon in self.tied_dynamics_horizons],
            device=clean_actions.device,
            dtype=torch.long,
        )
        return F.smooth_l1_loss(
            predicted_path.index_select(1, indices),
            target_path.index_select(1, indices),
            beta=self.tied_dynamics_beta,
        )

    @staticmethod
    def _dit_attention_bias(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert a [B,L] keep mask to Diffusers' additive cross-attention mask."""
        bias = torch.zeros(valid.shape, device=valid.device, dtype=dtype)
        bias.masked_fill_(~valid, -10_000.0)
        return bias[:, None, :]

    def _action_encoder_valid_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """Optionally restrict PI memory to masked-reasoning slots per training row."""
        rate = self.encoder_mlm_action_slot_dropout_rate
        enabled = rate > 0.0 and self.training and not self.is_inference
        if not enabled:
            self._last_encoder_mlm_action_slot_only_rate = 0.0
            return valid

        slots = getattr(self.qwen_vl_interface, "_last_encoder_mlm_slot_mask", None)
        if slots is None:
            raise RuntimeError(
                "encoder MLM action-slot dropout was enabled, but the VLM did not expose "
                "its masked reasoning-slot positions"
            )
        slots = slots[:, : valid.shape[1]].to(device=valid.device, dtype=torch.bool)
        if slots.shape != valid.shape:
            raise RuntimeError(
                "encoder MLM slot mask/PI memory mismatch: "
                f"{tuple(slots.shape)} vs {tuple(valid.shape)}"
            )
        slots = slots & valid
        if not bool(slots.any(dim=1).all()):
            raise RuntimeError("every PI row must contain at least one masked reasoning slot")

        slot_only_rows = torch.rand(valid.shape[0], device=valid.device) < rate
        self._last_encoder_mlm_action_slot_only_rate = float(
            slot_only_rows.float().mean().detach().cpu()
        )
        return torch.where(slot_only_rows[:, None], slots, valid)

    def _repeated_diffusion_steps(self) -> int:
        """Read the action-side Monte Carlo repeat count from its canonical config."""
        value = int(
            self.config.framework.action_model.get("repeated_diffusion_steps", 2)
        )
        if value < 1:
            raise ValueError(
                "framework.action_model.repeated_diffusion_steps must be at least 1, "
                f"got {value}"
            )
        return value

    def _structured_aux_forward(
        self,
        examples: List[dict],
        conversations: List[list | None],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Apply N's structured targets directly to captured raw encoder layers."""
        if not self.structured_aux_enabled:
            return None, {}
        layer_states = self.qwen_vl_interface._structured_layer_out
        encoder_valid = self.qwen_vl_interface._last_encoder_attention_mask.bool()
        parsed = [
            example.get("cot_structured_targets")
            or extract_structured_cot_targets(conversation)
            for example, conversation in zip(examples, conversations)
        ]
        losses: list[torch.Tensor] = []
        loss_weights: list[float] = []
        metrics: dict[str, torch.Tensor] = {}
        for name, spec in self.structured_aux_specs.items():
            layer = int(spec["layer"])
            dim = int(spec["dim"])
            weight = float(spec.get("weight", 1.0))
            if weight < 0.0:
                raise ValueError(f"structured auxiliary weight for {name} must be nonnegative")
            if layer not in layer_states:
                raise RuntimeError(f"structured auxiliary layer {layer} was not captured")
            hidden = layer_states[layer]
            valid_tokens = encoder_valid[:, : hidden.shape[1]]
            if valid_tokens.shape != hidden.shape[:2]:
                raise RuntimeError(
                    f"structured auxiliary mask mismatch at layer {layer}: "
                    f"{tuple(valid_tokens.shape)} vs {tuple(hidden.shape[:2])}"
                )
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
                loss_weights.append(weight)
                metrics[f"structured_aux/{name}_loss"] = head_loss.detach()
            else:
                # Preserve identical DeepSpeed parameter participation on all ranks.
                losses.append(prediction.sum() * 0.0)
                loss_weights.append(weight)
            metrics[f"structured_aux/{name}_weight"] = torch.tensor(
                weight, device=hidden.device
            )
            metrics[f"structured_aux/{name}_coverage"] = torch.tensor(
                sum(present) / max(len(present), 1), device=hidden.device
            )
        if not losses:
            zero = sum((parameter.sum() * 0.0 for parameter in self.structured_aux_heads.parameters()))
            return zero, metrics
        denominator = sum(loss_weights)
        if denominator <= 0.0:
            raise ValueError("structured auxiliary target weights must sum to a positive value")
        weighted = sum(weight * loss for weight, loss in zip(loss_weights, losses)) / denominator
        return weighted, metrics

    @staticmethod
    def _apply_cot_graph_guard(
        cot_loss: Optional[torch.Tensor],
        examples: List[dict],
        conversations: List[list | None],
    ) -> Optional[torch.Tensor]:
        """Keep the decoder graph but remove supervision from a synthetic guard row."""
        if cot_loss is None:
            return None
        guard_only = all(
            conversation is None or bool(example.get("cot_graph_guard", False))
            for example, conversation in zip(examples, conversations)
        )
        return cot_loss * 0.0 if guard_only else cot_loss

    def _add_state_conditioning(
        self,
        instructions: List[str],
        states: List[np.ndarray] | None,
    ) -> tuple[List[str], float]:
        """Append discretised state, with per-example training-only dropout.

        Dropped state is omitted rather than replaced by zeros: zero is a valid
        normalized robot state and should not also mean "state unavailable".
        """
        if states is None:
            return instructions, 0.0
        if len(instructions) != len(states):
            raise ValueError(
                f"instruction/state batch mismatch: {len(instructions)} vs {len(states)}"
            )

        if self.training and self.state_dropout_rate > 0.0:
            keep_state = np.random.random(len(states)) >= self.state_dropout_rate
        else:
            keep_state = np.ones(len(states), dtype=bool)

        conditioned = []
        for instruction, state, keep in zip(instructions, states, keep_state):
            if keep:
                state_str = self.state2str_transform(np.asarray(state)[0])
                conditioned.append(f"{instruction} [STATE] {state_str} [ACTION]")
            else:
                conditioned.append(instruction)
        return conditioned, float(np.mean(keep_state)) if len(keep_state) else 0.0

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  # List[List[PIL.Image]], length B
        instructions = [example["lang"] for example in examples]  # List[str], length B
        actions = [example["action"] for example in examples]  # List[ndarray (T, action_dim)]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Worker-resolved CoT is the production path. The direct resolver fallback keeps
        # smoke/manual batches equivalent without loading a 273k-row mapping per rank.
        cot_from_workers = all("cot_conversation" in ex for ex in examples)
        if not cot_from_workers and self.cot_resolver is None:
            self.cot_resolver = build_cot_resolver(self.config, is_inference=False)
            assert_cot_prompt_consistent(self.cot_resolver, self.config)
        cot_conversations = [
            ex.get("cot_conversation") if cot_from_workers else self.cot_resolver.resolve(
                ex.get("trajectory_name", ""), ex.get("frame_index", 0)
            )
            for ex in examples
        ]
        cot_available = [
            bool(ex.get("cot_available", c is not None)) if cot_from_workers else c is not None
            for ex, c in zip(examples, cot_conversations)
        ]
        if self.cot_dropout_enabled:
            if cot_from_workers:
                cot_modes = [
                    ex.get("cot_mode", "cot" if c is not None else "no_cot")
                    for ex, c in zip(examples, cot_conversations)
                ]
            else:
                for i, conversation in enumerate(cot_conversations):
                    if (
                        self.training
                        and conversation is not None
                        and np.random.random() < self.cot_dropout_rate
                    ):
                        cot_conversations[i] = None
                cot_modes = ["cot" if c is not None else "no_cot" for c in cot_conversations]
        else:
            cot_modes = None
        has_cot = any(c is not None for c in cot_conversations)
        if self.training and self.cot_dropout_enabled and not has_cot:
            raise RuntimeError(
                "CoT dropout removed every target from this local training batch. "
                "The worker-side collator must retain at least one target per rank."
            )
        cot_coverage = sum(cot_available) / max(len(cot_available), 1)
        available_count = sum(cot_available)
        cot_keep_rate = (
            sum(c is not None for c in cot_conversations) / available_count
            if available_count
            else 0.0
        )

        augmentation = str(self.config.datasets.vla_data.get("augmentation", "none")).lower()
        # Previously also required has_cot, which silently disabled augmentation for
        # every action-only config even though the YAML asked for it: the ervla_*_
        # actiononly runs all set augmentation: crop_photometric and never augmented
        # (train/cot_coverage 0.0). The gate existed because the transform's original
        # purpose was keeping a sampled crop consistent with coordinate-bearing CoT
        # targets, but the image half is useful on its own. cot_conversations is
        # always a correctly-sized list -- entries are None when there is no CoT --
        # and augment_cot_sample passes a None conversation through untouched, so the
        # images are augmented and nothing else changes.
        # cot_from_workers still suppresses it: CoTVideoAugment already augmented
        # worker-side, and doing it twice would compound crops.
        if (
            self.training
            and not cot_from_workers
            and augmentation in {"photometric", "crop_photometric"}
        ):
            batch_images, cot_conversations = augment_cot_batch(
                batch_images, cot_conversations, mode=augmentation
            )

        # Append discretised proprioceptive state, optionally dropping the full
        # state suffix per training example. Raw state never enters PI-v3's DiT.
        instructions, state_keep_rate = self._add_state_conditioning(instructions, state)
        state = None  # state is now encoded in the instruction tokens

        latent_targets = None
        latent_teacher_audit = {}
        if self.decoder_latent_alignment is not None:
            if self._decoder_latent_teacher is None:
                raise RuntimeError("Training/evaluation loss requested on inference-only alignment model")
            latent_targets = self._decoder_latent_teacher.targets(
                batch_images, instructions, cot_conversations,
                list(self.decoder_latent_alignment.offsets), next(self.parameters()).device)
            if self.training and not self._latent_alignment_audited:
                latent_teacher_audit = self._decoder_latent_teacher.audit(
                    batch_images, instructions, cot_conversations,
                    list(self.decoder_latent_alignment.offsets), latent_targets,
                    next(self.parameters()).device)
                if self._decoder_latent_teacher.benchmark_batches:
                    latent_teacher_audit.update(self._decoder_latent_teacher.benchmark(
                        batch_images, instructions, cot_conversations,
                        list(self.decoder_latent_alignment.offsets), latent_targets,
                        next(self.parameters()).device))
                self._latent_alignment_audited = True

        # Step 1: encode through QwenVL
        vl_embs_list, cot_loss, encoder_attention_bias, representation_anchor_loss, shared_z = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            cot_conversations=(
                cot_conversations if has_cot and self.cot_text_supervision else None
            ),
            cot_modes=cot_modes,
        )
        cot_loss = self._apply_cot_graph_guard(cot_loss, examples, cot_conversations)
        self._last_shared_z = shared_z
        structured_aux_loss, structured_aux_metrics = self._structured_aux_forward(
            examples, cot_conversations
        )
        if self.projected_numeric_aux is not None:
            numeric_targets = [ex.get("cot_structured_targets") or extract_structured_cot_targets(c)
                               for ex, c in zip(examples, cot_conversations)]
            structured_aux_loss, structured_aux_metrics = self.projected_numeric_aux(
                vl_embs_list, self.qwen_vl_interface._last_encoder_mlm_slot_mask, numeric_targets)
        if latent_targets is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                structured_aux_loss, structured_aux_metrics = self.decoder_latent_alignment(
                    vl_embs_list, self.qwen_vl_interface._last_encoder_mlm_slot_mask, *latent_targets)
        base_hidden = vl_embs_list[-1]

        # Step 2: compute flow-matching loss over the action chunk
        with torch.autocast("cuda", dtype=torch.float32):
            # Align labels: keep only the last action_horizon timesteps.
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            future_z = None
            if self.shared_z_enabled and self.shared_z_temporal_enabled:
                future_images = [example.get("future_image") for example in examples]
                if all(image is not None for image in future_images):
                    future_z = self._encode_future_shared_z(future_images, instructions)

            shared_z_loss = None
            shared_z_metrics = {}
            if self.shared_z_enabled:
                shared_z_loss, shared_z_metrics = self._shared_z_supervision(
                    shared_z, examples, cot_conversations, actions_target, future_z
                )

            repeated_diffusion_steps = self._repeated_diffusion_steps()
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat every VLM layer embedding to match the duplicated action batch.
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            encoder_attention_bias_repeated = encoder_attention_bias.repeat(
                repeated_diffusion_steps, 1, 1
            )
            shared_z_repeated = (
                shared_z.repeat(repeated_diffusion_steps, 1)
                if shared_z is not None else None
            )
            encoder_memory_keep = (
                self._shared_z_memory_keep(actions_target.shape[0], base_hidden.device)
                if self.shared_z_enabled else None
            )
            encoder_memory_keep_repeated = (
                encoder_memory_keep.repeat(repeated_diffusion_steps)
                if encoder_memory_keep is not None else None
            )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_output = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=encoder_attention_bias_repeated,
                return_clean_actions=self.tied_dynamics_enabled,
                z_conditioning=self._shared_z_action_conditioning(shared_z_repeated),
                encoder_memory_keep=encoder_memory_keep_repeated,
            )
            if self.tied_dynamics_enabled:
                action_loss, clean_actions = action_output
                tied_dynamics_loss = self._tied_dynamics_forward(
                    clean_actions, actions_target_repeated
                )
            else:
                action_loss = action_output
                tied_dynamics_loss = None

        result = {
            "action_loss": action_loss,
            "cot_coverage": cot_coverage,
            "cot_keep_rate": cot_keep_rate,
            "state_keep_rate": state_keep_rate,
        }
        result.update(latent_teacher_audit)
        if cot_loss is not None:
            result["cot_loss"] = cot_loss
            if self.encoder_mlm_enabled:
                # Keep cot_loss as the trainer's established weighted auxiliary slot while
                # exposing an unambiguous metric name for analysis and W&B.
                result["encoder_mlm_loss"] = cot_loss.detach()
        if self.encoder_mlm_action_slot_dropout_rate > 0.0:
            result["encoder_mlm_action_slot_only_rate"] = torch.tensor(
                self._last_encoder_mlm_action_slot_only_rate,
                device=action_loss.device,
                dtype=action_loss.dtype,
            )
        if representation_anchor_loss is not None:
            result["representation_anchor_loss"] = representation_anchor_loss
        if tied_dynamics_loss is not None:
            result["tied_dynamics_loss"] = tied_dynamics_loss
        if structured_aux_loss is not None:
            result["structured_aux_loss"] = structured_aux_loss
            result.update(structured_aux_metrics)
            if self.decoder_latent_alignment is not None and self.training:
                self._latent_alignment_steps += 1
                if self.latent_alignment_gradient_interval > 0 and (
                        self._latent_alignment_steps == 1 or
                        self._latent_alignment_steps % self.latent_alignment_gradient_interval == 0):
                    scale = float(self.config.trainer.loss_scale.get("structured_aux", 1.0))
                    alignment_gradients = self.decoder_latent_alignment.gradient_metrics(
                        action_loss, structured_aux_loss, vl_embs_list, scale)
                    result.update(alignment_gradients)
                    if self._latent_alignment_steps == 1 and (
                            not dist.is_initialized() or dist.get_rank() == 0):
                        # The trainer writes W&B only at logging_frequency multiples.
                        # Preserve the initial calibration in stdout as well.
                        print("DECODER_ALIGNMENT_INITIAL_GRADIENTS",
                              {k: float(v) for k, v in alignment_gradients.items()}, flush=True)
            if self.projected_numeric_aux is not None and self.training:
                self._numeric_aux_steps += 1
                if self.numeric_aux_grad_interval > 0 and (self._numeric_aux_steps == 1 or
                        self._numeric_aux_steps % self.numeric_aux_grad_interval == 0):
                    scale = float(self.config.trainer.loss_scale.get("structured_aux", 1.0))
                    result.update(self.projected_numeric_aux.gradient_metrics(
                        action_loss, structured_aux_loss, vl_embs_list, scale))
        if self.shared_z_enabled:
            if structured_aux_loss is not None:
                raise RuntimeError(
                    "shared_z and legacy structured_aux cannot share the same trainer loss slot"
                )
            result["structured_aux_loss"] = shared_z_loss
            result.update(shared_z_metrics)
            result["shared_z_memory_keep_rate"] = encoder_memory_keep.float().mean()
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Run inference and return a denoised action trajectory.

        Steps:
          1. Optionally resize images to the training observation resolution.
          2. Encode images + instruction (with discretised state prefix) through QwenVL.
          3. Project layer-wise VLM hidden states to the Action DiT latent space.
          4. Run the flow-matching sampler to produce the action chunk.

        Args:
            examples: List[dict], each entry requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - state: np.ndarray shaped (1, state_dim), optional

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape (B, action_horizon, action_dim),
                    denoised actions in the normalised action space.
        """

        if os.environ.get("STARVLA_PAIRED_ROLLOUT", "0") == "1":
            if len(examples) != 1 or "_paired_policy_seed" not in examples[0]:
                raise ValueError("Paired rollout requires batch size 1 and an explicit per-request seed")
            torch.use_deterministic_algorithms(True)
            seed = int(examples[0]["_paired_policy_seed"])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # List[List[PIL.Image]]
        instructions = [example["lang"] for example in examples]  # List[str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # List[ndarray] or None

        # _add_state_conditioning never applies dropout in eval/inference mode.
        instructions, _ = self._add_state_conditioning(instructions, state)
        state = None

        # Optionally resize images to the resolution used during training.
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: encode through QwenVL
        vl_embs_list, _, encoder_attention_bias, _, shared_z = self._encode_vl_hidden_states(
            batch_images, instructions
        )
        base_hidden = vl_embs_list[-1]

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 2: run the flow-matching sampler to produce the denoised action chunk.
        with torch.autocast("cuda", dtype=torch.float32):
            encoder_memory_keep = (
                self._shared_z_memory_keep(base_hidden.shape[0], base_hidden.device)
                if self.shared_z_enabled else None
            )
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state,
                encoder_attention_mask=encoder_attention_bias,
                z_conditioning=self._shared_z_action_conditioning(shared_z),
                encoder_memory_keep=encoder_memory_keep,
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise a state vector into 256 uniform bins and return it as a space-separated token string.

        Follows the π₀.5 convention: bins span [-1, 1] uniformly.
        Example: [-0.5, 0.1, 0.8] -> "95 133 203"
        """
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def add_discretized_state_to_instruction(self, instructions: List[str], states: List[np.ndarray]) -> List[str]:
        """Append discretised proprioceptive state tokens to each instruction.

        Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
        This lets the VLM attend to the robot state purely through its
        existing text-token pathway — no extra encoder required.
        """
        updated_instructions = []
        for instr, state in zip(instructions, states):
            state_str = self.state2str_transform(state[0])
            updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
        return updated_instructions


if __name__ == "__main__":
    import argparse

    import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    import os

    if os.getenv("DEBUG_MODE", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    cfg = OmegaConf.load(args.config_yaml)

    model = Qwen_PI_v3(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)

    def print_model_size(m: nn.Module, depth: int = 1):
        """Print parameter counts for each top-level submodule (depth=1)."""
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}")
        print(f"{'Module':<35} {'Params':>12}  {'%':>6}")
        print(f"{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}")
        print(f"  {'TOTAL':<33} {total:>12,}  100.0%")
        print(f"{'='*55}\n")

    print_model_size(model, depth=1)

    action_dim = int(cfg.framework.action_model.action_dim)
    state_dim = int(cfg.framework.action_model.state_dim)
    action_horizon = int(cfg.framework.action_model.action_horizon)
    obs_image_size = cfg.datasets.vla_data.get("obs_image_size", [224, 224])
    image_height = int(obs_image_size[0])
    image_width = int(obs_image_size[1])
    sample_video_groups = cfg.datasets.vla_data.get("sample_video_groups", None)
    num_views = len(sample_video_groups[0]) if sample_video_groups else 2

    # fake sample
    images = [
        Image.fromarray(np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8))
        for _ in range(num_views)
    ]
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(action_horizon, action_dim)).astype(np.float16),
        "image": images,
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float16),
    }

    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # # Advance: try forward model with dataloader
    # # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.include_state = True

    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # #
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch)

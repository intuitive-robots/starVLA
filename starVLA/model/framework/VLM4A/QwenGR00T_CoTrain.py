# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Nils Blank / Juelich / HKUST] in [2026].
"""
QwenGR00T_CoT — Knowledge-Insulated VLA with step-based CoT supervision
========================================================================

Works with train_starvla.py (single VLA dataloader) or train_starvla_cotrain.py
(adds a second VLM dataloader).  CoT supervision is self-contained in forward().

Architecture (π0 / Knowledge Insulation — pi.website/research/knowledge_insulation,
arxiv 2605.02881):

  1. **Knowledge insulation (stop-gradient)**
     Gradients from the flow-matching DiT expert do not flow into the VLM backbone.
     The backbone is updated only by:
       a) Step-based CoT CE loss  (per-phase spatial grounding annotations)
       b) VLM co-training CE loss (optional — only when using cotrain script)
     This prevents high-variance diffusion gradients from eroding the backbone's
     pre-trained language and visual understanding.

  2. **Step-based CoT supervision** (self-contained in forward)
     The cot_resolver maps (trajectory_name, frame_index) → ShareGPT conversations
     built by scripts/create_cot_mapping.py.  No second dataloader needed.

  3. **Explicit CoT inference** (optional, config-controlled)
     When framework.cot.generate_at_inference=true, predict_action does a two-pass:
       Pass 1: generate CoT text autoregressively from the user prompt
       Pass 2: re-encode [prompt + generated CoT] → richer hidden states
       Diffuse: DiT cross-attends to ALL tokens including the grounding reasoning
     When false (default), hidden states come from the prompt-only forward pass
     (implicit CoT — backbone has internalized grounding from training).

Config knobs (framework:)
  knowledge_insulation: true          # stop-grad DiT → backbone
  cot:
    generate_at_inference: false      # explicit two-pass CoT at inference
    max_new_tokens: 128               # max CoT generation length
  (datasets.vla_data.cot.*)          # CoT resolver config — see cot_resolver.py
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.cot_resolver import assert_cot_prompt_consistent, build_cot_resolver
from starVLA.model.modules.projector.readout import ReadoutProjector
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


# ──────────────────────────────────────────────────────────────────────────────
#  Default config  (YAML values override these; missing YAML keys fall back here)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QwenGR00TCoTDefaultConfig:
    name: str = "QwenGR00T_CoT"

    # ── VLM backbone ──────────────────────────────────────────────────────────
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct",
        "attn_implementation": "sdpa",
        "vl_hidden_dim": 2048,
    })

    # ── Flow-matching action expert ───────────────────────────────────────────
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 768,
        "hidden_size": 768,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 8,
        "state_dim": 0,
        "action_horizon": 24,
        "repeated_diffusion_steps": 4,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "diffusion_model_cfg": {
            "num_layers": 12,
            "output_dim": 768,
            "dropout": 0.1,
        },
    })

    # ── Knowledge insulation ──────────────────────────────────────────────────
    # When True: hidden states are detached before being passed to the DiT.
    # Backbone gradients come only from VLM co-training data and CoT CE loss.
    knowledge_insulation: bool = True

    # ── Inference-time CoT generation ─────────────────────────────────────────
    # generate_at_inference=True: two-pass predict_action —
    #   generate CoT → re-encode [prompt+CoT] → richer hidden states → diffuse.
    # generate_at_inference=False (default): prompt-only hidden states (implicit CoT).
    cot: dict = field(default_factory=lambda: {
        "generate_at_inference": False,
        "max_new_tokens": 128,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────────────────

@FRAMEWORK_REGISTRY.register("QwenGR00T_CoT")
@FRAMEWORK_REGISTRY.register("QwenGR00T_CoTrain")  # backward-compat alias
class QwenGR00T_CoT(baseframework):
    """
    Knowledge-insulated VLA model with step-based CoT supervision.

    Use with train_starvla.py (no VLM co-training needed):
      batch_vla → model.forward(batch_vla) → action_loss [+ cot_loss]

    Optionally use with train_starvla_cotrain.py for VLM co-training:
      batch_vlm → model.qwen_vl_interface(**vlm) → vlm_loss  (separate step)

    Gradient flow (knowledge_insulation=True):
      VLM backbone  ←  CoT CE loss  [+ VLM co-training CE if cotrain]
      DiT expert    ←  diffusion loss only
      backbone ✗←  DiT gradients  (detach applied to hidden states)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TCoTDefaultConfig, config)

        # VLM backbone
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        # Flow-matching action expert
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # Knowledge insulation toggle — read once and cache as a plain bool.
        # Avoids repeated OmegaConf attribute lookups in the hot forward path.
        self.knowledge_insulation: bool = bool(
            getattr(self.config.framework, "knowledge_insulation", True)
        )
        logger.info(
            f"[QwenGR00T_CoTrain] knowledge_insulation={'ON' if self.knowledge_insulation else 'OFF'}"
        )

        # Step-based CoT resolver
        self.cot_resolver = build_cot_resolver(self.config)
        assert_cot_prompt_consistent(self.cot_resolver, self.config)

        # GR00T N1.5-style readout projector (optional).
        # When enabled, compresses all VLM tokens into `num_tokens` learned
        # readout representations before passing to the DiT.
        # Sits on the action side (after detach) → updated only by diffusion loss.
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
            _ffn_dim    = _g("ffn_dim",     0) or None   # 0 → None → 2×hidden
            _dropout    = float(readout_cfg.get("dropout", 0.0) if hasattr(readout_cfg, "get") else getattr(readout_cfg, "dropout", 0.0))
            vlm_hidden  = self.qwen_vl_interface.model.config.hidden_size
            self.readout_projector: Optional[ReadoutProjector] = ReadoutProjector(
                vlm_hidden, _num_tokens, _num_layers, _num_heads, _ffn_dim, _dropout
            )
            logger.info(
                f"[QwenGR00T_CoT] Readout projector ON — "
                f"{_num_tokens} tokens, {_num_layers} layers, {_num_heads} heads"
            )
        else:
            self.readout_projector: Optional[ReadoutProjector] = None

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        """
        VLA forward pass.  Returns a dict with at least 'action_loss'; also
        returns 'cot_loss' when CoT annotations are available for this batch.

        Knowledge insulation:
          last_hidden is detached before the DiT sees it, so diffusion gradients
          cannot reach the backbone.  The backbone is updated only by VLM
          co-training CE loss (called separately in the cotrain loop) and the
          CoT CE loss computed here.
        """
        batch_images   = [ex["image"]  for ex in examples]
        instructions   = [ex["lang"]   for ex in examples]
        actions        = [ex["action"] for ex in examples]
        state          = [ex["state"]  for ex in examples] if "state" in examples[0] else None

        # Resolve per-step CoT texts (None → no CoT loss for that sample)
        cot_conversations = [
            self.cot_resolver.resolve(
                ex.get("trajectory_name", ""),
                ex.get("frame_index", 0),
            )
            for ex in examples
        ]
        has_cot = any(c is not None for c in cot_conversations)
        cot_coverage = sum(c is not None for c in cot_conversations) / max(len(cot_conversations), 1)

        # ── 1. VLM backbone forward ──────────────────────────────────────────
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            cot_conversations=cot_conversations if has_cot else None,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # (B, L, H)

        # CoT CE loss comes from the backbone's LM head over the CoT tokens.
        cot_loss = qwenvl_outputs.loss if (has_cot and qwenvl_outputs.loss is not None) else None

        # ── 2. Knowledge insulation: block DiT → backbone gradient path ──────
        #
        # With stop-gradient the backbone is updated only by:
        #   a) VLM co-training CE loss   (separate batch in cotrain loop)
        #   b) CoT CE loss               (computed just above)
        #
        # The DiT sees a frozen view of the representations, which empirically
        # leads to faster convergence and better VLM knowledge retention
        # (pi.website/research/knowledge_insulation).
        if self.knowledge_insulation:
            last_hidden = last_hidden.detach()

        # ── 3. Readout projection (optional) ─────────────────────────────────
        # Compress all VLM tokens → fixed-size readout tokens before DiT.
        # Readout projector is on the action side (post-detach) → only updated
        # by diffusion loss, consistent with knowledge insulation.
        if self.readout_projector is not None:
            dit_context = self.readout_projector(last_hidden)  # [B, num_tokens, H]
        else:
            dit_context = last_hidden                          # [B, L, H]

        # ── 4. DiT flow-matching action expert ───────────────────────────────
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=dit_context.device, dtype=dit_context.dtype
            )
            actions_target = actions_t[:, -self.action_horizon:, :]

            repeated = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_rep   = actions_target.repeat(repeated, 1, 1)
            dit_context_rep = dit_context.repeat(repeated, 1, 1)

            state_rep = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=dit_context.device, dtype=dit_context.dtype)
                state_rep = state_t.repeat(repeated, 1, 1)

            action_loss = self.action_model(dit_context_rep, actions_rep, state_rep)

        result = {"action_loss": action_loss, "cot_coverage": cot_coverage}
        if cot_loss is not None:
            result["cot_loss"] = cot_loss
        return result

    # ── Inference ─────────────────────────────────────────────────────────────

    def _generate_cot(self, batch_images: list, instructions: list) -> list[str]:
        """
        Pass 1 of explicit CoT inference: autoregressively generate one CoT string
        per sample from the user prompt alone.

        Uses greedy decoding for determinism. The generated text is in the same
        format as the training targets (phase description + Qwen point token).
        """
        cot_cfg = getattr(self.config.framework, "cot", {})
        max_new = cot_cfg.get("max_new_tokens", 128) if hasattr(cot_cfg, "get") else getattr(cot_cfg, "max_new_tokens", 128)

        # Prompt-only inputs — add_generation_prompt=True is set inside build_qwenvl_inputs
        # when cot_conversations is None.
        prompt_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_ids = self.qwen_vl_interface.generate(
                **prompt_inputs,
                max_new_tokens=max_new,
                do_sample=False,
            )

        new_ids = gen_ids[:, prompt_len:]
        return self.qwen_vl_interface.processor.tokenizer.batch_decode(
            new_ids, skip_special_tokens=True
        )

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs,
    ) -> dict:
        """
        Two-mode action prediction:

        Implicit CoT (generate_at_inference=false, default):
            Encode user prompt → hidden states → diffuse.
            CoT knowledge is implicit in the backbone from training.

        Explicit CoT (generate_at_inference=true):
            Pass 1 — generate CoT text autoregressively from user prompt.
            Pass 2 — re-encode [prompt + generated CoT] → richer hidden states
                     (DiT cross-attends to all tokens, including grounding reasoning).
            Pass 3 — diffuse from the conditioned hidden states.
        """
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        state        = [ex["state"] for ex in examples] if "state" in examples[0] else None

        train_obs_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_size:
            batch_images = resize_images(batch_images, target_size=train_obs_size)

        cot_cfg = getattr(self.config.framework, "cot", {})
        generate_at_inference = (
            cot_cfg.get("generate_at_inference", False)
            if hasattr(cot_cfg, "get")
            else getattr(cot_cfg, "generate_at_inference", False)
        )

        if generate_at_inference:
            # Pass 1: generate CoT
            generated_cot = self._generate_cot(batch_images, instructions)
            logger.debug(f"[CoT inference] generated: {generated_cot[0][:80]}...")

            # Pass 2: re-encode with generated CoT as assistant context.
            # Human prompt mirrors the CoT_prompt config so both passes use the same user message.
            cot_prompt_cfg = getattr(self.config.datasets.vla_data, "CoT_prompt", "{instruction}")
            cot_conversations = [
                [
                    {"from": "human", "value": cot_prompt_cfg},
                    {"from": "gpt",   "value": cot_text},
                ]
                for cot_text in generated_cot
            ]
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                cot_conversations=cot_conversations,
            )
            # Drop labels — inference only needs hidden states, not CE loss
            qwen_inputs.pop("labels", None)
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        if self.readout_projector is not None:
            dit_context = self.readout_projector(last_hidden)  # [B, num_tokens, H]
        else:
            dit_context = last_hidden                          # [B, L, H]

        state_t = (
            torch.from_numpy(np.array(state)).to(dit_context.device, dtype=dit_context.dtype)
            if state is not None else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(dit_context, state_t)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

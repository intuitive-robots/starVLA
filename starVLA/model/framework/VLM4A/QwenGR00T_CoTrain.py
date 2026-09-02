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
  knowledge_insulation: false         # stop-grad DiT → backbone
  cot:
    generate_at_inference: false      # explicit two-pass CoT at inference
    max_new_tokens: 128               # max CoT generation length
  (datasets.vla_data.cot.*)          # CoT resolver config — see cot_resolver.py
"""

import os
import re
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
from starVLA.dataloader.cot_resolver import NullCoTResolver, assert_cot_prompt_consistent, build_cot_resolver
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
        "repeated_diffusion_steps": 8,
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
    knowledge_insulation: bool = False

    # ── Inference-time CoT generation ─────────────────────────────────────────
    # generate_at_inference=True: two-pass predict_action —
    #   generate CoT → re-encode [prompt+CoT] → richer hidden states → diffuse.
    # generate_at_inference=False (default): prompt-only hidden states (implicit CoT).
    cot: dict = field(default_factory=lambda: {
        "generate_at_inference": False,
        "max_new_tokens": 128,
    })

    # What the DiT is conditioned on:
    #   "all"      (default) every VLM token -- image, prompt and CoT.
    #   "cot_only" ONLY the CoT/trace token hidden states. Turns the trace into a
    #              hard information bottleneck: the action head sees the reasoning
    #              and nothing else, so trace quality becomes the whole signal.
    #              Requires cot.generate_at_inference=true, since at inference the
    #              trace is then the only context that exists.
    dit_context: str = "all"
    cot_context_tokens: int = 64


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
        self.is_inference = bool(kwargs.get("is_inference", False))
        vla_data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        data_cot_cfg = getattr(vla_data_cfg, "cot", None) if vla_data_cfg is not None else None
        self.cot_source = str(
            data_cot_cfg.get("source", "none")
            if hasattr(data_cot_cfg, "get")
            else getattr(data_cot_cfg, "source", "none")
        ).lower()

        # Knowledge insulation toggle — read once and cache as a plain bool.
        # Avoids repeated OmegaConf attribute lookups in the hot forward path.
        self.dit_context_mode: str = str(
            getattr(self.config.framework, "dit_context", "all")
        )
        self.cot_context_tokens: int = int(
            getattr(self.config.framework, "cot_context_tokens", 64)
        )
        if self.dit_context_mode not in ("all", "cot_only"):
            raise ValueError(
                f"framework.dit_context must be 'all' or 'cot_only', got {self.dit_context_mode!r}"
            )
        if self.dit_context_mode == "cot_only":
            logger.info(
                f"[QwenGR00T_CoTrain] dit_context=cot_only -- DiT sees ONLY the "
                f"{self.cot_context_tokens} trailing CoT tokens"
            )

        self.knowledge_insulation: bool = bool(
            getattr(self.config.framework, "knowledge_insulation", False)
        )
        logger.info(
            f"[QwenGR00T_CoTrain] knowledge_insulation={'ON' if self.knowledge_insulation else 'OFF'}"
        )

        # Step-based CoT resolver. Offline conditioning diagnostics inject their
        # conversations directly and can skip loading million-row mappings.
        if bool(kwargs.get("skip_cot_resolver", False)):
            self.cot_resolver = NullCoTResolver()
        else:
            self.cot_resolver = build_cot_resolver(self.config, is_inference=self.is_inference)
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

    def _resolve_training_cot(self, examples: List[dict]) -> tuple[list, float]:
        """Prefer worker-resolved CoT so geometric label rewrites are preserved."""
        cot_from_workers = all("cot_conversation" in ex for ex in examples)
        if cot_from_workers:
            conversations = [ex.get("cot_conversation") for ex in examples]
            available = [
                bool(ex.get("cot_available", conversation is not None))
                for ex, conversation in zip(examples, conversations)
            ]
        else:
            conversations = [
                self.cot_resolver.resolve(
                    ex.get("trajectory_name", ""), ex.get("frame_index", 0)
                )
                for ex in examples
            ]
            available = [conversation is not None for conversation in conversations]
        if (
            not any(conversation is not None for conversation in conversations)
            and getattr(self, "training", False)
            and getattr(self, "cot_source", "none") in {"mapping", "sparc_sqlite"}
        ):
            raise RuntimeError(
                "mapping-backed CoT training reached forward() without a local CoT target; "
                "the dataloader collator must retain at least one target per rank"
            )
        coverage = sum(available) / max(len(available), 1)
        return conversations, coverage

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

        # Dataloader workers may already have jointly transformed images and
        # coordinate-bearing CoT. Re-resolving here would restore stale labels.
        cot_conversations, cot_coverage = self._resolve_training_cot(examples)
        has_cot = any(c is not None for c in cot_conversations)

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

        # ── 2b. Trace bottleneck (optional) ──────────────────────────────────
        # Keep only the CoT token hidden states, so the action head is forced to
        # route everything through the reasoning. Samples without a CoT
        # annotation carry no context in this mode and are dropped from the
        # action loss rather than conditioned on zeros.
        cot_keep = None
        if self.dit_context_mode == "cot_only":
            labels = qwen_inputs.get("labels")
            if labels is None:
                raise RuntimeError(
                    "dit_context='cot_only' needs CoT labels, but this batch has no CoT "
                    "annotations at all. Check datasets.vla_data.cot.mapping_path."
                )
            last_hidden, cot_keep = self._cot_only_context(
                last_hidden, labels, self.cot_context_tokens
            )
            if last_hidden.shape[0] == 0:
                # No annotated sample in this batch -- skip the action loss for
                # this step instead of emitting a degenerate one.
                return {"action_loss": last_hidden.sum() * 0.0,
                        "cot_coverage": cot_coverage,
                        **({"cot_loss": cot_loss} if cot_loss is not None else {})}

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
            # cot_only dropped the unannotated samples from the context; the
            # targets must be filtered identically or they misalign.
            if cot_keep is not None:
                actions_t = actions_t[cot_keep]
            actions_target = actions_t[:, -self.action_horizon:, :]

            repeated = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_rep   = actions_target.repeat(repeated, 1, 1)
            dit_context_rep = dit_context.repeat(repeated, 1, 1)

            state_rep = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=dit_context.device, dtype=dit_context.dtype)
                if cot_keep is not None:
                    state_t = state_t[cot_keep]
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

    # Ring buffer of recently generated traces, used by the "roll" corruption mode
    # when a batch is too small to donate a wrong-but-plausible trace to itself.
    _corrupt_trace_pool: list = []

    def _cot_only_context(self, last_hidden, labels, n_tokens: int):
        """Restrict the DiT's conditioning to the CoT (trace) tokens alone.

        `labels` is -100 everywhere except the supervised CoT answer span, so it
        already marks exactly the trace tokens in both the training and the
        explicit-inference path -- no re-tokenisation needed.

        Returns (context, keep) where context is [K, n_tokens, H] and keep is a
        bool mask over the batch. Samples with no CoT have nothing to condition
        on in this mode and are dropped by the caller rather than fed zeros,
        which would train the DiT on empty context.
        """
        mask = labels != -100                       # [B, L]
        keep = mask.any(dim=1)                      # [B]
        B, _, H = last_hidden.shape
        out = last_hidden.new_zeros((int(keep.sum()), n_tokens, H))
        row = 0
        for b in range(B):
            if not bool(keep[b]):
                continue
            span = last_hidden[b][mask[b]]          # [n_cot, H]
            # Keep the tail: the trace coordinates sit at the end of the answer.
            take = min(span.shape[0], n_tokens)
            out[row, :take] = span[-take:]
            row += 1
        return out, keep

    def _corrupt_cot_traces(self, generated_cot: list[str], mode: str) -> list[str]:
        """Replace the coordinates inside <|trace|>{...}<|/trace|>, keeping the rest intact.

        The point is to hold token count and syntax fixed while destroying the
        spatial content, so a flat success rate means the DiT is not reading the
        trace. Non-matching strings are returned unchanged.

        Modes:
          roll  — reuse another sample's trace (a real, well-formed trace of the
                  WRONG scene). The strongest control: plausible but misgrounded.
          rand  — uniform random points in the 0-1000 normalized coordinate space.
          shuf  — same points, permuted order. Kills temporal direction only.
        """
        if mode not in ("roll", "rand", "shuf"):
            raise ValueError(f"unknown STARVLA_COT_CORRUPT mode: {mode!r} (roll|rand|shuf)")

        pat = re.compile(r"(<\|trace\|>)(.*?)(<\|/trace\|>)", re.DOTALL)
        payloads = [m.group(2) if (m := pat.search(c)) else None for c in generated_cot]

        def _sub(cot: str, payload: str) -> str:
            return pat.sub(lambda m: m.group(1) + payload + m.group(3), cot, count=1)

        if mode == "roll":
            current = [p for p in payloads if p]
            # Pool spans calls so single-sample batches still find a foreign donor.
            available = current + [p for p in self._corrupt_trace_pool if p not in current]
            out = []
            for i, (cot, own) in enumerate(zip(generated_cot, payloads)):
                if own is None:
                    out.append(cot)
                    continue
                donors = [p for p in available if p != own]
                if not donors:
                    logger.warning("[CoT corrupt] no foreign trace available; passing sample through")
                    out.append(cot)
                    continue
                out.append(_sub(cot, donors[i % len(donors)]))
            self._corrupt_trace_pool = available[:64]
            return out

        # rand / shuf operate on the coordinate array only. Parsing the whole
        # payload would swallow the "2" in the "trace_2d" key as a coordinate.
        arr_pat = re.compile(r"\[\s*\[.*?\]\s*\]", re.DOTALL)

        def _replace(payload: str) -> str:
            arr = arr_pat.search(payload)
            if not arr:
                return payload
            nums = re.findall(r"-?\d+", arr.group(0))
            if len(nums) < 2:
                return payload
            pts = [(int(nums[i]), int(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]
            if mode == "rand":
                pts = [(int(np.random.randint(0, 1001)), int(np.random.randint(0, 1001))) for _ in pts]
            else:  # shuf
                pts = [pts[i] for i in np.random.permutation(len(pts))]
            body = "[" + ", ".join(f"[{x}, {y}]" for x, y in pts) + "]"
            return payload[: arr.start()] + body + payload[arr.end() :]

        return [
            _sub(cot, _replace(p)) if p is not None else cot
            for cot, p in zip(generated_cot, payloads)
        ]

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

            # Ablation hook: corrupt the trace coordinates while keeping the CoT
            # string syntactically valid, to test whether the DiT reads trace
            # *content* or merely benefits from the extra tokens in context.
            # Off unless STARVLA_COT_CORRUPT is set. See _corrupt_cot_traces.
            corrupt_mode = os.environ.get("STARVLA_COT_CORRUPT", "")
            if corrupt_mode:
                generated_cot = self._corrupt_cot_traces(generated_cot, corrupt_mode)

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
            # Keep labels when cot_only needs them to locate the trace span;
            # they must not reach the backbone, so pop either way.
            cot_labels = qwen_inputs.pop("labels", None)
        else:
            cot_labels = None
            if self.dit_context_mode == "cot_only":
                raise RuntimeError(
                    "dit_context='cot_only' requires cot.generate_at_inference=true -- "
                    "without a generated trace the DiT would have no context at all."
                )
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

        if self.dit_context_mode == "cot_only":
            if cot_labels is None:
                raise RuntimeError("dit_context='cot_only': no CoT label span available at inference")
            last_hidden, _ = self._cot_only_context(
                last_hidden, cot_labels.to(last_hidden.device), self.cot_context_tokens
            )

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

        result = {"normalized_actions": pred_actions.detach().cpu().numpy()}
        # Surface the reasoning that conditioned this chunk so eval clients can
        # visualise it. Only present in explicit-CoT mode; after any corruption
        # hook, so what ships is exactly what the DiT saw.
        if generate_at_inference:
            result["cot_text"] = list(generated_cot)
        return result

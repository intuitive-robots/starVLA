"""Encoder-only Qwen3.5 enc-dec backbone.

The Qwen3-VL enc-dec arms go through ``QWen3_EncDec``, which drives the model built by
``encdec-vlm``'s ``build_qwen3_vl_enc_dec``. Qwen3.5 needs its own path for two reasons:

1. **Different architecture.** Qwen3.5 is hybrid — each block is either ``full_attention``
   (``self_attn``) or ``linear_attention`` (``linear_attn``, a gated DeltaNet). Its encoder
   is therefore not "the same layers with a bidirectional mask": a DeltaNet is a directional
   scan, so ``Qwen35EncDecForConditionalGeneration`` runs each such layer forwards AND over
   the reversed valid positions and mixes them with a learned per-layer gate,
   ``fwd + bidir_gates[i] * bwd``, then applies ``enc_norm`` and ``enc_scale`` (and an
   optional residual ``enc_proj`` MLP) to the encoder output. None of that exists in the
   Qwen3-VL path, so loading those weights into it would silently run a different encoder
   than the one that was trained.

2. **Different module layout.** That class keeps ``encoder_layers`` on the OUTER model,
   while ``QWen3_EncDec`` looks for them on ``model.model.language_model`` — which is also
   why the checkpoint stores 318 top-level ``encoder_layers.*`` tensors that a plain
   ``load_state_dict`` drops on the floor.

We reuse the reference implementation rather than reimplementing it: the encoder forward is
``model._encode``, exactly as in training. This wrapper only reproduces the embedding /
position-id preamble of ``Qwen35EncDecForConditionalGeneration.forward`` (vision merge,
3-D position ids, prompt mask) and stops after the encoder, because every consumer here sets
``skip_decoder: true``.
"""

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_VL_Interface
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# The Qwen3.5 enc-dec model class lives with the runs that produce its checkpoints, the same
# arrangement QWen3_EncDec uses for encdec-vlm. Override with STARVLA_Q35_ENCDEC_REPO.
Q35_ENCDEC_REPO = os.environ.get(
    "STARVLA_Q35_ENCDEC_REPO", "/e/project1/m3/blank4/code/train_downstream"
)


class _QWen3_5_EncDec_Interface(_QWen3_5_VL_Interface):
    """Qwen3.5 encoder as a frozen-or-trainable feature extractor for an action head."""

    def __init__(self, config, **kwargs):
        nn.Module.__init__(self)  # the parent would from_pretrained a causal Qwen3.5

        qcfg = config.framework.get("qwenvl", {})
        model_id = qcfg.get("base_vlm")
        attn = qcfg.get("attn_implementation", "sdpa")
        if attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning("flash_attn not installed, falling back to sdpa")
                attn = "sdpa"

        if not bool(qcfg.get("skip_decoder", True)):
            raise ValueError(
                "QWen3_5_EncDec is encoder-only: it stops after _encode and never builds "
                "decoder inputs. Set framework.qwenvl.skip_decoder=true."
            )

        if Q35_ENCDEC_REPO not in sys.path:
            sys.path.insert(0, Q35_ENCDEC_REPO)
        from train.models.qwen35_enc_dec import build_qwen35_enc_dec

        # Build FROM the trained checkpoint directory when we have one. Its config.json
        # carries enc_proj, and `from_pretrained` resolves `encoder_layers.*`, `enc_norm`,
        # `enc_scale` and `bidir_gates` under their native names -- the prefix mismatch that
        # makes a bare load_state_dict drop all 318 encoder tensors never arises.
        ckpt = qcfg.get("encdec_ckpt", None)
        model, processor = build_qwen35_enc_dec(
            model_path=str(ckpt) if ckpt else model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            processor_path=model_id,  # checkpoint dirs carry weights, not processor files
            enc_proj=str(qcfg.get("enc_proj", "linear")),
            decoder_mode="prefix",
        )
        if ckpt:
            logger.info(f"Qwen3.5 enc-dec: built from checkpoint {ckpt}")

        processor.tokenizer.padding_side = "left"
        self.model = model
        self.processor = processor
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self.skip_decoder = True
        self._last_encoder_attention_mask = None

        # transformers renamed the per-block attention kind between the version the
        # reference encoder was written against (`layer.layer_type`) and the one in this
        # image, 5.14.1 (`layer.block_type`). _enc_layer reads `layer_type` to decide
        # between the DeltaNet scan and self-attention, so without this alias every
        # encoder forward dies with
        #   AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'layer_type'
        # Alias rather than fork: the two names carry the same value.
        for _blk in self.model.encoder_layers:
            if not hasattr(_blk, "layer_type"):
                if not hasattr(_blk, "block_type"):
                    raise RuntimeError(
                        "Qwen3.5 decoder layer exposes neither layer_type nor block_type; "
                        f"transformers layout changed again (have {type(_blk).__name__})"
                    )
                _blk.layer_type = _blk.block_type

        # An ablation switch in the reference: encoder DeltaNet layers run causally only.
        # Off by default, because every trained checkpoint so far used the backward scan.
        if bool(qcfg.get("skip_backward_scan", False)):
            self.model._skip_bwd_scan = True
            logger.warning("Qwen3.5 encoder: backward DeltaNet scan DISABLED by config")

        # PI consumes hidden_states[-N:], one VLM layer per DiT block, so every encoder
        # layer's output has to be collected -- _encode returns only the final state.
        self._collect_layers = bool(qcfg.get("collect_encoder_layers", False))
        self._layer_out: List[torch.Tensor] = []
        self._collecting = False
        encoder_blocks = list(self.model.encoder_layers)
        self.num_encoder_feature_layers = len(encoder_blocks)
        if self._collect_layers:
            # NOT register_forward_hook: _encode never calls a block's forward. It reaches
            # into the submodules (input_layernorm, linear_attn/self_attn, mlp) so it can run
            # the DeltaNet twice for the bidirectional scan, so a hook on the block never
            # fires and PI would silently get zero layer states. Wrap the per-layer function
            # instead, which returns exactly the residual-stream state after each layer.
            _orig_enc_layer = self.model._enc_layer

            def _enc_layer_collecting(i, layer, h, *args, **kw):
                out = _orig_enc_layer(i, layer, h, *args, **kw)
                # Guarded by _collecting so gradient-checkpoint recomputation in backward,
                # which re-runs this outside forward, cannot append a second copy.
                if self._collecting:
                    self._layer_out.append(out)
                return out

            self.model._enc_layer = _enc_layer_collecting
            logger.info(f"collecting all {len(encoder_blocks)} Qwen3.5 encoder layer states")

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        """Mirror of Qwen35EncDecForConditionalGeneration.forward up to ``_encode``.

        Kept deliberately parallel to that method (train_downstream
        ``train/models/qwen35_enc_dec.py``): same vision merge, same 3-D position ids, same
        prompt mask. Diverging here would mean evaluating a different encoder than training.
        """
        if labels is not None:
            raise RuntimeError(
                "labels were passed to the encoder-only Qwen3.5 backbone; it cannot produce "
                "a CoT loss. Drop the cot block, or use the Qwen3-VL enc-dec arm."
            )

        m = self.model.model
        embeds = m.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            img = m.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            img = torch.cat(img, 0).to(embeds.device, embeds.dtype)
            image_mask, _ = m.get_placeholder_mask(input_ids, inputs_embeds=embeds, image_features=img)
            embeds = embeds.masked_scatter(image_mask, img)
        if pixel_values_videos is not None:
            vid = m.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output
            vid = torch.cat(vid, 0).to(embeds.device, embeds.dtype)
            _, video_mask = m.get_placeholder_mask(input_ids, inputs_embeds=embeds, video_features=vid)
            embeds = embeds.masked_scatter(video_mask, vid)
        if position_ids is None:
            position_ids = m.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=mm_token_type_ids,
            )
        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        # No labels here, so the reference's prompt_mask reduces to the valid mask.
        prompt_mask = valid

        self._layer_out = []
        self._collecting = self._collect_layers
        try:
            enc_hidden = self.model._encode(embeds, prompt_mask, position_ids)
        finally:
            self._collecting = False

        if prompt_mask.shape != enc_hidden.shape[:2]:
            raise RuntimeError(
                "Qwen3.5 encoder mask/hidden-state mismatch: "
                f"{tuple(prompt_mask.shape)} vs {tuple(enc_hidden.shape[:2])}"
            )
        # Consumed by the action head alongside hidden_states[-1]; keeping the encoder's own
        # mask keeps left padding out of the DiT's cross-attention.
        self._last_encoder_attention_mask = prompt_mask

        if self._collect_layers:
            layers = tuple(self._layer_out)
            self._layer_out = []
            if len(layers) != self.num_encoder_feature_layers:
                raise RuntimeError(
                    f"expected {self.num_encoder_feature_layers} encoder layer states, "
                    f"hooks captured {len(layers)}"
                )
            # Last element is the post-norm encoder output, matching hidden_states[-1].
            return CausalLMOutputWithPast(
                loss=None, logits=None, hidden_states=layers[:-1] + (enc_hidden,)
            )
        return CausalLMOutputWithPast(loss=None, logits=None, hidden_states=(enc_hidden,))

    def generate(self, **kwargs):
        raise RuntimeError(
            "generate() needs a decoder; the Qwen3.5 enc-dec interface is encoder-only."
        )

# Encoder-decoder Qwen3-VL backbone for starVLA.
#
# Why this exists
# ---------------
# The encdec-vlm project converts causal Qwen3-VL into a T5Gemma-2-style encoder-decoder:
# a bidirectional encoder over [image ; instruction], plus a causal decoder with merged
# attention. Offline probes and behaviour-cloning on cached features (encdec-vlm
# results.md, Tier-1/Tier-2) say the encoder representation differs measurably from the
# causal one. Those are both PROXIES. This adapter exists so the same backbones can be
# measured the way the claim is actually stated: LIBERO rollout success.
#
# What the framework needs
# ------------------------
# VLM4A frameworks consume exactly one thing from the VLM:
#     qwenvl_outputs = self.qwen_vl_interface(**inputs, output_hidden_states=True)
#     last_hidden    = qwenvl_outputs.hidden_states[-1]      # [B, L, H]  -> action expert
# So the whole adapter is: run the bidirectional ENCODER and return its states in that
# slot. Prompt construction, generation and tokenisation are inherited unchanged, which
# keeps this arm identical to the causal arm everywhere except the feature source.
#
# The encoder-state handoff
# -------------------------
# The enc-dec model stashes encoder states in `_aux_ctx["enc_hidden"]`, but only when
#   (a) `encoder_prefix_length` is passed  (that flag is what selects training_mode), and
#   (b) `_aux_enabled` is set on the module owning the encoder stack.
# Both are normally set by the aux-loss training path. We need the states at INFERENCE
# too — every rollout step — so we enable them explicitly and pass the full sequence
# length, putting the entire prompt in the bidirectional encoder. Same convention as
# encdec-vlm's phase0d/tier2_extract, so features here match what the probes measured.

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.training.trainer_utils import initialize_overwatch

from .QWen3 import _QWen3_VL_Interface

logger = initialize_overwatch(__name__)

ENCDEC_REPO = "/e/project1/m3/blank4/code/encdec-vlm"


class _StopAfterEncoder(Exception):
    """Control-flow signal: the encoder has run, the decoder is not needed."""


class _QWen3_EncDec_Interface(_QWen3_VL_Interface):
    """Encoder-decoder Qwen3-VL. Returns ENCODER hidden states in hidden_states[-1].

    Config (framework.qwenvl):
        base_vlm            base Qwen3-VL weights the enc-dec is built from
        encdec_ckpt         optional dir with model.safetensors (a trained enc-dec run)
        encdec_maskft       optional phase0h blob; its encoder_tail is overlaid on top
        enc_dec_mode        full_duplicate (default) | layer_split | overlap_split
        n_encoder_layers    only used by layer_split
        n_decoder_layers    decoder-exclusive depth for overlap_split
        n_overlap_layers    duplicated boundary depth for overlap_split
        use_merged_attention  T5Gemma-2 merged attention (default true)
        separate_cross_attention  use independent per-layer decoder→encoder attention;
                            decoder embeddings/norm/head are also decoupled
        skip_decoder        don't run the decoder at all (default true). The action
                            expert never reads decoder states, and in full_duplicate the
                            decoder is a second 28-layer stack over a 2T merged sequence
                            — pure waste at every rollout step.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        # Deliberately NOT calling super().__init__: that builds the causal model.
        nn.Module.__init__(self)

        qcfg = config.framework.get("qwenvl", {})
        model_id = qcfg.get("base_vlm")
        attn = qcfg.get("attn_implementation", "sdpa")
        if attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning("flash_attn not installed, falling back to sdpa")
                attn = "sdpa"

        import sys
        if ENCDEC_REPO not in sys.path:
            sys.path.insert(0, ENCDEC_REPO)
        from train.models.qwen3_vl_enc_dec import build_qwen3_vl_enc_dec

        choice_cfg = dict(config.framework.get("choice_policy", {}) or {})
        if choice_cfg.get("enabled", False):
            choice_cfg.setdefault(
                "num_action_queries",
                int(config.framework.action_model.action_horizon),
            )

        model, processor = build_qwen3_vl_enc_dec(
            model_path=model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            trust_remote_code=True,
            mode=str(qcfg.get("enc_dec_mode", "full_duplicate")),
            n_encoder_layers=int(qcfg.get("n_encoder_layers", 14)),
            n_decoder_layers=(
                int(qcfg.get("n_decoder_layers"))
                if qcfg.get("n_decoder_layers", None) is not None else None
            ),
            n_overlap_layers=int(qcfg.get("n_overlap_layers", 0)),
            use_merged_attention=bool(qcfg.get("use_merged_attention", True)),
            separate_cross_attention=bool(qcfg.get("separate_cross_attention", False)),
            cross_attention_gate_init=float(qcfg.get("cross_attention_gate_init", 0.1)),
            choice_cfg=choice_cfg,
        )

        ckpt = qcfg.get("encdec_ckpt", None)
        if ckpt:
            from pathlib import Path

            from safetensors.torch import load_file
            sd = load_file(str(Path(ckpt) / "model.safetensors"))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"enc-dec ckpt {ckpt}: {len(sd)} tensors, "
                        f"missing {len(missing)}, unexpected {len(unexpected)}")
            adapter_markers = (
                ".cross_attn_adapters.", ".decoder_embed_tokens.", ".decoder_norm."
            )
            real_missing = [
                key for key in missing if not any(marker in key for marker in adapter_markers)
            ]
            if len(real_missing) > 50:
                raise RuntimeError(
                    f"{len(real_missing)} non-adapter keys missing loading {ckpt} — "
                    f"architecture mismatch. First few: {real_missing[:5]}")
            if bool(qcfg.get("separate_cross_attention", False)):
                # The base enc-dec checkpoint predates these modules. Refresh every copied
                # decoder component from the just-loaded checkpoint rather than retaining
                # the earlier from_pretrained initialization.
                lm = model.model.language_model
                lm.decoder_embed_tokens.load_state_dict(lm.embed_tokens.state_dict())
                lm.decoder_norm.load_state_dict(lm.norm.state_dict())
                for layer, adapter in zip(lm.layers, lm.cross_attn_adapters):
                    adapter.q_proj.load_state_dict(layer.self_attn.q_proj.state_dict())
                    adapter.k_proj.load_state_dict(layer.self_attn.k_proj.state_dict())
                    adapter.v_proj.load_state_dict(layer.self_attn.v_proj.state_dict())
                    adapter.o_proj.load_state_dict(layer.self_attn.o_proj.state_dict())
                logger.info("initialized decoupled decoder and separate cross-attention "
                            "from the loaded enc-dec checkpoint")

        maskft = qcfg.get("encdec_maskft", None)
        if maskft:
            self._apply_maskft(model, maskft)

        processor.tokenizer.padding_side = "left"
        self.model = model
        self.processor = processor
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # A frozen causal readout is central to the representation-shaping ablation:
        # autograd still traverses it into encoder memory, but its own blocks cannot adapt
        # into a language-only shortcut. Generic dotted-path freezing cannot express the
        # decoder suffix of a shared layer-split ModuleList, so handle that partition here.
        if bool(qcfg.get("freeze_decoder_blocks", False)):
            decoder_blocks = self._decoder_blocks(self._text_model())
            for block in decoder_blocks:
                for parameter in block.parameters():
                    parameter.requires_grad = False
            logger.info("froze %d causal decoder blocks (gradients still reach encoder)",
                        len(decoder_blocks))

        # _aux_ctx is only populated when this flag is on; it is off outside aux training.
        n_en = 0
        for m in self.model.modules():
            if hasattr(m, "_aux_enabled"):
                m._aux_enabled = True
                n_en += 1
        if n_en == 0:
            raise RuntimeError("no module exposes _aux_enabled — encoder states cannot "
                               "be captured; check the enc-dec build")
        logger.info(f"enabled encoder-state capture on {n_en} module(s)")

        # QwenPI consumes hidden_states[-N:] -- one VLM layer per DiT block. Returning a
        # 1-tuple would silently hand it the same tensor N times, so collect every encoder
        # layer's output when asked. Hooks, because the enc-dec stashes only the final
        # encoder state in _aux_ctx.
        self._collect_layers = bool(qcfg.get("collect_encoder_layers", False))
        self._layer_out = []
        self._structured_layer_out = {}
        self._last_choice_hidden = None
        lm_ = self._text_model()
        encoder_blocks = self._encoder_blocks(lm_)
        self.num_encoder_feature_layers = len(encoder_blocks)
        if self._collect_layers:
            def _grab(mod, args, out):
                self._layer_out.append(out[0] if isinstance(out, tuple) else out)
            for blk in encoder_blocks:
                blk.register_forward_hook(_grab)
            logger.info(f"collecting all {len(encoder_blocks)} encoder layer states")

        structured_cfg = dict(config.framework.get("structured_aux", {}) or {})
        structured_layers = sorted({
            int(spec.get("layer"))
            for spec in (structured_cfg.get("targets", {}) or {}).values()
            if spec.get("layer", None) is not None
        }) if structured_cfg.get("enabled", False) else []
        for layer_idx in structured_layers:
            if not 0 <= layer_idx < len(encoder_blocks):
                raise ValueError(
                    f"structured auxiliary layer {layer_idx} outside encoder depth "
                    f"{len(encoder_blocks)}"
                )
            def _grab_structured(mod, args, out, idx=layer_idx):
                self._structured_layer_out[idx] = out[0] if isinstance(out, tuple) else out
            encoder_blocks[layer_idx].register_forward_hook(_grab_structured)
        if structured_layers:
            logger.info("capturing structured auxiliary encoder layers %s", structured_layers)

        # `skip_decoder` is the CONFIG intent; `_skip_now` is the per-call decision, set in
        # forward(). The hook is registered unconditionally and consults _skip_now, because
        # the auxiliary-reasoning arms need the decoder ON for training steps (labels
        # present) and OFF at rollout -- no labels means nothing to score, and in
        # full_duplicate the decoder is a second 28-layer stack over a 2T merged sequence,
        # i.e. pure waste at every env step.
        self.skip_decoder = bool(qcfg.get("skip_decoder", True))
        self._skip_now = True

        def _maybe_stop(mod, args, kwargs=None):
            if self._skip_now:
                raise _StopAfterEncoder()

        lm = self._text_model()
        decoder_blocks = self._decoder_blocks(lm)
        if not decoder_blocks:
            raise RuntimeError("encoder-decoder backbone exposes no causal decoder blocks")
        decoder_blocks[0].register_forward_pre_hook(_maybe_stop)
        logger.info(f"decoder gate installed (skip_decoder={self.skip_decoder}); the decoder "
                    f"runs only when labels are passed and skip_decoder is false")

    def _text_model(self):
        m = self.model.model
        return m.language_model if hasattr(m, "language_model") else m

    @staticmethod
    def _encoder_blocks(lm):
        """Return encoder blocks for both separate-stack and disjoint split modes."""
        if hasattr(lm, "encoder_layers"):
            return list(lm.encoder_layers)
        if hasattr(lm, "n_encoder_layers"):
            return list(lm.layers[:int(lm.n_encoder_layers)])
        raise RuntimeError("enc-dec language model exposes no encoder block partition")

    @staticmethod
    def _decoder_blocks(lm):
        """Return only causal decoder blocks, never the split encoder prefix."""
        if hasattr(lm, "encoder_layers"):
            return list(lm.layers)
        if hasattr(lm, "n_encoder_layers"):
            return list(lm.layers[int(lm.n_encoder_layers):])
        return []

    @staticmethod
    def _apply_maskft(model, blob_path: str):
        """Overlay an encdec-vlm phase0h mask-fine-tuned encoder tail.

        phase0h stores only the unfrozen blocks as {"<offset>.<param>": tensor}, since the
        rest is untouched base weights.
        """
        # Prefer safetensors. torch.load performs lazy imports partway through
        # deserialisation, so a transient stale NFS handle on site-packages kills the
        # process mid-load with OSError 116 — observed on three separate nodes, while the
        # safetensors arms (base/encdec) loaded fine throughout. safetensors does a plain
        # mmap read with no imports, so it does not have that failure mode.
        st_path = str(blob_path)
        if st_path.endswith(".pt"):
            cand = st_path[:-3] + ".tail.safetensors"
            if os.path.exists(cand):
                st_path = cand
        if st_path.endswith(".safetensors"):
            from safetensors import safe_open
            tail = {}
            with safe_open(st_path, framework="pt", device="cpu") as f:
                cfg = json.loads(f.metadata()["cfg"])
                for k in f.keys():
                    tail[k] = f.get_tensor(k)
            logger.info(f"loaded mask-FT tail from safetensors: {st_path}")
            vis_tail = {k[len("visual."):]: v for k, v in tail.items()
                        if k.startswith("visual.")}
            tail = {k: v for k, v in tail.items() if not k.startswith("visual.")}
        else:
            blob = torch.load(blob_path, map_location="cpu")
            tail, cfg = blob["encoder_tail"], blob["cfg"]
            vis_tail = blob.get("vision_tail", {})
        m = model.model
        lm = m.language_model if hasattr(m, "language_model") else m
        blocks = lm.encoder_layers[-int(cfg["unfreeze"]):]
        per_block: dict = {}
        for k, v in tail.items():
            i, rest = k.split(".", 1)
            per_block.setdefault(int(i), {})[rest] = v
        for i, sd in per_block.items():
            blocks[i].load_state_dict(sd)
        logger.info(f"applied mask-FT encoder tail from {blob_path} (cfg={cfg})")

        # Vision tower, when the tail was trained with --unfreeze-vit. Older blobs have no
        # "vision_tail" key and land here as {}, so this is a no-op for every existing tail.
        n_vit = int(cfg.get("unfreeze_vit", 0) or 0)
        if vis_tail and n_vit > 0:
            visual = getattr(m, "visual", None) or getattr(model, "visual", None)
            if visual is None:
                raise RuntimeError("tail carries a vision_tail but no vision tower found")
            vstack = getattr(visual, "blocks", None) or getattr(visual, "layers", None)
            if vstack is None:
                raise RuntimeError(f"vision tower has no .blocks/.layers: {type(visual)}")
            vblocks = vstack[-n_vit:]
            per_v: dict = {}
            for k, v in vis_tail.items():
                i, rest = k.split(".", 1)
                per_v.setdefault(int(i), {})[rest] = v
            for i, sd in per_v.items():
                vblocks[i].load_state_dict(sd)
            logger.info(f"applied mask-FT VISION tail: last {n_vit} tower blocks")
        elif vis_tail or n_vit:
            raise RuntimeError(f"inconsistent vision tail: {len(vis_tail)} tensors vs "
                               f"unfreeze_vit={n_vit}")

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        """Run the bidirectional encoder; return its states as hidden_states[-1].

        Two regimes:

        * **no labels** (every rollout step, and the arms with no reasoning supervision) —
          the whole prompt goes into the encoder and the decoder is skipped. There is no
          response to leak: the action expert, not the LM, produces the output.
        * **labels present and skip_decoder=false** (auxiliary-reasoning arms) — we do NOT
          set encoder_prefix_length. `_EncDecOuterMixin.forward` derives it as
          ``(labels != -100).argmax(dim=1)``, so the bidirectional encoder sees exactly the
          prompt and the causal decoder sees exactly the CoT answer, and we return the
          decoder's CE loss alongside the encoder states. The action expert still reads only
          `hidden_states[-1]`, i.e. encoder states — it never sees the reasoning.
        """
        kwargs.pop("output_hidden_states", None)   # we supply the states ourselves
        self._layer_out = []
        self._structured_layer_out = {}
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("enc-dec interface needs input_ids to size the encoder prefix")
        B, T = input_ids.shape[:2]

        labels = kwargs.get("labels")
        cot_prefix_length = kwargs.pop("_cot_encoder_prefix_length", None)
        run_choice_queries = bool(kwargs.pop("_run_choice_queries", False))
        if labels is not None and self.skip_decoder:
            # Silently returning loss=None here would turn a reasoning arm into the plain
            # arm with no visible symptom -- exactly the failure mode that kept cot_loss
            # missing from every earlier run. Fail loudly instead.
            raise RuntimeError(
                "labels were passed but framework.qwenvl.skip_decoder=true, so the decoder "
                "cannot run and no CoT loss would be produced. Set skip_decoder=false on "
                "reasoning arms, or drop the cot block from the config.")

        run_decoder = labels is not None and not self.skip_decoder
        lm = self._text_model()
        lm._choice_queries_active = run_choice_queries
        self._last_choice_hidden = None
        self._skip_now = not run_decoder
        if run_decoder and cot_prefix_length is not None:
            kwargs["encoder_prefix_length"] = cot_prefix_length
        elif not run_decoder:
            kwargs["encoder_prefix_length"] = torch.full(
                (B,), T, device=input_ids.device, dtype=torch.long)

        out = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                out = self.model(**kwargs)
            except _StopAfterEncoder:
                pass

        ctx = None
        for m in self.model.modules():
            if getattr(m, "_aux_ctx", None) is not None:
                ctx = m._aux_ctx
                break
        if ctx is None:
            raise RuntimeError("no _aux_ctx captured — encoder_prefix_length was not "
                               "honoured by this enc-dec build")
        enc_hidden = ctx["enc_hidden"]                       # [B, T_prompt, H]
        enc_attention_mask = ctx.get("enc_attention_mask")
        if enc_attention_mask is None:
            input_attention_mask = kwargs.get("attention_mask")
            if input_attention_mask is None:
                enc_attention_mask = torch.ones(
                    enc_hidden.shape[:2], device=enc_hidden.device, dtype=torch.bool
                )
            else:
                enc_attention_mask = input_attention_mask[:, : enc_hidden.shape[1]].bool()
        if enc_attention_mask.shape != enc_hidden.shape[:2]:
            raise RuntimeError(
                "encoder attention mask/hidden-state mismatch: "
                f"{tuple(enc_attention_mask.shape)} vs {tuple(enc_hidden.shape[:2])}"
            )
        # QwenGR00T consumes this alongside hidden_states[-1]. Keeping the exact mask
        # captured by the encoder avoids exposing left padding or decoder-target slots to DiT.
        self._last_encoder_attention_mask = enc_attention_mask
        if run_choice_queries:
            location = str(
                self.config.framework.get("choice_policy", {}).get("location", "encoder")
            ).lower()
            key = "encoder_choice" if location == "encoder" else "decoder_choice"
            choice_hidden = ctx.get(key)
            if choice_hidden is None:
                raise RuntimeError(
                    f"choice queries were requested at {location!r} but {key!r} "
                    "was not captured by the encoder-decoder backbone"
                )
            self._last_choice_hidden = choice_hidden
        loss = getattr(out, "loss", None) if out is not None else None
        # Training intentionally drops the very large vocabulary logits after the
        # decoder CE has been computed.  Clean modality diagnostics need the
        # per-token CE, so allow them to retain logits explicitly without changing
        # the default training/evaluation memory behaviour.
        diagnostic_logits = (
            getattr(out, "logits", None)
            if getattr(self, "_return_decoder_logits_for_diagnostics", False)
            and out is not None
            else None
        )
        if run_decoder and loss is None:
            raise RuntimeError(
                "decoder ran but produced no loss -- labels reached the model without being "
                "scored. Check that build_qwenvl_inputs attached a `labels` tensor with at "
                "least one non-(-100) entry per sample.")
        if self._collect_layers:
            layers = tuple(self._layer_out)
            self._layer_out = []
            # last element must be the final encoder state, matching hidden_states[-1]
            return CausalLMOutputWithPast(loss=loss, logits=diagnostic_logits,
                                          hidden_states=layers[:-1] + (enc_hidden,))
        return CausalLMOutputWithPast(loss=loss, logits=diagnostic_logits,
                                      hidden_states=(enc_hidden,))

    def generate(self, **kwargs):
        if self.skip_decoder:
            raise RuntimeError(
                "generate() needs the decoder, which is disabled. Set "
                "framework.qwenvl.skip_decoder=false to use autoregressive decoding.")
        return super().generate(**kwargs)

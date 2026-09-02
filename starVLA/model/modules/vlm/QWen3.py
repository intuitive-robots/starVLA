# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        # Metadata used by the encoder-decoder adapter to split prompt from answer.
        # The causal Qwen model has no use for it and must not receive an unknown kwarg.
        kwargs.pop("_cot_encoder_prefix_length", None)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        # bf16, not fp16: weights are cast to bf16 elsewhere (matches forward()'s
        # autocast). An fp16 autocast here forces every op to cast bf16 weights to
        # fp16 and back each decode step -- pure overhead, no accuracy upside.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        cot_conversations=None,
        cot_modes=None,
        **kwargs,
    ):
        """
        Build model inputs from raw data (images + instructions + optional solutions/cot_conversations).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        Args:
            images:            List[List[PIL.Image]]  — one inner list per sample
            instructions:      List[str]
            solutions:         List[str] | None  — fast-tokenizer action token sequences
            cot_conversations: List[list|None] | None  — per-sample ShareGPT conversations,
                               each [{from: human, value: ...}, {from: gpt, value: ...}] or None
                               (unannotated → no CoT loss for that sample). The human value may
                               contain {instruction}, which is filled in here.
            cot_modes:         List[str|None] | None — explicit ERVLA-style ``cot`` /
                               ``no_cot`` condition. The marker is appended to the user prompt.

        NOTE: this parameter used to be absent, so every framework that passed
        ``cot_conversations=`` (QwenGR00T, QwenGR00T_CoTrain) had it silently swallowed by
        **kwargs — no labels were built, ``qwenvl_outputs.loss`` stayed None, and no
        ``cot_loss`` key ever reached the trainer. Ported from QWen3_5.py, which already had
        this path; see RESULTS_TIER3.md §3 for the runs that were affected.
        """

        # Create user-only messages (prompt side).
        # For CoT samples the human turn comes from the conversation, not the CoT_prompt config.
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        if cot_modes is not None and len(cot_modes) != len(instructions):
            raise ValueError("cot_modes length must match batch size")
        for i, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in imgs]

            conv = cot_conversations[i] if cot_conversations is not None else None
            if conv is not None:
                prompt = conv[0]["value"].replace("{instruction}", instruction)
            elif "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            mode = cot_modes[i] if cot_modes is not None else None
            if mode is not None:
                if mode not in {"cot", "no_cot"}:
                    raise ValueError(f"unsupported cot mode: {mode!r}")
                prompt = f"{prompt}\n/{mode}"

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # ── CoT conversation path ─────────────────────────────────────────────────
        # Loss is computed only on gpt (assistant) tokens; human/system tokens are masked.
        if cot_conversations is not None:
            assert len(cot_conversations) == len(messages), "cot_conversations length must match batch size"

            messages_with_cot = []
            for msg, conv in zip(messages, cot_conversations):
                gpt_text = conv[1]["value"] if conv is not None else ""
                messages_with_cot.append(
                    msg + [{"role": "assistant", "content": [{"type": "text", "text": gpt_text}]}]
                )

            # Encode prompt-only to measure per-sample prompt lengths. With
            # add_generation_prompt=True the encoding ends exactly where the assistant starts.
            prompt_only_inputs = self.processor.apply_chat_template(
                messages,  # user-only messages
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            batch_inputs = self.processor.apply_chat_template(
                messages_with_cot,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Build labels: mask everything up to the assistant turn, supervise only the CoT
            # tokens. With LEFT padding the valid tokens are right-aligned, so the assistant
            # span is the last (full_valid - prompt_valid) positions.
            #
            # For the enc-dec backbone this is also what defines the architecture split:
            # _EncDecOuterMixin sets encoder_prefix_length = (labels != -100).argmax(dim=1),
            # so the bidirectional encoder sees exactly the prompt and the causal decoder
            # sees exactly the CoT answer.
            max_full_len = batch_inputs["input_ids"].shape[1]
            labels = batch_inputs["input_ids"].clone()
            encoder_prefix_length = []
            for i in range(labels.size(0)):
                prompt_valid = int(prompt_only_inputs["attention_mask"][i].sum())
                full_valid = int(batch_inputs["attention_mask"][i].sum())
                asst_len = full_valid - prompt_valid
                asst_start = max_full_len - max(asst_len, 0)
                encoder_prefix_length.append(asst_start)
                labels[i, :asst_start] = IGNORE_INDEX
                # An unmatched frame deliberately has no auxiliary target.  Appending an
                # empty assistant turn still adds chat-template terminators, so mask the
                # complete row instead of accidentally training on those special tokens.
                if cot_conversations[i] is None:
                    labels[i, :] = IGNORE_INDEX
                # Mask any residual pad tokens that ended up in the label.
                if self.processor.tokenizer.pad_token_id is not None:
                    labels[i][labels[i] == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels
            # Keep the prompt boundary even for all-ignored rows.  Deriving it with
            # `(labels != -100).argmax()` would return zero and remove that sample's image
            # and instruction from the bidirectional encoder.  The ordinary causal adapter
            # drops this private metadata in forward(); QWen3_EncDec consumes it.
            batch_inputs["_cot_encoder_prefix_length"] = torch.tensor(
                encoder_prefix_length, device=labels.device, dtype=torch.long)
            return batch_inputs.to(self.model.device)

        # Preparation for inference (no CoT)

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass

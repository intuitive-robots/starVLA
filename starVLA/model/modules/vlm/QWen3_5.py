# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3.5 with qwen2.5
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
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

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
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, cot_conversations=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions/cot_conversations).
        Follow Official Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        Args:
            images:            List[List[PIL.Image]]  — one inner list per sample
            instructions:      List[str]
            solutions:         List[str] | None  — fast-tokenizer action token sequences (existing path)
            cot_conversations: List[list|None] | None  — per-sample ShareGPT conversations.
                               Each entry is [{from: human, value: ...}, {from: gpt, value: ...}]
                               or None (holdout / unannotated → no CoT loss for that sample).
                               The human value may contain {instruction} which is filled in here.
        """

        # Create user-only messages (prompt side).
        # For CoT samples, the human turn comes from the conversation (not CoT_prompt config).
        # For non-CoT samples, fall back to CoT_prompt config or bare instruction.
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for i, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in imgs]

            conv = cot_conversations[i] if cot_conversations is not None else None
            if conv is not None:
                prompt = conv[0]["value"].replace("{instruction}", instruction)
            elif "CoT_prompt" in self.config.datasets.vla_data:
                prompt = self.config.datasets.vla_data.get("CoT_prompt", "").replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[i]}]})
            messages.append(msg)

        # ── CoT conversation path ─────────────────────────────────────────────────
        # Loss is computed only on gpt (assistant) tokens; human/system tokens are masked.
        if cot_conversations is not None:
            assert len(cot_conversations) == len(messages), "cot_conversations length must match batch size"

            # Append assistant (gpt) turn. None entries → empty string → no loss contribution.
            messages_with_cot = []
            for msg, conv in zip(messages, cot_conversations):
                msg_cot = list(msg)
                gpt_text = conv[1]["value"] if conv is not None else ""
                msg_cot.append({"role": "assistant", "content": [{"type": "text", "text": gpt_text}]})
                messages_with_cot.append(msg_cot)

            # Encode prompt-only (no assistant) to measure per-sample prompt lengths.
            # With add_generation_prompt=True the encoding ends exactly where the assistant starts.
            prompt_only_inputs = self.processor.apply_chat_template(
                messages,  # user-only messages
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Encode full sequence (user + assistant CoT).
            batch_inputs = self.processor.apply_chat_template(
                messages_with_cot,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Build labels: mask everything up to (and including) the prompt, supervise only
            # the assistant CoT tokens. With left padding the valid tokens are right-aligned.
            max_full_len = batch_inputs["input_ids"].shape[1]
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                prompt_valid = int(prompt_only_inputs["attention_mask"][i].sum())
                full_valid = int(batch_inputs["attention_mask"][i].sum())
                asst_len = full_valid - prompt_valid
                # With left padding: last `full_valid` positions are real tokens;
                # last `asst_len` of those are the assistant response.
                asst_start = max_full_len - max(asst_len, 0)
                labels[i, :asst_start] = IGNORE_INDEX
                # Mask any residual pad tokens that ended up in the label.
                if self.processor.tokenizer.pad_token_id is not None:
                    labels[i][labels[i] == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels
            return batch_inputs.to(self.model.device)

        # Preparation for inference (no CoT / no solutions)
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
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
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

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass

"""Instruction dropout with mask slots (default off).

``framework.instruction_dropout_rate`` > 0 replaces, in a training row drawn with that rate, every token of
the instruction with ONE learned mask id after tokenization, so the prompt layout is identical to the
unmasked row. The mask tokens can then carry a reconstruction target (``fusion_registers.instruction``):
the EMA copy's block-L tokens of the true instruction at the same positions. Inference never masks.

The instruction is located by tokenizing it alone (with and without a leading space) and finding that exact
id sequence in the row; a row where neither matches is left unmasked and counted
(``structured_aux/instr_mask_fallback``), never guessed.
"""
from typing import List, Optional

import numpy as np
import torch

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class InstructionMaskMixin:
    def _configure_instruction_mask(self, config) -> None:
        cfg = config.framework
        self.instruction_dropout_rate = float(cfg.get("instruction_dropout_rate", 0.0) or 0.0)
        self._instr_mask = None
        self._instr_mask_fallbacks = 0
        self._instr_mask_rows = 0
        if self.instruction_dropout_rate <= 0.0:
            return
        if not 0.0 < self.instruction_dropout_rate < 1.0:
            raise ValueError("framework.instruction_dropout_rate must be in [0, 1)")
        if getattr(self, "state_token_source", None) != "rare_real":
            raise ValueError("instruction_dropout_rate needs state_token_source rare_real (for the mask id)")
        n_state = len(self._state_bin_ids) + len(self._state_group_ids)
        # well past the state ids, the fusion register ids (n_state .. n_state + 8) and the state mask id (+64)
        self._instr_mask_id = int(list(self._rare_token_ids(n_state + 66))[-1])
        if getattr(self, "state_dropout_mask", False) and self._instr_mask_id == getattr(self, "_state_mask_id", None):
            raise RuntimeError("instruction mask id collides with the state mask id")
        self._install_trainable_bin_embedding([self._instr_mask_id], name="instr_mask_embed")
        self._instr_mask_logged = False
        logger.info("instruction dropout ON: rate=%.3f mask_id=%d", self.instruction_dropout_rate, self._instr_mask_id)

    @staticmethod
    def _find_subseq(row: List[int], sub: List[int]) -> int:
        n = len(sub)
        if n == 0:
            return -1
        first = sub[0]
        for i in range(len(row) - n + 1):
            if row[i] == first and row[i:i + n] == sub:
                return i
        return -1

    def _mask_instruction_tokens(self, qwen_inputs: dict, instructions: List[str]) -> None:
        """Training only: overwrite the instruction tokens of the drawn rows with the mask id, in place."""
        self._instr_mask = None
        if getattr(self, "instruction_dropout_rate", 0.0) <= 0.0 or not self.training:
            return
        pend, self._instr_draw_pending = getattr(self, "_instr_draw_pending", None), None
        if getattr(self, "state_follows_instruction", False):
            if pend is None or len(pend) != len(instructions):
                raise RuntimeError("state_follows_instruction: no instruction draw from the state conditioning")
            draw = pend  # the rows whose state was masked
        else:
            draw = np.random.random(len(instructions)) < self.instruction_dropout_rate
        if not draw.any():
            return
        tok = self._resolve_tokenizer()
        ids = qwen_inputs["input_ids"]
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row in np.nonzero(draw)[0].tolist():
            row_ids = ids[row].tolist()
            text = str(instructions[row]).strip()
            start, sub = -1, []
            for variant in (" " + text, text):
                sub = tok.encode(variant, add_special_tokens=False)
                start = self._find_subseq(row_ids, sub)
                if start >= 0:
                    break
            self._instr_mask_rows += 1
            if start < 0:
                self._instr_mask_fallbacks += 1
                continue
            mask[row, start:start + len(sub)] = True
        if not bool(mask.any()):
            return
        self._instr_mask_orig = ids[mask].clone()
        ids[mask] = int(self._instr_mask_id)
        self._instr_mask = mask
        if not self._instr_mask_logged:
            self._instr_mask_logged = True
            print(f"__INSTR_MASK__ rows={int(mask.any(1).sum())} tokens={int(mask.sum())} "
                  f"fallbacks={self._instr_mask_fallbacks}/{self._instr_mask_rows}", flush=True)

    def _instr_mask_metrics(self) -> dict:
        if getattr(self, "instruction_dropout_rate", 0.0) <= 0.0 or not self._instr_mask_rows:
            return {}
        return {"structured_aux/instr_mask_fallback": torch.tensor(
            float(self._instr_mask_fallbacks) / float(self._instr_mask_rows))}

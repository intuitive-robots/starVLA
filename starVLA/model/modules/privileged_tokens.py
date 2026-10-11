"""Privileged geometric supervision fed as INPUT tokens during an image-free stage.

The LIT pattern (arXiv:2609.12641, "Breaking the Vision-Action Shortcut"). LIT trains the
action expert without images, conditioned on language + state + the action chunk's terminal
SE(3) pose, then in stage 2 drops that conditioning and instead RECONSTRUCTS the same pose
from vision-derived latents. At inference both the stage-1 encoder and the stage-2 decoder
are omitted: "The policy requires only visual observations, language, and robot state."

Why this module exists: our own image-free curriculum collapsed. The blind arm and both
annealed arms scored 35.7 / 39.0 / 36.3 on LIBERO-Plus against a 77.7 control, and on Robot
Initial States they scored 5-7% against the control's 61% -- i.e. indistinguishable from a
model whose images are switched off permanently. Predicting an action chunk from language
and state alone is underdetermined, so the model learns the degenerate marginal and the
gradient through the vision path never recovers. LIT's stage 1 is not underdetermined,
because the goal pose largely determines the chunk.

We have richer signals than LIT: the CoT annotations already parsed by
``cot_resolver.extract_structured_cot_targets`` and already used as shared-z REGRESSION
TARGETS. This module inverts them into INPUTS for stage 1. Stage 2 needs no new code --
shared_z already reconstructs exactly these targets, which is LIT's pose decoder.

Quantization, measured over 40,000 annotations (see test_privileged_tokens.py):

    target             dim   native   range     error          notes
    trajectory3d        15   cm/20    [-1, 1]   0.060 cm       100% exact -- integer cm grid
    trajectory2d        10   px/1000  [ 0, 1]   1.13 px
    target_point         2   px/1000  [ 0, 1]   1.16 px        5.1% missing
    object_box           4   px/1000  [ 0, 1]   1.71 px       14.1% missing
    ground_visibility    2   flag     [ 0, 1]   binary
    phase                1   class                             4-way categorical

Ranges are taken from the ANNOTATION FORMAT, not from data percentiles. That matters:
trajectory3d is integer centimetres divided by 20, so it is already discrete (~41 distinct
values). Re-ranging it to observed percentiles misaligns the bin centres with that grid and
quadruples RMSE (0.0030 -> 0.0121). A fixed semantic range keeps it lossless.

Each value maps into the SAME 256-bin codebook after a per-target affine map to [0, 1],
then to reserved embedding rows exactly as in ``StateTokenMixin`` (id = vocab_size-1-bin).
Position in the sequence, not the id, identifies which target a token belongs to. Missing
targets emit a dedicated sentinel id so "absent" is never confusable with a real bin --
zero is a valid coordinate.
"""

from typing import List, Optional

import numpy as np

# (dim, lo, hi) per target. lo/hi come from the annotation format; see module docstring.
# (dim, lo, hi, bins). Bin counts are sized to MEASURED occupancy over 40k annotations, not
# set uniformly: trajectory3d is integer centimetres / 20, so it holds ~41 distinct values
# and occupied only 38/256 bins (3.22 bits) -- 64 bins is lossless for it and frees rows.
# The image-coordinate targets are px/1000 and genuinely used 169-223 of 256.
PRIVILEGED_SPEC = {
    "target_point": (2, 0.0, 1.0, 256),
    "object_box": (4, 0.0, 1.0, 256),
    "trajectory2d": (10, 0.0, 1.0, 256),
    "trajectory3d": (15, -1.0, 1.0, 64),
    "ground_visibility": (2, 0.0, 1.0, 2),
    "phase": (1, 0.0, 3.0, 4),
}

PRIVILEGED_PLACEHOLDER = "<|repo_name|>"


class PrivilegedTokenMixin:
    """Stage-1 privileged geometry, delivered as reserved-vocabulary input tokens."""

    PRIVILEGED_PLACEHOLDER = PRIVILEGED_PLACEHOLDER

    def _configure_privileged_tokens(self, config) -> None:
        cfg = dict(config.framework.get("privileged_tokens", {}) or {})
        self.privileged_enabled = bool(cfg.get("enabled", False))
        self._pending_privileged_ids: Optional[List[Optional[List[int]]]] = None
        self._privileged_splice_logged = False
        if not self.privileged_enabled:
            self.privileged_targets: List[str] = []
            return

        targets = list(cfg.get("targets") or ["trajectory3d", "object_box", "target_point"])
        unknown = [t for t in targets if t not in PRIVILEGED_SPEC]
        if unknown:
            raise ValueError(
                f"framework.privileged_tokens.targets has unknown entries {unknown}; "
                f"known: {sorted(PRIVILEGED_SPEC)}"
            )
        self.privileged_targets = targets
        self.privileged_num_bins = int(cfg.get("num_bins", 256))
        self.privileged_width = sum(PRIVILEGED_SPEC[t][0] for t in targets)

        tokenizer = self._resolve_tokenizer()
        ids = tokenizer.convert_tokens_to_ids([self.PRIVILEGED_PLACEHOLDER])
        unk = getattr(tokenizer, "unk_token_id", None)
        if not ids or ids[0] is None or (unk is not None and ids[0] == unk):
            raise RuntimeError(
                f"privileged placeholder {self.PRIVILEGED_PLACEHOLDER!r} is not a single "
                "token in this tokenizer"
            )
        self._privileged_placeholder_id = int(ids[0])
        if self._privileged_placeholder_id == getattr(self, "_state_placeholder_id", None):
            raise ValueError(
                "privileged and state placeholders must differ, otherwise the two splices "
                "cannot tell their slots apart"
            )

        # Each target gets its OWN contiguous block of Latin-free real-token rows, so a bin
        # index is never ambiguous across targets, plus one shared "absent" sentinel.
        # Rare-real rows rather than the dead reserved window: those hold 7 distinct vectors
        # in 257 rows (pairwise distance 0.0008), so with a frozen embedding table every bin
        # is the same input and the signal is exactly zero -- the failure that silently
        # invalidated jobs 1898597/1899196.
        need = sum(PRIVILEGED_SPEC[t][3] for t in targets) + 1
        pool = list(self._rare_token_ids(need))
        self._privileged_missing_id = pool[0]
        self._privileged_bin_ids = {}
        off = 1
        for t in targets:
            nb = PRIVILEGED_SPEC[t][3]
            self._privileged_bin_ids[t] = pool[off:off + nb]
            off += nb
        if bool(cfg.get("trainable_bin_embedding", True)):
            # Includes the sentinel: "target absent" must be learnable too, otherwise a
            # missing object_box (14% of frames) is pinned to an arbitrary Cyrillic vector.
            self._install_trainable_bin_embedding(pool, name="privileged_bin_embed")

    def _privileged_ids_for(self, targets: Optional[dict]) -> List[int]:
        """Quantise one sample's structured targets into reserved ids, fixed width."""
        out: List[int] = []
        targets = targets or {}
        for name in self.privileged_targets:
            dim, lo, hi, nbins = PRIVILEGED_SPEC[name]
            table = self._privileged_bin_ids[name]
            raw = targets.get(name)
            if raw is None:
                out.extend([self._privileged_missing_id] * dim)
                continue
            vec = np.atleast_1d(np.asarray(raw, dtype=np.float64)).ravel()
            if vec.size != dim:
                # Malformed annotation: treat as absent rather than silently mis-binning.
                out.extend([self._privileged_missing_id] * dim)
                continue
            unit = (np.clip(vec, lo, hi) - lo) / (hi - lo)
            bins = np.clip((unit * nbins).astype(np.int64), 0, nbins - 1)
            out.extend(int(table[b]) for b in bins)
        return out

    def _privileged_slots(self) -> str:
        return " ".join([self.PRIVILEGED_PLACEHOLDER] * self.privileged_width)

    def _add_privileged_conditioning(
        self, instructions: List[str], structured_targets: Optional[List[Optional[dict]]]
    ) -> List[str]:
        """Append privileged geometry slots to each instruction (stage 1 only)."""
        if not self.privileged_enabled:
            self._pending_privileged_ids = None
            return instructions
        if structured_targets is None:
            structured_targets = [None] * len(instructions)
        conditioned, pending = [], []
        for instr, tgt in zip(instructions, structured_targets):
            conditioned.append(f"{instr} [TRACE] {self._privileged_slots()} [/TRACE]")
            pending.append(self._privileged_ids_for(tgt))
        self._pending_privileged_ids = pending
        return conditioned

    def _splice_privileged_tokens(self, qwen_inputs: dict) -> None:
        pending = self._pending_privileged_ids
        self._pending_privileged_ids = None
        if not self.privileged_enabled or pending is None:
            return
        self._splice_reserved(
            qwen_inputs, pending, self._privileged_placeholder_id,
            tag="PRIVILEGED_SPLICE", log_attr="_privileged_splice_logged",
        )

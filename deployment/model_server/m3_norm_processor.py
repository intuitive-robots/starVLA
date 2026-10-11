"""State / action normalisation for checkpoints trained through the m3_data bridge.

Checkpoints trained with ``datasets.vla_data.dataset_py: m3_data_datasets`` (ABC-130K and other m3
embodiments) have no entry in ``DATASET_NAMED_MIXTURES`` / ``ROBOT_TYPE_CONFIG_MAP``: their
``data_mix`` is only a label, and the bridge normalises in its own workers
(``starVLA/dataloader/m3_data_datasets.py::_Q99Normalizer``).  ``PolicyNormProcessor`` therefore
cannot serve them.  This processor reproduces the bridge exactly from ``dataset_statistics.json``
(one top-level key per m3 embodiment, written by ``StarVLAM3DataReader.save_dataset_statistics``):

    normalised = clip(2 * (x - q01) / (q99 - q01) - 1, -1, 1)      on dims with q99 > q01
    raw        = (clip(a, -1, 1) + 1) / 2 * (q99 - q01) + q01       inverse, same dims

Dims with q99 <= q01 pass through, as in training.  The wire format is the m3 field order recorded
in the statistics (``m3.action_fields`` / ``m3.state_fields``; ABC: [L arm 6, L grip, R arm 6,
R grip]).  Used only when the checkpoint's training config names the m3 bridge; every other
checkpoint keeps ``PolicyNormProcessor``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from starVLA.model.framework.share_tools import read_mode_config

logger = logging.getLogger(__name__)


def is_m3_checkpoint(contract_cfg: Dict[str, Any]) -> bool:
    """True when the training config used the m3_data bridge."""
    vla = ((contract_cfg or {}).get("datasets") or {}).get("vla_data") or {}
    return str(vla.get("dataset_py", "")).strip() == "m3_data_datasets"


def _field_names(fields: List[Any], prefix: str) -> List[str]:
    names = []
    for group, start, end in fields:
        for i in range(int(start), int(end)):
            names.append(f"{prefix}.{group}[{i}]")
    return names


class M3NormProcessor:
    """Same public surface as ``PolicyNormProcessor`` (the parts ``PolicyServerWrapper`` uses)."""

    def __init__(self, ckpt_path: str, unnorm_key: Optional[str] = None) -> None:
        self._ckpt_path = str(ckpt_path)
        _, norm_stats = read_mode_config(self._ckpt_path)
        self._norm_stats = norm_stats
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    f"m3 checkpoint has several embodiments {list(norm_stats)}; pass unnorm_key explicitly"
                )
            unnorm_key = next(iter(norm_stats))
        if unnorm_key not in norm_stats:
            raise KeyError(f"unnorm_key={unnorm_key!r} not in {list(norm_stats)}")
        self._unnorm_key = unnorm_key
        entry = norm_stats[unnorm_key]
        m3 = entry.get("m3") or {}
        self._m3 = m3
        self._data_config = None  # no gr00t DataConfig (keeps serve_image_indices lookups a no-op)

        def _vec(kind: str, stat: str) -> np.ndarray:
            return np.asarray(entry[kind][stat], dtype=np.float32)

        self._a_q01, self._a_q99 = _vec("action", "q01"), _vec("action", "q99")
        self._s_q01, self._s_q99 = _vec("state", "q01"), _vec("state", "q99")
        self._a_live = self._a_q99 > self._a_q01
        self._s_live = self._s_q99 > self._s_q01
        self._action_keys = _field_names(m3.get("action_fields") or [], "action") or [
            f"action[{i}]" for i in range(len(self._a_q01))
        ]
        self._state_keys = _field_names(m3.get("state_fields") or [], "state") or [
            f"state[{i}]" for i in range(len(self._s_q01))
        ]
        logger.info(
            "M3NormProcessor ready: unnorm_key=%s, action_dim=%d, state_dim=%d, cameras=%s, letterbox=%s",
            unnorm_key, len(self._a_q01), len(self._s_q01), m3.get("cameras"), m3.get("letterbox"),
        )

    # -- PolicyNormProcessor surface --------------------------------------------------------
    @property
    def action_keys(self) -> List[str]:
        return list(self._action_keys)

    @property
    def state_keys(self) -> List[str]:
        return list(self._state_keys)

    @property
    def unnorm_key(self) -> str:
        return self._unnorm_key

    @property
    def available_unnorm_keys(self) -> List[str]:
        return list(self._norm_stats.keys())

    @property
    def m3_contract(self) -> Dict[str, Any]:
        """What an eval client needs to preprocess exactly as the bridge did."""
        return {
            "cameras": list(self._m3.get("cameras") or []),
            "letterbox": bool(self._m3.get("letterbox", False)),
            "native_hw": self._m3.get("native_hw"),
            "action_fields": self._m3.get("action_fields"),
            "state_fields": self._m3.get("state_fields"),
            # bridge m3_instruction_map (training prompts of this embodiment); None when the bridge did not rewrite
            "instruction_map": self._m3.get("instruction_map"),
        }

    def unapply_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        a = np.asarray(normalized_actions, dtype=np.float32)
        if a.ndim != 2 or a.shape[-1] != len(self._a_q01):
            raise ValueError(f"expected (T, {len(self._a_q01)}) normalised actions, got {a.shape}")
        out = a.copy()
        live = self._a_live
        out[:, live] = (np.clip(a[:, live], -1.0, 1.0) + 1.0) / 2.0 * (
            self._a_q99[live] - self._a_q01[live]
        ) + self._a_q01[live]
        return out

    def apply_state(self, raw_state: np.ndarray) -> np.ndarray:
        s = np.asarray(raw_state, dtype=np.float32)
        if s.ndim == 1:
            s = s[None, :]
        if s.ndim != 2 or s.shape[-1] != len(self._s_q01):
            raise ValueError(f"expected state (D,) or (T, D) with D={len(self._s_q01)}, got {s.shape}")
        out = s.copy()
        live = self._s_live
        out[:, live] = np.clip(
            2.0 * (s[:, live] - self._s_q01[live]) / (self._s_q99[live] - self._s_q01[live]) - 1.0, -1.0, 1.0
        )
        return out

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


_STATE_CORRUPT_MODES = {"shuffle", "zero", "random"}


def _corrupt_normalized_state(
    state: np.ndarray,
    *,
    mode: str,
    seed: int,
    key: str,
) -> np.ndarray:
    """Apply a deterministic diagnostic corruption after the training state transform.

    New configs can opt in to q99 normalization. Historical LIBERO configs have an
    identity state transform and are clipped by the token quantizer; shuffle commutes
    with that elementwise clipping, so it still permutes the exact token-bin values the
    historical model consumes.
    """
    mode = str(mode).strip().lower()
    if mode not in _STATE_CORRUPT_MODES:
        raise ValueError(
            f"STARVLA_STATE_CORRUPT={mode!r}; expected one of "
            f"{sorted(_STATE_CORRUPT_MODES)}"
        )
    out = np.array(state, dtype=np.float32, copy=True)
    if mode == "zero":
        out.fill(0.0)
        return out

    digest = hashlib.blake2b(
        f"{int(seed)}|{key}".encode("utf-8"), digest_size=8
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    if mode == "random":
        return rng.uniform(-1.0, 1.0, size=out.shape).astype(np.float32)
    for index in np.ndindex(out.shape[:-1]):
        out[index] = rng.permutation(out[index])
    return out


def _training_obs_image_size(model_cfg: Dict[str, Any]) -> Optional[List[int]]:
    """Return the image size that was actually applied during training.

    ``datasets.vla_data.image_size`` is deliberately excluded: historical
    loaders accepted that field but did not apply it, so treating it as an
    observation contract would break evaluation of existing checkpoints.
    """
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    size = vla_data_cfg.get("pack_image_size") or vla_data_cfg.get("obs_image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    return [int(size[0]), int(size[1])]


def _training_action_mode(model_cfg: Dict[str, Any]) -> str:
    """Return the action representation encoded in the checkpoint config."""
    mode = str(
        ((model_cfg.get("datasets") or {}).get("vla_data") or {}).get(
            "action_mode", "abs"
        )
    ).strip().lower()
    aliases = {"absolute": "abs", "relative": "rel"}
    mode = aliases.get(mode, mode)
    if mode not in {"abs", "delta", "rel"}:
        raise ValueError(f"Unsupported training action_mode in checkpoint: {mode!r}")
    return mode


def _load_training_contract_config(
    ckpt_path: str, compact_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Load the unpruned launch config used to define the data contract.

    Framework checkpoints intentionally save a compact ``config.yaml`` containing
    model-construction fields. Dataset execution fields added by the launcher (notably
    ``action_mode`` / ``action_mode_apply_keys``) can be absent there even though they
    governed training and remain in ``config.full.yaml`` at the run root. Serving must
    advertise the latter contract; silently defaulting an omitted field to ``abs`` can
    invalidate an otherwise correctly trained relative-action policy.
    """
    checkpoint = Path(ckpt_path).resolve()
    candidates = [checkpoint.parent.parent / "config.full.yaml"]
    if checkpoint.is_dir():
        candidates.insert(0, checkpoint / "config.full.yaml")
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            full_config = yaml.safe_load(handle) or {}
        if not isinstance(full_config, dict):
            raise ValueError(f"training contract config is not a mapping: {path}")
        logging.info("PolicyServerWrapper: training contract loaded from %s", path)
        return full_config
    logging.warning(
        "PolicyServerWrapper: config.full.yaml not found beside %s; metadata falls "
        "back to compact checkpoint config",
        ckpt_path,
    )
    return compact_config


def _state_token_contract(model_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the checkpoint's exact token-state layout for client handshakes.

    The state array sent over the wire is in the DataConfig's concatenation order.
    ``state_token_permutation`` is applied only after training-time normalization,
    immediately before quantization/group-marker insertion in ``StateTokenMixin``.
    Advertising all of those fields lets embodiment adapters fail closed instead of
    silently swapping arms or omitting the left/right marker tokens.
    """
    framework_cfg = model_cfg.get("framework") or {}
    if framework_cfg.get("state_token_mode") not in {"reserved_ids", "text"}:
        return None

    permutation = framework_cfg.get("state_token_permutation") or []
    group_sizes = framework_cfg.get("state_token_group_sizes") or []
    group_names = framework_cfg.get("state_token_group_names") or []
    return {
        "mode": str(framework_cfg.get("state_token_mode")),
        "routing": str(framework_cfg.get("state_routing", "")),
        "num_bins": int(framework_cfg.get("state_num_bins", 0) or 0),
        "token_source": str(framework_cfg.get("state_token_source", "")),
        "permutation": [int(value) for value in permutation],
        "group_sizes": [int(value) for value in group_sizes],
        "group_names": [str(value) for value in group_names],
        "num_value_tokens": len(permutation) if permutation else None,
        "num_tokens_with_markers": (
            len(permutation) + len(group_sizes) if permutation else None
        ),
    }


def _expects_state(model_cfg, framework=None) -> bool:
    """True when the checkpoint actually CONSUMES proprioceptive state.

    State can enter by either of two independent paths:

    * early, as tokens in the VLM (``include_state`` + ``state_token_mode``); or
    * late, through ``action_model.state_encoder`` (``state_dim > 0``).

    The modality-z arms intentionally use only the first path and therefore set
    ``state_dim: 0``.  Do not use that field as an early return: doing so makes the
    evaluator withhold the eight token-ID state values and the strict modality-z
    interface then has no state span to pool.

    Conversely, ``framework.action_model.state_dim > 0`` alone is NOT sufficient. Several
    QwenOFT configs carry a stale ``state_dim: 7`` default with no state pathway
    behind it -- ``state_token_mode: null`` and no ``state_encoder`` weights --
    e.g. ``libero_qwen08b_base``. Trusting the number alone makes the server
    advertise ``expects_state`` for such a checkpoint, and a client that honours
    the handshake then appends a [STATE] suffix the model never saw in training.

    So accept an explicit training-time token pathway first. For the late pathway,
    require real state weights (``action_model.state_encoder.*`` /
    ``state_bin_embed``), not just the dimension.
    """
    try:
        framework_cfg = model_cfg.get("framework") or {}
        vla_data_cfg = (model_cfg.get("datasets") or {}).get("vla_data") or {}
        token_mode = framework_cfg.get("state_token_mode")
        if bool(vla_data_cfg.get("include_state", False)) and token_mode in {
            "reserved_ids",
            "text",
        }:
            return True
    except Exception:
        pass

    try:
        action_model = model_cfg["framework"]["action_model"]
        dim = int(action_model.get("state_dim", 0) or 0)
    except Exception:
        return False
    if dim <= 0:
        return False

    if framework is not None:
        try:
            return any(
                ("state_encoder" in k) or ("state_bin_embed" in k)
                for k in framework.state_dict().keys()
            )
        except Exception:
            pass

    # With no model to inspect, the bare dimension is not trustworthy.
    return False


# LIBERO-Goal: all 10 tasks share one scene. Each instruction maps to the next task's (cyclic, sorted),
# used by STARVLA_PERMUTE_GOAL_INSTRUCTION for wrong-instruction rollouts.
_GOAL_TASKS = ['open the middle drawer of the cabinet', 'open the top drawer and put the bowl inside', 'push the plate to the front of the stove', 'put the bowl on the plate', 'put the bowl on the stove', 'put the bowl on top of the cabinet', 'put the cream cheese in the bowl', 'put the wine bottle on the rack', 'put the wine bottle on top of the cabinet', 'turn on the stove']
_GOAL_NEXT = {t: _GOAL_TASKS[(i + 1) % len(_GOAL_TASKS)] for i, t in enumerate(_GOAL_TASKS)}
_GOAL_PREV = {t: _GOAL_TASKS[(i - 1) % len(_GOAL_TASKS)] for i, t in enumerate(_GOAL_TASKS)}  # inverse of _GOAL_NEXT

class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        is_inference: bool = False,
        cot_max_new_tokens: Optional[int] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(
            self._ckpt_path,
            is_inference=is_inference,
            attn_implementation=attn_implementation,
        )
        if cot_max_new_tokens is not None:
            if cot_max_new_tokens <= 0:
                raise ValueError(
                    f"cot_max_new_tokens must be positive, got {cot_max_new_tokens}"
                )
            framework_cfg = getattr(framework.config, "framework", None)
            cot_cfg = getattr(framework_cfg, "cot", None)
            if cot_cfg is None:
                logging.info(
                    "Ignoring cot_max_new_tokens=%d because this checkpoint has no CoT config",
                    cot_max_new_tokens,
                )
            else:
                if hasattr(cot_cfg, "__setitem__"):
                    cot_cfg["max_new_tokens"] = int(cot_max_new_tokens)
                else:
                    setattr(cot_cfg, "max_new_tokens", int(cot_max_new_tokens))
                logging.info(
                    "Inference override: framework.cot.max_new_tokens=%d",
                    cot_max_new_tokens,
                )
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Test-time attention-path knockout (STARVLA_ATTN_KNOCKOUT) or dense-probe parity
        # run (STARVLA_ATTN_PROBE=1). Off unless requested; see attn_probe.py.
        from starVLA.model.modules.attn_probe import attach_from_env

        self._attn_probe = attach_from_env(framework)
        self._attn_knockout = os.environ.get("STARVLA_ATTN_KNOCKOUT", "").strip()
        self._obs_dump_counts: Dict[str, int] = {}

        # E4/E5 aim bias on instruction -> image attention (STARVLA_OBJ_BIAS). Off unless
        # requested; needs the eval client's per-query ``_obj_tok`` masks (online_obj_mask.py).
        self._obj_bias = None
        if os.environ.get("STARVLA_OBJ_BIAS", "").strip():
            from starVLA.model.modules.attn_probe import AttnProbe, parse_obj_bias

            self._obj_bias = parse_obj_bias(os.environ["STARVLA_OBJ_BIAS"])
            if self._attn_probe is None:
                self._attn_probe = AttnProbe(framework, knockout=None, patch_attention=True)
            self._obj_bias_calls = 0
            self._obj_bias_empty = [0, 0]
            logging.warning("OBJ BIAS ACTIVE %s (probe topology=%s, knockout=%r)", self._obj_bias,
                            self._attn_probe.topology, self._attn_probe.knockout_spec)

        # Closed-loop image-token transplant (STARVLA_TRANSPLANT_IMAGE_FROM). Off unless
        # requested; see image_transplant.py.
        self._transplant = None
        if os.environ.get("STARVLA_TRANSPLANT_IMAGE_FROM", "").strip():
            from deployment.model_server.image_transplant import attach_transplant_from_env

            self._transplant = attach_transplant_from_env(framework, _GOAL_NEXT, probe=self._attn_probe)

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        self._contract_cfg = _load_training_contract_config(self._ckpt_path, model_cfg)

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]
        
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _dump_observations(self, examples: List[dict], prepared: List[dict]) -> None:
        """Save the exact framework input for analysis frame banks (STARVLA_OBS_DUMP_DIR).

        Keeps every ``STARVLA_OBS_DUMP_EVERY``-th query (default 6) of each episode, at
        most ``STARVLA_OBS_DUMP_MAX`` (default 5) per episode. The episode is identified by
        the client's ``_state_corrupt_key`` (``suite|task_id|episode_idx|step``); requests
        without one are skipped. ``state`` is the normalized state the framework consumes.
        """
        root = os.environ["STARVLA_OBS_DUMP_DIR"]
        every = max(1, int(os.environ.get("STARVLA_OBS_DUMP_EVERY", "6")))
        limit = int(os.environ.get("STARVLA_OBS_DUMP_MAX", "5"))
        os.makedirs(root, exist_ok=True)
        for raw, prep in zip(examples, prepared):
            key = str(raw.get("_state_corrupt_key", ""))
            parts = key.split("|")
            if len(parts) != 4:
                continue
            suite, task_id, episode_idx, step = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            episode = f"{suite}|{task_id}|{episode_idx}"
            query = self._obs_dump_counts.get(episode, 0)
            self._obs_dump_counts[episode] = query + 1
            if query % every or query // every >= limit:
                continue
            stem = f"{suite}_t{task_id:04d}_e{episode_idx:02d}_q{query:03d}_s{step:04d}"
            path = os.path.join(root, stem + ".npz")
            tmp = os.path.join(root, f".{stem}.{os.getpid()}.tmp.npz")
            payload = {
                f"image_{i}": np.asarray(img, dtype=np.uint8) for i, img in enumerate(raw["image"])
            }
            payload.update({
                "lang": np.asarray(str(raw.get("lang", ""))),
                "key": np.asarray(key),
                "suite": np.asarray(suite),
                "task_id": np.asarray(task_id),
                "episode_idx": np.asarray(episode_idx),
                "step": np.asarray(step),
                "query": np.asarray(query),
                "ckpt": np.asarray(self._ckpt_path),
            })
            if "state" in prep:
                payload["state"] = np.asarray(prep["state"], dtype=np.float32)
            if "state" in raw:
                payload["raw_state"] = np.asarray(raw["state"], dtype=np.float32)
            np.savez(tmp, **payload)
            os.replace(tmp, path)

    def _apply_obj_bias(self, examples: List[dict], prepared: List[dict]) -> List[dict]:
        """Per-row regions for STARVLA_OBJ_BIAS from each request's ``_obj_tok`` [2, 8, 8]."""
        from starVLA.model.modules import attn_probe as ap
        from starVLA.model.modules.token_regions import random_region, token_sets

        cfg = self._obj_bias
        rows, out = [], []
        for i, (raw, prep) in enumerate(zip(examples, prepared)):
            if "_obj_tok" not in raw:
                raise ValueError(
                    "STARVLA_OBJ_BIAS is set but the request has no _obj_tok; run the client with "
                    "STARVLA_ONLINE_OBJ_MASK=1 (APPTAINERENV_STARVLA_ONLINE_OBJ_MASK=1)")
            tok = np.asarray(raw["_obj_tok"], dtype=np.float32).reshape(2, 8, 8)
            rng = None
            if cfg["region"] == "random":
                # One seed per episode (suite|task|episode), so the control region's offset is stable
                # while the object's shape is; identical across arms.
                key = "|".join(str(raw.get("_state_corrupt_key", "")).split("|")[:3])
                h = hashlib.blake2b(f"{cfg['seed']}|{key}".encode(), digest_size=8).digest()
                rng = np.random.default_rng(int.from_bytes(h, "little"))
            row = {}
            for c, gid in ((0, ap.CAM0), (1, ap.CAM1)):
                obj, non = token_sets(tok[c], tau=cfg["tau"])
                if not len(obj):
                    self._obj_bias_empty[c] += 1
                    continue
                row[gid] = {"object": obj, "background": non}.get(cfg["region"])
                if cfg["region"] == "random":
                    row[gid] = random_region(obj, rng)
            rows.append(row)
            ex = dict(prep)
            ex.pop("_obj_tok", None)
            out.append(ex)
        self._attn_probe.key_bias = {"b": cfg["b"], "layers": cfg["layers"], "rows": rows}
        self._obj_bias_calls += len(rows)
        if self._obj_bias_calls <= len(rows) or self._obj_bias_calls % 2000 < len(rows):
            logging.warning("OBJ BIAS rows=%d total=%d empty_masks(ext,wrist)=%s row0_tokens=%s",
                            len(rows), self._obj_bias_calls, self._obj_bias_empty,
                            {k: len(v) for k, v in rows[0].items()})
        return out

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            if self._is_m3:
                # m3_data-bridge checkpoints (ABC-130K): q01/q99 per embodiment exactly as the
                # bridge normalised; no gr00t DataConfig exists for their data_mix label.
                from deployment.model_server.m3_norm_processor import M3NormProcessor

                self._norm_processors[cache_key] = M3NormProcessor(
                    self._ckpt_path, unnorm_key=unnorm_key
                )
            else:
                self._norm_processors[cache_key] = PolicyNormProcessor(
                    self._ckpt_path, unnorm_key=unnorm_key
                )
        return self._norm_processors[cache_key]

    @property
    def _is_m3(self) -> bool:
        from deployment.model_server.m3_norm_processor import is_m3_checkpoint

        return is_m3_checkpoint(getattr(self, "_contract_cfg", None) or {})

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "image_preprocessing_contract_version": 2,
            "training_obs_image_size": _training_obs_image_size(self._contract_cfg),
            # Environment adapters must invert the same action representation used for
            # training.  Advertising it makes a rel-vs-abs mismatch fail closed instead
            # of silently driving the robot with joint deltas as absolute targets.
            "training_action_mode": _training_action_mode(self._contract_cfg),
            # Whether this checkpoint was TRAINED with proprioceptive state. A client that
            # sends state to a model trained without it appends a [STATE] suffix the model
            # has never seen; a client that withholds state from a model trained WITH it
            # removes a channel that was present in every training sample. Both are silent
            # at the API level and catastrophic at rollout, so the contract is advertised
            # here rather than left to each client to guess.
            "expects_state": _expects_state(self._contract_cfg, self._framework),
            "state_token_contract": _state_token_contract(self._contract_cfg),
            "attn_knockout": self._attn_knockout,
            "attn_probe": self._attn_probe is not None,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
            serve_idx = getattr(getattr(proc, "_data_config", None), "serve_image_indices", None)
            if serve_idx is not None:  # single-view checkpoints only; key absent otherwise
                base["serve_image_indices"] = list(serve_idx)
        if self._is_m3:
            # m3 checkpoints only: per-embodiment camera order / letterbox / field layout, so an
            # eval client (examples/ABC/eval_files) can preprocess exactly as the bridge did.
            base["m3_contract"] = {
                key: self._get_processor(key).m3_contract for key in self._available_unnorm_keys
            }
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        expects_state = _expects_state(self._contract_cfg, self._framework)
        state_corrupt_mode = os.environ.get("STARVLA_STATE_CORRUPT", "").strip().lower()
        if state_corrupt_mode and not expects_state:
            raise ValueError(
                "STARVLA_STATE_CORRUPT was requested for a checkpoint that does not "
                "consume state"
            )
        prepared_examples = examples
        if expects_state:
            prepared_examples = []
            for row, example in enumerate(examples):
                if "state" not in example:
                    raise ValueError(
                        f"checkpoint expects proprioceptive state, but example {row} has no state"
                    )
                prepared = dict(example)
                normalized_state = proc.apply_state(np.asarray(example["state"]))
                corrupt_key = str(prepared.pop("_state_corrupt_key", ""))
                if state_corrupt_mode:
                    if not corrupt_key:
                        raise ValueError(
                            "state corruption requires each request to provide the stable "
                            "_state_corrupt_key used for paired deterministic interventions"
                        )
                    clean_state = normalized_state.copy()
                    normalized_state = _corrupt_normalized_state(
                        normalized_state,
                        mode=state_corrupt_mode,
                        seed=int(os.environ.get("STARVLA_STATE_CORRUPT_SEED", "0")),
                        key=corrupt_key,
                    )
                    if not getattr(self, "_state_corruption_logged", False):
                        self._state_corruption_logged = True
                        logging.warning(
                            "STATE CORRUPTION ACTIVE mode=%s key=%s clean=%s corrupt=%s",
                            state_corrupt_mode,
                            corrupt_key,
                            np.asarray(clean_state).reshape(-1).tolist(),
                            np.asarray(normalized_state).reshape(-1).tolist(),
                        )
                # Framework examples conventionally carry a single observation step as
                # (1, state_dim), matching the training loader.
                prepared["state"] = normalized_state
                prepared_examples.append(prepared)

        # Single-view checkpoints (rebuttal H-cam): a DataConfig that declares
        # ``serve_image_indices`` (LIBERO ``data_mix: libero_mixed_primary`` -> [0]) was
        # trained on a subset of the cameras a client sends, so keep only those positions.
        # No other DataConfig has the attribute, so every other checkpoint is untouched.
        serve_idx = getattr(getattr(proc, "_data_config", None), "serve_image_indices", None)
        if serve_idx is not None:
            selected = []
            for row, example in enumerate(prepared_examples):
                images = list(example["image"])
                if max(serve_idx) >= len(images):
                    raise ValueError(
                        f"checkpoint serves image positions {list(serve_idx)} but example {row} "
                        f"has only {len(images)} image(s)"
                    )
                ex = dict(example)
                ex["image"] = [images[i] for i in serve_idx]
                selected.append(ex)
            if not getattr(self, "_serve_views_logged", False):
                self._serve_views_logged = True
                logging.warning("SERVE IMAGE VIEWS %s: client sent %d image(s), model gets %d",
                                list(serve_idx), len(prepared_examples[0]["image"]),
                                len(selected[0]["image"]))
            prepared_examples = selected

        # Eval-time camera removal (camdrop rebuttal; off by default). STARVLA_ZERO_IMAGE_INDICES
        # ="1" keeps every image position in the prompt but has the framework zero the
        # listed positions' pixel_values after the processor, exactly as training-time
        # framework.modality_dropout {mode: camera, removal: pixels} removes a camera
        # (image-token count, positions and prompt layout unchanged). Positions index the
        # list the model receives (after serve_image_indices). LIBERO: 0 = agent view,
        # 1 = wrist.
        zero_spec = os.environ.get("STARVLA_ZERO_IMAGE_INDICES", "").strip()
        if zero_spec:
            if not getattr(self._framework, "supports_zero_image_indices", False):
                raise ValueError(
                    "STARVLA_ZERO_IMAGE_INDICES was set but framework "
                    f"{type(self._framework).__name__} does not implement pixel zeroing"
                )
            zero_idx = sorted({int(x) for x in zero_spec.split(",") if x.strip()})
            n_img = len(prepared_examples[0]["image"])
            if not zero_idx or min(zero_idx) < 0 or max(zero_idx) >= n_img or len(zero_idx) >= n_img:
                raise ValueError(
                    f"STARVLA_ZERO_IMAGE_INDICES={zero_spec!r} invalid for {n_img} image(s)"
                )
            zeroed = []
            for example in prepared_examples:
                ex = dict(example)
                ex["_zero_image_indices"] = list(zero_idx)
                zeroed.append(ex)
            if not getattr(self, "_zero_views_logged", False):
                self._zero_views_logged = True
                logging.warning("ZERO IMAGE POSITIONS %s of %d (pixel_values zeroed, tokens kept)",
                                zero_idx, n_img)
            prepared_examples = zeroed

        # Instruction interventions (insight analyses; off by default). STARVLA_OVERRIDE_INSTRUCTION
        # replaces every instruction with a fixed string ("__EMPTY__" = ""); STARVLA_PERMUTE_GOAL_INSTRUCTION=1
        # replaces each LIBERO-Goal instruction with the next Goal task's (same scene, wrong task).
        override = os.environ.get("STARVLA_OVERRIDE_INSTRUCTION")
        permute_goal = os.environ.get("STARVLA_PERMUTE_GOAL_INSTRUCTION", "") == "1"
        # "prev" (transplant override condition): each Goal instruction -> the previous Goal task's.
        permute_goal_prev = os.environ.get("STARVLA_PERMUTE_GOAL_INSTRUCTION", "") == "prev"
        if override is not None or permute_goal or permute_goal_prev:
            rewritten = []
            for example in prepared_examples:
                ex = dict(example)
                lang = str(ex.get("lang", ""))
                if override is not None:
                    ex["lang"] = "" if override == "__EMPTY__" else override
                elif permute_goal:
                    key = " ".join(lang.lower().replace("_", " ").strip().rstrip(".").split())
                    if key in _GOAL_NEXT:
                        ex["lang"] = _GOAL_NEXT[key]
                elif permute_goal_prev:
                    key = " ".join(lang.lower().replace("_", " ").strip().rstrip(".").split())
                    if key in _GOAL_PREV:
                        ex["lang"] = _GOAL_PREV[key]
                rewritten.append(ex)
            if not getattr(self, "_instruction_intervention_logged", False):
                self._instruction_intervention_logged = True
                logging.warning("INSTRUCTION INTERVENTION override=%r permute_goal=%s: %r -> %r",
                                override, "prev" if permute_goal_prev else permute_goal,
                                str(prepared_examples[0].get("lang", "")),
                                str(rewritten[0].get("lang", "")))
            prepared_examples = rewritten

        if self._obj_bias is not None:
            prepared_examples = self._apply_obj_bias(examples, prepared_examples)
        if os.environ.get("STARVLA_OBS_DUMP_DIR"):
            self._dump_observations(examples, prepared_examples)
        if self._transplant is not None:
            out = self._transplant.predict(
                self._framework.predict_action, examples, prepared_examples, **kwargs
            )
        else:
            out = self._framework.predict_action(examples=prepared_examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        response = {"actions": unnorm}
        # Pass reasoning text through untouched when the framework produced it
        # (explicit-CoT models only), so clients can overlay it on rollout video.
        if "cot_text" in out:
            response["cot_text"] = out["cot_text"]
        return response

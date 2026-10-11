"""Future-latent auxiliary: predict the policy backbone's own view of a camera 8 steps ahead.

What it is
----------
A default-off training-only objective (``framework.future_latent``). It reuses the
shared-z temporal machinery (the dataloader's ``future_image`` at ``shared_z_future_offset``
and a no-grad encoding of that future frame with a cosine loss) WITHOUT the shared-z
bottleneck: nothing is added to the action head's input, the prompt is unchanged, and
``predict_action`` never runs it.

    online    = mean over camera c's image tokens of the output of backbone block L
                (this frame, the training forward, camera dropout applied, with gradient)
    pred      = MLP_c(online)                                    (LayerNorm-Linear-GELU-Linear)
    target    = mean over camera c's image tokens of the output of block L in a no-grad
                pass of the SAME backbone on the future frames t+offset of the same episode
                (both cameras, no dropout, the same instruction and state tokens)
    loss      = mean over valid (sample, camera) of 1 - cos(pred, target)

``target: delta`` (opt-in) replaces the target by target - online.detach(): the pooled
block-L latent barely moves over the offset (copying the current frame already scores
cos 0.997 on LIBERO), so the level target is solved by an identity map and exerts no
pressure; the delta target makes the head predict the change itself (copy scores 0).

``pool: token`` (opt-in, requires ``target: delta``) drops the pooling: the MLP maps every
online image token of camera c to that token's own change at t+offset,
delta_i = h_target_i - h_online_i.detach(), and the loss is the per-channel normalised MSE
    mean_{i,d} ((pred_i,d - delta_i,d) / rms_d)^2,  rms_d = stop-grad RMS of delta_:,d
over the batch's valid tokens, so a few large-norm channels cannot dominate and predicting
zero scores exactly 1 (logged R2 = 1 - loss). Motion is local (arm, object patches), so a
pooled delta averages it away; the token target keeps where the change happens.

Block L uses the analysis convention (0-based block index, forward hook on the block, as
in interchange_factorial.py / path_patching.py): for the converted encoder (P6) it is
encoder block L, for the causal backbone (P2) decoder-only block L. The target is taken
at the same layer L, so prediction and target live in the same space; L=16 sits between
L12 and L20, where the transplant and path-patching results place the task being written
into the image tokens.

A (sample, camera) pair is skipped when that camera was the one camera-dropout zeroed on
the sample (the online tokens carry no pixels of it), and a sample is skipped when the
dataloader has no future frame (the last ``offset`` frames of an episode; with
``datasets.vla_data.shared_z_future_keep_tail: true`` those frames stay in the sampler so
the arm trains on exactly the frames of its no-aux control).

The target pass stops at block L (a private exception raised from the block hook), so it
costs about (L+1)/n_layers of a forward, and it runs after the action loss so it cannot
overwrite any per-forward interface state the head reads.

Loss slot: ``structured_aux_loss`` (weighted by ``trainer.loss_scale.structured_aux``,
which accepts the {start, end, start_step, end_step} ramp). Mutually exclusive with
shared_z and the legacy structured_aux targets, which use the same slot.

Task-identity auxiliary (``framework.task_pred``), same hook and slot
------------------------------------------------------------------------
The external view's image tokens at block L, mean-pooled, go through a small MLP
classifier over the dataset's task identities (LIBERO: the 40 base tasks, the rows of
``meta/tasks.parquet``; a sample's label is the index of its instruction string there).
Cross-entropy, skipped for a sample whose external camera was dropped. For a causal
backbone the external image tokens precede the instruction and cannot attend to it, so
the task can only be inferred from the scene there. Mutually exclusive with future_latent.

    enabled: false
    layer: 16
    views: [external]
    tasks_file: <data_root_dir>/libero/meta/tasks.parquet
    num_tasks: 40
    weight: 1.0
    mlp_hidden: 1024

Config (``framework.future_latent``)::

    enabled: false
    layer: 16
    views: [external]          # or [external, wrist]; external = image 0, wrist = image 1
    offset: 8                  # must equal datasets.vla_data.shared_z_future_offset
    weight: 1.0                # inner weight; the ramp lives in loss_scale.structured_aux
    loss: cosine
    target: level              # or delta (predict target - online, see above)
    pool: mean                 # or token (per-token delta, normalised MSE; needs delta)
    mlp_hidden: 2048
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

_VIEW_INDEX = {"external": 0, "wrist": 1, "cam0": 0, "cam1": 1, "cam2": 2}  # camN = N-th image of the prompt (RoboCasa 3-cam: left, right, eye_in_hand)


class _StopAfterFutureLatentLayer(Exception):
    """Raised from the block hook to end the no-grad target pass at block L."""


class _AuxLayerHookMixin:
    """One forward hook on backbone block L, armed only around the passes that need it."""

    def _aux_blocks(self) -> List[nn.Module]:
        """The attention blocks the head reads through (same rule as analysis/common.py)."""
        iface = self.qwen_vl_interface
        if hasattr(iface, "_encoder_blocks") and hasattr(iface, "_text_model"):
            return list(iface._encoder_blocks(iface._text_model()))
        return list(iface.model.model.language_model.layers)

    def _aux_register_hook(self, layer: int) -> None:
        blocks = self._aux_blocks()
        if not 0 <= int(layer) < len(blocks):
            raise ValueError(f"aux layer {layer} outside the {len(blocks)} backbone blocks")
        if getattr(self, "_fl_hook_handle", None) is not None:
            if getattr(self, "_fl_hook_layer", None) == int(layer):
                return  # future_latent + task_pred on the same block share one hook
            raise RuntimeError("aux block hook registered twice at different layers")
        self._fl_hook_layer = int(layer)
        self._fl_capture = None

        def hook(module, inputs, output):
            capture = self._fl_capture
            if capture is None:
                return None
            if getattr(self, "_loop_inside", False):
                return None  # backbone_loop re-runs of this block: keep the first-pass output
            capture["hidden"] = output[0] if isinstance(output, tuple) else output
            if capture.get("stop", False):
                raise _StopAfterFutureLatentLayer()
            return None

        self._fl_hook_handle = blocks[int(layer)].register_forward_hook(hook)

    def _aux_active(self) -> bool:
        return bool(
            (getattr(self, "future_latent_enabled", False)
             or getattr(self, "task_pred_enabled", False)
             or getattr(self, "fusion_registers_enabled", False))
            and self.training and torch.is_grad_enabled()
        )

    def _future_latent_arm(self, stop: bool = False) -> None:
        self._fl_capture = {"hidden": None, "stop": bool(stop)}

    def _future_latent_take(self) -> Optional[torch.Tensor]:
        capture, self._fl_capture = getattr(self, "_fl_capture", None), None
        return None if capture is None else capture.get("hidden")

    def _aux_camera_masks(self, input_ids: torch.Tensor, views: Sequence[str]) -> Dict[str, torch.Tensor]:
        """[B, L] bool masks of each camera's image tokens.

        Camera k = the image tokens after the (k+1)-th vision_start token of the row, the
        attn_probe TokenLabeler rule (cam0 = first image = external view in LIBERO).
        """
        cfg = self.qwen_vl_interface.model.config
        image_token_id = int(getattr(cfg, "image_token_id", 151655))
        vision_start_id = int(cfg.vision_start_token_id)
        cam = torch.cumsum((input_ids == vision_start_id).long(), dim=1) - 1
        is_image = input_ids == image_token_id
        return {view: is_image & (cam == _VIEW_INDEX[view]) for view in views}

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask[:, : hidden.shape[1]].to(hidden.device)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _aux_check_online(online_hidden: Optional[torch.Tensor]) -> None:
        if online_hidden is None:
            raise RuntimeError("aux: block hook captured no hidden state")
        if not online_hidden.requires_grad:
            raise RuntimeError(
                "aux: the captured block output has no gradient (gradient checkpointing "
                "re-runs blocks under no_grad; this aux needs it off)"
            )


class FutureLatentMixin(_AuxLayerHookMixin):
    """Config, predictor MLPs, and the loss of the future-latent aux."""

    # ── config ───────────────────────────────────────────────────────────────
    def _configure_future_latent(self, config) -> None:
        cfg = dict(config.framework.get("future_latent", {}) or {})
        self.future_latent_enabled = bool(cfg.get("enabled", False))
        if not self.future_latent_enabled:
            return
        self.future_latent_layer = int(cfg.get("layer", 16))
        tp = config.framework.get("task_pred", {}) or {}
        if bool(tp.get("enabled", False)) and int(tp.get("layer", 16)) != self.future_latent_layer:
            raise ValueError("future_latent and task_pred together must use the same block layer "
                             "(one hook); their losses are summed in the structured_aux slot")
        views = cfg.get("views", ["external"])
        if isinstance(views, str):
            views = [views]
        self.future_latent_views = [str(v).lower() for v in views]
        unknown = [v for v in self.future_latent_views if v not in _VIEW_INDEX]
        if unknown or not self.future_latent_views:
            raise ValueError(
                f"framework.future_latent.views must be a non-empty subset of "
                f"{sorted(_VIEW_INDEX)}, got {self.future_latent_views}"
            )
        self.future_latent_offset = int(cfg.get("offset", 8))
        self.future_latent_weight = float(cfg.get("weight", 1.0))
        self.future_latent_loss = str(cfg.get("loss", "cosine")).lower()
        self.future_latent_mlp_hidden = int(cfg.get("mlp_hidden", 2048))
        self.future_latent_target = str(cfg.get("target", "level")).lower()
        if self.future_latent_target not in ("level", "delta", "dynaflip"):
            raise ValueError("framework.future_latent.target must be 'level', 'delta' or 'dynaflip'")
        # dynaflip: a FROZEN external target, the same for every backbone: per image token, the
        # change F(t+offset) - F(t) of DynaFLIP-base patch features (DINOv2-B, 16x16 patches at
        # 224 px) average-pooled to the camera's token grid. Token i of Qwen's merged grid covers
        # the same image region as pooled patch i (verified: analysis/dyn_probe.py alignment check).
        self.future_latent_dynaflip_dir = str(cfg.get("dynaflip_dir", "/e/project1/m3/blank4/weights"))
        if self.future_latent_target == "dynaflip" and str(cfg.get("pool", "mean")).lower() != "token":
            raise ValueError("framework.future_latent.target: dynaflip requires pool: token")
        self.future_latent_pool = str(cfg.get("pool", "mean")).lower()
        if self.future_latent_pool not in ("mean", "token"):
            raise ValueError("framework.future_latent.pool must be 'mean' or 'token'")
        if self.future_latent_pool == "token" and self.future_latent_target not in ("delta", "dynaflip"):
            raise ValueError("framework.future_latent.pool: token requires target: delta or dynaflip")
        if self.future_latent_loss != "cosine":
            raise ValueError("framework.future_latent.loss supports only 'cosine'")
        if self.future_latent_weight <= 0:
            raise ValueError("framework.future_latent.weight must be positive")

        vla_data = config.datasets.get("vla_data", {}) if hasattr(config, "datasets") else {}
        data_offset = int(vla_data.get("shared_z_future_offset", 0) or 0)
        if data_offset != self.future_latent_offset:
            raise ValueError(
                "framework.future_latent.offset must equal "
                f"datasets.vla_data.shared_z_future_offset ({self.future_latent_offset} vs "
                f"{data_offset}); the dataloader supplies the future frame at that offset"
            )
        if not bool(vla_data.get("shared_z_future_keep_tail", False)):
            raise ValueError(
                "framework.future_latent requires datasets.vla_data.shared_z_future_keep_tail: "
                "true, otherwise the sampler drops the last offset frames of every episode "
                "and the arm no longer trains on the frames of its no-aux control"
            )
        if bool(config.framework.get("shared_z", {}).get("enabled", False)):
            raise ValueError("future_latent and shared_z share the future frame and loss slot")
        if bool(config.framework.get("structured_aux", {}).get("enabled", False)):
            raise ValueError("future_latent and structured_aux share the structured_aux loss slot")
        if bool(config.framework.get("privileged_tokens", {}).get("enabled", False)):
            raise ValueError("future_latent does not rebuild privileged tokens for the target pass")
        logger.info(
            "future latent aux ON: layer=%d views=%s offset=%d weight=%.3f loss=%s target=%s pool=%s",
            self.future_latent_layer, self.future_latent_views, self.future_latent_offset,
            self.future_latent_weight, self.future_latent_loss, self.future_latent_target,
            self.future_latent_pool,
        )

    # ── modules ──────────────────────────────────────────────────────────────
    def _build_future_latent_modules(self, hidden_size: int) -> None:
        if not getattr(self, "future_latent_enabled", False):
            return
        h, m = int(hidden_size), self.future_latent_mlp_hidden
        out = 768 if self.future_latent_target == "dynaflip" else h
        self.future_latent_predictor = nn.ModuleDict({
            view: nn.Sequential(
                nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(), nn.Linear(m, out)
            )
            for view in self.future_latent_views
        })
        if self.future_latent_target == "dynaflip":
            # DynaFLIP changes are small (a few % of the feature norm) and the loss divides by
            # their per-channel RMS: start from "no change" so the normalised loss starts at 1,
            # not ~1e3 (smoke 2115565: 1170 at step 1 with a default-initialised last layer).
            for mlp in self.future_latent_predictor.values():
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
        self._aux_register_hook(self.future_latent_layer)

    # ── loss ─────────────────────────────────────────────────────────────────
    def _future_latent_loss(
        self,
        online_hidden: Optional[torch.Tensor],
        online_input_ids: torch.Tensor,
        examples: List[dict],
        instructions: List[str],
        state_bins,
        dropped_images: Optional[List[List[int]]],
    ):
        """Return (loss, metrics). ``instructions`` are the state-conditioned prompts of
        the online pass and ``state_bins`` its pending state bins, so the target prompt is
        token-for-token the online prompt with the future images in place."""
        self._aux_check_online(online_hidden)
        device = online_hidden.device
        batch = len(examples)
        zero = sum(p.sum() for p in self.future_latent_predictor.parameters()) * 0.0
        future_images = [ex.get("future_image") for ex in examples]
        has_future = [
            img is not None and len(img) == len(ex["image"])
            for img, ex in zip(future_images, examples)
        ]
        metrics = {
            "structured_aux/future_latent_has_future": torch.tensor(
                float(np.mean(has_future)), device=device
            )
        }
        rows = [i for i in range(batch) if has_future[i]]
        online_masks = self._aux_camera_masks(online_input_ids, self.future_latent_views)
        if not rows:
            return zero, metrics
        if self.future_latent_target == "dynaflip":
            return self._future_latent_dynaflip_loss(
                online_hidden, online_masks, examples, future_images, rows, dropped_images, zero, metrics
            )

        # Target: same backbone, no grad, future frames (both cameras, no dropout), the
        # same prompt and state tokens, stopped right after block L.
        target_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[future_images[i] for i in rows],
            instructions=[instructions[i] for i in rows],
        )
        if state_bins is not None:
            self._pending_state_bins = [state_bins[i] for i in rows]
            self._splice_state_tokens(target_inputs)
        self._future_latent_arm(stop=True)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    self.qwen_vl_interface(
                        **target_inputs, output_attentions=False,
                        output_hidden_states=False, return_dict=True,
                    )
                except _StopAfterFutureLatentLayer:
                    pass
            target_hidden = self._future_latent_take()
        finally:
            self._fl_capture = None
        if target_hidden is None:
            raise RuntimeError("future_latent: the target pass never reached block L")
        target_masks = self._aux_camera_masks(target_inputs["input_ids"], self.future_latent_views)

        losses = []
        rows_t = torch.as_tensor(rows, device=device)
        for view in self.future_latent_views:
            cam = _VIEW_INDEX[view]
            valid = torch.tensor(
                [
                    not (dropped_images is not None and cam in dropped_images[i])
                    for i in rows
                ],
                device=device, dtype=torch.bool,
            )
            online_mask = online_masks[view][rows_t]
            target_mask = target_masks[view]
            n_online = online_mask.sum(dim=1)
            n_target = target_mask.sum(dim=1)
            if bool((n_online == 0).any()) or bool((n_online != n_target).any()):
                raise RuntimeError(
                    f"future_latent: camera {view} token counts differ or are empty "
                    f"(online {n_online.tolist()}, target {n_target.tolist()})"
                )
            if getattr(self, "future_latent_pool", "mean") == "token":
                # Equal token counts per row (checked above), so the flattened online and
                # target tokens align position for position.
                on_tok = online_hidden[rows_t].float()[online_mask]
                tg_tok = target_hidden.float()[target_mask].detach()
                tok_valid = valid.repeat_interleave(n_online)
                metrics[f"structured_aux/future_latent_valid_{view}"] = torch.tensor(
                    int(valid.sum()) / batch, device=device
                )
                if not bool(tok_valid.any()):
                    continue
                delta = (tg_tok - on_tok.detach())[tok_valid]
                pred = self.future_latent_predictor[view](on_tok[tok_valid].to(
                    self.future_latent_predictor[view][1].weight.dtype
                )).float()
                rms = delta.pow(2).mean(dim=0).sqrt().clamp_min(1e-6).detach()
                nmse = ((pred - delta) / rms).pow(2).mean()
                losses.append(nmse.reshape(1))
                metrics[f"structured_aux/future_latent_r2_{view}"] = (1.0 - nmse).detach()
                metrics[f"structured_aux/future_latent_delta_rel_{view}"] = (
                    delta.norm(dim=-1) / (delta + on_tok.detach()[tok_valid]).norm(dim=-1).clamp_min(1e-6)
                ).mean().detach()
                continue
            online_pool = self._masked_mean(online_hidden[rows_t].float(), online_mask)
            target_pool = self._masked_mean(target_hidden.float(), target_mask).detach()
            pred = self.future_latent_predictor[view](online_pool.to(
                self.future_latent_predictor[view][1].weight.dtype
            )).float()
            copy_cos = F.cosine_similarity(online_pool.detach(), target_pool, dim=-1)
            if getattr(self, "future_latent_target", "level") == "delta":
                delta = target_pool - online_pool.detach()
                metrics[f"structured_aux/future_latent_delta_rel_{view}"] = (
                    delta.norm(dim=-1) / target_pool.norm(dim=-1).clamp_min(1e-6)
                ).mean().detach()
                target_pool = delta
            cos = F.cosine_similarity(pred, target_pool, dim=-1)
            n_valid = int(valid.sum())
            metrics[f"structured_aux/future_latent_valid_{view}"] = torch.tensor(
                n_valid / batch, device=device
            )
            if n_valid:
                losses.append((1.0 - cos)[valid])
                metrics[f"structured_aux/future_latent_cos_{view}"] = cos[valid].detach().mean()
                metrics[f"structured_aux/future_latent_copy_cos_{view}"] = (
                    copy_cos[valid].mean()
                )
        if not losses:
            return zero, metrics
        loss = torch.cat(losses).mean() * self.future_latent_weight + zero
        return loss, metrics

    # ── frozen DynaFLIP target ───────────────────────────────────────────────
    def _dynaflip_encoder(self, device):
        """Frozen DynaFLIP-base vision backbone, loaded on first use. Kept out of the module
        tree (self.__dict__), so it is neither trained, nor saved, nor built at inference."""
        enc = self.__dict__.get("_dynaflip_enc")
        if enc is None:
            from pathlib import Path

            from safetensors.torch import load_file
            from transformers import Dinov2Model

            root = Path(self.future_latent_dynaflip_dir)
            enc = Dinov2Model.from_pretrained(str(root / "dinov2-base"))
            sd = load_file(str(root / "dynaflip-base/model.safetensors"))
            pre = "vision_encoder.backbone."
            missing, unexpected = enc.load_state_dict(
                {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"DynaFLIP load: missing {missing[:3]} unexpected {unexpected[:3]}")
            enc.requires_grad_(False).eval()
            self.__dict__["_dynaflip_enc"] = enc
            logger.info("future latent: frozen DynaFLIP target loaded from %s", root)
        return enc.to(device)

    @torch.no_grad()
    def _dynaflip_grid(self, images, grid: int, device) -> torch.Tensor:
        """[N, grid*grid, 768] pooled DynaFLIP patch tokens of N PIL images, row-major."""
        x = torch.stack([
            torch.from_numpy(np.asarray(im.convert("RGB").resize((224, 224)), dtype=np.float32))
            for im in images
        ]).to(device).permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self._dynaflip_encoder(device)(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
        h = h.float()
        g = int(round(h.shape[1] ** 0.5))
        h = h.transpose(1, 2).reshape(h.shape[0], h.shape[2], g, g)
        return F.adaptive_avg_pool2d(h, grid).flatten(2).transpose(1, 2)

    def _future_latent_dynaflip_loss(self, online_hidden, online_masks, examples, future_images,
                                     rows, dropped_images, zero, metrics):
        device = online_hidden.device
        batch = len(examples)
        rows_t = torch.as_tensor(rows, device=device)
        losses = []
        for view in self.future_latent_views:
            cam = _VIEW_INDEX[view]
            valid = torch.tensor(
                [not (dropped_images is not None and cam in dropped_images[i]) for i in rows],
                device=device, dtype=torch.bool,
            )
            online_mask = online_masks[view][rows_t]
            n_online = online_mask.sum(dim=1)
            n = int(n_online[0])
            grid = int(round(n ** 0.5))
            if bool((n_online != n).any()) or grid * grid != n:
                raise RuntimeError(f"future_latent dynaflip: camera {view} tokens {n_online.tolist()} "
                                   "are not one square grid per row")
            metrics[f"structured_aux/future_latent_valid_{view}"] = torch.tensor(
                int(valid.sum()) / batch, device=device)
            if not bool(valid.any()):
                continue
            keep = [r for r, v in zip(rows, valid.tolist()) if v]
            f_now = self._dynaflip_grid([examples[i]["image"][cam] for i in keep], grid, device)
            f_fut = self._dynaflip_grid([future_images[i][cam] for i in keep], grid, device)
            delta = (f_fut - f_now).reshape(-1, f_now.shape[-1])  # rows x tokens, row-major
            on_tok = online_hidden[rows_t].float()[online_mask].reshape(len(rows), n, -1)[valid].reshape(len(keep) * n, -1)
            pred = self.future_latent_predictor[view](on_tok.to(
                self.future_latent_predictor[view][1].weight.dtype)).float()
            rms = delta.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
            nmse = ((pred - delta) / rms).pow(2).mean()
            losses.append(nmse.reshape(1))
            metrics[f"structured_aux/future_latent_r2_{view}"] = (1.0 - nmse).detach()
            metrics[f"structured_aux/future_latent_delta_rel_{view}"] = (
                delta.norm(dim=-1) / f_fut.reshape(-1, f_fut.shape[-1]).norm(dim=-1).clamp_min(1e-6)
            ).mean().detach()
        if not losses:
            return zero, metrics
        return torch.cat(losses).mean() * self.future_latent_weight + zero, metrics


class TaskPredMixin(_AuxLayerHookMixin):
    """Task-identity classifier on the pooled image tokens of block L (training only)."""

    def _configure_task_pred(self, config) -> None:
        cfg = dict(config.framework.get("task_pred", {}) or {})
        self.task_pred_enabled = bool(cfg.get("enabled", False))
        if not self.task_pred_enabled:
            return
        self.task_pred_layer = int(cfg.get("layer", 16))
        views = cfg.get("views", ["external"])
        if isinstance(views, str):
            views = [views]
        self.task_pred_views = [str(v).lower() for v in views]
        if not self.task_pred_views or any(v not in _VIEW_INDEX for v in self.task_pred_views):
            raise ValueError(f"framework.task_pred.views must be a subset of {sorted(_VIEW_INDEX)}")
        self.task_pred_num_tasks = int(cfg.get("num_tasks", 40))
        self.task_pred_label_map_file = cfg.get("label_map_file", None) or None
        self.task_pred_weight = float(cfg.get("weight", 1.0))
        self.task_pred_mlp_hidden = int(cfg.get("mlp_hidden", 1024))
        tasks_file = cfg.get("tasks_file", None)
        if not tasks_file:
            root = config.datasets.vla_data.get("data_root_dir", "")
            tasks_file = f"{root}/libero/meta/tasks.parquet"
        self.task_pred_tasks_file = str(tasks_file)
        self._task_pred_vocab: Optional[Dict[str, int]] = None
        if bool(config.framework.get("shared_z", {}).get("enabled", False)):
            raise ValueError("task_pred and shared_z share the structured_aux loss slot")
        if bool(config.framework.get("structured_aux", {}).get("enabled", False)):
            raise ValueError("task_pred and structured_aux share the structured_aux loss slot")
        logger.info(
            "task-identity aux ON: layer=%d views=%s num_tasks=%d tasks_file=%s",
            self.task_pred_layer, self.task_pred_views, self.task_pred_num_tasks,
            self.task_pred_tasks_file,
        )

    def _build_task_pred_modules(self, hidden_size: int) -> None:
        if not getattr(self, "task_pred_enabled", False):
            return
        h, m = int(hidden_size), self.task_pred_mlp_hidden
        self.task_pred_head = nn.ModuleDict({
            view: nn.Sequential(
                nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(),
                nn.Linear(m, self.task_pred_num_tasks),
            )
            for view in self.task_pred_views
        })
        self._aux_register_hook(self.task_pred_layer)

    @staticmethod
    def _task_key(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _task_pred_labels(self, examples: List[dict]) -> List[int]:
        """Task index per sample from the dataset's task table (loaded lazily, training only)."""
        if self._task_pred_vocab is None and getattr(self, "task_pred_label_map_file", None):
            # Optional explicit instruction -> class map (e.g. RoboCasa: 213 instruction
            # variants -> 18 atomic task types), JSON {"num_tasks": N, "map": {text: id}}.
            import json

            spec = json.load(open(self.task_pred_label_map_file))
            vocab = {self._task_key(t): int(i) for t, i in spec["map"].items()}
            if set(vocab.values()) != set(range(self.task_pred_num_tasks)):
                raise ValueError(
                    f"task_pred: {self.task_pred_label_map_file} covers {len(set(vocab.values()))} "
                    f"classes, config says num_tasks={self.task_pred_num_tasks}"
                )
            self._task_pred_vocab = vocab
        if self._task_pred_vocab is None:
            import pandas as pd

            df = pd.read_parquet(self.task_pred_tasks_file).reset_index()
            text_col = [c for c in df.columns if c != "task_index"][0]
            vocab = {self._task_key(t): int(i) for t, i in zip(df[text_col], df["task_index"])}
            if len(vocab) != self.task_pred_num_tasks or set(vocab.values()) != set(
                range(self.task_pred_num_tasks)
            ):
                raise ValueError(
                    f"task_pred: {self.task_pred_tasks_file} gives {len(vocab)} distinct "
                    f"tasks, config says num_tasks={self.task_pred_num_tasks}"
                )
            self._task_pred_vocab = vocab
        labels = []
        for ex in examples:
            key = self._task_key(ex["lang"])
            if key not in self._task_pred_vocab:
                raise KeyError(f"task_pred: instruction {ex['lang']!r} not in the task table")
            labels.append(self._task_pred_vocab[key])
        return labels

    def _task_pred_loss(
        self,
        online_hidden: Optional[torch.Tensor],
        online_input_ids: torch.Tensor,
        examples: List[dict],
        dropped_images: Optional[List[List[int]]],
    ):
        self._aux_check_online(online_hidden)
        device = online_hidden.device
        batch = len(examples)
        zero = sum(p.sum() for p in self.task_pred_head.parameters()) * 0.0
        labels = torch.as_tensor(self._task_pred_labels(examples), device=device)
        masks = self._aux_camera_masks(online_input_ids, self.task_pred_views)
        metrics = {}
        losses = []
        for view in self.task_pred_views:
            cam = _VIEW_INDEX[view]
            valid = torch.tensor(
                [not (dropped_images is not None and cam in dropped_images[i]) for i in range(batch)],
                device=device, dtype=torch.bool,
            )
            if bool((masks[view].sum(dim=1) == 0).any()):
                raise RuntimeError(f"task_pred: a row has no {view} image tokens")
            pooled = self._masked_mean(online_hidden.float(), masks[view])
            head = self.task_pred_head[view]
            logits = head(pooled.to(head[1].weight.dtype)).float()
            n_valid = int(valid.sum())
            metrics[f"structured_aux/task_pred_valid_{view}"] = torch.tensor(n_valid / batch, device=device)
            if n_valid:
                ce = F.cross_entropy(logits[valid], labels[valid], reduction="none")
                losses.append(ce)
                metrics[f"structured_aux/task_pred_ce_{view}"] = ce.detach().mean()
                metrics[f"structured_aux/task_pred_acc_{view}"] = (
                    (logits[valid].argmax(-1) == labels[valid]).float().mean().detach()
                )
        if not losses:
            return zero, metrics
        return torch.cat(losses).mean() * self.task_pred_weight + zero, metrics

"""Teacher alignment of camera tokens (T1-T6, default off; 2026-10-08).

``framework.teacher_align`` trains a GIFT-style head (arXiv 2609.04193) to predict a frozen vision teacher's patch
features from the policy's camera tokens. Targets are precomputed on the CLEAN current frame
(scripts/teacher_feats/precompute_teacher_feats.py; [n_cams, 14*14, D] per sample, attached by the dataloader as
``teacher_feats`` with ``datasets.vla_data.teacher_feats``) and resampled to the token grid they supervise
(2x2 average pool when the grid is half the teacher's, adaptive average pool when smaller, bilinear when larger).

Head: LayerNorm -> Linear(in, mlp_hidden) -> GELU, then a direction layer (-> D) and a scalar log-scale layer.
Per token: ``w_ang * (1 - cos(dir, t)) + w_scale * (logscale + offset - log(1 + |t|))^2``, averaged over the
valid tokens, times ``weight``; the sum enters ``structured_aux_loss`` (so trainer.loss_scale.structured_aux
scales it too). ``offset`` is a buffer set once from the first batch's mean target log-norm (all ranks agree),
so the scale term starts near its floor instead of at log(1+|t|)^2.

Modes:
  masked       (T1, T6) the online block-``layer`` states (= fusion_registers.layer, the same capture) of the image
               tokens of every camera dropped by modality dropout predict that camera's teacher grid. REPLACES the
               fusion_registers EMA target for dropped cameras (state / instruction EMA targets unchanged).
  masked_ema   (T2) as masked, but the EMA camera target is kept and this loss is added.
  present_vit  (T3) the vision tower's last block output (before the 2x2 merger: one token per 16 px patch) of every
               PRESENT camera predicts the teacher grid at that resolution.
  copies       (T5) the future-copy tokens (framework.future_copies with target: teacher; copy layer and greying
               from future_copies) of every camera, present or dropped, predict the current-frame teacher grid.

Training only: nothing is armed at inference, and predict_action never reads ``teacher_feats``. Rows without
``teacher_feats`` are skipped (their heads stay in the graph with a zero term).

Config (``framework.teacher_align``)::

    enabled: false
    mode: masked            # masked | masked_ema | present_vit | copies
    teacher: dinov3         # dinov3 | dinov2 | dynaflip (D 768) | vggt (D 2048); or set dim
    layer: 16               # masked modes: must equal fusion_registers.layer
    weight: 1.0             # present_vit default 0.2
    w_ang: 1.0
    w_scale: 0.25
    mlp_hidden: 2048
    target_norm: raw        # raw | center_ln: subtract the fixed per-camera dataset mean of the teacher features,
                            # then LayerNorm each token (no affine), as in Temporal Forcing (arXiv 2608.30643).
                            # Raw VGGT tokens are ~88% a shared offset (cos to their frame mean 0.94), so the raw
                            # cosine is near its floor before any geometry is learned.
    mean_path: null         # center_ln: [n_cams, D] .npy; default <teacher_feats.root>/libero/<teacher>/mean_cam.npy
"""
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.future_latent import _VIEW_INDEX
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

_TEACHER_DIM = {"dinov3": 768, "dinov2": 768, "dynaflip": 768, "vggt": 2048}
_MODES = ("masked", "masked_ema", "present_vit", "copies")
_P = "structured_aux/teacher_align/"


class TeacherHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU())
        self.direction = nn.Linear(hidden, out_dim)
        self.logscale = nn.Linear(hidden, 1)
        nn.init.zeros_(self.logscale.weight)
        nn.init.zeros_(self.logscale.bias)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.direction(h), self.logscale(h).squeeze(-1)


def _resample_grid(t: torch.Tensor, g: int) -> torch.Tensor:
    """[..., G*G, D] row-major teacher grid -> [..., g*g, D] row-major."""
    lead, n, d = t.shape[:-2], t.shape[-2], t.shape[-1]
    G = int(round(n ** 0.5))
    if G * G != n:
        raise RuntimeError(f"teacher_align: {n} teacher tokens are not a square grid")
    if G == g:
        return t
    x = t.reshape(-1, G, G, d).permute(0, 3, 1, 2)
    if G == 2 * g:
        x = F.avg_pool2d(x, 2)
    elif g < G:
        x = F.adaptive_avg_pool2d(x, g)
    else:
        x = F.interpolate(x, size=(g, g), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(*lead, g * g, d)


class TeacherAlignMixin:
    # ── config / modules ─────────────────────────────────────────────────────
    def _configure_teacher_align(self, config) -> None:
        cfg = dict(config.framework.get("teacher_align", {}) or {})
        self.teacher_align_enabled = bool(cfg.get("enabled", False))
        self.ta_replace_camera_ema = False
        self._ta_vit_capture = None
        self._ta_vit_online = None
        if not self.teacher_align_enabled:
            return
        self.ta_mode = str(cfg.get("mode", "masked")).lower()
        if self.ta_mode not in _MODES:
            raise ValueError(f"framework.teacher_align.mode must be one of {_MODES}")
        self.ta_teacher = str(cfg.get("teacher", "dinov3")).lower()
        dim = cfg.get("dim", None)
        if dim in (None, "", "null"):
            if self.ta_teacher not in _TEACHER_DIM:
                raise ValueError(f"teacher_align: unknown teacher {self.ta_teacher!r}; set framework.teacher_align.dim")
            dim = _TEACHER_DIM[self.ta_teacher]
        self.ta_dim = int(dim)
        self.ta_layer = int(cfg.get("layer", 16))
        self.ta_weight = float(cfg.get("weight", 0.2 if self.ta_mode == "present_vit" else 1.0))
        self.ta_w_ang = float(cfg.get("w_ang", 1.0))
        self.ta_w_scale = float(cfg.get("w_scale", 0.25))
        self.ta_mlp_hidden = int(cfg.get("mlp_hidden", 2048))
        self.ta_target_norm = str(cfg.get("target_norm", "raw")).lower()
        if self.ta_target_norm not in ("raw", "center_ln"):
            raise ValueError("framework.teacher_align.target_norm must be raw or center_ln")
        fr = dict(config.framework.get("fusion_registers", {}) or {})
        if not (bool(fr.get("enabled", False)) and str(fr.get("mode", "")) == "masked_query"):
            # the aux path (training-only block-L capture, dropped-camera bookkeeping) is the fusion one
            raise ValueError("teacher_align needs framework.fusion_registers {enabled: true, mode: masked_query}")
        if self.ta_mode in ("masked", "masked_ema") and self.ta_layer != int(fr.get("layer", 16)):
            raise ValueError("teacher_align.layer must equal fusion_registers.layer (the masked modes reuse its capture)")
        if self.ta_mode == "copies":
            fc = dict(config.framework.get("future_copies", {}) or {})
            if not (bool(fc.get("enabled", False)) and str(fc.get("target", "ema")).lower() == "teacher"):
                raise ValueError("teacher_align mode copies needs framework.future_copies {enabled: true, target: teacher}")
        # The dataloader block may be absent from an inference config (config.yaml keeps accessed keys only);
        # a training batch without any target then fails in the loss.
        tf = config.datasets.vla_data.get("teacher_feats", None)
        tf_teacher = None
        if tf:
            tf_teacher = tf.get("teacher", None) if hasattr(tf, "get") else getattr(tf, "teacher", None)
        if tf_teacher is not None and str(tf_teacher).lower() != self.ta_teacher:
            raise ValueError(f"teacher_align.teacher {self.ta_teacher} but datasets.vla_data.teacher_feats.teacher "
                             f"{tf_teacher}")
        self._ta_data_configured = tf_teacher is not None
        self._ta_mean = None
        if self.ta_target_norm == "center_ln":
            mp = cfg.get("mean_path", None)
            if mp in (None, "", "null"):
                root = (tf.get("root", None) if hasattr(tf, "get") else getattr(tf, "root", None)) if tf else None
                if root is None:
                    raise ValueError("teacher_align center_ln: set mean_path or datasets.vla_data.teacher_feats.root")
                mp = f"{root}/libero/{self.ta_teacher}/mean_cam.npy"
            mean = np.load(str(mp)).astype(np.float32)
            if mean.ndim != 2 or mean.shape[1] != self.ta_dim:
                raise ValueError(f"teacher_align: mean {mp} has shape {mean.shape}, expected [n_cams, {self.ta_dim}]")
            self._ta_mean = torch.from_numpy(mean)
            logger.info("teacher align: center_ln targets, per-camera mean %s %s", mp, tuple(mean.shape))
        self.ta_replace_camera_ema = self.ta_mode == "masked"
        self._ta_logged = False
        logger.info("teacher align ON: mode=%s teacher=%s dim=%d layer=%d weight=%.3f w_ang=%.2f w_scale=%.2f "
                    "target_norm=%s", self.ta_mode, self.ta_teacher, self.ta_dim, self.ta_layer, self.ta_weight,
                    self.ta_w_ang, self.ta_w_scale, self.ta_target_norm)

    def _build_teacher_align(self, hidden_size: int) -> None:
        if not getattr(self, "teacher_align_enabled", False):
            return
        in_dim = int(hidden_size)
        if self.ta_mode == "present_vit":
            visual = self.qwen_vl_interface.model.model.visual
            in_dim = int(visual.config.hidden_size)

            def vit_hook(module, args):
                cap = self._ta_vit_capture
                if cap is None or cap.get("hidden") is not None:
                    return None  # unarmed (inference, EMA passes) or already taken this forward
                cap["hidden"] = args[0]
                return None

            visual.merger.register_forward_pre_hook(vit_hook)
        self.teacher_head = TeacherHead(in_dim, self.ta_dim, self.ta_mlp_hidden)
        self.register_buffer("teacher_logscale_offset", torch.full((1,), float("nan")), persistent=True)

    # ── online pass (present_vit) ────────────────────────────────────────────
    def _ta_arm(self, qwen_inputs: dict, images_per_row: List[int]) -> None:
        self._ta_vit_capture = None
        self._ta_vit_online = None
        if (not getattr(self, "teacher_align_enabled", False) or self.ta_mode != "present_vit"
                or not self.training or not torch.is_grad_enabled()):
            return
        self._ta_vit_capture = {"hidden": None, "grid_thw": qwen_inputs.get("image_grid_thw"),
                                "images_per_row": list(images_per_row)}

    def _ta_take_online(self) -> None:
        cap, self._ta_vit_capture = self._ta_vit_capture, None
        self._ta_vit_online = cap

    # ── loss ─────────────────────────────────────────────────────────────────
    def _ta_target(self, t: np.ndarray, cam: int, g: int, device) -> torch.Tensor:
        """Teacher grid [G*G, D] of camera ``cam`` -> [g*g, D] target (centred + LayerNorm under center_ln)."""
        t = torch.from_numpy(np.asarray(t)).to(device).float()
        if getattr(self, "_ta_mean", None) is not None:
            t = t - self._ta_mean[min(cam, self._ta_mean.shape[0] - 1)].to(device)
        t = _resample_grid(t, g)
        if getattr(self, "_ta_mean", None) is not None:
            t = F.layer_norm(t, t.shape[-1:])
        return t

    def _ta_zero(self):
        return sum(p.sum() for p in self.teacher_head.parameters()) * 0.0

    def _ta_teacher_rows(self, examples):
        rows = [None if ex.get("teacher_feats") is None else np.asarray(ex["teacher_feats"]) for ex in examples]
        if all(r is None for r in rows) and not getattr(self, "_ta_data_configured", False):
            raise RuntimeError(f"teacher_align: no sample carries teacher_feats; set datasets.vla_data.teacher_feats "
                               f"{{teacher: {self.ta_teacher}, root: ...}}")
        return rows

    @torch.no_grad()
    def _ta_init_offset(self, logn: torch.Tensor) -> None:
        if not bool(torch.isnan(self.teacher_logscale_offset).any()):
            return
        # every rank calls this on every step until it is set (also with no tokens), so the all_reduce matches
        s = torch.stack([logn.float().sum(), torch.tensor(float(logn.numel()), device=logn.device)])
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(s)
        if float(s[1]) <= 0:
            return
        self.teacher_logscale_offset.fill_(float(s[0] / s[1]))
        logger.info("teacher align: log-scale offset %.3f", float(self.teacher_logscale_offset))

    def _ta_head_loss(self, src: torch.Tensor, tgt: torch.Tensor, zero, metrics: dict, n_rows: int):
        """src [N, in], tgt [N, D] -> weighted loss and the metrics (N may be 0)."""
        device = zero.device if torch.is_tensor(zero) else src.device
        metrics[_P + "n_tokens"] = torch.tensor(float(src.shape[0]), device=device)
        metrics[_P + "rows"] = torch.tensor(float(n_rows), device=device)
        tgt = tgt.float()
        norm = tgt.norm(dim=-1)
        keep = norm > 0
        src, tgt, norm = src[keep], tgt[keep], norm[keep]
        logn = torch.log1p(norm)
        self._ta_init_offset(logn.to(zero.device))
        if src.shape[0] == 0:
            return zero, metrics
        head = self.teacher_head
        d, ls = head(src.to(head.direction.weight.dtype))
        cos = F.cosine_similarity(d.float(), tgt, dim=-1, eps=1e-6)
        ang = (1.0 - cos).mean()
        pred_ls = ls.float() + self.teacher_logscale_offset.float()
        scale = (pred_ls - logn).pow(2).mean()
        metrics[_P + "ang"] = ang.detach()
        metrics[_P + "scale"] = scale.detach()
        metrics[_P + "cos"] = cos.mean().detach()
        metrics[_P + "n_tokens"] = torch.tensor(float(src.shape[0]), device=device)
        if not getattr(self, "_ta_logged", False):
            self._ta_logged = True
            print(f"__TEACHER_ALIGN__ mode={self.ta_mode} teacher={self.ta_teacher} tokens={int(src.shape[0])} "
                  f"src_dim={int(src.shape[-1])} tgt_dim={int(tgt.shape[-1])} rows={n_rows} "
                  f"offset={float(self.teacher_logscale_offset):.3f} ang={float(ang.detach()):.4f} "
                  f"scale={float(scale.detach()):.4f}",
                  flush=True)
        loss = self.ta_weight * (self.ta_w_ang * ang + self.ta_w_scale * scale)
        return loss + zero, metrics

    def _teacher_align_loss(self, online_hidden, online_input_ids, examples, dropped_images):
        """masked / masked_ema / present_vit (copies runs through future_copies -> _teacher_copies_loss)."""
        if not getattr(self, "teacher_align_enabled", False) or self.ta_mode == "copies":
            return None, {}
        zero = self._ta_zero()
        metrics = {}
        teach = self._ta_teacher_rows(examples)
        srcs, tgts, rows_used = [], [], set()
        if self.ta_mode in ("masked", "masked_ema"):
            self._aux_check_online(online_hidden)
            views = list(self.fusion_views)
            masks = self._aux_camera_masks(online_input_ids, views)
            for view in views:
                cam = _VIEW_INDEX[view]
                for r in range(len(examples)):
                    if teach[r] is None or dropped_images is None or cam not in (dropped_images[r] or []):
                        continue
                    pos = masks[view][r].nonzero(as_tuple=True)[0]
                    g = int(round(pos.numel() ** 0.5))
                    if g * g != pos.numel():
                        raise RuntimeError(f"teacher_align: camera {view} has {pos.numel()} tokens, not a square grid")
                    srcs.append(online_hidden[r, pos])
                    tgts.append(self._ta_target(teach[r][cam], cam, g, online_hidden.device))
                    rows_used.add(r)
        else:  # present_vit
            cap, self._ta_vit_online = self._ta_vit_online, None
            if cap is None or cap.get("hidden") is None:
                raise RuntimeError("teacher_align present_vit: no vision-tower capture from the online pass")
            hid = cap["hidden"]
            if not hid.requires_grad:
                raise RuntimeError("teacher_align present_vit: the vision-tower capture has no gradient")
            thw = cap["grid_thw"].tolist()
            offsets = np.concatenate(([0], np.cumsum([int(t) * int(h) * int(w) for t, h, w in thw]))).astype(int)
            if int(offsets[-1]) != int(hid.shape[0]):
                raise RuntimeError(f"teacher_align present_vit: {hid.shape[0]} patch states for {offsets[-1]} patches")
            perm_cache = {}
            cursor = 0
            for r, n_img in enumerate(cap["images_per_row"]):
                n_cams = n_img if not getattr(self, "fcopy_enabled", False) else n_img // 2
                for cam in range(n_cams):
                    k = cursor + cam
                    if teach[r] is None or cam >= teach[r].shape[0]:
                        continue
                    if dropped_images is not None and cam in (dropped_images[r] or []):
                        continue
                    t_, h, w = (int(v) for v in thw[k])
                    if t_ != 1 or h != w or h % 2:
                        raise RuntimeError(f"teacher_align present_vit: image grid {thw[k]} is not 1 x even square")
                    if h not in perm_cache:
                        # pixel_values order (Qwen-VL processor): (block row, block col, row in block, col in block)
                        bh, bw, ih, iw = torch.meshgrid(torch.arange(h // 2), torch.arange(w // 2), torch.arange(2),
                                                        torch.arange(2), indexing="ij")
                        perm_cache[h] = ((2 * bh + ih) * w + (2 * bw + iw)).reshape(-1).to(hid.device)
                    srcs.append(hid[offsets[k]:offsets[k + 1]])
                    tgts.append(self._ta_target(teach[r][cam], cam, h, hid.device)[perm_cache[h]])
                    rows_used.add(r)
                cursor += n_img
        if srcs:
            src, tgt = torch.cat(srcs, 0), torch.cat(tgts, 0)
        else:
            dev = online_hidden.device if online_hidden is not None else zero.device
            src, tgt = torch.zeros(0, 1, device=dev), torch.zeros(0, self.ta_dim, device=dev)
        return self._ta_head_loss(src, tgt, zero, metrics, len(rows_used))

    def _teacher_copies_loss(self, online, online_input_ids, examples, zero):
        """T5: every future-copy camera token grid predicts the current frame's teacher grid of that camera."""
        zero = zero + self._ta_zero()
        metrics = {}
        teach = self._ta_teacher_rows(examples)
        srcs, tgts, rows_used = [], [], set()
        for j in range(self._fc_n):
            m = self._fc_image_masks(online_input_ids, self._fc_n + j)
            for r in range(len(examples)):
                if teach[r] is None or j >= teach[r].shape[0]:
                    continue
                pos = m[r].nonzero(as_tuple=True)[0]
                g = int(round(pos.numel() ** 0.5))
                if g * g != pos.numel():
                    raise RuntimeError(f"teacher_align copies: copy {j} has {pos.numel()} tokens, not a square grid")
                srcs.append(online[r, pos])
                tgts.append(self._ta_target(teach[r][j], j, g, online.device))
                rows_used.add(r)
        if srcs:
            src, tgt = torch.cat(srcs, 0), torch.cat(tgts, 0)
        else:
            src, tgt = torch.zeros(0, 1, device=online.device), torch.zeros(0, self.ta_dim, device=online.device)
        return self._ta_head_loss(src, tgt, zero, metrics, len(rows_used))

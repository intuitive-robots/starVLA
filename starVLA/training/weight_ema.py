"""Weight EMA of the trainable parameters, sharded across ranks (2026-10-11, pi0.5 DROID per ARC: EMA 0.99).

``trainer.weight_ema: {enabled: true, decay: 0.99}``. Under DeepSpeed ZeRO-2 every rank holds the full bf16 model, so
the fp32 shadow is split by parameter across ranks (greedy by size): each rank keeps and updates only its own share
(~1/world of 4 B/param), so the per-GPU cost is ~3 GB for pi0.5 on 4 GPUs instead of ~13 GB.

* update after every synchronised optimizer step: shadow <- decay * shadow + (1 - decay) * param, initialised to the
  parameters at construction (the pretrained init, as openpi) or restored from a full training state;
* checkpoints: ``steps_N_ema_model.safetensors`` / ``steps_N_ema_pytorch_model.pt`` next to the online weights (the
  same keys and dtypes as the online state dict, frozen tensors copied from it). The name does not match the resume
  scanner's ``steps_N_(model.safetensors|pytorch_model.pt)``, so resume always continues from the online weights;
* full training state: every rank writes ``weight_ema_rank{r}.pt`` (its fp32 shards) into the state directory and
  reloads it on resume, so the EMA survives chained 12 h segments exactly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ShardedWeightEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        # one entry per trainable parameter object, with every name it is reachable under (tied weights)
        names: dict[int, list[str]] = {}
        params: dict[int, torch.nn.Parameter] = {}
        for n, p in model.named_parameters(remove_duplicate=False):
            if p.requires_grad:
                names.setdefault(id(p), []).append(n)
                params[id(p)] = p
        order = sorted(params, key=lambda k: names[k][0])   # deterministic, identical on every rank
        load = [0] * self.world
        self.entries = []   # (names, param, owner)
        for k in sorted(order, key=lambda k: -params[k].numel()):
            owner = min(range(self.world), key=lambda r: load[r])
            load[owner] += params[k].numel()
            self.entries.append((names[k], params[k], owner))
        self.entries.sort(key=lambda e: e[0][0])
        # first name -> fp32 tensor, own shares only; starts at the current (initial or resumed) weights
        self.shadow: dict[str, torch.Tensor] | None = None
        self._init_shadow()
        self.num_updates = 0
        mine = sum(p.numel() for _, p, o in self.entries if o == self.rank)
        logger.info("weight EMA: decay %.4f, %d trainable tensors (%.1fM params), %.1fM on rank %d (%.2f GB fp32)",
                    self.decay, len(self.entries), sum(p.numel() for _, p, _ in self.entries) / 1e6, mine / 1e6,
                    self.rank, mine * 4 / 1e9)

    def _init_shadow(self):
        self.shadow = {n[0]: p.detach().to(torch.float32, copy=True) for n, p, o in self.entries if o == self.rank}

    @torch.no_grad()
    def update(self):
        if self.shadow is None:
            self._init_shadow()
        w = 1.0 - self.decay
        for n, p, o in self.entries:
            if o == self.rank:
                # bf16 param promoted element-wise inside the kernel (no full fp32 temporary)
                self.shadow[n[0]].mul_(self.decay).add_(p.detach(), alpha=w)
        self.num_updates += 1

    @torch.no_grad()
    def gather_state_dict(self, online_state_dict: dict | None) -> dict | None:
        """Collective: every rank must call. Rank 0 gets the online state dict with the trainable tensors replaced
        by their EMA (cast to the online dtype, on CPU); other ranks get None."""
        if self.shadow is None:
            self._init_shadow()
        out = dict(online_state_dict) if self.rank == 0 else None
        for n, p, o in self.entries:
            if self.world > 1:
                buf = self.shadow[n[0]] if o == self.rank else torch.empty(p.shape, dtype=torch.float32,
                                                                             device=p.device)
                dist.broadcast(buf, src=o)
            else:
                buf = self.shadow[n[0]]
            if self.rank == 0:
                for name in n:
                    if name not in out:
                        raise KeyError(f"weight EMA: trainable parameter {name} missing from the online state dict")
                    out[name] = buf.to(dtype=out[name].dtype, device="cpu", copy=True)
            del buf
        return out

    def save_shards(self, state_dir: str | Path) -> None:
        if self.shadow is None:
            self._init_shadow()
        torch.save({"decay": self.decay, "world": self.world, "num_updates": self.num_updates,
                    "shadow": {k: v.cpu() for k, v in self.shadow.items()}},
                   Path(state_dir) / f"weight_ema_rank{self.rank}.pt")

    def load_shards(self, state_dir: str | Path) -> bool:
        path = Path(state_dir) / f"weight_ema_rank{self.rank}.pt"
        if not path.is_file():
            logger.warning("weight EMA: no %s, the EMA restarts from the resumed weights", path)
            return False
        st = torch.load(path, map_location="cpu", weights_only=True)
        if int(st["world"]) != self.world:
            raise RuntimeError(f"weight EMA: saved with world size {st['world']}, resumed with {self.world}")
        own = {n[0]: p for n, p, o in self.entries if o == self.rank}
        if set(st["shadow"]) != set(own):
            raise RuntimeError("weight EMA: saved shard names differ from this rank's share")
        self.shadow = {k: st["shadow"][k].to(device=own[k].device, dtype=torch.float32) for k in own}
        self.num_updates = int(st["num_updates"])
        logger.info("weight EMA: restored rank %d shards (%d updates) from %s", self.rank, self.num_updates, path)
        return True

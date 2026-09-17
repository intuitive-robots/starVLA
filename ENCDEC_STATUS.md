# Encoder–decoder backbone: where we stand, and what to do next

Written for the team working on the enc-dec VLA arms. Status as of 2026-09-17.
All numbers below are from result JSONs on disk; every table says how many episodes it rests on.

---

## 1. Bottom line

**The enc-dec backbone is not negative. It is a small positive on LIBERO-plus and neutral on
RoboCasa365.** The disappointment is the size of the effect, not its sign.

| arm (LIBERO-plus, 1000 eps/suite) | pooled | ±SE | eps | vs causal |
|---|---|---|---|---|
| v5 enc-dec + **GR00T** head | **0.746** | 0.005 | 8000 | **+0.029 ± 0.007** (≈4σ) |
| v5 enc-dec + PI head (aug) | 0.734 | 0.005 | 8000 | +0.017 ± 0.007 |
| v5 enc-dec + PI head | 0.731 | 0.005 | 8000 | +0.014 ± 0.007 (≈2σ) |
| v5 enc-dec + PI head (aux) | 0.721 | 0.005 | 8000 | +0.004 |
| **old** enc-dec backbone + PI ("bidir" control) | 0.721 | 0.004 | 12000 | +0.004 (n.s.) |
| **causal** Qwen3-VL + PI (no enc-dec) | 0.717 | 0.004 | 12000 | — |

Reading:

* The **head matters more than the backbone**. v5+GR00T − v5+PI = +0.015 ± 0.007. Whatever the
  encoder provides, the GR00T readout path extracts more of it than PI's layerwise cross-attention.
* **Backbone quality matters little**: v5 vs the June backbone is +0.010 ± 0.007, not significant.
* On **RoboCasa365** (17 atomic tasks, 48 eps each, 50k checkpoints, seed 4, 816 rollouts/arm):
  causal **0.363** vs v5 **0.373**, a difference of 0.010 ± 0.024 — indistinguishable.
* The **shared-z bottleneck is our strongest LIBERO-plus result** at full protocol: 0.783 (v5
  backbone, seed 43) and 0.776 (old backbone), both ~4000 eps, against 0.746 for the best
  action-only enc-dec arm. Inside zonly the backbone again makes no difference (0.783 vs 0.776).

So the ranking at full protocol is: **shared-z bottleneck (0.78) > enc-dec + GR00T (0.75) >
enc-dec + PI (0.73) ≈ causal (0.72)**.

### What is NOT true

The "enc-dec is much worse on RoboCasa" result reported earlier today does not survive. It came
from 30k checkpoints on one to three hand-picked tasks (pick-and-place, causal 0.44 vs v5 0.21).
At 50k across all 17 tasks the arms are equal. Treat any single-task robocasa comparison as noise.

---

## 2. Things ruled out (so nobody re-investigates them)

| hypothesis | verdict | evidence |
|---|---|---|
| Eval feeds the wrong state | **was true, now fixed** | sin/cos was applied to the concatenated 16-d vector instead of per key; 27 of 32 slots wrong. Fixed in `5c8e5a1`/`fc5f324` |
| The upstream merge broke the enc-dec path | **no** | `QWen3_EncDec` and the action heads untouched by the merge; only `interleave_self_attention`'s default flipped, and every config overrides it |
| `use_merged_attention: false` means the arms trained wrong | **no** | it only selects decoder behaviour; under `skip_decoder: true` the encoder output is **bitwise identical** either way |
| v5 needs more flow-matching integration steps | **no** | 96 eps/cell: v5 0.28→0.32 with 4→10 steps (n.s.); causal 0.51→0.38 |
| v5 predicts different actions at inference | **no** | paired open loop on identical sim states: cosine **0.964**, gripper sign agreement 98%, divergence symmetric in both drive directions |
| The enc-dec arms fit worse | **no, the opposite** | RoboCasa `action_dit_loss` is lower for v5 at every bin in both seeds (0.0387 vs 0.0401 @25k; 0.0212 vs 0.0218 @45k) |

The last two together are the interesting part: **v5 fits better, acts the same**. The extra
capacity is going somewhere that closed-loop success does not reward.

---

## 3. The state hypothesis, and how to test it

`QwenPI_v3` conditions on proprioception **only** as text: the 32-d sin/cos state is quantised
into 256 uniform bins over [-1,1] and appended to the instruction as
`<instruction> [STATE] 95 133 203 … [ACTION]`, then `state = None` (QwenPI_v3.py:1308-1310 in
training, 1518-1520 at inference). Raw state never reaches the DiT — `state_dim: 32` in the
action-model block is vestigial for this framework. `framework.state_dropout_rate` exists and
defaults to **0.0**, so every training example carried the full state suffix.

That makes "the encoder leans on state" a live hypothesis: an encoder with more capacity can
memorise the state→action mapping on the training distribution (lower loss, as observed) while
learning less from pixels — which costs nothing in teacher-forced loss and everything in closed
loop, where its own drift takes it off that mapping.

**Cheap tests, no retraining:**

1. **State ablation at eval.** The RoboCasa bridge already takes `--args.include-state false`.
   Run both arms with and without state, 48 eps × several tasks. If v5's success collapses
   further than causal's, it depends more on state. ~30 min/node for 4 cells.
2. **State corruption at eval.** Feed a permuted or noised state and measure the drop per arm.
   We have an accidental data point: with the permuted-state bug, OpenDrawer scored 0/2 where the
   fixed version scored 0.70 — state clearly matters a lot to *both* arms, but it was never
   measured per arm.

**Tests that need training:**

3. **`state_dropout_rate: 0.1–0.3`** on both arms. If enc-dec gains more from state dropout than
   causal does, over-reliance is confirmed and partially cured in one move.
4. **Continuous state into the DiT** instead of (or alongside) 256-bin text. 256 bins over
   [-1,1] is ~0.008 resolution on a sin/cos value; the DiT already has a `state_dim` input that
   nothing feeds. This is a modest code change in `QwenPI_v3`.

**On reordering:** permuting the state layout is not expected to help by itself — the model learns
whatever fixed order it is trained with, and information content is unchanged. The ordering only
mattered because train and eval disagreed (now fixed). The one ordering question worth an
experiment is *where* the state sits: appended after the instruction (current) vs. before it, which
changes which tokens the bidirectional encoder can attend to when forming the visual summary.

---

## 4. What we built today (reusable)

* **RoboCasa365 eval works end to end** on compute nodes: `run_eval.sh` fixed (directory overlay,
  the Apptainer session dir is capped at 16 MiB and ENOSPCs on the first `reset()`), per-key
  sin/cos state, seed plumbed into `env.reset`, `--args.result-tag`, throughput instrumentation.
* **Throughput measured**: `n_envs=24` is the operating point, ~929 rollouts/h/GPU, ~3,700/h/node
  — about 1.5× the reference setup a colleague measured (2,400/h/node). Videos cost VRAM: use
  `n_envs=12` when recording.
* **A 17-task RoboCasa benchmark sweep** (34 units/seed) runs in ~40 min/node.
* **`JOBS.md` + `scripts/jobs_status.sh`**: every job with its launch command, and the script
  flags queued jobs missing from the ledger.
* **Qwen3.5 encoder ported** (`QWen3_5_EncDec.py`, verified as an exact numerical match to the
  reference encoder). The q35 arm is blocked one layer lower, on `fla`/triton device detection.

Known blockers, all recorded in `JOBS.md`: degraded render nodes (EGL aborts, node-scoped);
one-client-per-server (per-request state lives on the module, so concurrent clients corrupt each
other); `PickPlaceSinkToCounter` fails in the renderer 3/3 attempts.

---

## 5. What to do next, in order of information per GPU-hour

1. **Finish the protocol clean-up.** Four CoT-trace re-evals are queued (`1850350-53`). That
   family currently leads at 0.826/0.822 but on 256-354 eps/suite; zonly fell 0.812→0.776 when
   re-run at full protocol. Until they land, the leaderboard's top is unverified.
   *Do not compare any 256-episode number against a 4000-episode one.*
2. **State-ablation eval** (test 1 above). Half a node-hour, no training, and it directly
   addresses the "what is the encoder actually using" question.
3. **Put the GR00T head on the best backbone in the zonly setting.** The two largest effects we
   have are the shared-z bottleneck (+0.06 over causal) and the GR00T head (+0.015 over PI). They
   have never been combined.
4. **`state_dropout_rate` sweep** on the enc-dec + GR00T arm (the arm where the backbone actually
   pays). Two seeds, one node-day.
5. **Only then** consider more backbone pretraining. Two different backbones (June and v5) give
   the same downstream number on two benchmarks and inside zonly; more of the same is unlikely to
   move it.

### On the resource question

The enc-dec investment is not lost — it produced a reproducible +3pp with the GR00T head and the
shared-z result that sits on top of our board. But the evidence says **backbone quality is not the
bottleneck**; how the action head reads the encoder is. That is where the next runs should go.

---

## 6. Update — 2026-09-17

### The state ablation says the opposite of the overfitting hypothesis

Serving both 50k seed-4 checkpoints with `--args.include-state False`, paired against the
benchmark cells (same seed, same scenes, 48 eps):

| task | causal +state | causal **−state** | v5 +state | v5 **−state** |
|---|---|---|---|---|
| OpenStandMixerHead | 0.83 | **0.00** | 0.81 | **0.83** |
| PickPlaceCounterToStove | 0.65 | **0.00** | 0.67 | **0.48** |
| TurnOnElectricKettle | 0.60 | **0.00** | 0.71 | **0.46** |

The causal arm needs proprioception almost totally; the enc-dec arm barely misses it. That is
a point FOR the enc-dec backbone that equal mean success rates hide: same score, far more
robustness to a degraded prompt.

**Caveat, and the follow-up.** Dropping state also removes the `[STATE] … [ACTION]` suffix, so
this conflates information with prompt format. In a causal model the trailing tokens are where
the summary accumulates; a bidirectional encoder spreads it. The clean test keeps the format
and corrupts the values (shuffle or zero the bins) — a few lines in the eval bridge behind an
env var, run on the same three tasks.

### GR00T head + shared-z now exists

`shared_z` was 123 call sites inside `QwenPI_v3`, which is the only reason the two largest
effects had never been combined. It is now `starVLA/model/modules/shared_z.py` (`SharedZMixin`),
with the classes and five helpers moved verbatim — QwenPI_v3's losses and parameter count are
byte-identical before and after (`scripts/`-adjacent harness, 25 terms compared).

`GR00T_ActionHeader` now forwards `z_conditioning` / `encoder_memory_keep` to the DiT as
`extra_conditioning` / `cross_attention_row_mask`. Both already existed in the shared
`cross_attention_dit`; only PI's wrapper passed them. Defaults are `None`, so existing GR00T
runs are unchanged.

Running as **1851382** (seeds 42/43), evals 1851383/1851384.

### Still blocked: q35 encoder-only

Fifth attempt printed `fla/triton preflight OK` and then died with the same
`module 'torch.cpu' has no attribute 'device'`. The binding happens **per process**: fla
resolves its device at import from triton's driver, and a probe in a separate process cannot
predict what the training process will see. The fix has to run inside the training process —
initialise CUDA before the import chain that pulls in fla, and repair the binding if it still
came out as CPU.

# Upstream StarVLA LIBERO-plus audit — 2026-09-17

## Decision

The upstream 75.0% Qwen3-VL-OFT and 77.0% Qwen3-VL-PI LIBERO-plus scores are important reported baselines. They are **not reproductions of our causal configuration**, and our 4,000-episode numbers cannot be compared directly with them. Upstream evaluated all 10,030 perturbed tasks once; our canonical internal protocol evaluates 1,000 tasks from each suite, 4,000 total.

Do not start a new camera-routing or auxiliary-objective training run until the already-running shared-z+GR00T pair finishes and its dependent evaluations land. In parallel, make the released upstream checkpoints runnable and evaluate them first on our frozen 4,000-task list, then on the official 10,030-task list. No job or checkpoint download was launched by this audit.

Sources inspected at upstream `starVLA_dev` commit `a376315d328b7c47a18497f09ef59ecbc37e37d1`:

- [Upstream LIBERO-plus instructions and reported table](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO-plus/README.md)
- [Released Qwen3-VL-PI checkpoint](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1)
- [Released Qwen3-VL-OFT checkpoint](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1)
- [Upstream LIBERO-plus evaluator](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.py)

## What upstream reports

| method | LIBERO-plus | official task count | released checkpoint | plain LIBERO reported by model card |
|---|---:|---:|---|---:|
| Qwen3-VL-OFT | 75.0% | 10,030, one rollout/task | 50k | 98.0% |
| Qwen3-VL-PI | 77.0% | 10,030, one rollout/task | 100k | 97.5% |

The LIBERO-plus repository contains the aggregate table but not raw per-task rollout outcomes. The Hugging Face model cards also say the plain-LIBERO numbers are retained reported results without raw episode logs. Treat all four numbers as author-reported until we run the released weights ourselves.

## Why our causal result is not a failed reproduction

| setting | upstream Qwen3-VL-PI | our causal PI baseline |
|---|---|---|
| backbone | Qwen3-VL-4B | Qwen3-VL-2B |
| framework | historical `QwenPI` | custom `QwenPI_v3` causal path |
| action expert | 36 layers, width 2,560, 40 heads | 28 layers, width 1,024, 16 heads |
| VLM/action interface | last 36 VLM layers, 32 target vision tokens configured | 28 projected layers, zero target vision tokens |
| action/state contract | horizon 8, state dimension 7 configured | horizon 16, state dimension 0 |
| inference | 4 flow steps; execute the full 8-step chunk | 4 flow steps; execute the full 16-step chunk |
| training | 100k steps, 16 GPUs × batch 8 = global 128 | 20k steps, 2 GPUs × batch 32 = global 64 |
| approximate sample presentations | 12.8M | 1.28M |
| VLM tuning | full fine-tuning (`freeze_modules` ultimately empty in archived launcher) | full fine-tuning (`freeze_modules: ''`) |
| data | four original LIBERO suites, 1,693 trajectories / 272,104 transitions in released statistics | consolidated four-suite data, 1,692 trajectories / 273,356 transitions |
| action statistics | gripper stored as 0/1; translation/rotation statistics from old conversion | gripper stored as −1/+1; measurably different normalization statistics |
| LIBERO-plus protocol | all 10,030 tasks: 2,519/2,591/2,518/2,402 by suite | 1,000 tasks/suite = 4,000 |
| compatibility | historical noncanonical LayerwiseFM forward required | corrected alternating PI layout |

The upstream PI therefore uses roughly ten times as many sample presentations, a 4B VLM, and a much larger action expert. Its 77.0% does not show that our 2B/20k causal baseline should have reached 77%. It does show that 71.66% is too weak to call a generally competitive causal reference without qualifying compute and architecture.

OFT is also not a cheap matched control: its released checkpoint uses Qwen3-VL-4B, full fine-tuning, batch 16 on 8 GPUs, and 50k steps—about 6.4M sample presentations—plus a different L1 action-token head.

## Unresolved upstream contract problems

1. The PI model card says one `image_0` view at 224×224. The checkpoint-era `Libero4in1DataConfig` selects both primary and wrist views, and the LIBERO-plus evaluator supplies both. The saved YAML's `obs: [image_0]` is not referenced by the inspected generic data-config path. Camera count for the reported checkpoint is therefore not established by the packaged artifacts.
2. The PI YAML configures `state_dim: 7`, but the released evaluator omits state. The implementation treats state as optional, so the evaluated model likely ran without it; this needs a handshake/input capture during reproduction.
3. The PI YAML contains fields that do not match the constructed checkpoint architecture. Its model card explicitly requires a historical `use_canonical_forward=false` compatibility override.
4. Upstream resizes both camera images to 224×224 at evaluation. Our saved checkpoint config does not declare an observation resize, and the logs used for the current result predate explicit image-contract metadata. Confirm the actual training/evaluation tensor sizes before attributing any difference to model architecture.
5. The upstream results provide no raw task outcomes or seed variation. We cannot compute uncertainty, verify category weighting, or determine checkpoint selection from the LIBERO-plus table alone.

## Next work, in order

1. Let shared-z+GR00T job 1851382 and dependent evaluations 1851383/1851384 complete. Do not idle: prepare the external checkpoint reproduction while it runs.
2. Download the released PI checkpoint and its complete config/statistics bundle. Use the historical forward override. Capture the actual image count/order, resolution, state presence, normalized action range and action chunk returned by one inference request.
3. Run a fixed 50–100-task smoke test that includes every perturbation category. Compare successful-task identities and action scales, not only the mean.
4. Evaluate released PI on the exact same 4,000 task IDs as our causal/shared-z runs. This is the fastest valid score comparison. Evaluate OFT next if the contract is stable.
5. Run the official 10,030-task protocol for released PI and our frozen causal/shared-z checkpoints. The full task list is required before claiming equality or superiority to the reported 77.0%.
6. Only then choose one new training intervention. If GR00T+z is clearly stronger, confirm it. If it is neutral and the upstream reproduction exposes a camera/chunk issue, use that evidence to choose camera routing or horizon 8. Otherwise prioritize causal+shared-z attribution.

The upstream checkpoint comparison is a baseline-reproduction task, not another architecture branch. It should displace camera-routing work on September 17–18 rather than expand the schedule.

# starVLA — working agreements

## Job ledger: JOBS.md

Every SLURM job goes into [JOBS.md](JOBS.md) **at submit time**, with its launch command
and the directory it must run from. Check status with:

```bash
bash scripts/jobs_status.sh                      # queue, plus anything untracked
bash scripts/jobs_status.sh --since 2026-09-16   # also finished / failed
```

Lost a launch command? `sacct -j <id> -X -o SubmitLine%400` while it is still in accounting.

## Two trees

- `/e/project1/m3/blank4/code/starVLA` — live tree, branch `branch_starVLA_dev_dataloading_juelich`
- `/e/project1/m3/blank4/code/starVLA-upstream-merge` — worktree, branch `merge_upstream_2026_09`

**RoboCasa365 lives in the worktree only** (`examples/simBenchmarks/Robocasa_365`, its
launchers, and its checkpoints under `playground/Checkpoints/ervla_robocasa365_*`). Submitting
its launchers from the live tree fails with `Unable to open file`. Fixes that apply to both
trees must be committed in both until the branch is merged.

## Cluster failure signatures that are NOT our code

| Signature | Meaning | Action |
|---|---|---|
| `CANCELLED by 0`, no log file, ~2 min elapsed | admin/node failure during CONFIGURING | resubmit as-is |
| `EGL_NOT_INITIALIZED`, or `Aborted` in `mjr_readPixels` | degraded render node | resubmit; it lands elsewhere |
| `EOFError` from a vector-env worker pipe | a sim worker was killed (usually VRAM) | lower `n_envs` |

Dependent jobs queued with `afterok` are cancelled when their parent dies — resubmit the
parent *and* its evals.

## Sim evaluation

MuJoCo renders on the GPU via EGL inside the Apptainer images; do not fall back to OSMesa.
The **login node's GPU is degraded** — sim aborts there within a few hundred `mjr_readPixels`
calls, so run sim work on a compute node. See the operating notes in JOBS.md for RoboCasa365
`n_envs` settings and the server seed contract.

"""
Standalone open-loop eval that loads training data from a config YAML,
sends inference requests to a running model server, and saves trajectory
comparison plots for up to N trajectories per subdataset.

Output is saved under:
    <run_root_dir>/<run_id>/open_loop_eval_train/
(i.e. alongside training checkpoints, mirroring the in-training eval convention)
Override with --output_dir if needed.

Usage:
    python -m starVLA.training.open_loop_eval_server \
        --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
        --host 127.0.0.1 --port 10093 \
        --num_traj 5
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sample_to_server_format(sample: dict) -> dict:
    """Convert a dataset sample to a msgpack-serializable dict for the server.

    dataset.__getitem__ returns images as PIL Images; msgpack cannot serialize
    them, so we convert to uint8 numpy arrays here (same representation the
    env-side evals use).
    """
    images = sample["image"]
    np_images = [
        np.array(img, dtype=np.uint8) if isinstance(img, Image.Image) else img
        for img in images
    ]
    out = {k: v for k, v in sample.items() if k != "image"}
    out["image"] = np_images
    return out


def _select_trajectories(mixture_dataset, num_traj: int):
    """Return list of (dataset_name, dataset, [trajectory_id, ...]) tuples.

    Picks the first ``num_traj`` trajectories from each subdataset so the
    selection is deterministic and comes from the training split.
    """
    selected = []
    for dataset in getattr(mixture_dataset, "datasets", []):
        traj_ids = list(getattr(dataset, "trajectory_ids", []))
        dataset_name = getattr(dataset, "dataset_name", getattr(dataset, "_dataset_name", "dataset"))
        if not traj_ids:
            continue
        chosen = traj_ids[: num_traj]
        selected.append((dataset_name, dataset, chosen))
    return selected


def _plot_trajectory(gt_traj: np.ndarray, pred_traj: np.ndarray, save_path: Path, title: str) -> None:
    action_dim = gt_traj.shape[-1]
    ncols = min(4, action_dim)
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for dim_idx in range(action_dim):
        ax = axes_flat[dim_idx]
        ax.plot(gt_traj[:, dim_idx], label="gt", linewidth=2)
        ax.plot(pred_traj[:, dim_idx], label="pred", linewidth=1.5)
        ax.set_title(f"action_dim_{dim_idx}")
        ax.set_xlabel("timestep")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[action_dim:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# core eval
# ---------------------------------------------------------------------------

def run_open_loop_eval(
    cfg,
    host: str,
    port: int,
    num_traj: int,
    output_dir: Path,
    unnorm_key: str | None,
    use_ddim: bool,
    num_ddim_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train dataset from config
    print("Building train dataset from config ...")
    train_dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train")

    action_horizon = int(getattr(cfg.framework.action_model, "action_horizon", 1))
    print(f"Action horizon: {action_horizon}")

    # Connect to server
    print(f"Connecting to model server at ws://{host}:{port} ...")
    client = WebsocketClientPolicy(host=host, port=port)
    meta = client.get_server_metadata()
    print(f"Server metadata: {meta}")

    # Build GT unnormalizer from the server's own ckpt path so GT and pred
    # are compared in the same real-scale space. The server returns unnormalized
    # actions; dataset samples carry normalized actions (training-time scale).
    ckpt_path = meta.get("ckpt_path")
    if ckpt_path is None:
        raise RuntimeError("Server metadata missing 'ckpt_path' — cannot build GT unnormalizer.")
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    gt_unnormalizer = PolicyNormProcessor(ckpt_path=ckpt_path, unnorm_key=unnorm_key)
    print(f"GT unnormalizer ready (unnorm_key={gt_unnormalizer.unnorm_key})")

    trajectories = _select_trajectories(train_dataset, num_traj)
    if not trajectories:
        print("No trajectories found in dataset.")
        return

    all_results = {}

    for dataset_name, dataset, traj_ids in trajectories:
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Dataset: {dataset_name} | evaluating {len(traj_ids)} trajectories ===")

        dataset_results = []

        for trajectory_id in traj_ids:
            traj_idx = dataset.get_trajectory_index(trajectory_id)
            trajectory_length = int(dataset.trajectory_lengths[traj_idx])

            # Build step -> sample_index map
            trajectory_step_to_index = {
                base_index: sample_index
                for sample_index, (traj_id, base_index) in enumerate(dataset.all_steps)
                if traj_id == trajectory_id
            }

            gt_chunks, pred_chunks = [], []

            for step in tqdm(
                range(0, trajectory_length, action_horizon),
                desc=f"traj {trajectory_id}",
                leave=False,
            ):
                if step not in trajectory_step_to_index:
                    print(f"  Warning: step {step} missing from all_steps for traj {trajectory_id}, skipping.")
                    continue

                sample = dataset[trajectory_step_to_index[step]]
                # dataset returns normalized actions; unnormalize to real scale
                # so GT and server predictions are in the same space
                gt_actions = gt_unnormalizer.unapply_actions(
                    np.asarray(sample["action"], dtype=np.float32)
                )

                vla_input = {
                    "examples": [_sample_to_server_format(sample)],
                    "unnorm_key": unnorm_key,
                    "do_sample": False,
                    "use_ddim": use_ddim,
                    "num_ddim_steps": num_ddim_steps,
                }
                response = client.predict_action(vla_input)
                try:
                    actions_batch = response["data"]["actions"]  # (B, T, D) unnormalized
                except KeyError:
                    raise KeyError(
                        f"Key 'actions' not found in server response. Keys: {list(response.get('data', {}).keys())}"
                    )
                pred_actions = np.asarray(actions_batch)[0]  # (T, D)

                valid_horizon = min(action_horizon, trajectory_length - step)
                gt_chunks.append(gt_actions[:valid_horizon])
                pred_chunks.append(pred_actions[:valid_horizon])

            if not gt_chunks:
                continue

            gt_traj = np.concatenate(gt_chunks, axis=0)
            pred_traj = np.concatenate(pred_chunks, axis=0)

            squared_error = (pred_traj - gt_traj) ** 2
            per_dim_mse = squared_error.mean(axis=0)
            mse_mean = float(per_dim_mse.mean())

            print(f"  traj {trajectory_id}: MSE={mse_mean:.6f}, length={gt_traj.shape[0]}")

            plot_path = dataset_dir / f"traj_{trajectory_id}.png"
            _plot_trajectory(
                gt_traj=gt_traj,
                pred_traj=pred_traj,
                save_path=plot_path,
                title=f"{dataset_name} / traj {trajectory_id}",
            )

            traj_result = {
                "trajectory_id": int(trajectory_id),
                "trajectory_length": int(gt_traj.shape[0]),
                "mse_mean": mse_mean,
                "mse_per_dim": per_dim_mse.tolist(),
                "plot": str(plot_path),
            }
            dataset_results.append(traj_result)

            # Save per-trajectory arrays
            # np_path = dataset_dir / f"traj_{trajectory_id}.npz"
            # np.savez_compressed(np_path, gt=gt_traj, pred=pred_traj)

        all_results[dataset_name] = dataset_results

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print overall stats
    all_mse = [r["mse_mean"] for res_list in all_results.values() for r in res_list]
    if all_mse:
        print(f"Overall mean MSE across {len(all_mse)} trajectories: {np.mean(all_mse):.6f}")

    client.close()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _resolve_output_dir(cfg, override: str | None) -> Path:
    """Derive output dir from config (mirrors training convention) unless overridden."""
    if override:
        return Path(override)
    # Same path construction as setup_directories() in train_starvla.py
    run_root = getattr(cfg, "run_root_dir", None)
    run_id = getattr(cfg, "run_id", None)
    if run_root and run_id:
        return Path(run_root) / run_id / "open_loop_eval_train"
    # Fallback if config doesn't have these keys
    return Path("results/open_loop_eval_train")


def main():
    parser = argparse.ArgumentParser(description="Open-loop eval via model server")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to training YAML config")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Model server host")
    parser.add_argument("--port", type=int, default=10093, help="Model server port")
    parser.add_argument("--num_traj", type=int, default=5, help="Number of trajectories per subdataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory. Defaults to <run_root_dir>/<run_id>/open_loop_eval_train/",
    )
    parser.add_argument("--unnorm_key", type=str, default=None, help="Unnorm key for the server")
    parser.add_argument("--no_ddim", action="store_true", help="Disable DDIM sampling")
    parser.add_argument("--num_ddim_steps", type=int, default=20, help="Number of DDIM steps")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    if dotlist:
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)

    output_dir = _resolve_output_dir(cfg, args.output_dir)
    print(f"Output directory: {output_dir}")

    run_open_loop_eval(
        cfg=cfg,
        host=args.host,
        port=args.port,
        num_traj=args.num_traj,
        output_dir=output_dir,
        unnorm_key=args.unnorm_key,
        use_ddim=not args.no_ddim,
        num_ddim_steps=args.num_ddim_steps,
    )


if __name__ == "__main__":
    main()

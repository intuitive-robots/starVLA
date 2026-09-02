# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import inspect
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf
import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    DATASET_NAMED_MIXTURES,
    EmbodimentTag,
)

def collate_fn(batch):
    """Collate samples and apply CoT dropout at local-batch granularity.

    CoT-capable distributed runs must execute the decoder on every rank.  Independent
    sample-side dropout could remove every target from a rank (especially in an epoch's
    short final batch), giving DeepSpeed different parameter graphs and deadlocking its
    gradient collectives.  Keeping resolution/augmentation in ``__getitem__`` but sampling
    the keep mask here preserves worker-side overlap and lets us enforce one target per
    local batch.
    """
    if not batch:
        return batch

    enabled = [bool(sample.pop("_cot_dropout_enabled", False)) for sample in batch]
    rates = [float(sample.pop("_cot_dropout_rate", 0.0)) for sample in batch]
    if not any(enabled):
        return batch
    if not all(enabled):
        raise RuntimeError("mixed CoT-dropout settings in one local batch")
    if not np.allclose(rates, rates[0]):
        raise RuntimeError(f"mixed CoT-dropout rates in one local batch: {rates}")

    candidates = [
        i for i, sample in enumerate(batch)
        if sample.get("cot_available", False) and sample.get("cot_conversation") is not None
    ]
    if not candidates:
        raise RuntimeError(
            "CoT-enabled local training batch contains no mapped targets; refusing to "
            "enter a rank-dependent decoder graph"
        )

    keep = {i for i in candidates if np.random.random() >= rates[0]}
    if not keep:
        # Conditioning on a nonempty set changes the requested 0.5 keep rate by only the
        # all-dropped probability (2^-16 for the current micro-batch), while guaranteeing
        # identical decoder participation across ranks.
        keep.add(candidates[int(np.random.randint(len(candidates)))])

    for i in candidates:
        if i in keep:
            batch[i]["cot_mode"] = "cot"
        else:
            batch[i]["cot_conversation"] = None
            batch[i]["cot_mode"] = "no_cot"
    return batch


def _apply_trajectory_split(
    dataset: LeRobotSingleDataset,
    split: str,
    holdout_trajectories_per_dataset: int = 1,
    holdout_trajectories_per_task: int = 0,
) -> LeRobotSingleDataset:
    """Apply a deterministic per-dataset trajectory split in-place."""
    trajectory_ids = np.asarray(dataset.trajectory_ids)
    trajectory_lengths = np.asarray(dataset.trajectory_lengths)
    num_trajectories = len(trajectory_ids)

    # Consolidated LeRobot-v3 releases may contain many benchmark tasks in one
    # dataset directory.  Holding out only the last N trajectories would then
    # evaluate one task and train on all examples of the other tasks.  This opt-in
    # path recreates StarVLA's legacy one-directory-per-task split exactly.
    if holdout_trajectories_per_task > 0 and num_trajectories:
        grouped: dict[str, list[int]] = {}
        for trajectory_id in trajectory_ids:
            metadata = dataset.trajectory_ids_to_metadata.get(int(trajectory_id), {})
            task_name = metadata.get("task_name")
            if task_name is None:
                raise ValueError(
                    f"holdout_trajectories_per_task requires task_name episode metadata; "
                    f"missing for {dataset.dataset_name} episode {trajectory_id}"
                )
            grouped.setdefault(str(task_name), []).append(int(trajectory_id))
        held_out: set[int] = set()
        for task_name, task_ids in grouped.items():
            task_ids.sort()
            if len(task_ids) <= holdout_trajectories_per_task:
                raise ValueError(
                    f"task {task_name!r} has only {len(task_ids)} trajectories, cannot "
                    f"hold out {holdout_trajectories_per_task}"
                )
            held_out.update(task_ids[-holdout_trajectories_per_task:])
        select_holdout = split == "eval"
        mask = np.asarray(
            [(int(trajectory_id) in held_out) == select_holdout for trajectory_id in trajectory_ids],
            dtype=bool,
        )
        selected_ids = trajectory_ids[mask]
        selected_lengths = trajectory_lengths[mask]
    elif holdout_trajectories_per_dataset <= 0 or num_trajectories == 0:
        return dataset
    elif num_trajectories <= holdout_trajectories_per_dataset:
        if split == "eval":
            print(
                f"Warning: Dataset {dataset.dataset_name} has only {num_trajectories} trajectories; "
                "skipping eval holdout for this dataset."
            )
            selected_ids = trajectory_ids[:0]
            selected_lengths = trajectory_lengths[:0]
        else:
            selected_ids = trajectory_ids
            selected_lengths = trajectory_lengths
    else:
        split_index = num_trajectories - holdout_trajectories_per_dataset
        if split == "eval":
            selected_ids = trajectory_ids[split_index:]
            selected_lengths = trajectory_lengths[split_index:]
        else:
            selected_ids = trajectory_ids[:split_index]
            selected_lengths = trajectory_lengths[:split_index]

    dataset._trajectory_ids = np.asarray(selected_ids)
    dataset._trajectory_lengths = np.asarray(selected_lengths)
    dataset._refresh_trajectory_index_cache()
    dataset._all_steps = dataset._get_all_steps_single_process()
    dataset._valid_base_indices_by_trajectory = dataset._build_valid_base_indices_by_trajectory()
    dataset._valid_trajectory_ids = np.array(list(dataset._valid_base_indices_by_trajectory.keys()))
    dataset._valid_trajectory_lengths = np.array(
        [len(indices) for indices in dataset._valid_base_indices_by_trajectory.values()]
    )
    return dataset

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    default_data_cfg = getattr(data_config, "default_data_cfg", None)
    if data_cfg is None:
        data_cfg = {}
    elif not isinstance(data_cfg, dict):
        data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
    if default_data_cfg:
        merged_data_cfg = dict(default_data_cfg)
        merged_data_cfg.update(data_cfg)
        data_cfg = merged_data_cfg
    action_horizon_override = data_cfg.get("action_horizon") if data_cfg else None
    if action_horizon_override and "action_horizon" in inspect.signature(data_config.modality_config).parameters:
        modality_config = data_config.modality_config(action_horizon=int(action_horizon_override))
    else:
        modality_config = data_config.modality_config()
    # Benchmark transforms may use dataset-level options (for example LIBERO's opt-in
    # photometric/crop augmentation). Keep backward compatibility with configs whose
    # transform() takes no arguments.
    if "data_cfg" in inspect.signature(data_config.transform).parameters:
        transforms = data_config.transform(data_cfg=data_cfg)
    else:
        transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    holdout_trajectories_per_dataset = int(data_cfg.get("holdout_trajectories_per_dataset", 1))
    holdout_trajectories_per_task = int(data_cfg.get("holdout_trajectories_per_task", 0))
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )
        if mode == "train":
            dataset.transforms.train()
        else:
            dataset.transforms.eval()
        if mode in {"train", "eval"}:
            dataset = _apply_trajectory_split(
                dataset,
                split=mode,
                holdout_trajectories_per_dataset=holdout_trajectories_per_dataset,
                holdout_trajectories_per_task=holdout_trajectories_per_task,
            )
        if len(dataset) == 0:
            print(f"Skipping {mode} dataset {dataset.dataset_name} because the split is empty.")
            continue
        dataset_mixture.append((dataset, d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    #args.config_yaml = "/e/home/jusers/blank4/jupiter/blank4/code/starVLA/examples/LIBERO/train_files/starvla_real_robot_qwen_08_base.yaml"

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 100:
            break
        count += 1
        pass

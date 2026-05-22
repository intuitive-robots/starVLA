# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

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
    return batch


def _apply_trajectory_split(
    dataset: LeRobotSingleDataset,
    split: str,
    holdout_trajectories_per_dataset: int = 1,
) -> LeRobotSingleDataset:
    """Apply a deterministic per-dataset trajectory split in-place."""
    trajectory_ids = np.asarray(dataset.trajectory_ids)
    trajectory_lengths = np.asarray(dataset.trajectory_lengths)
    num_trajectories = len(trajectory_ids)

    if holdout_trajectories_per_dataset <= 0 or num_trajectories == 0:
        return dataset

    if num_trajectories <= holdout_trajectories_per_dataset:
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
    modality_config = data_config.modality_config()
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
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )
        if mode in {"train", "eval"}:
            dataset = _apply_trajectory_split(
                dataset,
                split=mode,
                holdout_trajectories_per_dataset=holdout_trajectories_per_dataset,
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

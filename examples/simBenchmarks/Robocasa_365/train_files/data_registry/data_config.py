"""RoboCasa365 (PandaOmron, single-arm Franka) — data config, embodiment tags, and mixtures.

Loaded automatically by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``.

Schema follows the official robocasa LeRobot conversion (see
``robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py``):

* observation.state (16d): base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)
* action (12d): eef_pos(3) + eef_rot(3) + gripper_close(1) + base_motion(4) + control_mode(1)
* 3 cameras at 256x256, 20 fps. We only use ``robot0_agentview_left`` for training.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class PandaOmronRoboCasa365DataConfig:
    """Single-arm Franka PandaOmron used by upstream RoboCasa365."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand"]
    state_keys = [
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    # Per-key dims for PolicyNormProcessor
    # action (12-D): eef_pos(3) + eef_rot(3) + gripper_close(1) + base_motion(4) + control_mode(1)
    action_key_dims = {
        "action.end_effector_position": 3,
        "action.end_effector_rotation": 3,
        "action.gripper_close": 1,
        "action.base_motion": 4,
        "action.control_mode": 1,
    }
    # state (16-D): base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)
    state_key_dims = {
        "state.base_position": 3,
        "state.base_rotation": 4,
        "state.end_effector_position_relative": 3,
        "state.end_effector_rotation_relative": 4,
        "state.gripper_qpos": 2,
    }
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        import warnings

        from starVLA.dataloader.cot_augmentation import CoTVideoAugment

        data_cfg = data_cfg or {}
        mode = str(data_cfg.get("augmentation", "none")).lower()
        if mode not in {"none", "photometric", "crop_photometric"}:
            raise ValueError(f"Unsupported RoboCasa augmentation: {mode}")
        # Versioned opt-in: historical configs carried an ignored augmentation flag.
        # Do not silently change a resumed control's inputs. New matched arms must
        # all set this flag, including the control without native auxiliary labels.
        transforms = []
        enabled = bool(data_cfg.get("robocasa_joint_augmentation", False))
        if data_cfg.get("native_supervision") and mode != "none" and not enabled:
            raise ValueError("Native augmentation requires robocasa_joint_augmentation: true")
        if mode != "none" and not enabled:
            warnings.warn(
                "RoboCasa augmentation is configured but remains disabled for historical "
                "checkpoint compatibility; set robocasa_joint_augmentation: true in every "
                "matched new arm to enable it.",
                stacklevel=2,
            )
        if enabled and mode != "none":
            transforms.append(CoTVideoAugment(apply_to=self.video_keys, mode=mode))
        return ComposedModalityTransform(transforms=transforms + [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])


class PandaOmronRoboCasa365SingleCamDataConfig(PandaOmronRoboCasa365DataConfig):
    """Single-view variant that matches what the evaluation bridge actually sends.

    ``model2robocasa365_interface.PolicyWarper`` reads only
    ``video.robot0_agentview_left`` from the environment observation. A policy
    trained on the three-camera config above therefore sees a different number of
    images at evaluation time than it did during training, which silently
    degrades the rollout instead of raising. Train against this config whenever
    the checked-in bridge is used unmodified; use the three-camera config only
    after extending the bridge to send all three views.
    """

    video_keys = ["video.robot0_agentview_left"]


class PandaOmronRoboCasa365TwoCamDataConfig(PandaOmronRoboCasa365DataConfig):
    """Left agentview + wrist camera.

    The eye-in-hand view carries the contact and grasp-alignment detail that a
    fixed agentview cannot resolve, and 14 of the 18 Atomic-Seen tasks are
    manipulation rather than navigation. The environment publishes all three
    views (see PandaOmron_modality.json), so this only requires the evaluation
    bridge to forward both, which model2robocasa365_interface now does.

    Key order here is the contract: the loader stacks frames in video_keys order
    and the bridge must send them in the same order.
    """

    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_eye_in_hand",
    ]


ROBOT_TYPE_CONFIG_MAP = {
    "panda_omron_robocasa365": PandaOmronRoboCasa365DataConfig(),
    "panda_omron_robocasa365_1cam": PandaOmronRoboCasa365SingleCamDataConfig(),
    "panda_omron_robocasa365_2cam": PandaOmronRoboCasa365TwoCamDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}

# Each task lives at
#   ``playground/Datasets/robocasa365/<RELATIVE_PATH>/lerobot``
# where RELATIVE_PATH below is exactly what robocasa's upstream
# ``ATOMIC_TASK_DATASETS[name]['target']['human_path']`` (or the composite
# equivalent) returns. Lists below are a one-shot snapshot of those tables for
# every task that ships a ``target/human`` LeRobot bundle (50 tasks total: 18
# atomic + 32 composite). Re-generate via:
#
#   python -m robocasa.utils.dataset_registry  # has constants
#   # or run the helper at examples/simBenchmarks/Robocasa_365/train_files/dump_target_human_paths.py
_ROBOT_TAG = "panda_omron_robocasa365"
_ROBOT_TAG_1CAM = "panda_omron_robocasa365_1cam"
_ROBOT_TAG_2CAM = "panda_omron_robocasa365_2cam"

# Atomic single-skill tasks (target/human split, 18 tasks).
_TARGET_HUMAN_ATOMIC = {
    "CloseBlenderLid":           "v1.0/target/atomic/CloseBlenderLid/20250822",
    "CloseFridge":               "v1.0/target/atomic/CloseFridge/20250816",
    "CloseToasterOvenDoor":      "v1.0/target/atomic/CloseToasterOvenDoor/20250818",
    "CoffeeSetupMug":            "v1.0/target/atomic/CoffeeSetupMug/20250813",
    "NavigateKitchen":           "v1.0/target/atomic/NavigateKitchen/20250821",
    "OpenCabinet":               "v1.0/target/atomic/OpenCabinet/20250813",
    "OpenDrawer":                "v1.0/target/atomic/OpenDrawer/20250816",
    "OpenStandMixerHead":        "v1.0/target/atomic/OpenStandMixerHead/20250818",
    "PickPlaceCounterToCabinet": "v1.0/target/atomic/PickPlaceCounterToCabinet/20250811",
    "PickPlaceCounterToStove":   "v1.0/target/atomic/PickPlaceCounterToStove/20250818",
    "PickPlaceDrawerToCounter":  "v1.0/target/atomic/PickPlaceDrawerToCounter/20250820",
    "PickPlaceSinkToCounter":    "v1.0/target/atomic/PickPlaceSinkToCounter/20250813",
    "PickPlaceToasterToCounter": "v1.0/target/atomic/PickPlaceToasterToCounter/20250817",
    "SlideDishwasherRack":       "v1.0/target/atomic/SlideDishwasherRack/20250820",
    "TurnOffStove":              "v1.0/target/atomic/TurnOffStove/20250812",
    "TurnOnElectricKettle":      "v1.0/target/atomic/TurnOnElectricKettle/20250817",
    "TurnOnMicrowave":           "v1.0/target/atomic/TurnOnMicrowave/20250813",
    "TurnOnSinkFaucet":          "v1.0/target/atomic/TurnOnSinkFaucet/20250812",
}

# Composite multi-step tasks (target/human split, 32 tasks).
# Includes both ``composite_seen`` (16) and ``composite_unseen`` (16) — these
# are *training* data; ``unseen`` only refers to the eval task list.
_TARGET_HUMAN_COMPOSITE = {
    "ArrangeBreadBasket":   "v1.0/target/composite/ArrangeBreadBasket/20250809",
    "ArrangeTea":           "v1.0/target/composite/ArrangeTea/20250812",
    "BreadSelection":       "v1.0/target/composite/BreadSelection/20250815",
    "CategorizeCondiments": "v1.0/target/composite/CategorizeCondiments/20250814",
    "CuttingToolSelection": "v1.0/target/composite/CuttingToolSelection/20250814",
    "DeliverStraw":         "v1.0/target/composite/DeliverStraw/20250813",
    "GarnishPancake":       "v1.0/target/composite/GarnishPancake/20250815",
    "GatherTableware":      "v1.0/target/composite/GatherTableware/20250815",
    "GetToastedBread":      "v1.0/target/composite/GetToastedBread/20250812",
    "HeatKebabSandwich":    "v1.0/target/composite/HeatKebabSandwich/20250813",
    "KettleBoiling":        "v1.0/target/composite/KettleBoiling/20250814",
    "LoadDishwasher":       "v1.0/target/composite/LoadDishwasher/20250811",
    "MakeIceLemonade":      "v1.0/target/composite/MakeIceLemonade/20250813",
    "PackIdenticalLunches": "v1.0/target/composite/PackIdenticalLunches/20250815",
    "PanTransfer":          "v1.0/target/composite/PanTransfer/20250817",
    "PortionHotDogs":       "v1.0/target/composite/PortionHotDogs/20250816",
    "PreSoakPan":           "v1.0/target/composite/PreSoakPan/20250809",
    "PrepareCoffee":        "v1.0/target/composite/PrepareCoffee/20250812",
    "RecycleBottlesByType": "v1.0/target/composite/RecycleBottlesByType/20250812",
    "RinseSinkBasin":       "v1.0/target/composite/RinseSinkBasin/20250816",
    "ScrubCuttingBoard":    "v1.0/target/composite/ScrubCuttingBoard/20250816",
    "SearingMeat":          "v1.0/target/composite/SearingMeat/20250812",
    "SeparateFreezerRack":  "v1.0/target/composite/SeparateFreezerRack/20250815",
    "SetUpCuttingStation":  "v1.0/target/composite/SetUpCuttingStation/20250817",
    "StackBowlsCabinet":    "v1.0/target/composite/StackBowlsCabinet/20250815",
    "SteamInMicrowave":     "v1.0/target/composite/SteamInMicrowave/20250814",
    "StirVegetables":       "v1.0/target/composite/StirVegetables/20250814",
    "StoreLeftoversInBowl": "v1.0/target/composite/StoreLeftoversInBowl/20250813",
    "WaffleReheat":         "v1.0/target/composite/WaffleReheat/20250817",
    "WashFruitColander":    "v1.0/target/composite/WashFruitColander/20250811",
    "WashLettuce":          "v1.0/target/composite/WashLettuce/20250814",
    "WeighIngredients":     "v1.0/target/composite/WeighIngredients/20250812",
}


def _entries(path_dict):
    """Build mixture entries (relpath/lerobot, weight=1.0, robot_tag) from a path dict."""
    return [(f"{p}/lerobot", 1.0, _ROBOT_TAG) for p in path_dict.values()]


# ---------------------------------------------------------------------------
# LeRobot v3.0 mirrors (ember-lab-berkeley on HuggingFace)
# ---------------------------------------------------------------------------
# The official robocasa Box tarballs are LeRobot **v2.1**, one directory per
# task (the _TARGET_HUMAN_* tables above). ember-lab-berkeley publishes the same
# demonstrations re-chunked as LeRobot **v3.0**, one repo per benchmark group:
#
#   ember-lab-berkeley/robocasa365-target-atomic            6.5 GB   9126 eps
#   ember-lab-berkeley/robocasa365-target-composite-seen   18.1 GB   8077 eps
#   ember-lab-berkeley/robocasa365-target-composite-unseen 20.3 GB   8104 eps
#
# Each repo has data/ meta/ videos/ at its top level, so the mixture entry is the
# bare directory name and ``data_root_dir`` must point at its PARENT. Set
# ``lerobot_version: "v3.0"`` on the dataset config when using these.
#
# These mirrors do NOT ship ``meta/modality.json``; copy the official
# ``robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json`` into
# each dataset's ``meta/`` or the state/action keys above cannot be resolved.
_V3_MIRRORS = {
    "robocasa365_v3_target_atomic":           "robocasa365_target_atomic",
    "robocasa365_v3_target_composite_seen":   "robocasa365_target_composite_seen",
    "robocasa365_v3_target_composite_unseen": "robocasa365_target_composite_unseen",
}


DATASET_NAMED_MIXTURES = {
    # ------- minimal walk-through mixture (1 atomic task) -------
    "robocasa365_open_drawer_target_human": [
        ("v1.0/target/atomic/OpenDrawer/20250816/lerobot", 1.0, _ROBOT_TAG),
    ],
    # ------- full mixtures (each task weighted 1.0; equal sampling per task) -------
    "robocasa365_atomic_target_human_all":    _entries(_TARGET_HUMAN_ATOMIC),
    "robocasa365_composite_target_human_all": _entries(_TARGET_HUMAN_COMPOSITE),
    "robocasa365_target_human_all":           _entries({**_TARGET_HUMAN_ATOMIC,
                                                       **_TARGET_HUMAN_COMPOSITE}),
    # ------- LeRobot v3.0 mirrors (one merged dataset per benchmark group) -------
    **{name: [(directory, 1.0, _ROBOT_TAG)] for name, directory in _V3_MIRRORS.items()},
    # Same data, single-camera contract (left agentview only).
    **{f"{name}_1cam": [(directory, 1.0, _ROBOT_TAG_1CAM)]
       for name, directory in _V3_MIRRORS.items()},
    # Left agentview + wrist. Requires the two-camera eval bridge.
    **{f"{name}_2cam": [(directory, 1.0, _ROBOT_TAG_2CAM)]
       for name, directory in _V3_MIRRORS.items()},
    # "seen-34" = 18 atomic + 16 composite-seen, the scope StarVLA-PI and
    # StarVLA-GR00T were trained on in the archived snapshot. It holds out the
    # 16 composite-unseen tasks, which the target-50 mixtures below do not.
    # Only expressible on the v3.0 mirrors: the per-task v2.1 tables above lump
    # composite-seen and composite-unseen into one dict.
    "robocasa365_v3_seen34": [
        ("robocasa365_target_atomic", 1.0, _ROBOT_TAG),
        ("robocasa365_target_composite_seen", 1.0, _ROBOT_TAG),
    ],
    "robocasa365_v3_seen34_1cam": [
        ("robocasa365_target_atomic", 1.0, _ROBOT_TAG_1CAM),
        ("robocasa365_target_composite_seen", 1.0, _ROBOT_TAG_1CAM),
    ],
    # All 50 target tasks in v3.0, sampled evenly across the three groups.
    "robocasa365_v3_target_all": [
        (directory, 1.0, _ROBOT_TAG) for directory in _V3_MIRRORS.values()
    ],
}

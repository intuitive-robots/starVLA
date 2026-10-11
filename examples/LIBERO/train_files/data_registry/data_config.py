"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

import os

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform, StateBinarize
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoRandomRotation,
    VideoToNumpy,
    VideoToTensor,
)
from starVLA.dataloader.cot_augmentation import AnnotVideoAugment, CoTVideoAugment
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self, action_horizon: int | None = None):
        # action_horizon lets the ground-truth action chunk length track
        # framework.action_model.action_horizon (passed via datasets.vla_data.action_horizon);
        # otherwise falls back to the class default of 8.
        action_indices = list(range(action_horizon)) if action_horizon else self.action_indices
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    # Gripper normalization mode. IMPORTANT: this transform is rebuilt from CODE at
    # inference time (deployment/model_server/policy_norm_processor.py fetches this
    # DataConfig and calls .transform()), but the checkpoint does NOT record which
    # mode it was trained with. So changing the default silently corrupts every
    # checkpoint trained under the old default: the server would un-normalize with a
    # rule the model never saw.
    #
    # Pick per checkpoint via STARVLA_LIBERO_GRIPPER_NORM:
    #   "none"          -> gripper left un-normalized (checkpoints whose
    #                      dataset_statistics.json has action.min[6] == 0, i.e. data
    #                      already stored gripper as {0,1}; e.g. 1229_libero4in1_qwen35oft)
    #   "binary_invert" -> {-1,+1} raw LIBERO -> {1=open, 0=close} target, invertible
    #                      back to raw (checkpoints with action.min[6] == -1)
    # auto_eval_libero.sh sets this from the checkpoint's stats file.
    GRIPPER_NORM_MODE = os.environ.get("STARVLA_LIBERO_GRIPPER_NORM", "binary_invert")

    def transform(self, data_cfg: dict | None = None):
        """Build LIBERO preprocessing with an explicitly selected augmentation policy.

        ``augmentation`` is deliberately opt-in so old checkpoints and the ERVLA A--E
        comparison retain their original recipe. Supported values:

        * ``none``: historical LIBERO preprocessing.
        * ``photometric``: coordinate-safe color jitter; valid with CoT supervision.
        * ``crop_photometric``: π0.5 recipe: 0.95 random crop, resize back to 256,
          rotation in [-5°, 5°], then brightness/contrast/saturation jitter. CoT datasets
          use the joint worker transform so all coordinate fields follow the geometry.
        """
        data_cfg = data_cfg or {}
        augmentation = str(data_cfg.get("augmentation", "none")).lower()
        allowed = {"none", "photometric", "crop_photometric"}
        if augmentation not in allowed:
            raise ValueError(
                f"unknown LIBERO augmentation {augmentation!r}; expected one of {sorted(allowed)}")
        has_cot = bool(data_cfg.get("cot"))
        has_annot = bool(data_cfg.get("annot", False))
        if has_annot and has_cot:
            raise ValueError("datasets.vla_data.annot and cot cannot be combined (both own the joint image geometry)")

        transforms = []
        if augmentation != "none" and has_annot:
            # Simulator annotations (masks, 2D boxes) must follow the image geometry: one sampled crop and
            # angle for every camera and the annotation, same recipe and parameters as below (2026-10-03).
            transforms.append(AnnotVideoAugment(
                apply_to=self.video_keys, mode=augmentation, crop_scale=0.95))
        elif augmentation != "none" and has_cot:
            # The dataset attaches the resolved conversation before transforms run, letting
            # workers overlap image augmentation and coordinate rewriting with GPU compute.
            transforms.append(CoTVideoAugment(
                apply_to=self.video_keys, mode=augmentation, crop_scale=0.95))
        elif augmentation != "none":
            transforms.append(VideoToTensor(apply_to=self.video_keys))
            if augmentation == "crop_photometric":
                # LIBERO training videos are 256x256. Resize restores the model-facing
                # resolution after the crop, making this a translation/zoom perturbation.
                transforms.extend([
                    VideoCrop(apply_to=self.video_keys, scale=0.95),
                    VideoResize(
                        apply_to=self.video_keys,
                        height=256,
                        width=256,
                        interpolation="linear",
                    ),
                    VideoRandomRotation(
                        apply_to=self.video_keys,
                        degrees=(-5.0, 5.0),
                        interpolation="linear",
                    ),
                ])
            transforms.extend([
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.0,
                ),
                VideoToNumpy(apply_to=self.video_keys),
            ])

        # Opt-in q99 state normalization (``datasets.vla_data.state_normalization: q99``).
        # Default stays raw: every LIBERO ladder checkpoint was trained on raw state, which
        # the state-token quantizer clips to [-1, 1] (roll collapses to one bin). The base
        # Libero4in1DataConfig's q99 switch is shadowed by this registry class.
        state_normalization = data_cfg.get("state_normalization", None)
        if state_normalization in ("", "none", "raw", False):
            state_normalization = None
        if state_normalization not in (None, "q99"):
            raise ValueError(f"LIBERO state_normalization must be raw or 'q99', got {state_normalization!r}")
        # 2026-10-09: optional open / closed gripper (``state_gripper_binary: true``, q99 only): both finger joints
        # snapped to -1 / +1 after normalisation. The state-resolution probe showed q99 policies use the precise
        # finger opening as a replay cue; raw-state policies see it in only ~6 bins.
        gripper_binary = bool(data_cfg.get("state_gripper_binary", False))
        if gripper_binary and state_normalization != "q99":
            raise ValueError("LIBERO state_gripper_binary needs state_normalization: q99")
        if state_normalization == "q99":
            transforms.extend([
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(apply_to=self.state_keys,
                                     normalization_modes={key: "q99" for key in self.state_keys}),
            ])
            if gripper_binary:
                transforms.append(StateBinarize(apply_to=["state.pad", "state.gripper"]))

        normalization_modes = {
            "action.x": "min_max",
            "action.y": "min_max",
            "action.z": "min_max",
            "action.roll": "min_max",
            "action.pitch": "min_max",
            "action.yaw": "min_max",
        }
        if self.GRIPPER_NORM_MODE != "none":
            normalization_modes["action.gripper"] = self.GRIPPER_NORM_MODE

        transforms.extend([
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=normalization_modes,
            ),
        ])
        return ComposedModalityTransform(transforms=transforms)


class Libero4in1PrimaryOnlyDataConfig(Libero4in1DataConfig):
    """Single-view LIBERO: agent-view camera only, no wrist camera (rebuttal H-cam).

    Identical to ``Libero4in1DataConfig`` except ``video_keys``: same state/action keys,
    same transforms and the same FRANKA embodiment tag, so dataset_statistics.json keeps
    its ``franka`` key. Selected with ``datasets.vla_data.data_mix: libero_mixed_primary``.

    ``serve_image_indices`` is the serving contract: the policy server
    (deployment/model_server/policy_wrapper.py) keeps only these positions of the
    client's image list, so an eval client that sends [agentview, wrist] feeds the model
    exactly the one view it was trained on. Absent on every other DataConfig (no-op).
    """

    video_keys = [
        "video.primary_image",
    ]
    serve_image_indices = [0]


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "libero_franka_primary": Libero4in1PrimaryOnlyDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
    # The loader (lerobot_datasets.make_LeRobotSingleDataset) still reads this map and
    # falls back to NEW_EMBODIMENT, so the single-view robot type is listed explicitly.
    "libero_franka_primary": EmbodimentTag.FRANKA,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    # Same data as the base ``libero_mixed`` (mixtures.py), agent-view camera only.
    "libero_mixed_primary": [
        ("libero", 1.0, "libero_franka_primary"),
    ],
}

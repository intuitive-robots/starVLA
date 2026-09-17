from pathlib import Path
from unittest import skipUnless

import numpy as np

from starVLA.dataloader.cot_augmentation import CoTVideoAugment
from starVLA.dataloader.robocasa_supervision import (
    RoboCasaSupervision,
    augment_native_targets,
    pack_native_supervision,
)


LABEL_ROOTS = [
    "/e/scratch/m3/blank4/rc365_supervision/labels_v1",
    "/e/scratch/m3/blank4/rc365_supervision/retry_labels_v1",
]
VIDEO_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_eye_in_hand",
]
requires_labels = skipUnless(
    Path(LABEL_ROOTS[0]).is_dir(),
    "RoboCasa supervision package is not mounted",
)


def _resolver(include_unreviewed=False):
    return RoboCasaSupervision(
        {
            "labels_roots": LABEL_ROOTS,
            "include_unreviewed_boundaries": include_unreviewed,
            "trace_subject": "gripper",
            "trace_span": "full_subtask",
        },
        "robocasa365_target_atomic",
        VIDEO_KEYS,
        horizon=16,
        future_offset=8,
    )


@requires_labels
def test_two_door_boundary_keeps_targets_inside_each_subtask():
    resolver = _resolver(include_unreviewed=True)
    before = resolver.resolve(502, 221)
    after = resolver.resolve(502, 222)

    assert (before["subtask_id"], before["subtask_start"], before["subtask_end"]) == (0, 0, 222)
    assert (after["subtask_id"], after["subtask_start"], after["subtask_end"]) == (1, 222, 429)
    assert "right_door" in before["entity"]
    assert "left_door" in after["entity"]
    assert not before["future_frame_valid"]
    assert after["future_frame_index"] == 230
    assert after["future_frame_valid"]
    assert not np.allclose(
        before["traces"]["object_full_subtask"],
        after["traces"]["object_full_subtask"],
    )


@requires_labels
def test_unreviewed_boundary_is_masked_by_default():
    record = _resolver(include_unreviewed=False).resolve(2529, 193)
    assert record["boundary_needs_review"]
    assert not record["boundary_accepted"]
    assert not any(np.asarray(value).any() for value in record["valid"].values())
    assert not any(value.any() for value in record["trace_valid"].values())
    assert not record["future_frame_valid"]


@requires_labels
def test_repaired_episode_is_present_in_canonical_root():
    resolver = _resolver(include_unreviewed=True)
    # Episode 2645 failed in the first pass and was repaired. The finalized package
    # copied it into labels_v1, so training never interprets that old failure as zero.
    assert (Path(LABEL_ROOTS[1]) / "episode_002645" / "COMPLETE.json").is_file()
    record = resolver.resolve(2645, 0)
    assert record["source_path"].startswith(LABEL_ROOTS[0])
    assert record["valid"]["phase"]


@requires_labels
def test_crop_rotation_updates_targets_and_masks_without_edge_clipping():
    record = _resolver().resolve(0, 100)
    record["targets"]["target_point"] = np.array([0.01, 0.5], dtype=np.float32)
    record["valid"]["target_point"] = True
    augmented = augment_native_targets(
        record,
        left=13,
        top=0,
        crop_width=243,
        crop_height=256,
        image_width=256,
        image_height=256,
        angle_degrees=0,
    )
    assert augmented["targets"]["target_point"][0] < 0
    assert not augmented["valid"]["target_point"]
    assert "target_point" not in _packed(augmented)["cot_structured_targets"]


@requires_labels
def test_video_augment_uses_same_geometry_for_native_targets():
    record = _resolver().resolve(0, 100)
    data = {
        VIDEO_KEYS[0]: np.zeros((1, 256, 256, 3), dtype=np.uint8),
        VIDEO_KEYS[1]: np.zeros((1, 256, 256, 3), dtype=np.uint8),
        "_cot_conversation": None,
        "_native_supervision": record,
    }
    transform = CoTVideoAugment(apply_to=VIDEO_KEYS, mode="crop_photometric")
    np.random.seed(0)
    transformed = transform(data)
    native = transformed["_native_supervision"]
    assert transformed[VIDEO_KEYS[0]].shape == (1, 256, 256, 3)
    assert set(native["augmentation"]) == {
        "left", "top", "crop_width", "crop_height",
        "image_width", "image_height", "angle_degrees",
    }
    packed = _packed(native)
    assert packed["cot_structured_targets"]["_native_robocasa"]


def _packed(record):
    sample = {"future_image": [object(), object()]}
    pack_native_supervision(sample, record, VIDEO_KEYS)
    return sample

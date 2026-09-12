import numpy as np

from shellshock.perception.world import build_world_from_image_with_diagnostics
from shellshock.perception.self_center_model import (
    crop_self_roi,
    make_self_center_label,
    map_roi_point_to_image,
    valid_roi_keypoint,
)
from shellshock.domain.world import DetectionBox


def test_crop_self_roi_adds_padding_and_stays_inside_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box = DetectionBox("self", 80, 30, 40, 20, 0.95)

    roi, origin = crop_self_roi(image, box, padding=0.25)

    assert roi.shape[:2] == (30, 60)
    assert origin == (70, 25)


def test_crop_self_roi_expands_when_physical_center_is_outside_box_padding():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box = DetectionBox("self", 80, 30, 40, 20, 0.95)

    roi, origin = crop_self_roi(image, box, padding=0.15, required_point=(120.0, 70.0))

    assert origin[0] <= 120.0 < origin[0] + roi.shape[1]
    assert origin[1] <= 70.0 < origin[1] + roi.shape[0]


def test_map_roi_point_to_image_returns_full_frame_coordinates():
    assert map_roi_point_to_image((0.25, 0.5), origin=(70, 25), size=(60, 30)) == (85.0, 40.0)


def test_invalid_keypoint_is_rejected_for_fallback():
    assert valid_roi_keypoint((0.5, 0.5, 0), 0.7) is False
    assert valid_roi_keypoint((1.2, 0.5, 2), 0.7) is False
    assert valid_roi_keypoint((0.5, 0.5, 2), 0.7) is True


def test_make_self_center_label_uses_crop_relative_keypoint():
    box = DetectionBox("self", 80, 30, 40, 20, 0.95)
    label = make_self_center_label(center=(85.0, 40.0), origin=(70, 25), size=(60, 30), box=box)

    assert label == "0 0.500000 0.500000 0.666667 0.666667 0.250000 0.500000 2"


def test_world_prefers_second_stage_refined_center():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box = DetectionBox("self", 80, 30, 40, 20, 0.95, refined_center=(91.0, 42.0))

    world, diagnostics = build_world_from_image_with_diagnostics([box], image)

    assert world.self_position == (91.0, 42.0)
    assert diagnostics[0]["source"] == "model"

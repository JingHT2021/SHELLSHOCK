from pathlib import Path
import pytest
import numpy as np

from shellshock_detector.world_geometry import DetectionBox
from shellshock_detector_yolo.annotation_conversion import (
    AnnotationBox,
    SceneAnnotation,
    load_manual_scene,
    merge_annotations,
    annotations_to_world,
    yolo_detections_to_annotations,
)


def test_yolo_detections_convert_to_pixel_annotation_boxes():
    annotations = yolo_detections_to_annotations(
        [DetectionBox("enemy", 10, 20, 30, 40, 0.9)], 100, 200
    )
    assert annotations.boxes == [AnnotationBox("enemy", 10, 20, 30, 40, 0.9, "yolo")]


FIXTURES = Path(__file__).parent / "fixtures"


def test_manual_override_replaces_matching_yolo_and_keeps_unmatched_yolo():
    base = yolo_detections_to_annotations(
        [
            DetectionBox("enemy", 10, 20, 30, 40, 0.9),
            DetectionBox("enemy", 70, 20, 10, 10, 0.8),
        ],
        100,
        100,
    )
    label = FIXTURES / "replay_manual.txt"
    manual = load_manual_scene(label, None, None, 100, 100)
    merged = merge_annotations(base, manual)
    assert len(merged.boxes) == 2
    assert merged.boxes[0].source == "manual"
    assert merged.boxes[0].x == 0
    assert merged.boxes[0].y == pytest.approx(5)
    assert merged.boxes[1].source == "yolo"


def test_manual_geometry_has_priority_and_manual_scene_can_be_annotation_only():
    label = FIXTURES / "replay_geometry.txt"
    geometry = FIXTURES / "replay_geometry.json"
    manual = load_manual_scene(label, geometry, None, 100, 100)
    assert manual.self_muzzle == (8.0, 9.0)
    assert manual.self_center is None
    assert manual.lines[0].start == (1.0, 2.0)
    assert manual.lines[0].end == (3.0, 4.0)
    assert manual.boxes[0].name == "self"


def test_manual_geometry_loads_both_self_keypoints():
    geometry = FIXTURES / "replay_keypoints.json"
    manual = load_manual_scene(None, geometry, None, 100, 100)
    assert manual.self_center == (12.0, 13.0)
    assert manual.self_muzzle == (20.0, 21.0)


def test_replay_world_prefers_self_center_over_muzzle_derived_position():
    scene = SceneAnnotation(
        100,
        100,
        [AnnotationBox("self", 70, 70, 10, 10)],
        self_center=(12.0, 13.0),
        self_muzzle=(80.0, 80.0),
    )
    world, muzzle = annotations_to_world(scene, np.zeros((100, 100, 3), dtype=np.uint8))
    assert world.self_position == (12.0, 13.0)
    assert muzzle == (80.0, 80.0)


def test_deleted_marker_removes_matching_yolo_object():
    base = yolo_detections_to_annotations([DetectionBox("enemy", 10, 20, 30, 40, 0.9)], 100, 100)
    manual = SceneAnnotation(100, 100, deleted=[{"name": "enemy", "x": 25, "y": 40}])
    assert merge_annotations(base, manual).boxes == []

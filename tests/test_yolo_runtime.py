import numpy as np
import pytest

from shellshock_detector.yolo_runtime import REQUIRED_CLASSES, YoloDetector, validate_class_names


def test_validate_class_names_accepts_required_sparse_ids():
    names = {
        0: "enemy", 2: "self", 3: "obstacle_circle", 4: "obstacle_line",
        5: "portal_orange", 6: "portal_blue",
    }
    assert validate_class_names(names) == REQUIRED_CLASSES


def test_validate_class_names_reports_missing_required_name():
    with pytest.raises(ValueError, match="portal_blue"):
        validate_class_names({0: "enemy", 2: "self"})


class _Boxes:
    xyxy = np.array([[20, 30, 60, 70], [1, 2, 3, 4], [10, 5, 30, 25]], dtype=float)
    conf = np.array([0.9, 0.7, 0.5], dtype=float)
    cls = np.array([2, 99, 5], dtype=float)


class _Result:
    boxes = _Boxes()


class _Keypoints:
    xy = np.array([[35, 50], [0, 0], [20, 10]], dtype=float)
    conf = np.array([[0.95, 0.0], [0.0, 0.0], [0.8, 0.7]], dtype=float)
    data = np.array([[[35, 50, 2], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[20, 10, 2], [25, 10, 2]]], dtype=float)


_Result.keypoints = _Keypoints()


class _Model:
    names = {0: "enemy", 2: "self", 3: "obstacle_circle", 4: "obstacle_line", 5: "portal_orange", 6: "portal_blue"}

    def predict(self, **kwargs):
        assert kwargs["verbose"] is False
        assert kwargs["conf"] == 0.6
        return [_Result()]


def test_detector_discards_unknown_and_low_confidence_boxes():
    detector = YoloDetector("fake.pt", confidence=0.6, model_factory=lambda _: _Model())
    boxes = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert [(box.name, box.x, box.y, box.width, box.height) for box in boxes] == [
        ("self", 20.0, 30.0, 40.0, 40.0),
    ]
    assert boxes[0].keypoints[0].x == 35.0
    assert boxes[0].keypoints[0].visible == 2


def test_detector_handles_results_without_pose_keypoints():
    class DetectionOnlyResult:
        boxes = _Boxes()

    class DetectionOnlyModel(_Model):
        def predict(self, **kwargs):
            return [DetectionOnlyResult()]

    detector = YoloDetector("fake.pt", confidence=0.6, model_factory=lambda _: DetectionOnlyModel())
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))[0].keypoints == ()

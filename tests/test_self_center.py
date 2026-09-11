import cv2
import numpy as np
import pytest

from shellshock.domain.world import DetectionBox, PoseKeypoint
from shellshock.perception.self_center import fit_self_center
from shellshock.perception.world import build_world_from_image_with_diagnostics


def tank():
    image = np.zeros((120, 140, 3), np.uint8)
    green = (30, 240, 30)
    # Turret and angled barrel bias the overall silhouette away from the wheels.
    cv2.fillPoly(image, [np.array([[45, 35], [70, 35], [83, 59], [33, 59]])], green)
    cv2.line(image, (75, 43), (93, 23), green, 3)
    cv2.line(image, (28, 65), (92, 65), green, 1)
    cv2.line(image, (29, 81), (91, 81), green, 1)
    for x in [34, 47, 60, 73, 86]:
        cv2.circle(image, (x, 73), 5, green, 1)
    return image, DetectionBox('self', 20, 15, 80, 85, .9)


def test_tracks_ignore_turret_and_barrel():
    image, box = tank()
    assert fit_self_center(image, box) == pytest.approx((60, 73), abs=1)


@pytest.mark.parametrize('kind', ['empty', 'noise', 'solid', 'one_rail'])
def test_unreliable_images_decline_fit(kind):
    image, box = tank()
    image[:] = 0
    if kind == 'noise':
        image[:] = np.random.default_rng(7).integers(0, 256, image.shape, dtype=np.uint8)
    elif kind == 'solid':
        image[40:85, 25:95] = (30, 240, 30)
    elif kind == 'one_rail':
        image[80, 25:95] = (30, 240, 30)
    assert fit_self_center(image, box) is None


def test_world_uses_fit_then_pose_then_bbox_and_preserves_manual():
    image, box = tank()
    pose_box = DetectionBox(**{**box.__dict__, 'keypoints': (PoseKeypoint(80, 55, 2, .9),)})
    world, diagnostics = build_world_from_image_with_diagnostics([pose_box], image)
    assert world.self_position == pytest.approx((60, 73), abs=1)
    assert diagnostics[0]['source'] == 'fit'
    empty = np.zeros_like(image)
    assert build_world_from_image_with_diagnostics([pose_box], empty)[0].self_position == (80, 55)
    assert build_world_from_image_with_diagnostics([box], empty)[0].self_position == (60, 57.5)
    manual = DetectionBox(**{**pose_box.__dict__, 'source': 'manual'})
    world, diagnostics = build_world_from_image_with_diagnostics([manual], image)
    assert world.self_position == (80, 55)
    assert diagnostics[0]['source'] == 'manual'


def test_world_selects_winning_detection_and_rejects_confidence_ties():
    image, box = tank()
    stronger = DetectionBox('self', 105, 5, 20, 20, .95)
    assert build_world_from_image_with_diagnostics([box, stronger], image)[0].self_position == (115, 15)
    tied = DetectionBox(**{**stronger.__dict__, 'confidence': box.confidence})
    assert build_world_from_image_with_diagnostics([box, tied], image)[0].self_position is None


def test_clipped_and_invalid_rois_decline_fit():
    image, _ = tank()
    for box in [DetectionBox('self', -100, -100, 10, 10, .9),
                DetectionBox('self', 200, 200, 40, 40, .9),
                DetectionBox('self', float('nan'), 10, 40, 40, .9)]:
        assert fit_self_center(image, box) is None

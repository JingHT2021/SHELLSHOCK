import cv2
import numpy as np
import pytest

from shellshock.perception.world import DetectionBox, PoseKeypoint, World, build_world, build_world_from_image_with_diagnostics


def box(name, x=20, y=20, size=80, keypoints=()):
    return DetectionBox(name, x, y, size, size, .9, keypoints)


@pytest.mark.parametrize('name', ['obstacle_circle', 'portal_orange', 'portal_blue', 'blackhole', 'double_damage', 'Triple_damage'])
def test_standard_circle_refines_all_classes(name):
    from shellshock.perception.world import _fit_visible_circle
    image = np.zeros((140, 140, 3), np.uint8)
    cv2.circle(image, (57, 63), 30, (255, 255, 255), 2)
    fitted = _fit_visible_circle(box(name), image)
    assert fitted is not None
    assert fitted.center == pytest.approx((57, 63), abs=1)
    assert fitted.radius == pytest.approx(30, abs=2)


def test_noise_rejected_and_bbox_retained():
    from shellshock.perception.world import _fit_visible_circle
    image = np.random.default_rng(42).integers(0, 256, (140, 140, 3), dtype=np.uint8)
    assert _fit_visible_circle(box('blackhole'), image) is None
    world, _ = build_world_from_image_with_diagnostics([box('blackhole')], image)
    assert world.blackholes[0].center == (60, 60)
    assert world.blackholes[0].radius == 40


def test_missing_pixels_preserves_bbox_and_pose():
    image = np.zeros((140, 140, 3), np.uint8)
    pose = (PoseKeypoint(58, 61, 2, .9), PoseKeypoint(33, 61, 2, .9))
    world, _ = build_world_from_image_with_diagnostics([
        box('self', keypoints=pose), box('obstacle_circle'), box('blackhole', keypoints=pose),
        box('double_damage'), box('Triple_damage')], image)
    assert world.self_position == (58, 61)
    assert world.circles[0].radius == 40
    assert world.blackholes[0].radius == 25
    assert [r.multiplier for r in world.rewards] == [2, 3]
    assert len({r.object_id for r in world.rewards}) == 2
    assert build_world([box('blackhole'), box('double_damage')], 140, 140).rewards


def test_weak_self_pose_falls_back():
    world, _ = build_world_from_image_with_diagnostics([
        box('self', keypoints=(PoseKeypoint(55, 65, 2, .1),))], np.zeros((140, 140, 3), np.uint8))
    assert world.self_position == (60, 60)


def test_portal_fit_survives_number_pairing(monkeypatch):
    monkeypatch.setattr('shellshock.perception.world.recognize_trained_digits', lambda image: ('2', .99))
    image = np.zeros((140, 260, 3), np.uint8)
    cv2.circle(image, (57, 63), 30, (0, 150, 255), 2)
    cv2.circle(image, (180, 61), 29, (255, 140, 0), 2)
    world, _ = build_world_from_image_with_diagnostics([box('portal_orange'), box('portal_blue', x=140)], image)
    pair = world.portal_pairs[0]
    assert pair.orange.number == pair.blue.number == 2
    assert pair.orange.radius == pytest.approx(30, abs=2)
    assert pair.blue.radius == pytest.approx(29, abs=2)


def test_new_world_collections_are_immutable_tuples():
    world = World(blackholes=[], rewards=[])
    assert world.blackholes == world.rewards == ()


def test_visible_colored_arc_recovers_standard_circle():
    from shellshock.perception.world import _fit_visible_circle
    image = np.zeros((140, 140, 3), np.uint8)
    cv2.ellipse(image, (60, 60), (32, 32), 0, 20, 220, (255, 0, 255), 2)
    fitted = _fit_visible_circle(box('obstacle_circle'), image)
    assert fitted is not None
    assert fitted.center == pytest.approx((60, 60), abs=1.5)
    assert fitted.radius == pytest.approx(32, abs=1.5)


def test_short_arc_is_insufficient_evidence():
    from shellshock.perception.world import _fit_visible_circle
    image = np.zeros((140, 140, 3), np.uint8)
    cv2.ellipse(image, (60, 60), (32, 32), 0, 20, 45, (255, 0, 255), 1)
    assert _fit_visible_circle(box('obstacle_circle'), image) is None


@pytest.mark.parametrize('name', ['blackhole', 'double_damage', 'Triple_damage'])
def test_unknown_class_color_does_not_override_trusted_pose_with_white_graphic(name):
    image = np.zeros((140, 140, 3), np.uint8)
    cv2.circle(image, (60, 60), 25, (255, 255, 255), 2)
    pose = (PoseKeypoint(58, 61, 2, .9), PoseKeypoint(23, 61, 2, .9))
    world, diagnostics = build_world_from_image_with_diagnostics([box(name, keypoints=pose)], image)
    chosen = (world.blackholes or world.rewards)[0]
    assert chosen.center == (58, 61) and chosen.radius == 35
    assert diagnostics[0]['source'] == 'pose'

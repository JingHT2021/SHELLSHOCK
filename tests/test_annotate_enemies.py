import cv2
import numpy as np

from annotate_enemies import (
    CLASS_KEY_MAP,
    CircleAnnotation,
    build_parser,
    circle_to_yolo_box,
    display_to_image_point,
    delete_nearest_annotation,
    key_action,
    select_images,
    write_enemy_supplement,
)


def test_select_images_includes_timestamp_boundaries(tmp_path):
    for stem in (
        "20260906_161533",
        "20260906_161534",
        "20260906_171759",
        "20260906_171800",
    ):
        assert cv2.imwrite(str(tmp_path / f"{stem}.png"), np.zeros((10, 10, 3), dtype=np.uint8))

    selected = select_images(tmp_path, "20260906_161534", "20260906_171759")

    assert [path.stem for path in selected] == ["20260906_161534", "20260906_171759"]


def test_display_point_maps_back_to_original_coordinates():
    assert display_to_image_point((480, 225), (960, 450), (3840, 1800)) == (1920, 900)


def test_write_supplemental_enemy_label_uses_reference_box_size(tmp_path):
    write_enemy_supplement(tmp_path, "scene", [(1920, 900)], 3840, 1800)

    assert (tmp_path / "scene.txt").read_text(encoding="utf-8").startswith(
        "0 0.500000 0.500000"
    )


def test_key_action_maps_arrow_and_escape_keys():
    assert key_action(81) == "previous"
    assert key_action(83) == "next"
    assert key_action(27) == "quit"


def test_parser_can_select_every_image_in_a_custom_review_directory():
    args = build_parser().parse_args(
        ["--raw-dir", "train/review_portal_mismatch", "--all-images"]
    )

    assert str(args.raw_dir) == "train\\review_portal_mismatch"
    assert args.all_images is True


def test_numeric_keys_select_all_requested_classes():
    assert CLASS_KEY_MAP == {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
        "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    }


def test_circle_is_saved_as_clipped_square_yolo_box():
    annotation = CircleAnnotation(7, 5, 5, 50)
    line = circle_to_yolo_box(annotation, 100, 80)
    assert line == "7 0.275000 0.343750 0.550000 0.687500"


def test_right_click_deletes_nearest_matching_annotation_only():
    annotations = [CircleAnnotation(0, 10, 10, 10), CircleAnnotation(1, 12, 10, 10), CircleAnnotation(0, 80, 80, 10)]
    remaining = delete_nearest_annotation(annotations, 11, 10, 0)
    assert remaining == [CircleAnnotation(1, 12, 10, 10), CircleAnnotation(0, 80, 80, 10)]

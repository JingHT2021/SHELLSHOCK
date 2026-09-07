import cv2
import numpy as np
import pytest

from shellshock_detector.yolo_dataset import (
    PreparationConfig,
    YoloBox,
    build_merged_corpus,
    export_portal_mismatches,
    make_multilabel_folds,
    merge_supplemental_boxes,
    parse_yolo_label_text,
    prepare_dataset,
    transform_yolo_box_for_crop,
)
from shellshock_detector import yolo_dataset


def test_crop_clips_and_renormalizes_bottom_intersection():
    result = transform_yolo_box_for_crop(0, 0.5, 0.95, 0.1, 0.2, 3840, 2000, 1850)
    assert result[0] == 0
    assert result[1:] == pytest.approx((0.5, 1775 / 1850, 0.1, 150 / 1850))


def test_crop_discards_box_entirely_below_boundary():
    assert transform_yolo_box_for_crop(0, 0.5, 0.975, 0.1, 0.05, 3840, 2000, 1850) is None


def test_serialized_crop_box_that_touches_top_edge_remains_strictly_valid():
    transformed = transform_yolo_box_for_crop(
        3, 0.782292, 0.07625, 0.091146, 0.1525, 3840, 2000, 1850
    )

    boxes, errors = parse_yolo_label_text(
        yolo_dataset._format_box(YoloBox(*transformed)), 1, 1
    )

    assert errors == []
    assert boxes[0].class_id == 3


def test_parser_accepts_class_nine_and_rejects_out_of_range_identifier():
    valid, errors = parse_yolo_label_text(
        "9 0.5 0.5 0.1 0.1\n10 0.5 0.5 0.1 0.1\n", 3840, 1800
    )
    assert [box.class_id for box in valid] == [9]
    assert "class_id_out_of_range" in errors[0]


def test_enemy_candidates_are_review_artifacts_not_automatic_exclusions():
    assert PreparationConfig().exclude_unlabelled_enemy_candidates is False


def test_prepare_creates_sparse_id_yaml_and_never_modifies_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()
    image_path = raw_dir / "scene.png"
    label_path = raw_dir / "scene.txt"
    assert cv2.imwrite(str(image_path), np.zeros((2000, 3840, 3), dtype=np.uint8))
    original_label = "9 0.5 0.90 0.1 0.15\n"
    label_path.write_text(original_label, encoding="utf-8")

    summary = prepare_dataset(raw_dir, output_dir, PreparationConfig(preview_count=0))

    assert image_path.exists()
    assert label_path.read_text(encoding="utf-8") == original_label
    prepared_image = next((output_dir / "images" / "train").glob("scene.*"))
    assert cv2.imread(str(prepared_image)).shape[:2] == (1850, 3840)
    assert "9: Triple_damage" in (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    prepared_label = (output_dir / "labels" / "train" / "scene.txt").read_text(encoding="utf-8")
    assert prepared_label.startswith("9 ")
    assert summary["total_images"] == 1


def test_export_portal_mismatches_copies_only_unequal_pairs_and_writes_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for stem, labels in {
        "unequal": "5 0.5 0.5 0.1 0.1\n",
        "equal": "5 0.5 0.5 0.1 0.1\n6 0.6 0.5 0.1 0.1\n",
    }.items():
        image_path = raw_dir / f"{stem}.png"
        assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
        (raw_dir / f"{stem}.txt").write_text(labels, encoding="utf-8")

    summary = export_portal_mismatches(raw_dir, tmp_path / "review")

    assert summary["mismatch_images"] == 1
    assert (tmp_path / "review" / "unequal.png").exists()
    assert (tmp_path / "review" / "unequal.txt").exists()
    assert not (tmp_path / "review" / "equal.png").exists()
    manifest = (tmp_path / "review" / "manifest.csv").read_text(encoding="utf-8")
    assert "unequal.png,1,0,orange_blue_count_mismatch" in manifest


def test_supplemental_enemy_boxes_append_new_boxes_but_not_duplicate_existing_box(tmp_path):
    supplemental_path = tmp_path / "scene.txt"
    supplemental_path.write_text(
        "0 0.5 0.5 0.1 0.1\n0 0.8 0.5 0.1 0.1\n", encoding="utf-8"
    )

    merged, errors, added, duplicates = merge_supplemental_boxes(
        [YoloBox(0, 0.5, 0.5, 0.1, 0.1)], supplemental_path, 3840, 1800
    )

    assert errors == []
    assert added == 1
    assert duplicates == 1
    assert [(box.center_x, box.center_y) for box in merged] == [(0.5, 0.5), (0.8, 0.5)]


def test_prepare_uses_optional_portal_review_label_override_without_editing_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    review_dir = tmp_path / "review"
    raw_dir.mkdir()
    review_dir.mkdir()
    image_path = raw_dir / "scene.png"
    raw_label = raw_dir / "scene.txt"
    assert cv2.imwrite(str(image_path), np.zeros((1800, 3840, 3), dtype=np.uint8))
    raw_label.write_text("5 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (review_dir / "scene.txt").write_text("6 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    summary = prepare_dataset(
        raw_dir,
        tmp_path / "output",
        PreparationConfig(preview_count=0, portal_review_dir=review_dir),
    )

    prepared_label = next((tmp_path / "output" / "labels").glob("*/*.txt"))
    assert prepared_label.read_text(encoding="utf-8").startswith("6 ")
    assert raw_label.read_text(encoding="utf-8").startswith("5 ")
    assert summary["portal_review_label_overrides"] == 1


def test_build_merged_corpus_prefers_review_labels_and_merges_enemy_supplements(tmp_path):
    raw_dir, review_dir, supplemental_dir = (
        tmp_path / "raw",
        tmp_path / "review",
        tmp_path / "supplemental",
    )
    for directory in (raw_dir, review_dir, supplemental_dir):
        directory.mkdir()
    assert cv2.imwrite(str(raw_dir / "scene.png"), np.zeros((1800, 3840, 3), dtype=np.uint8))
    assert cv2.imwrite(str(review_dir / "scene.png"), np.zeros((1800, 3840, 3), dtype=np.uint8))
    (raw_dir / "scene.txt").write_text("5 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (review_dir / "scene.txt").write_text("6 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (supplemental_dir / "scene.txt").write_text("0 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    summary = build_merged_corpus(raw_dir, review_dir, supplemental_dir, tmp_path / "corpus")

    label_text = (tmp_path / "corpus" / "labels" / "scene.txt").read_text(encoding="utf-8")
    assert summary["unique_images"] == 1
    assert summary["review_source_images"] == 1
    assert label_text.splitlines()[0].startswith("6 ")
    assert "0 0.200000" in label_text


def test_build_merged_corpus_can_limit_only_the_raw_source_stems(tmp_path):
    raw_dir, review_dir, supplemental_dir = (
        tmp_path / "raw",
        tmp_path / "review",
        tmp_path / "supplemental",
    )
    for directory in (raw_dir, review_dir, supplemental_dir):
        directory.mkdir()
    for stem in ("keep", "exclude"):
        assert cv2.imwrite(str(raw_dir / f"{stem}.png"), np.zeros((1800, 3840, 3), dtype=np.uint8))
        (raw_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    summary = build_merged_corpus(
        raw_dir,
        review_dir,
        supplemental_dir,
        tmp_path / "corpus",
        raw_stems={"keep"},
    )

    assert summary["unique_images"] == 1
    assert (tmp_path / "corpus" / "images" / "keep.png").exists()
    assert not (tmp_path / "corpus" / "images" / "exclude.png").exists()


def test_make_multilabel_folds_assigns_every_stem_to_exactly_one_validation_fold():
    folds = make_multilabel_folds(
        {"a": {0}, "b": {0, 5}, "c": {2}, "d": {6}, "e": {3}, "f": {4}},
        n_splits=5,
        seed=42,
    )

    assigned = [stem for fold in folds for stem in fold]
    assert sorted(assigned) == ["a", "b", "c", "d", "e", "f"]
    assert len(set(assigned)) == 6

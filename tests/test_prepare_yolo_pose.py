from pathlib import Path

import cv2
import numpy as np
import yaml

from prepare_yolo_pose import prepare


def _write_sample(raw: Path, geometry: Path, name: str, label: str, *, center=(50, 50)):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(raw / f"{name}.png"), image)
    (raw / f"{name}.txt").write_text(label, encoding="utf-8")
    (geometry / f"{name}.json").write_text(
        '{"lines": [{"start": [10, 20], "end": [90, 80]}], "self_center": [%s, %s]}' % center,
        encoding="utf-8",
    )


def test_prepare_writes_pose_split_and_filters_shortcut_point(tmp_path):
    raw, geometry, output = (tmp_path / item for item in ("raw", "geometry", "output"))
    raw.mkdir(); geometry.mkdir()
    _write_sample(raw, geometry, "20260910_120001_a", "2 0.4 0.5 0.1 0.2\n1 0.4 0.5 0.01 0.01\n")
    _write_sample(raw, geometry, "20260910_120002_b", "4 0.5 0.5 0.2 0.2\n")
    _write_sample(raw, geometry, "20260910_130001_c", "3 0.5 0.5 0.2 0.2\n")

    assert prepare(raw, output, geometry, raw, val_fraction=0.5, seed=42) == 3
    yaml_data = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert yaml_data["kpt_shape"] == [2, 3]
    assert yaml_data["train"] == "images/train"
    assert yaml_data["val"] == "images/val"
    pose_lines = [
        line
        for label_path in (output / "labels").glob("**/*.txt")
        for line in label_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(pose_lines) == 3
    assert all(len(line.split()) == 11 for line in pose_lines)
    assert all(not line.startswith("1 ") for line in pose_lines)
    assert (output / "images" / "train").exists()
    assert (output / "images" / "val").exists()


def test_prepare_assigns_same_minute_group_to_one_split(tmp_path):
    raw, geometry, output = (tmp_path / item for item in ("raw", "geometry", "output"))
    raw.mkdir(); geometry.mkdir()
    for suffix in ("a", "b"):
        _write_sample(raw, geometry, f"20260910_120001_{suffix}", "0 0.5 0.5 0.1 0.1\n")
    _write_sample(raw, geometry, "20260910_130001_c", "0 0.5 0.5 0.1 0.1\n")

    prepare(raw, output, geometry, raw, val_fraction=0.5, seed=42)
    split_for = {}
    for split in ("train", "val"):
        for image in (output / "images" / split).glob("*.png"):
            split_for[image.stem] = split
    assert split_for["20260910_120001_a"] == split_for["20260910_120001_b"]


def test_prepare_accepts_annotate_check_subdirectories(tmp_path):
    root = tmp_path / "annotate_check"
    images, labels, geometry, output = (root / item for item in ("images", "labels", "pose_geometry", "output"))
    images.mkdir(parents=True); labels.mkdir(); geometry.mkdir()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images / "20260910_120001_a.png"), image)
    (labels / "20260910_120001_a.txt").write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (geometry / "20260910_120001_a.json").write_text('{"self_center": [50, 50], "lines": []}', encoding="utf-8")
    assert prepare(root, output, geometry, labels) == 1
    assert (output / "images" / "train" / "20260910_120001_a.png").exists()

import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import prepare_wind_digit_dataset as module


def test_manifest_uses_absolute_wind_value_and_one_row_per_digit():
    root = Path("tests/.wind_dataset_test")
    shutil.rmtree(root, ignore_errors=True)
    labels = root / "labels"
    crops = root / "crops"
    output = root / "output"
    labels.mkdir(parents=True)
    crops.mkdir()
    image = np.full((70, 130, 3), 220, dtype=np.uint8)
    cv2.putText(image, "64", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.imwrite(str(crops / "scene.png"), image)
    (labels / "scene.json").write_text(json.dumps({"image": str(crops / "scene.png"), "wind_value": 64, "wind_direction": "left"}), encoding="utf-8")

    module.build_dataset(labels, crops, output)
    rows = list(csv.DictReader((output / "manifest.csv").open(encoding="utf-8")))
    assert [(row["digit_label"], row["position"]) for row in rows] == [("6", "0"), ("4", "1")]
    assert all(row["source_stem"] == "scene" and row["wind_value"] == "64" for row in rows)


def test_split_is_by_source_stem_not_by_digit_crop():
    splits = module.split_stems(["a", "a", "b", "b", "c", "c", "d", "d"], seed=42)
    memberships = {split: set(stems) for split, stems in splits.items()}
    assert not memberships["train"] & memberships["validation"]
    assert not memberships["train"] & memberships["test"]
    assert not memberships["validation"] & memberships["test"]

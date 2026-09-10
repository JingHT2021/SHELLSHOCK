import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import extract_capture_crops as module
from shellshock_detector.models import Wind


def _write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((40, 60, 3), 200, dtype=np.uint8))


def test_extract_one_preserves_manual_value_and_repairs_image_path(monkeypatch):
    tmp_path = Path("tests/.wind_extract_test_1")
    shutil.rmtree(tmp_path, ignore_errors=True)
    image_path = tmp_path / "images" / "scene.png"
    _write_image(image_path)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    wind_labels_dir = tmp_path / "wind_labels"
    wind_labels_dir.mkdir()
    (wind_labels_dir / "scene.json").write_text(
        json.dumps({"wind_value": 64, "wind_direction": "left", "source": "manual"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "detect_wind", lambda image: (Wind(64, "left", 0.9), None, (5, 5, 20, 10)))

    result = module.extract_one(image_path, labels_dir, tmp_path / "output", wind_labels_dir)
    payload = json.loads((wind_labels_dir / "scene.json").read_text(encoding="utf-8"))

    assert result["wind"] == 1
    assert payload["wind_value"] == 64
    assert payload["wind_direction"] == "left"
    assert Path(payload["image"]).exists()


def test_audit_reports_missing_wind_crops():
    tmp_path = Path("tests/.wind_extract_test_2")
    shutil.rmtree(tmp_path, ignore_errors=True)
    image_dir = tmp_path / "images"
    wind_dir = tmp_path / "wind"
    labels_dir = tmp_path / "wind_labels"
    _write_image(image_dir / "one.png")
    _write_image(image_dir / "two.png")
    wind_dir.mkdir()
    labels_dir.mkdir()
    _write_image(wind_dir / "one.png")
    (labels_dir / "one.json").write_text(json.dumps({"wind_value": 1}), encoding="utf-8")
    (labels_dir / "two.json").write_text(json.dumps({"wind_value": 2}), encoding="utf-8")

    summary = module.audit_dataset(image_dir, wind_dir, labels_dir)

    assert summary["images"] == 2
    assert summary["wind_crops"] == 1
    assert summary["wind_labels"] == 2
    assert summary["missing_wind"] == 1

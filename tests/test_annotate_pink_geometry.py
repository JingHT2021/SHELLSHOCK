import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from annotate_pink_geometry import annotate_geometry_directory


def test_geometry_batch_rebuilds_only_obstacle_labels_from_reflection_geometry():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        annotated = root / "annotated"
        raw = root / "raw_cropped"
        geometry = root / "geometry"
        annotated.mkdir()
        raw.mkdir()
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        pink = (255, 255, 255)
        cv2.circle(image, (500, 360), 150, pink, 8)
        assert cv2.imwrite(str(raw / "scene.png"), image)
        assert cv2.imwrite(str(annotated / "scene.png"), image)
        label = raw / "scene.txt"
        label.write_text("2 0.100000 0.200000 0.020000 0.030000\n3 0.500000 0.500000 0.200000 0.200000\n4 0.600000 0.500000 0.200000 0.020000\n", encoding="utf-8")

        summary = annotate_geometry_directory(annotated, raw, geometry)

        assert summary["images"] == 1
        assert summary["circles"] == 1
        assert summary["lines"] == 0
        assert summary["removed_yolo_boxes"] == 2
        labels = label.read_text(encoding="utf-8").splitlines()
        assert labels[0] == "2 0.100000 0.200000 0.020000 0.030000"
        assert [line.split()[0] for line in labels].count("3") == 1
        assert "4" not in {line.split()[0] for line in labels}
        payload = json.loads((geometry / "scene.json").read_text(encoding="utf-8"))
        assert len(payload["circles"]) == 1
        assert payload["lines"] == []
        assert {item["class_id"] for item in payload["objects"]} >= {2, 3}

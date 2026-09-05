import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from annotate_pink_geometry import annotate_geometry_directory


def test_geometry_batch_writes_circle_parameters_and_removes_only_old_obstacle_boxes():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        annotated = root / "annotated"
        raw = root / "raw_cropped"
        geometry = root / "geometry"
        annotated.mkdir()
        raw.mkdir()
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        pink = tuple(map(int, cv2.cvtColor(np.uint8([[[153, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
        cv2.circle(image, (500, 360), 150, pink, 8)
        assert cv2.imwrite(str(raw / "scene.png"), image)
        assert cv2.imwrite(str(annotated / "scene.png"), image)
        label = raw / "scene.txt"
        label.write_text("2 0.100000 0.200000 0.020000 0.030000\n3 0.500000 0.500000 0.200000 0.200000\n4 0.600000 0.500000 0.200000 0.020000\n", encoding="utf-8")

        summary = annotate_geometry_directory(annotated, raw, geometry)

        assert summary == {"images": 1, "circles": 1, "lines": 0, "removed_yolo_boxes": 2, "missing_raw": 0}
        assert label.read_text(encoding="utf-8") == "2 0.100000 0.200000 0.020000 0.030000\n"
        payload = json.loads((geometry / "scene.json").read_text(encoding="utf-8"))
        assert len(payload["circles"]) == 1
        assert payload["lines"] == []

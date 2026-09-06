from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from annotate_pink_obstacles import annotate_directory


def test_annotate_directory_reports_additions_without_replacing_existing_labels():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        annotated = root / "annotated"
        labels = root / "raw_cropped"
        annotated.mkdir()
        labels.mkdir()
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        pink = (255, 255, 255)
        cv2.line(image, (80, 300), (360, 150), pink, 8)
        assert cv2.imwrite(str(annotated / "scene.png"), image)
        (labels / "scene.txt").write_text("0 0.500000 0.500000 0.020000 0.020000\n", encoding="utf-8")

        summary = annotate_directory(annotated, labels)
        second_summary = annotate_directory(annotated, labels)

        assert summary == {"images": 1, "added_circle": 0, "added_line": 1, "missing_labels": 0}
        assert second_summary == {"images": 1, "added_circle": 0, "added_line": 0, "missing_labels": 0}
        assert (labels / "scene.txt").read_text(encoding="utf-8").startswith("0 0.500000")

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from correct_portal_annotations import correct_directory


def test_correction_removes_only_lower_hud_blue_portals_and_keeps_game_portals():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        raw, annotated, geometry = root / "raw", root / "annotated", root / "geometry"
        raw.mkdir(); annotated.mkdir(); geometry.mkdir()
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        assert cv2.imwrite(str(raw / "20260906_130000.png"), image)
        assert cv2.imwrite(str(annotated / "20260906_130000.png"), image)
        (raw / "20260906_130000.txt").write_text(
            "2 0.100000 0.200000 0.020000 0.030000\n6 0.300000 0.250000 0.100000 0.100000\n6 0.500000 0.900000 0.100000 0.100000\n",
            encoding="utf-8",
        )
        (geometry / "20260906_130000.json").write_text(json.dumps({
            "circles": [], "lines": [],
            "portals": [
                {"class_id": 6, "name": "portal_blue", "center": [180, 100], "radius": 30, "pair_id": None},
                {"class_id": 6, "name": "portal_blue", "center": [300, 360], "radius": 30, "pair_id": None},
            ],
            "objects": [],
        }), encoding="utf-8")

        summary = correct_directory(annotated, raw, geometry, blue_ui_min_y=300)

        labels = (raw / "20260906_130000.txt").read_text(encoding="utf-8")
        payload = json.loads((geometry / "20260906_130000.json").read_text(encoding="utf-8"))
        assert [line.split()[0] for line in labels.splitlines()].count("6") == 1
        assert [portal["center"] for portal in payload["portals"]] == [[180, 100]]
        assert summary["removed_hud_blue"] == 1

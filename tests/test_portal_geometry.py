import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from annotate_pink_geometry import annotate_geometry_directory
from shellshock_detector.obstacle_geometry import (
    PortalConfig,
    detect_portal_geometry,
    geometry_to_dict,
)


def _hsv_color(hue: int) -> tuple[int, int, int]:
    return tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))


def test_portal_detector_reports_colored_circles_and_shared_pair_id():
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    cv2.circle(image, (220, 340), 90, _hsv_color(15), 8)
    cv2.circle(image, (900, 430), 91, _hsv_color(100), 8)

    portals = detect_portal_geometry(image)

    assert [(item.class_id, item.name, item.pair_id) for item in portals] == [
        (5, "portal_orange", 1),
        (6, "portal_blue", 1),
    ]
    assert abs(portals[0].center[0] - 220) <= 4
    assert abs(portals[0].center[1] - 340) <= 4
    assert abs(portals[0].radius - 90) <= 5
    assert abs(portals[1].radius - 91) <= 5


def test_portal_detector_keeps_unpaired_portal():
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    cv2.circle(image, (220, 340), 90, _hsv_color(15), 8)

    portals = detect_portal_geometry(image)

    assert len(portals) == 1
    assert portals[0].class_id == 5
    assert portals[0].pair_id is None


def test_geometry_json_contains_all_yolo_objects_and_portals_without_removing_existing_labels():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        annotated, raw, geometry = root / "annotated", root / "raw", root / "geometry"
        annotated.mkdir()
        raw.mkdir()
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        cv2.line(image, (100, 650), (400, 500), (255, 255, 255), 7)
        cv2.circle(image, (220, 340), 90, _hsv_color(15), 8)
        cv2.circle(image, (900, 430), 90, _hsv_color(100), 8)
        assert cv2.imwrite(str(raw / "scene.png"), image)
        assert cv2.imwrite(str(annotated / "scene.png"), image)
        label_path = raw / "scene.txt"
        existing = "0 0.700000 0.500000 0.020000 0.030000\n2 0.100000 0.200000 0.020000 0.030000\n"
        label_path.write_text(existing, encoding="utf-8")

        summary = annotate_geometry_directory(annotated, raw, geometry)

        labels = label_path.read_text(encoding="utf-8")
        assert labels.startswith(existing)
        assert {line.split()[0] for line in labels.splitlines()} >= {"0", "2", "4", "5", "6"}
        payload = json.loads((geometry / "scene.json").read_text(encoding="utf-8"))
        assert {item["class_id"] for item in payload["objects"]} >= {0, 2, 4, 5, 6}
        assert [item["pair_id"] for item in payload["portals"]] == [1, 1]
        assert summary["added_portal_orange"] == 1
        assert summary["added_portal_blue"] == 1
        assert summary["removed_yolo_boxes"] == 0


def test_dry_run_does_not_write_labels_preview_or_geometry():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        annotated, raw, geometry = root / "annotated", root / "raw", root / "geometry"
        annotated.mkdir()
        raw.mkdir()
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(image, (200, 200), 70, _hsv_color(15), 6)
        assert cv2.imwrite(str(raw / "scene.png"), image)
        assert cv2.imwrite(str(annotated / "scene.png"), image)
        label_path = raw / "scene.txt"
        original = "2 0.100000 0.200000 0.020000 0.030000\n"
        label_path.write_text(original, encoding="utf-8")
        preview_before = (annotated / "scene.png").read_bytes()

        annotate_geometry_directory(annotated, raw, geometry, dry_run=True)

        assert label_path.read_text(encoding="utf-8") == original
        assert (annotated / "scene.png").read_bytes() == preview_before
        assert not geometry.exists()

"""Apply reviewed portal and triple-damage corrections without re-detecting obstacles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from shellshock_detector.obstacle_geometry import (
    CircleObstacle,
    LineObstacle,
    ObstacleGeometry,
    PortalGeometry,
    detect_portal_geometry,
    draw_geometry_preview,
    portal_yolo_detections,
    save_geometry,
    yolo_objects,
)
from shellshock_detector.training_data import ObstacleDetection, _draw_yolo_preview, append_yolo_obstacle_labels


TRIPLE_DAMAGE_CIRCLES = {
    "20260906_011224": CircleObstacle((3283, 197), 57),
    "20260906_013049": CircleObstacle((2848, 148), 72),
    "20260906_013112": CircleObstacle((2011, 274), 64),
}
BLUE_MISSING_STEM = "20260906_010628"
OCCLUDED_ORANGE_STEM = "20260906_014608"
OCCLUDED_ORANGE = PortalGeometry(5, "portal_orange", (2983, 323), 95, 1)
HUD_START_STEM = "20260906_125936"


def _remove_classes(text: str, classes: set[int]) -> str:
    return "".join(line for line in text.splitlines(keepends=True) if not (line.split() and int(line.split()[0]) in classes))


def _geometry(payload: dict[str, object]) -> ObstacleGeometry:
    return ObstacleGeometry(
        circles=[CircleObstacle(tuple(item["center"]), item["radius"]) for item in payload.get("circles", [])],
        lines=[LineObstacle(tuple(item["start"]), tuple(item["end"])) for item in payload.get("lines", [])],
        portals=[PortalGeometry(item["class_id"], item["name"], tuple(item["center"]), item["radius"], item.get("pair_id")) for item in payload.get("portals", [])],
    )


def _normalise_pairs(portals: list[PortalGeometry]) -> list[PortalGeometry]:
    ids = {portal.pair_id for portal in portals if portal.pair_id is not None}
    valid = {
        pair_id for pair_id in ids
        if {portal.class_id for portal in portals if portal.pair_id == pair_id} == {5, 6}
    }
    return [
        PortalGeometry(portal.class_id, portal.name, portal.center, portal.radius, portal.pair_id if portal.pair_id in valid else None)
        for portal in portals
    ]


def correct_directory(
    annotated_dir: Path,
    raw_dir: Path,
    geometry_dir: Path,
    *,
    blue_ui_min_y: int = 1800,
) -> dict[str, int]:
    summary = {"images": 0, "triple_damage": 0, "added_missing_blue": 0, "corrected_occluded_orange": 0, "removed_hud_blue": 0}
    for geometry_path in sorted(geometry_dir.glob("*.json")):
        stem = geometry_path.stem
        raw_path = raw_dir / f"{stem}.png"
        label_path = raw_dir / f"{stem}.txt"
        preview_path = annotated_dir / f"{stem}.png"
        if not raw_path.exists() or not label_path.exists() or not preview_path.exists():
            continue
        image = cv2.imread(str(raw_path))
        if image is None:
            continue
        payload = json.loads(geometry_path.read_text(encoding="utf-8"))
        geometry = _geometry(payload)
        labels = label_path.read_text(encoding="utf-8")
        additions: list[ObstacleDetection] = []
        remove: set[int] = set()

        if stem in TRIPLE_DAMAGE_CIRCLES:
            remove.update({5, 6, 9})
            circle = TRIPLE_DAMAGE_CIRCLES[stem]
            additions.append(ObstacleDetection(9, (circle.center[0] - circle.radius, circle.center[1] - circle.radius, circle.radius * 2, circle.radius * 2)))
            geometry = ObstacleGeometry(geometry.circles, geometry.lines, [])
            summary["triple_damage"] += 1
        elif stem == BLUE_MISSING_STEM:
            remove.update({5, 6})
            geometry = ObstacleGeometry(geometry.circles, geometry.lines, detect_portal_geometry(image))
            additions.extend(portal_yolo_detections(geometry.portals))
            summary["added_missing_blue"] += 1
        else:
            portals = geometry.portals
            if stem == OCCLUDED_ORANGE_STEM:
                remove.add(5)
                portals = [OCCLUDED_ORANGE if portal.class_id == 5 else portal for portal in portals]
                summary["corrected_occluded_orange"] += 1
            if stem >= HUD_START_STEM:
                removed = [portal for portal in portals if portal.class_id == 6 and portal.center[1] >= blue_ui_min_y]
                if removed:
                    remove.add(6)
                    portals = [portal for portal in portals if portal not in removed]
                    additions.extend(portal_yolo_detections([portal for portal in portals if portal.class_id == 6]))
                    summary["removed_hud_blue"] += len(removed)
            geometry = ObstacleGeometry(geometry.circles, geometry.lines, _normalise_pairs(portals))
            if stem == OCCLUDED_ORANGE_STEM:
                additions.extend(portal_yolo_detections([portal for portal in geometry.portals if portal.class_id == 5]))

        merged, _ = append_yolo_obstacle_labels(_remove_classes(labels, remove), additions, image.shape[1], image.shape[0])
        if merged != labels:
            label_path.write_text(merged, encoding="utf-8")
        save_geometry(geometry_path, geometry, yolo_objects(merged, image.shape[1], image.shape[0]))
        preview = draw_geometry_preview(_draw_yolo_preview(image, merged, hidden_class_ids={3, 4, 5, 6}), geometry)
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"failed to write preview: {preview_path}")
        summary["images"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated-dir", type=Path, default=Path("train/annotated"))
    parser.add_argument("--raw-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("train/geometry"))
    args = parser.parse_args()
    print(json.dumps(correct_directory(args.annotated_dir, args.raw_dir, args.geometry_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

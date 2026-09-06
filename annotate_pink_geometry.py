"""Create center/radius and endpoint geometry annotations for pink obstacles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from shellshock_detector.obstacle_geometry import (
    detect_pink_obstacle_geometry,
    detect_portal_geometry,
    draw_geometry_preview,
    obstacle_yolo_detections,
    portal_yolo_detections,
    remove_obstacle_yolo_labels,
    save_geometry,
    yolo_objects,
)
from shellshock_detector.training_data import (
    _draw_yolo_preview,
    append_yolo_obstacle_labels,
)


def annotate_geometry_directory(
    annotated_dir: Path,
    raw_dir: Path,
    geometry_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Use raw same-stem images for geometry while updating selected preview files."""
    summary = {
        "images": 0, "circles": 0, "lines": 0,
        "added_circle": 0, "added_line": 0,
        "added_portal_orange": 0, "added_portal_blue": 0,
        "removed_yolo_boxes": 0, "missing_raw": 0,
    }
    for preview_path in sorted(annotated_dir.glob("*.png")):
        summary["images"] += 1
        raw_image_path = raw_dir / preview_path.name
        label_path = raw_dir / f"{preview_path.stem}.txt"
        if not raw_image_path.exists() or not label_path.exists():
            summary["missing_raw"] += 1
            continue
        image = cv2.imread(str(raw_image_path))
        if image is None:
            summary["missing_raw"] += 1
            continue
        obstacle_geometry = detect_pink_obstacle_geometry(image)
        portals = detect_portal_geometry(image)
        geometry = obstacle_geometry.__class__(
            circles=obstacle_geometry.circles, lines=obstacle_geometry.lines, portals=portals
        )
        summary["circles"] += len(geometry.circles)
        summary["lines"] += len(geometry.lines)
        original_labels = label_path.read_text(encoding="utf-8")
        cleaned_labels = remove_obstacle_yolo_labels(original_labels)
        summary["removed_yolo_boxes"] += sum(
            1 for line in original_labels.splitlines() if line.split() and line.split()[0] in {"3", "4"}
        )
        detections = obstacle_yolo_detections(geometry, image.shape[1]) + portal_yolo_detections(portals)
        merged_labels, _ = append_yolo_obstacle_labels(
            cleaned_labels, detections, image.shape[1], image.shape[0]
        )
        old_count = len(cleaned_labels.splitlines())
        for line in merged_labels.splitlines()[old_count:]:
            class_id = int(line.split()[0])
            if class_id == 3:
                summary["added_circle"] += 1
            elif class_id == 4:
                summary["added_line"] += 1
            elif class_id == 5:
                summary["added_portal_orange"] += 1
            elif class_id == 6:
                summary["added_portal_blue"] += 1
        if dry_run:
            continue
        if merged_labels != original_labels:
            label_path.write_text(merged_labels, encoding="utf-8")
        save_geometry(
            geometry_dir / f"{preview_path.stem}.json",
            geometry,
            yolo_objects(merged_labels, image.shape[1], image.shape[0]),
        )
        preview = draw_geometry_preview(
            _draw_yolo_preview(image, merged_labels, hidden_class_ids={3, 4, 5, 6}), geometry
        )
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"failed to write preview: {preview_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated-dir", type=Path, default=Path("train/annotated"))
    parser.add_argument("--raw-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("train/geometry"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(annotate_geometry_directory(args.annotated_dir, args.raw_dir, args.geometry_dir, dry_run=args.dry_run), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

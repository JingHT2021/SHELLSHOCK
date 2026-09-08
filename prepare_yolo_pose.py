"""Create a two-keypoint YOLO Pose dataset from existing detection labels."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import json

from shellshock_detector.obstacle_geometry import detect_pink_obstacle_geometry, LineObstacle, ObstacleGeometry
from shellshock_detector.pose_dataset import convert_boxes, format_pose_label
from shellshock_detector.yolo_dataset import parse_yolo_label_text


def prepare(raw_dir: Path, output_dir: Path, geometry_dir: Path = Path("train/pose_geometry"), override_dir: Path = Path("train/annotation_overrides")) -> int:
    image_dir, label_dir = output_dir / "images", output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(raw_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}: continue
        raw_label_path = raw_dir / f"{image_path.stem}.txt"
        label_path = override_dir / f"{image_path.stem}.txt" if (override_dir / f"{image_path.stem}.txt").exists() else raw_label_path
        if not label_path.exists(): continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None: continue
        height, width = image.shape[:2]
        boxes, errors = parse_yolo_label_text(label_path.read_text(encoding="utf-8"), width, height)
        if errors: raise ValueError(f"invalid labels in {label_path}: {' | '.join(errors)}")
        geometry = detect_pink_obstacle_geometry(image)
        sidecar = geometry_dir / f"{image_path.stem}.json"
        self_muzzle = None
        if sidecar.exists():
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            lines = [LineObstacle(tuple(item["start"]), tuple(item["end"])) for item in payload.get("lines", [])]
            geometry = ObstacleGeometry([], lines)
            if payload.get("self_muzzle"):
                self_muzzle = tuple(payload["self_muzzle"])
        # Class 1 is now a keypoint action (muzzle), not a detection object.
        # This also discards legacy class-1 ally boxes from the old dataset.
        annotations = convert_boxes([box for box in boxes if box.class_id != 1], width, height, geometry, self_muzzle)
        shutil.copy2(image_path, image_dir / image_path.name)
        lines = [format_pose_label(item.box.class_id, (item.box.center_x, item.box.center_y, item.box.width, item.box.height), item.keypoints, width, height) for item in annotations]
        (label_dir / label_path.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        count += 1
    (output_dir / "dataset.yaml").write_text(f"path: {output_dir.resolve()}\ntrain: images\nval: images\nkpt_shape: [2, 3]\nnames:\n  0: enemy\n  1: self_muzzle\n  2: self\n  3: obstacle_circle\n  4: obstacle_line\n  5: portal_orange\n  6: portal_blue\n  7: blackhole\n  8: double_damage\n  9: Triple_damage\n", encoding="utf-8")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("train/yolo_captures/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/yolo_pose_dataset"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("train/pose_geometry"))
    parser.add_argument("--override-dir", type=Path, default=Path("train/annotation_overrides"))
    args = parser.parse_args()
    print(f"Prepared {prepare(args.raw_dir, args.output_dir, args.geometry_dir, args.override_dir)} images in {args.output_dir}")


if __name__ == "__main__": main()

"""Prepare a grouped YOLO Pose dataset from the manual ShellShock labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

import cv2

from shellshock_detector.obstacle_geometry import LineObstacle, ObstacleGeometry
from shellshock_detector.pose_dataset import convert_boxes, format_pose_label
from shellshock_detector.yolo_dataset import parse_yolo_label_text

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "enemy", 1: "self_center_keypoint", 2: "self", 3: "obstacle_circle",
    4: "obstacle_line", 5: "portal_orange", 6: "portal_blue", 7: "blackhole",
    8: "double_damage", 9: "Triple_damage",
}


def _group_key(stem: str) -> str:
    return stem[:13]


def _load_geometry(path: Path):
    if not path.exists():
        return ObstacleGeometry([], []), None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [LineObstacle(tuple(item["start"]), tuple(item["end"])) for item in payload.get("lines", [])]
    center = payload.get("self_center")
    self_center = tuple(center) if isinstance(center, (list, tuple)) and len(center) == 2 else None
    circles = payload.get("circles")
    return ObstacleGeometry([], lines), self_center, circles if isinstance(circles, list) else None


def _write_yaml(output_dir: Path) -> None:
    lines = [f"path: {output_dir.resolve().as_posix()}", "train: images/train", "val: images/val", "kpt_shape: [2, 3]", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_stems(stems: list[str], val_fraction: float, seed: int) -> set[str]:
    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(_group_key(stem), []).append(stem)
    group_names = sorted(groups)
    if len(group_names) < 2:
        return set()
    random.Random(seed).shuffle(group_names)
    count = max(1, min(len(group_names) - 1, round(len(group_names) * val_fraction)))
    return {stem for name in group_names[:count] for stem in groups[name]}


def prepare(raw_dir: Path, output_dir: Path, geometry_dir: Path = Path("train/annotate_check/pose_geometry"),
            override_dir: Path = Path("train/annotate_check/labels"), val_fraction: float = 0.2,
            seed: int = 42, overwrite: bool = False) -> int:
    raw_dir, output_dir, geometry_dir, override_dir = map(Path, (raw_dir, output_dir, geometry_dir, override_dir))
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    image_root = raw_dir / "images" if (raw_dir / "images").is_dir() else raw_dir
    label_root = raw_dir / "labels" if (raw_dir / "labels").is_dir() else raw_dir
    samples = []
    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        raw_label = label_root / f"{image_path.stem}.txt"
        label_path = override_dir / raw_label.name if (override_dir / raw_label.name).exists() else raw_label
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read image: {image_path}")
        height, width = image.shape[:2]
        boxes, errors = parse_yolo_label_text(label_path.read_text(encoding="utf-8"), width, height)
        if errors:
            raise ValueError(f"invalid labels in {label_path}: {' | '.join(errors)}")
        boxes = [box for box in boxes if box.class_id != 1]
        geometry, self_center, circles = _load_geometry(geometry_dir / f"{image_path.stem}.json")
        annotations = convert_boxes(boxes, width, height, geometry, None, self_center, circles)
        text = "\n".join(format_pose_label(item.box.class_id,
            (item.box.center_x, item.box.center_y, item.box.width, item.box.height),
            item.keypoints, width, height) for item in annotations)
        samples.append((image_path, text + ("\n" if text else "")))
    val_stems = _split_stems([image.stem for image, _ in samples], val_fraction, seed)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for image_path, text in samples:
        split = "val" if image_path.stem in val_stems else "train"
        shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
        (output_dir / "labels" / split / f"{image_path.stem}.txt").write_text(text, encoding="utf-8")
    _write_yaml(output_dir)
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("train/annotate_check"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/yolo_pose_dataset_v2"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("train/annotate_check/pose_geometry"))
    parser.add_argument("--override-dir", type=Path, default=Path("train/annotate_check/labels"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(f"Prepared {prepare(**vars(args))} images in {args.output_dir}")


if __name__ == "__main__":
    main()

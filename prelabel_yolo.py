"""Run the existing detection model to seed annotation overrides for review."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2

CANONICAL = {
    "enemy": 0,
    "self": 2,
    "obstacle_circle": 3,
    "obstacle_line": 4,
    "portal_orange": 5,
    "portal_blue": 6,
    "blackhole": 7,
    "double_damage": 8,
    "triple_damage": 9,
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _name_map(names):
    values = names.values() if isinstance(names, dict) else names
    return {str(name).lower(): index for index, name in (names.items() if isinstance(names, dict) else enumerate(values))}


def _label_lines(result, names, width, height, confidence):
    name_by_id = {int(k): str(v).lower() for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    lines = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return lines
    for bounds, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        if float(score) < confidence:
            continue
        source_name = name_by_id.get(int(class_id), "")
        target_id = CANONICAL.get(source_name)
        # The old model's class 1 was ally. It must not become the new muzzle point.
        if target_id is None:
            continue
        left, top, right, bottom = (float(v) for v in bounds)
        if right <= left or bottom <= top:
            continue
        lines.append(f"{target_id} {(left + right) / 2 / width:.6f} {(top + bottom) / 2 / height:.6f} {(right - left) / width:.6f} {(bottom - top) / height:.6f}")
    return lines


def run(weights: Path, image_dir: Path, override_dir: Path, confidence: float, overwrite: bool) -> int:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    created = 0
    for image_path in images:
        output_path = override_dir / f"{image_path.stem}.txt"
        if output_path.exists() and not overwrite:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        results = model.predict(source=image, verbose=False, conf=confidence)
        lines = _label_lines(results[0], model.names, width, height, confidence) if results else []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        created += 1
    print(f"预标注完成：写入 {created} 个覆盖层；已有覆盖层默认保留。")
    return created


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("train/runs/shellshock_yolo11n_cv65_final_all/weights/best.pt"))
    parser.add_argument("--image-dir", type=Path, default=Path("train/yolo_captures/full"))
    parser.add_argument("--override-dir", type=Path, default=Path("train/yolo_captures/labels"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.weights, args.image_dir, args.override_dir, args.confidence, args.overwrite)

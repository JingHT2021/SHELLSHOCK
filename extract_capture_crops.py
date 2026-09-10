"""Extract wind and wormhole crops from existing full captures.

This is intentionally separate from the interactive annotator.  It can repair
old captures without adding wind OCR or image processing latency to labeling.
Wormhole boxes come from the per-image YOLO labels (classes 5 and 6).
"""

from __future__ import annotations

import argparse
import json
import json
from pathlib import Path

import cv2

from shellshock_detector.dataset_capture import _crop
from shellshock_detector.wind import detect_wind


PORTAL_CLASS_IDS = {5, 6}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def audit_dataset(image_dir: Path, wind_dir: Path, wind_labels_dir: Path) -> dict[str, int]:
    """Report alignment between full screenshots, wind crops and labels."""
    image_stems = {path.stem for path in Path(image_dir).iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    crop_stems = {path.stem for path in Path(wind_dir).iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES} if Path(wind_dir).exists() else set()
    label_paths = list(Path(wind_labels_dir).glob("*.json")) if Path(wind_labels_dir).exists() else []
    label_stems = {path.stem for path in label_paths}
    invalid_labels = 0
    for path in label_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = float(payload["wind_value"])
            if not 0 <= value <= 100:
                invalid_labels += 1
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            invalid_labels += 1
    return {
        "images": len(image_stems),
        "wind_crops": len(crop_stems),
        "wind_labels": len(label_stems),
        "missing_wind": len(image_stems - crop_stems),
        "orphan_labels": len(label_stems - image_stems),
        "invalid_labels": invalid_labels,
    }


def yolo_boxes(label_path: Path, width: int, height: int):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, box_width, box_height = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        if class_id not in PORTAL_CLASS_IDS:
            continue
        boxes.append(
            (
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                box_width * width,
                box_height * height,
            )
        )
    return boxes


def extract_one(image_path: Path, labels_dir: Path, output_dir: Path, wind_labels_dir: Path, padding: int = 12):
    image = cv2.imread(str(image_path))
    if image is None:
        return {"image": image_path.name, "wind": 0, "wormholes": 0, "error": "unreadable"}
    height, width = image.shape[:2]
    output_dir = Path(output_dir)
    wind_dir = output_dir / "wind"
    wormhole_dir = output_dir / "wormholes"
    wind_dir.mkdir(parents=True, exist_ok=True)
    wormhole_dir.mkdir(parents=True, exist_ok=True)
    wind, _, wind_box = detect_wind(image)
    wind_crop = _crop(image, wind_box, padding)
    wind_count = 0
    if wind_crop is not None:
        wind_path = wind_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(wind_path), wind_crop):
            raise RuntimeError(f"failed to save {wind_path}")
        wind_count = 1
        wind_labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = Path(wind_labels_dir) / f"{image_path.stem}.json"
        if label_path.exists():
            # Keep manual wind corrections.  This tool repairs image crops;
            # it must not replace values edited in the annotation workflow.
            try:
                payload = json.loads(label_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload["image"] = str(wind_path)
            payload.setdefault("stem", image_path.stem)
        else:
            payload = {
                "image": str(wind_path),
                "stem": image_path.stem,
                "wind_value": abs(float(wind.value or 0)),
                "wind_direction": str(wind.direction or "right"),
                "wind_signed": abs(float(wind.value or 0)) * (-1 if wind.direction == "left" else 1),
                "source": "detected",
            }
        (wind_labels_dir / f"{image_path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    wormhole_count = 0
    for index, box in enumerate(yolo_boxes(Path(labels_dir) / f"{image_path.stem}.txt", width, height)):
        crop = _crop(image, box, padding)
        if crop is None:
            continue
        path = wormhole_dir / f"{image_path.stem}_{index:02d}.png"
        if cv2.imwrite(str(path), crop):
            wormhole_count += 1
    return {"image": image_path.name, "wind": wind_count, "wormholes": wormhole_count}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("train/annotate_check/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("train/annotate_check/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/annotate_check"))
    parser.add_argument("--wind-labels-dir", type=Path, default=Path("train/annotate_check/wind_labels"))
    parser.add_argument("--padding", type=int, default=12)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    images = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    totals = {"images": 0, "wind": 0, "wormholes": 0}
    for image_path in images:
        result = extract_one(image_path, args.labels_dir, args.output_dir, args.wind_labels_dir, args.padding)
        totals["images"] += 1
        totals["wind"] += result.get("wind", 0)
        totals["wormholes"] += result.get("wormholes", 0)
    print(totals)


if __name__ == "__main__":
    main()

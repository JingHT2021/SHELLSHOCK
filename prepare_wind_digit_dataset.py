"""Build grouped single-digit samples from wind HUD crops and labels."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def digit_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = gray.shape[:2]
    boxes = []
    for x, y, box_width, box_height, area in stats[1:count]:
        if (box_height >= max(8, int(height * 0.28))
                and box_width <= max(40, width * 0.7)
                and area >= max(8, box_width * box_height * 0.08)
                and y >= height * 0.15
                and y + box_height <= height * 0.9):
            boxes.append((int(x), int(y), int(box_width), int(box_height)))
    return sorted(boxes)


def normalize_digit(image: np.ndarray, size: tuple[int, int] = (32, 48)) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("digit crop has no foreground")
    crop = mask[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    target_w, target_h = size
    scale = min((target_h - 4) / crop.shape[0], (target_w - 4) / crop.shape[1])
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    top = (target_h - resized.shape[0]) // 2
    left = (target_w - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def split_stems(stems: list[str], seed: int = 42) -> dict[str, list[str]]:
    ordered = sorted(set(stems))
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    train_end = max(1, round(n * 0.70))
    validation_end = min(n - 1, train_end + max(1, round(n * 0.15)))
    return {"train": sorted(ordered[:train_end]), "validation": sorted(ordered[train_end:validation_end]), "test": sorted(ordered[validation_end:])}


def build_dataset(labels_dir: Path, crop_dir: Path, output_dir: Path, seed: int = 42) -> dict[str, int]:
    output_dir = Path(output_dir)
    digit_dir = output_dir / "digits"
    qa_dir = output_dir / "qa"
    digit_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    records = []
    review = []
    for label_path in sorted(Path(labels_dir).glob("*.json")):
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            numeric = float(payload["wind_value"])
            value = int(numeric)
            if numeric != value or not 0 <= value <= 100:
                raise ValueError("wind value must be an integer from 0 to 100")
            crop_path = Path(payload.get("image", crop_dir / f"{label_path.stem}.png"))
            if not crop_path.is_absolute():
                crop_path = Path(crop_path)
                if not crop_path.exists():
                    crop_path = Path(crop_dir) / crop_path.name
            image = cv2.imread(str(crop_path))
            if image is None:
                raise ValueError("wind crop is missing or unreadable")
            boxes = digit_boxes(image)
            expected = str(value)
            if len(boxes) != len(expected):
                review.append((label_path.stem, value, len(boxes), "component_count_mismatch"))
                continue
            for position, (digit, (x, y, width, height)) in enumerate(zip(expected, boxes)):
                sample = normalize_digit(image[y:y + height, x:x + width])
                split_path = digit_dir / f"{label_path.stem}_{position}_{digit}.png"
                cv2.imwrite(str(split_path), sample)
                records.append({"source_stem": label_path.stem, "digit_path": str(split_path), "digit_label": digit,
                                "position": position, "wind_value": value, "direction": payload.get("wind_direction", "right"),
                                "source": payload.get("source", "unknown")})
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            review.append((label_path.stem, "", "", str(error)))
    splits = split_stems([row["source_stem"] for row in records], seed)
    split_by_stem = {name: set(values) for name, values in splits.items()}
    for row in records:
        row["split"] = next(name for name, stems in split_by_stem.items() if row["source_stem"] in stems)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_stem", "digit_path", "digit_label", "position", "wind_value", "direction", "source", "split"])
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "splits.json").write_text(json.dumps({k: v for k, v in splits.items()}, indent=2) + "\n", encoding="utf-8")
    with (qa_dir / "review_required.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_stem", "wind_value", "detected_components", "reason"])
        writer.writerows(review)
    return {"labels": len(list(Path(labels_dir).glob("*.json"))), "samples": len(records), "review_required": len(review),
            "train_stems": len(splits["train"]), "validation_stems": len(splits["validation"]), "test_stems": len(splits["test"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=Path("train/annotate_check/wind_labels"))
    parser.add_argument("--crop-dir", type=Path, default=Path("train/annotate_check/wind"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/annotate_check/wind_model_data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.labels_dir, args.crop_dir, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()

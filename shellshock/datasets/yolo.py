"""Non-destructive preparation utilities for the ShellShock YOLO dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import random
import shutil

import cv2
import numpy as np


CLASS_NAMES = {
    0: "enemy",
    1: "self_muzzle",
    2: "self",
    3: "obstacle_circle",
    4: "obstacle_line",
    5: "portal_orange",
    6: "portal_blue",
    7: "blackhole",
    8: "double_damage",
    9: "Triple_damage",
}
VALID_CLASS_IDS = frozenset(CLASS_NAMES)


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class PreparationConfig:
    seed: int = 42
    val_fraction: float = 0.20
    crop_height: int = 1850
    preview_count: int = 12
    detect_enemy_candidates: bool = True
    exclude_unlabelled_enemy_candidates: bool = False
    portal_review_dir: Path | None = None
    annotation_override_dir: Path | None = None


@dataclass
class PreparedSample:
    stem: str
    image_path: Path
    label_path: Path
    image: np.ndarray
    boxes: list[YoloBox]
    candidate_boxes: list[YoloBox]


def parse_yolo_label_text(
    text: str, image_width: int, image_height: int
) -> tuple[list[YoloBox], list[str]]:
    """Parse valid YOLO boxes and collect human-readable errors for invalid rows."""
    boxes: list[YoloBox] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        parts = raw_line.split()
        if not parts:
            continue
        prefix = f"line_{line_number}"
        if len(parts) != 5:
            errors.append(f"{prefix}:field_count_{len(parts)}")
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"{prefix}:class_id_not_integer")
            continue
        try:
            center_x, center_y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"{prefix}:coordinate_not_float")
            continue
        if class_id not in VALID_CLASS_IDS:
            errors.append(f"{prefix}:class_id_out_of_range_{class_id}")
            continue
        if not all(math.isfinite(value) for value in (center_x, center_y, width, height)):
            errors.append(f"{prefix}:coordinate_not_finite")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"{prefix}:non_positive_size")
            continue
        left, top = center_x - width / 2, center_y - height / 2
        right, bottom = center_x + width / 2, center_y + height / 2
        # Six-decimal YOLO serialization can put an edge a few ulps outside
        # the image (for example 1.0000005). Accept only this rounding noise
        # and clip it; genuinely invalid boxes remain rejected.
        epsilon = 2e-6
        if left < -epsilon or top < -epsilon or right > 1 + epsilon or bottom > 1 + epsilon:
            errors.append(f"{prefix}:box_outside_image")
            continue
        left, top, right, bottom = max(0., left), max(0., top), min(1., right), min(1., bottom)
        boxes.append(YoloBox(class_id, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top))
    return boxes, errors


def _image_paths(directory: Path) -> list[Path]:
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in image_extensions),
        key=lambda path: path.name,
    )


def portal_counts(boxes: list[YoloBox]) -> tuple[int, int]:
    """Return the number of orange and blue portal boxes in one image."""
    return (
        sum(box.class_id == 5 for box in boxes),
        sum(box.class_id == 6 for box in boxes),
    )


def export_portal_mismatches(
    raw_dir: Path, review_dir: Path, *, overwrite: bool = False
) -> dict[str, int]:
    """Copy unequal orange/blue portal samples to a separate manual-review directory."""
    raw_dir, review_dir = Path(raw_dir), Path(review_dir)
    if review_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"review directory already exists: {review_dir}; choose another path or pass overwrite=True"
            )
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True)
    geometry_dir = raw_dir.parent / "geometry"
    rows: list[tuple[str, int, int, str]] = []
    invalid_labels = 0
    for image_path in _image_paths(raw_dir):
        label_path = raw_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        boxes, errors = parse_yolo_label_text(
            label_path.read_text(encoding="utf-8"), image.shape[1], image.shape[0]
        )
        if errors:
            invalid_labels += 1
            continue
        orange, blue = portal_counts(boxes)
        if orange == blue:
            continue
        shutil.copy2(image_path, review_dir / image_path.name)
        shutil.copy2(label_path, review_dir / label_path.name)
        geometry_path = geometry_dir / f"{image_path.stem}.json"
        if geometry_path.exists():
            shutil.copy2(geometry_path, review_dir / geometry_path.name)
        rows.append((image_path.name, orange, blue, "orange_blue_count_mismatch"))
    with (review_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "portal_orange", "portal_blue", "reason"))
        writer.writerows(rows)
    return {"mismatch_images": len(rows), "invalid_label_images": invalid_labels}


def build_merged_corpus(
    raw_dir: Path,
    review_dir: Path,
    supplemental_dir: Path,
    output_dir: Path,
    *,
    crop_height: int = 1850,
    raw_stems: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build a deduplicated corpus, giving reviewed image/labels priority by stem.

    ``raw_stems`` limits only the raw source.  Review samples always remain in the
    corpus so manual portal corrections are never accidentally omitted.
    """
    raw_dir, review_dir = Path(raw_dir), Path(review_dir)
    supplemental_dir, output_dir = Path(supplemental_dir), Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"corpus output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    sources = {
        path.stem: (path, raw_dir / f"{path.stem}.txt", "raw")
        for path in _image_paths(raw_dir)
        if raw_stems is None or path.stem in raw_stems
    }
    sources.update({path.stem: (path, review_dir / f"{path.stem}.txt", "review") for path in _image_paths(review_dir)})
    images_dir, labels_dir = output_dir / "images", output_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    samples: list[PreparedSample] = []
    supplemental_added = 0
    for stem, (image_path, label_path, source) in sorted(sources.items()):
        if not label_path.exists():
            raise FileNotFoundError(f"missing label for {image_path}: {label_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read {image_path}")
        height, width = image.shape[:2]
        boxes, errors = parse_yolo_label_text(label_path.read_text(encoding="utf-8"), width, height)
        if errors:
            raise ValueError(f"invalid labels in {label_path}: {' | '.join(errors)}")
        boxes, supplement_errors, added, _ = merge_supplemental_boxes(
            boxes, supplemental_dir / f"{stem}.txt", width, height
        )
        if supplement_errors:
            raise ValueError(f"invalid supplemental labels for {stem}: {' | '.join(supplement_errors)}")
        supplemental_added += added
        if width == 3840 and height == 2000:
            image = image[:crop_height, :]
            boxes = [
                YoloBox(*result)
                for box in boxes
                if (result := transform_yolo_box_for_crop(
                    box.class_id, box.center_x, box.center_y, box.width, box.height,
                    width, height, crop_height,
                )) is not None
            ]
        destination_image = images_dir / image_path.name
        if not cv2.imwrite(str(destination_image), image):
            raise RuntimeError(f"unable to write {destination_image}")
        destination_label = labels_dir / f"{stem}.txt"
        destination_label.write_text(
            "\n".join(_format_box(box) for box in boxes) + ("\n" if boxes else ""), encoding="utf-8"
        )
        samples.append(PreparedSample(stem, destination_image, destination_label, image, boxes, []))
        rows.append({"stem": stem, "source": source, "image": image_path.name, "boxes": len(boxes)})
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "source", "image", "boxes"))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {
        "unique_images": len(samples),
        "raw_source_images": sum(row["source"] == "raw" for row in rows),
        "review_source_images": sum(row["source"] == "review" for row in rows),
        "supplemental_enemy_boxes_added": supplemental_added,
        "classes": _statistics(samples),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def make_multilabel_folds(
    stem_classes: dict[str, set[int]], *, n_splits: int = 5, seed: int = 42
) -> list[set[str]]:
    """Assign every stem to exactly one deterministic, approximately multilabel-balanced fold."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if len(stem_classes) < n_splits:
        raise ValueError("need at least n_splits samples")
    rng = random.Random(seed)
    class_frequency = Counter(class_id for classes in stem_classes.values() for class_id in classes)
    randomized = {stem: rng.random() for stem in stem_classes}
    ordered = sorted(
        stem_classes,
        key=lambda stem: (
            min((class_frequency[class_id] for class_id in stem_classes[stem]), default=len(stem_classes)),
            -len(stem_classes[stem]), randomized[stem], stem,
        ),
    )
    folds = [set() for _ in range(n_splits)]
    fold_counts = [Counter() for _ in range(n_splits)]
    for stem in ordered:
        classes = stem_classes[stem]
        fold_index = min(
            range(n_splits),
            key=lambda index: (
                sum(fold_counts[index][class_id] / max(1, class_frequency[class_id]) for class_id in classes),
                len(folds[index]),
                index,
            ),
        )
        folds[fold_index].add(stem)
        fold_counts[fold_index].update(classes)
    return folds


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _write_dataset_yaml_at(dataset_dir: Path, train_path: str, val_path: str) -> None:
    lines = [f"path: {dataset_dir.resolve().as_posix()}", f"train: {train_path}", f"val: {val_path}", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    (dataset_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fivefold_datasets(
    corpus_dir: Path, output_dir: Path, folds: list[set[str]], *, overwrite: bool = False
) -> dict[str, object]:
    """Materialize five YOLO datasets and an all-data training dataset from a corpus."""
    corpus_dir, output_dir = Path(corpus_dir), Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"five-fold output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    all_stems = sorted(path.stem for path in (corpus_dir / "images").iterdir() if path.is_file())
    rows: list[dict[str, object]] = []
    for index, validation_stems in enumerate(folds, start=1):
        fold_dir = output_dir / f"fold{index}"
        for split, stems in (("train", [stem for stem in all_stems if stem not in validation_stems]), ("val", sorted(validation_stems))):
            image_dir, label_dir = fold_dir / "images" / split, fold_dir / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for stem in stems:
                source_image = next((corpus_dir / "images").glob(f"{stem}.*"))
                _link_or_copy(source_image, image_dir / source_image.name)
                _link_or_copy(corpus_dir / "labels" / f"{stem}.txt", label_dir / f"{stem}.txt")
                rows.append({"fold": index, "split": split, "stem": stem})
        _write_dataset_yaml_at(fold_dir, "images/train", "images/val")
    final_dir = output_dir / "final_all"
    final_images, final_labels = final_dir / "images" / "train", final_dir / "labels" / "train"
    final_images.mkdir(parents=True)
    final_labels.mkdir(parents=True)
    for stem in all_stems:
        source_image = next((corpus_dir / "images").glob(f"{stem}.*"))
        _link_or_copy(source_image, final_images / source_image.name)
        _link_or_copy(corpus_dir / "labels" / f"{stem}.txt", final_labels / f"{stem}.txt")
    _write_dataset_yaml_at(final_dir, "images/train", "images/train")
    with (output_dir / "folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("fold", "split", "stem"))
        writer.writeheader()
        writer.writerows(rows)
    return {"folds": len(folds), "unique_images": len(all_stems), "final_dataset": str(final_dir)}

def transform_yolo_box_for_crop(
    class_id: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    crop_height: int,
) -> tuple[int, float, float, float, float] | None:
    """Clip a normalized YOLO box to a top-aligned crop and renormalize it."""
    left = (center_x - width / 2) * image_width
    top = (center_y - height / 2) * image_height
    right = (center_x + width / 2) * image_width
    bottom = (center_y + height / 2) * image_height
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(image_width), right)
    bottom = min(float(image_height), float(crop_height), bottom)
    if right <= left or bottom <= top:
        return None
    return (
        class_id,
        (left + right) / 2 / image_width,
        (top + bottom) / 2 / crop_height,
        (right - left) / image_width,
        (bottom - top) / crop_height,
    )


def _format_box(box: YoloBox) -> str:
    """Serialize a box without introducing a rounding-only boundary violation."""
    precision = 8
    scale = 10**precision
    width = float(f"{box.width:.{precision}f}")
    height = float(f"{box.height:.{precision}f}")
    center_x = float(f"{box.center_x:.{precision}f}")
    center_y = float(f"{box.center_y:.{precision}f}")

    def clamp_center(center: float, size: float) -> float:
        lower = math.ceil((size / 2) * scale - 1e-9) / scale
        upper = math.floor((1 - size / 2) * scale + 1e-9) / scale
        return min(max(center, lower), upper)

    center_x = clamp_center(center_x, width)
    center_y = clamp_center(center_y, height)
    return (
        f"{box.class_id} {center_x:.{precision}f} {center_y:.{precision}f} "
        f"{width:.{precision}f} {height:.{precision}f}"
    )


def _box_iou(first: YoloBox, second: YoloBox) -> float:
    first_left, first_top = first.center_x - first.width / 2, first.center_y - first.height / 2
    first_right, first_bottom = first.center_x + first.width / 2, first.center_y + first.height / 2
    second_left, second_top = second.center_x - second.width / 2, second.center_y - second.height / 2
    second_right, second_bottom = second.center_x + second.width / 2, second.center_y + second.height / 2
    left, top = max(first_left, second_left), max(first_top, second_top)
    right, bottom = min(first_right, second_right), min(first_bottom, second_bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def find_red_enemy_candidates(image: np.ndarray) -> list[YoloBox]:
    """Return conservative red connected-component candidates for manual review.

    They are not training annotations: appearance candidates merely prevent a partly
    annotated frame from silently teaching YOLO that a possible enemy is background.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 110, 90)), np.array((10, 255, 255)))
    mask |= cv2.inRange(hsv, np.array((170, 110, 90)), np.array((180, 255, 255)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = image.shape[:2]
    candidates: list[YoloBox] = []
    min_area = max(80, int(width * height * 0.00001))
    max_area = int(width * height * 0.02)
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if not (min_area <= area <= max_area and 8 <= component_width <= width * 0.18 and 6 <= component_height <= height * 0.2):
            continue
        aspect = component_width / component_height
        if not 0.20 <= aspect <= 5.0:
            continue
        candidates.append(YoloBox(0, (x + component_width / 2) / width, (y + component_height / 2) / height, component_width / width, component_height / height))
    return candidates


def merge_supplemental_boxes(
    boxes: list[YoloBox], supplemental_path: Path, image_width: int, image_height: int
) -> tuple[list[YoloBox], list[str], int, int]:
    """Append distinct supplemental enemy boxes and reject non-enemy supplemental labels."""
    if not supplemental_path.exists():
        return list(boxes), [], 0, 0
    supplemental, errors = parse_yolo_label_text(
        supplemental_path.read_text(encoding="utf-8"), image_width, image_height
    )
    merged = list(boxes)
    added = 0
    duplicates = 0
    for candidate in supplemental:
        if candidate.class_id != 0:
            errors.append("class_id_must_be_enemy")
            continue
        if any(box.class_id == 0 and _box_iou(box, candidate) >= 0.90 for box in merged):
            duplicates += 1
            continue
        merged.append(candidate)
        added += 1
    return merged, [f"supplemental:{error}" for error in errors], added, duplicates


def _stratified_split(samples: list[PreparedSample], config: PreparationConfig) -> tuple[set[str], list[str]]:
    """Choose validation stems with deterministic greedy coverage of rare labels."""
    if len(samples) < 2:
        return set(), ["fewer_than_two_eligible_images; validation set is empty"]
    target_count = max(1, min(len(samples) - 1, round(len(samples) * config.val_fraction)))
    rng = random.Random(config.seed)
    by_class: dict[int, list[PreparedSample]] = {class_id: [] for class_id in CLASS_NAMES}
    for sample in samples:
        for class_id in {box.class_id for box in sample.boxes}:
            by_class[class_id].append(sample)
    warnings: list[str] = []
    for class_id, matching in by_class.items():
        if not matching:
            warnings.append(f"{CLASS_NAMES[class_id]}: 0 eligible images")
        elif len(matching) < 2:
            warnings.append(f"{CLASS_NAMES[class_id]}: only {len(matching)} eligible image(s), cannot cover both splits")
    validation: set[str] = set()
    for class_id in sorted(CLASS_NAMES, key=lambda item: (len(by_class[item]), item)):
        matching = by_class[class_id]
        if len(validation) >= target_count or len(matching) < 2:
            continue
        shuffled = matching[:]
        rng.shuffle(shuffled)
        choice = next((sample for sample in shuffled if sample.stem not in validation), None)
        if choice:
            validation.add(choice.stem)
    remaining = [sample for sample in samples if sample.stem not in validation]
    rng.shuffle(remaining)
    validation.update(sample.stem for sample in remaining[: target_count - len(validation)])
    return validation, warnings


def _statistics(samples: list[PreparedSample]) -> dict[str, dict[str, int]]:
    counts = {CLASS_NAMES[class_id]: {"images": 0, "boxes": 0} for class_id in CLASS_NAMES}
    for sample in samples:
        present = {box.class_id for box in sample.boxes}
        for class_id in present:
            counts[CLASS_NAMES[class_id]]["images"] += 1
        for box in sample.boxes:
            counts[CLASS_NAMES[box.class_id]]["boxes"] += 1
    return counts


def _draw_preview(image: np.ndarray, boxes: list[YoloBox], candidates: list[YoloBox]) -> np.ndarray:
    preview = image.copy()
    colors = {0: (0, 0, 255), 2: (0, 255, 0), 3: (0, 255, 255), 4: (255, 255, 0), 5: (0, 165, 255), 6: (255, 0, 0), 9: (255, 0, 255)}
    height, width = preview.shape[:2]
    for box in boxes:
        left, top = int((box.center_x - box.width / 2) * width), int((box.center_y - box.height / 2) * height)
        right, bottom = int((box.center_x + box.width / 2) * width), int((box.center_y + box.height / 2) * height)
        color = colors.get(box.class_id, (200, 200, 200))
        cv2.rectangle(preview, (left, top), (right, bottom), color, 3)
        cv2.putText(preview, CLASS_NAMES[box.class_id], (left, max(22, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for candidate in candidates:
        left, top = int((candidate.center_x - candidate.width / 2) * width), int((candidate.center_y - candidate.height / 2) * height)
        right, bottom = int((candidate.center_x + candidate.width / 2) * width), int((candidate.center_y + candidate.height / 2) * height)
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(preview, "review_enemy", (left, max(22, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return preview


def _write_yaml(output_dir: Path) -> None:
    path = output_dir.resolve().as_posix()
    lines = [f"path: {path}", "train: images/train", "val: images/val", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(raw_dir: Path, output_dir: Path, config: PreparationConfig = PreparationConfig(), *, overwrite: bool = False) -> dict[str, object]:
    """Create a validated derived dataset, without writing anything into ``raw_dir``."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}; choose a new directory or pass overwrite=True explicitly")
        shutil.rmtree(output_dir)
    images = {path.stem: path for path in _image_paths(raw_dir)}
    labels = {path.stem: path for path in raw_dir.glob("*.txt")}
    excluded: list[dict[str, str]] = []
    for stem in sorted(set(images) - set(labels)):
        excluded.append({"stem": stem, "reason": "image_missing_matching_txt"})
    for stem in sorted(set(labels) - set(images)):
        excluded.append({"stem": stem, "reason": "txt_missing_matching_image"})

    supplemental_dir = raw_dir.parent / "supplemental_labels"
    eligible: list[PreparedSample] = []
    candidate_rows: list[dict[str, object]] = []
    empty_label_images = 0
    supplemental_enemy_boxes_added = 0
    supplemental_enemy_duplicates_skipped = 0
    portal_review_label_overrides = 0
    for stem in sorted(set(images) & set(labels)):
        image_path, label_path = images[stem], labels[stem]
        review_label_path = (
            Path(config.portal_review_dir) / f"{stem}.txt"
            if config.portal_review_dir is not None
            else None
        )
        override_label_path = (
            Path(config.annotation_override_dir) / f"{stem}.txt"
            if config.annotation_override_dir is not None
            else raw_dir.parent / "annotation_overrides" / f"{stem}.txt"
        )
        if override_label_path.exists():
            label_path = override_label_path
            source = "annotation_override"
        elif review_label_path is not None and review_label_path.exists():
            label_path = review_label_path
            source = "portal_review"
            portal_review_label_overrides += 1
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            excluded.append({"stem": stem, "reason": "unreadable_image"})
            continue
        image_height, image_width = image.shape[:2]
        boxes, errors = parse_yolo_label_text(label_path.read_text(encoding="utf-8"), image_width, image_height)
        if errors:
            excluded.append({"stem": stem, "reason": "invalid_label:" + "|".join(errors)})
            continue
        boxes, supplemental_errors, added, duplicates = merge_supplemental_boxes(
            boxes, supplemental_dir / f"{stem}.txt", image_width, image_height
        )
        supplemental_enemy_boxes_added += added
        supplemental_enemy_duplicates_skipped += duplicates
        if supplemental_errors:
            excluded.append({"stem": stem, "reason": "invalid_supplemental_label:" + "|".join(supplemental_errors)})
            continue
        if image_width == 3840 and image_height == 2000:
            image = image[: config.crop_height, :]
            transformed: list[YoloBox] = []
            for box in boxes:
                result = transform_yolo_box_for_crop(box.class_id, box.center_x, box.center_y, box.width, box.height, image_width, image_height, config.crop_height)
                if result is not None:
                    transformed.append(YoloBox(*result))
            boxes = transformed
        candidates = find_red_enemy_candidates(image) if config.detect_enemy_candidates else []
        unmatched = [candidate for candidate in candidates if not any(box.class_id == 0 and _box_iou(candidate, box) >= 0.35 for box in boxes)]
        for candidate in unmatched:
            candidate_rows.append({"stem": stem, "x_center": candidate.center_x, "y_center": candidate.center_y, "width": candidate.width, "height": candidate.height})
        if unmatched and config.exclude_unlabelled_enemy_candidates:
            excluded.append({"stem": stem, "reason": f"unlabelled_enemy_candidate_count_{len(unmatched)}"})
            continue
        if not boxes:
            empty_label_images += 1
        eligible.append(PreparedSample(stem, image_path, label_path, image, boxes, unmatched))

    validation_stems, warnings = _stratified_split(eligible, config)
    output_dir.mkdir(parents=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True)
        (output_dir / "labels" / split).mkdir(parents=True)
    for sample in eligible:
        split = "val" if sample.stem in validation_stems else "train"
        destination_image = output_dir / "images" / split / sample.image_path.name
        if not cv2.imwrite(str(destination_image), sample.image):
            raise RuntimeError(f"unable to write {destination_image}")
        (output_dir / "labels" / split / f"{sample.stem}.txt").write_text("\n".join(_format_box(box) for box in sample.boxes) + ("\n" if sample.boxes else ""), encoding="utf-8")
    _write_yaml(output_dir)
    previews_dir = output_dir / "previews"
    previews_dir.mkdir()
    preview_rng = random.Random(config.seed)
    preview_samples = eligible[:]
    preview_rng.shuffle(preview_samples)
    for sample in preview_samples[: config.preview_count]:
        cv2.imwrite(str(previews_dir / f"{sample.stem}.jpg"), _draw_preview(sample.image, sample.boxes, sample.candidate_boxes))
    with (output_dir / "excluded_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "reason"))
        writer.writeheader()
        writer.writerows(excluded)
    with (output_dir / "enemy_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "x_center", "y_center", "width", "height"))
        writer.writeheader()
        writer.writerows(candidate_rows)
    train_samples = [sample for sample in eligible if sample.stem not in validation_stems]
    val_samples = [sample for sample in eligible if sample.stem in validation_stems]
    summary: dict[str, object] = {
        "total_images": len(images), "total_labels": len(labels), "eligible_images": len(eligible),
        "train_images": len(train_samples), "val_images": len(val_samples), "excluded_images": len(excluded),
        "empty_label_images": empty_label_images, "unlabelled_enemy_candidates": len(candidate_rows),
        "supplemental_enemy_boxes_added": supplemental_enemy_boxes_added,
        "supplemental_enemy_duplicates_skipped": supplemental_enemy_duplicates_skipped,
        "portal_review_label_overrides": portal_review_label_overrides,
        "warnings": warnings, "all": _statistics(eligible), "train": _statistics(train_samples), "val": _statistics(val_samples),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_id", "class_name", "all_images", "all_boxes", "train_images", "train_boxes", "val_images", "val_boxes"))
        for class_id, class_name in CLASS_NAMES.items():
            writer.writerow((class_id, class_name, summary["all"][class_name]["images"], summary["all"][class_name]["boxes"], summary["train"][class_name]["images"], summary["train"][class_name]["boxes"], summary["val"][class_name]["images"], summary["val"][class_name]["boxes"]))
    return summary

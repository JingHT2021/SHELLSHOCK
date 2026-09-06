"""YOLO Detection sample export for manually marked game positions."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import cv2
import numpy as np


DEFAULT_BOX_SIZE_AT_REFERENCE = (38.4, 28.8)
"""(width, height) of a manual label box at a 1920-pixel client width."""


@dataclass(frozen=True)
class PinkObstacleConfig:
    """Strict HSV and geometry thresholds for reflective pink obstacles."""

    hsv_lower: tuple[int, int, int] = (0, 0, 230)
    hsv_upper: tuple[int, int, int] = (0, 0, 255)
    reference_width: int = 1920
    circle_min_radius: float = 25.0
    circle_max_radius: float = 350.0
    circle_min_circularity: float = 0.72
    line_min_length: float = 55.0
    line_min_thickness: float = 2.0
    line_max_thickness: float = 32.0
    line_min_aspect_ratio: float = 3.0
    hough_circle_min_radius: float = 80.0
    hough_circle_min_support: float = 0.45


@dataclass(frozen=True)
class ObstacleDetection:
    """One obstacle class and its pixel-space axis-aligned bounding box."""

    class_id: int
    box: tuple[int, int, int, int]


CLASS_NAMES = {
    0: "enemy",
    1: "ally",
    2: "self",
    3: "obstacle_circle",
    4: "obstacle_line",
    5: "portal_orange",
    6: "portal_blue",
}


def _scaled(value: float, image_width: int, config: PinkObstacleConfig) -> float:
    return value * image_width / config.reference_width


def detect_pink_obstacles(
    image: np.ndarray, config: PinkObstacleConfig = PinkObstacleConfig()
) -> list[ObstacleDetection]:
    """Return only high-confidence pink circle and line obstacle candidates."""
    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(config.hsv_lower), np.array(config.hsv_upper))
    kernel_size = max(3, int(round(_scaled(3, image.shape[1], config))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ring_support_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_radius = _scaled(config.circle_min_radius, image.shape[1], config)
    max_radius = _scaled(config.circle_max_radius, image.shape[1], config)
    min_line_length = _scaled(config.line_min_length, image.shape[1], config)
    min_line_thickness = _scaled(config.line_min_thickness, image.shape[1], config)
    max_line_thickness = _scaled(config.line_max_thickness, image.shape[1], config)
    detections: list[ObstacleDetection] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        circularity = 4 * pi * area / (perimeter * perimeter)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        aspect = width / height if height else 0.0
        if (
            min_radius <= radius <= max_radius
            and config.circle_min_circularity <= circularity
            and 0.80 <= aspect <= 1.25
        ):
            detections.append(ObstacleDetection(3, (x, y, width, height)))
            continue

        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        long_side, short_side = sorted((rect_width, rect_height), reverse=True)
        if (
            short_side > 0
            and long_side >= min_line_length
            and short_side >= min_line_thickness
            and short_side <= max_line_thickness
            and long_side / short_side >= config.line_min_aspect_ratio
        ):
            detections.append(ObstacleDetection(4, (x, y, width, height)))

    min_distance = max(20, int(round(_scaled(80, image.shape[1], config))))
    hough_circles = cv2.HoughCircles(
        cv2.GaussianBlur(mask, (9, 9), 2),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_distance,
        param1=50,
        param2=12,
        minRadius=max(1, int(round(max(min_radius, _scaled(config.hough_circle_min_radius, image.shape[1], config))))),
        maxRadius=max(1, int(round(max_radius))),
    )
    if hough_circles is not None:
        for center_x, center_y, radius in np.round(hough_circles[0]).astype(int):
            angles = np.linspace(0, 2 * pi, 720, endpoint=False)
            sample_x = np.rint(center_x + radius * np.cos(angles)).astype(int)
            sample_y = np.rint(center_y + radius * np.sin(angles)).astype(int)
            valid = (
                (sample_x >= 0)
                & (sample_x < image.shape[1])
                & (sample_y >= 0)
                & (sample_y < image.shape[0])
            )
            if (
                not valid.any()
                or float(np.mean(ring_support_mask[sample_y[valid], sample_x[valid]] > 0))
                < config.hough_circle_min_support
            ):
                continue
            circle_box = (center_x - radius, center_y - radius, radius * 2, radius * 2)
            candidate = ObstacleDetection(3, circle_box)
            if any(item.class_id == 3 and _iou(_box_to_center_box(item.box), _box_to_center_box(circle_box)) >= 0.50 for item in detections):
                continue
            detections.append(candidate)
    return sorted(detections, key=lambda item: (item.class_id, item.box[0], item.box[1]))


def _box_to_center_box(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return x + width / 2, y + height / 2, float(width), float(height)


def yolo_box_label_line(
    class_id: int,
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> str:
    """Return a clipped normalized YOLO line for an axis-aligned pixel box."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if class_id < 0:
        raise ValueError("class_id must be non-negative")
    x, y, width, height = box
    left = max(0, x)
    top = max(0, y)
    right = min(image_width, x + width)
    bottom = min(image_height, y + height)
    if right <= left or bottom <= top:
        raise ValueError("box is outside the saved image")
    return (
        f"{class_id} {(left + right) / 2 / image_width:.6f} "
        f"{(top + bottom) / 2 / image_height:.6f} "
        f"{(right - left) / image_width:.6f} {(bottom - top) / image_height:.6f}"
    )


def _parse_yolo_boxes(text: str, image_width: int, image_height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError:
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append((class_id, (center_x * image_width, center_y * image_height, width * image_width, height * image_height)))
    return boxes


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    first_left, first_top = first_x - first_width / 2, first_y - first_height / 2
    second_left, second_top = second_x - second_width / 2, second_y - second_height / 2
    left, top = max(first_left, second_left), max(first_top, second_top)
    right = min(first_left + first_width, second_left + second_width)
    bottom = min(first_top + first_height, second_top + second_height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union else 0.0


def append_yolo_obstacle_labels(
    existing_text: str,
    detections: list[ObstacleDetection],
    image_width: int,
    image_height: int,
    duplicate_iou: float = 0.90,
) -> tuple[str, int]:
    """Append unique obstacle labels while leaving every existing byte untouched."""
    parsed = _parse_yolo_boxes(existing_text, image_width, image_height)
    additions: list[str] = []
    for detection in detections:
        x, y, width, height = detection.box
        candidate = (x + width / 2, y + height / 2, float(width), float(height))
        if any(class_id == detection.class_id and _iou(box, candidate) >= duplicate_iou for class_id, box in parsed):
            continue
        additions.append(yolo_box_label_line(detection.class_id, detection.box, image_width, image_height))
        parsed.append((detection.class_id, candidate))
    if not additions:
        return existing_text, 0
    separator = "" if not existing_text or existing_text.endswith("\n") else "\n"
    return existing_text + separator + "\n".join(additions) + "\n", len(additions)


def _draw_yolo_preview(
    image: np.ndarray, label_text: str, hidden_class_ids: set[int] | None = None
) -> np.ndarray:
    preview = image.copy()
    hidden_class_ids = hidden_class_ids or set()
    # These preview colours deliberately fall outside PinkObstacleConfig's HSV range.
    colors = {3: (0, 255, 255), 4: (255, 255, 0)}
    for class_id, (center_x, center_y, width, height) in _parse_yolo_boxes(label_text, image.shape[1], image.shape[0]):
        if class_id in hidden_class_ids:
            continue
        left = int(round(center_x - width / 2))
        top = int(round(center_y - height / 2))
        right = int(round(center_x + width / 2))
        bottom = int(round(center_y + height / 2))
        color = colors.get(class_id, (0, 255, 255))
        cv2.rectangle(preview, (left, top), (right, bottom), color, 2)
        cv2.putText(preview, CLASS_NAMES.get(class_id, str(class_id)), (left, max(16, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return preview


def annotate_pink_obstacle_sample(
    source_image_path: Path,
    label_path: Path,
    preview_path: Path,
    config: PinkObstacleConfig = PinkObstacleConfig(),
    dry_run: bool = False,
) -> list[int]:
    """Append detected obstacle labels and render a preview; return classes newly added."""
    image = cv2.imread(str(source_image_path))
    if image is None:
        raise ValueError(f"unable to read image: {source_image_path}")
    existing = label_path.read_text(encoding="utf-8")
    detections = detect_pink_obstacles(image, config)
    merged = existing
    added_classes: list[int] = []
    for detection in detections:
        merged_candidate, added = append_yolo_obstacle_labels(
            merged, [detection], image.shape[1], image.shape[0]
        )
        if added:
            merged = merged_candidate
            added_classes.append(detection.class_id)
    if not dry_run and added_classes:
        label_path.write_text(merged, encoding="utf-8")
    if not dry_run:
        if not cv2.imwrite(str(preview_path), _draw_yolo_preview(image, merged)):
            raise RuntimeError(f"failed to write preview: {preview_path}")
    return added_classes


def yolo_label_line(
    class_id: int,
    point: tuple[int, int],
    image_width: int,
    image_height: int,
    box_size_at_reference: tuple[float, float] = DEFAULT_BOX_SIZE_AT_REFERENCE,
) -> str:
    """Return one clipped, normalized YOLO Detection label line."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if class_id < 0:
        raise ValueError("class_id must be non-negative")
    scale = image_width / 1920.0
    box_width = box_size_at_reference[0] * scale
    box_height = box_size_at_reference[1] * scale
    left = max(0.0, point[0] - box_width / 2)
    top = max(0.0, point[1] - box_height / 2)
    right = min(float(image_width), point[0] + box_width / 2)
    bottom = min(float(image_height), point[1] + box_height / 2)
    if right <= left or bottom <= top:
        raise ValueError("manual point is outside the saved image")
    return (
        f"{class_id} {(left + right) / 2 / image_width:.6f} "
        f"{(top + bottom) / 2 / image_height:.6f} "
        f"{(right - left) / image_width:.6f} {(bottom - top) / image_height:.6f}"
    )


def save_training_sample(
    image: np.ndarray,
    train_dir: Path,
    timestamp: str,
    annotations: list[tuple[int, tuple[int, int]]],
) -> tuple[Path, Path, Path]:
    """Save cropped image, same-stem YOLO labels, and an annotated preview."""
    raw_dir = train_dir / "raw_cropped"
    annotated_dir = train_dir / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{timestamp}.png"
    label_path = raw_dir / f"{timestamp}.txt"
    annotated_path = annotated_dir / f"{timestamp}.png"
    if not cv2.imwrite(str(raw_path), image):
        raise RuntimeError("failed to write training image")
    lines = [yolo_label_line(class_id, point, image.shape[1], image.shape[0]) for class_id, point in annotations]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    preview = image.copy()
    for class_id, point in annotations:
        scale = image.shape[1] / 1920.0
        width = int(round(DEFAULT_BOX_SIZE_AT_REFERENCE[0] * scale))
        height = int(round(DEFAULT_BOX_SIZE_AT_REFERENCE[1] * scale))
        cv2.rectangle(preview, (point[0] - width // 2, point[1] - height // 2), (point[0] + width // 2, point[1] + height // 2), (0, 255, 255), 2)
        cv2.putText(preview, str(class_id), (point[0], point[1] - height // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if not cv2.imwrite(str(annotated_path), preview):
        raise RuntimeError("failed to write annotated training image")
    return raw_path, label_path, annotated_path

"""Convert existing YOLO detection labels into two-keypoint pose labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .obstacle_geometry import ObstacleGeometry
from .yolo_dataset import YoloBox

POSE_KEYPOINT_COUNT = 2
CIRCLE_CLASSES = frozenset({3, 5, 6, 7, 8, 9})


@dataclass(frozen=True)
class PoseAnnotation:
    box: YoloBox
    keypoints: tuple[tuple[float, float, int], tuple[float, float, int]]


def circle_keypoints_from_box(box: tuple[float, float, float, float], image_width: int, image_height: int):
    center_x, center_y, width, height = box
    radius = max(width, height) * image_width / 2
    x, y = center_x * image_width, center_y * image_height
    return ((float(x), float(y), 2), (float(x + radius), float(y), 2))


def line_keypoints(start: tuple[int, int], end: tuple[int, int], image_width: int, image_height: int):
    return ((start[0] / image_width, start[1] / image_height, 2), (end[0] / image_width, end[1] / image_height, 2))


def _invisible_keypoints():
    return ((0.0, 0.0, 0), (0.0, 0.0, 0))


def format_pose_label(class_id: int, box: tuple[float, float, float, float], keypoints, image_width: int, image_height: int) -> str:
    tokens = [str(class_id), *(f"{float(value):.6f}" for value in box)]
    for x, y, visible in keypoints:
        normalized_x = x / image_width if abs(x) > 1 else x
        normalized_y = y / image_height if abs(y) > 1 else y
        tokens.extend((f"{normalized_x:.6f}", f"{normalized_y:.6f}", str(int(visible))))
    return " ".join(tokens)


def convert_boxes(boxes: list[YoloBox], image_width: int, image_height: int, geometry: ObstacleGeometry | None = None, self_muzzle: tuple[float, float] | None = None) -> list[PoseAnnotation]:
    result: list[PoseAnnotation] = []
    for box in boxes:
        keypoints = _invisible_keypoints()
        if box.class_id in CIRCLE_CLASSES:
            keypoints = circle_keypoints_from_box((box.center_x, box.center_y, box.width, box.height), image_width, image_height)
        elif box.class_id == 2 and self_muzzle is not None:
            keypoints = ((box.center_x * image_width, box.center_y * image_height, 2), (self_muzzle[0], self_muzzle[1], 2))
        elif box.class_id == 4 and geometry is not None and geometry.lines:
            cx, cy = box.center_x * image_width, box.center_y * image_height
            line = min(geometry.lines, key=lambda item: hypot((item.start[0] + item.end[0]) / 2 - cx, (item.start[1] + item.end[1]) / 2 - cy))
            keypoints = line_keypoints(line.start, line.end, image_width, image_height)
        result.append(PoseAnnotation(box, keypoints))
    return result

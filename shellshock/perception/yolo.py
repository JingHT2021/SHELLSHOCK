"""Validated, lazily loaded YOLO inference for the independent aim entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from shellshock.domain.world import DetectionBox, Point, PoseKeypoint
from shellshock.perception.self_center_model import crop_self_roi, map_roi_point_to_image, valid_roi_keypoint


REQUIRED_CLASSES = frozenset({"self", "obstacle_circle", "obstacle_line", "portal_orange", "portal_blue"})


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def validate_class_names(names: dict[int, str] | list[str]) -> frozenset[str]:
    """Reject a weight whose class map cannot describe the required world."""
    present = set(names.values()) if isinstance(names, dict) else set(names)
    missing = sorted(REQUIRED_CLASSES - present)
    if missing:
        raise ValueError("YOLO weight is missing required classes: " + ", ".join(missing))
    return REQUIRED_CLASSES


def merge_duplicate_detection_boxes(detections: list[DetectionBox], overlap_threshold: float = 0.60) -> list[DetectionBox]:
    """Merge highly overlapping detections of one reflection board."""
    merged: list[DetectionBox] = []
    for candidate in detections:
        if candidate.name != "obstacle_line":
            merged.append(candidate)
            continue
        right = candidate.x + candidate.width
        bottom = candidate.y + candidate.height
        match = None
        for index, existing in enumerate(merged):
            if existing.name != candidate.name:
                continue
            left = max(candidate.x, existing.x)
            top = max(candidate.y, existing.y)
            overlap_right = min(right, existing.x + existing.width)
            overlap_bottom = min(bottom, existing.y + existing.height)
            intersection = max(0.0, overlap_right - left) * max(0.0, overlap_bottom - top)
            smaller_area = min(candidate.width * candidate.height, existing.width * existing.height)
            if smaller_area > 0 and intersection / smaller_area >= overlap_threshold:
                match = index
                break
        if match is None:
            merged.append(candidate)
            continue
        existing = merged[match]
        winner = candidate if candidate.confidence > existing.confidence else existing
        left = min(existing.x, candidate.x)
        top = min(existing.y, candidate.y)
        merged[match] = replace(
            winner,
            x=left,
            y=top,
            width=max(existing.x + existing.width, right) - left,
            height=max(existing.y + existing.height, bottom) - top,
        )
    return merged


class YoloDetector:
    """Convert one Ultralytics prediction into deterministic named boxes."""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.6,
        self_center_weights: str | None = None,
        self_center_confidence: float = 0.5,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self._model = model_factory(weights)
        raw_names = self._model.names
        self._names = dict(raw_names) if isinstance(raw_names, dict) else dict(enumerate(raw_names))
        validate_class_names(self._names)
        self.confidence = confidence
        self._self_center_confidence = self_center_confidence
        self._self_center_model = model_factory(self_center_weights) if self_center_weights else None

    def _refine_self_center(self, image: np.ndarray, box: DetectionBox) -> tuple[Point, float] | None:
        if self._self_center_model is None:
            return None
        try:
            roi, origin = crop_self_roi(image, box)
            results = self._self_center_model.predict(source=roi, verbose=False, conf=0.01, imgsz=384)
            if not results:
                return None
            pose = getattr(results[0], "keypoints", None)
            xy = getattr(pose, "xy", None) if pose is not None else None
            conf = getattr(pose, "conf", None) if pose is not None else None
            if xy is None:
                return None
            raw_xy = _numpy(xy[0])
            point = raw_xy[0] if raw_xy.ndim > 1 else raw_xy
            score = float(_numpy(conf[0])[0]) if conf is not None else 1.0
            normalized = (float(point[0]) / roi.shape[1], float(point[1]) / roi.shape[0], 2)
            if not valid_roi_keypoint(normalized, score, threshold=self._self_center_confidence):
                return None
            return map_roi_point_to_image(normalized[:2], origin=origin, size=(roi.shape[1], roi.shape[0])), score
        except (IndexError, TypeError, ValueError, AttributeError, RuntimeError):
            return None

    def detect(self, image: np.ndarray) -> list[DetectionBox]:
        results = self._model.predict(source=image, verbose=False, conf=self.confidence)
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        detections: list[DetectionBox] = []
        pose = getattr(results[0], "keypoints", None)
        pose_xy = getattr(pose, "xy", None) if pose is not None else None
        pose_conf = getattr(pose, "conf", None) if pose is not None else None
        pose_data = getattr(pose, "data", None) if pose is not None else None
        for index, (bounds, confidence, class_id) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            score = float(confidence)
            name = self._names.get(int(class_id))
            if name is None or score < self.confidence:
                continue
            left, top, right, bottom = (float(value) for value in bounds)
            if right <= left or bottom <= top:
                continue
            keypoints: list[PoseKeypoint] = []
            if pose_xy is not None and index < len(pose_xy):
                raw_points = _numpy(pose_xy[index])
                points = [raw_points] if raw_points.ndim == 1 else raw_points
                for kp_index, point in enumerate(points):
                    kp_score = float(_numpy(pose_conf[index])[kp_index]) if pose_conf is not None and index < len(pose_conf) and kp_index < len(pose_conf[index]) else 0.0
                    data_score = float(_numpy(pose_data[index])[kp_index][2]) if pose_data is not None and index < len(pose_data) and kp_index < len(pose_data[index]) else kp_score
                    visible = 2 if data_score > 0.0 or kp_score > 0.0 else 0
                    keypoints.append(PoseKeypoint(float(point[0]), float(point[1]), visible, kp_score))
            detections.append(DetectionBox(name, left, top, right - left, bottom - top, score, tuple(keypoints)))
        self_candidates = [item for item in detections if item.name == "self"]
        if self_candidates and self._self_center_model is not None:
            selected = max(self_candidates, key=lambda item: item.confidence)
            refined = self._refine_self_center(image, selected)
            if refined is not None:
                index = detections.index(selected)
                detections[index] = replace(selected, refined_center=refined[0])
        detections = merge_duplicate_detection_boxes(detections)
        return sorted(detections, key=lambda item: (item.name, item.y, item.x, -item.confidence))

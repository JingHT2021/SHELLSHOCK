"""Validated, lazily loaded YOLO inference for the independent aim entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .world_geometry import DetectionBox


REQUIRED_CLASSES = frozenset({"self", "obstacle_circle", "obstacle_line", "portal_orange", "portal_blue"})


def validate_class_names(names: dict[int, str] | list[str]) -> frozenset[str]:
    """Reject a weight whose class map cannot describe the required world."""
    present = set(names.values()) if isinstance(names, dict) else set(names)
    missing = sorted(REQUIRED_CLASSES - present)
    if missing:
        raise ValueError("YOLO weight is missing required classes: " + ", ".join(missing))
    return REQUIRED_CLASSES


class YoloDetector:
    """Convert one Ultralytics prediction into deterministic named boxes."""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.6,
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

    def detect(self, image: np.ndarray) -> list[DetectionBox]:
        results = self._model.predict(source=image, verbose=False, conf=self.confidence)
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        detections: list[DetectionBox] = []
        for bounds, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            score = float(confidence)
            name = self._names.get(int(class_id))
            if name is None or score < self.confidence:
                continue
            left, top, right, bottom = (float(value) for value in bounds)
            if right <= left or bottom <= top:
                continue
            detections.append(DetectionBox(name, left, top, right - left, bottom - top, score))
        return sorted(detections, key=lambda item: (item.name, item.y, item.x, -item.confidence))

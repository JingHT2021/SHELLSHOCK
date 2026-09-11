from __future__ import annotations

import cv2
import numpy as np

from shellshock.domain.detection import Wind
from shellshock.perception.digits import recognize_trained_digits

TRAINED_DIGIT_CONFIDENCE_THRESHOLD = 0.50


def detect_direction(panel: np.ndarray) -> str | None:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 15:
            continue
        vertices = cv2.approxPolyDP(contour, 0.08 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(vertices) != 3:
            continue
        x_values = vertices[:, 0].astype(float)
        span = max(x_values.max() - x_values.min(), 1.0)
        # Anti-aliased triangles have a slightly slanted base, so group
        # vertices within 25% of the width instead of requiring equal x.
        minimum_count = int(np.count_nonzero(x_values <= x_values.min() + span * 0.25))
        maximum_count = int(np.count_nonzero(x_values >= x_values.max() - span * 0.25))
        if maximum_count == 1 and minimum_count >= 2:
            return "right"
        if minimum_count == 1 and maximum_count >= 2:
            return "left"
    return None


def _number_from_panel(panel: np.ndarray) -> int | None:
    # The trained recognizer is the sole numeric recognition path.  It performs
    # its own digit component filtering on the complete HUD panel.
    trained_value, trained_confidence = recognize_trained_digits(panel)
    if trained_value is not None and trained_confidence >= TRAINED_DIGIT_CONFIDENCE_THRESHOLD:
        try:
            value = int(trained_value)
            if 0 <= value <= 100:
                return value
        except ValueError:
            pass
    return None


def _find_panel_box(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    top = bgr[: int(bgr.shape[0] * 0.35)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 145)), np.array((179, 90, 255)))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [cv2.boundingRect(item) for item in contours if cv2.contourArea(item) > 50]
    # The game always presents wind near the top centre.  Controls such as
    # EDIT in the upper-left otherwise resemble the cloud-and-arrow panel.
    centre_left = int(bgr.shape[1] * 0.25)
    centre_right = int(bgr.shape[1] * 0.75)
    candidates = [
        item
        for item in candidates
        if centre_left <= item[0] + item[2] / 2 <= centre_right
        # The wind HUD is fixed near the top edge.  Scoreboards and player
        # lists can contain similarly pale icons lower in the screen.
        and item[1] + item[3] <= bgr.shape[0] * 0.12
    ]
    if not candidates:
        return None
    # The cloud and arrow can be disconnected; join near, aligned components.
    candidates.sort(key=lambda item: abs((item[0] + item[2] / 2) - bgr.shape[1] / 2))
    minimum_width = 70 * (bgr.shape[1] / 2560)
    best: tuple[int, int, int, int] | None = None
    for first in candidates:
        for second in candidates:
            left, right = sorted((first, second), key=lambda item: item[0])
            gap = right[0] - (left[0] + left[2])
            aligned = abs((left[1] + left[3] / 2) - (right[1] + right[3] / 2)) < max(left[3], right[3])
            if 0 <= gap <= 60 and aligned:
                x = left[0]
                y = min(left[1], right[1])
                panel_right = right[0] + right[2]
                bottom = max(left[1] + left[3], right[1] + right[3])
                if panel_right - x < minimum_width:
                    continue
                best = (x, y, panel_right - x, bottom - y)
                break
        if best:
            break
    return best


def detect_wind(bgr: np.ndarray) -> tuple[Wind, str | None, tuple[int, int, int, int] | None]:
    box = _find_panel_box(bgr)
    if box is None:
        return Wind(value=0, direction=None, confidence=0.0), None, None
    x, y, width, height = box
    panel = bgr[y : y + height, x : x + width]
    direction = detect_direction(panel)
    value = _number_from_panel(panel)
    errors: list[str] = []
    if direction is None:
        errors.append("wind direction not found")
    if value is None:
        errors.append("wind value not found (trained model unavailable or low confidence)")
    confidence = 0.9 if not errors else 0.45 if direction else 0.25
    return Wind(value=value, direction=direction, confidence=confidence), "; ".join(errors) or None, box

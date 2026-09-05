from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import Detection

REFERENCE_WIDTH = 2560
PLAYFIELD_BOTTOM = 1800
SELF_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "self_green.png"


def _detections_from_mask(mask: np.ndarray, image_width: int, image_height: int) -> list[Detection]:
    scale = image_width / REFERENCE_WIDTH
    minimum_area = 80 * scale * scale
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / max(height, 1)
        center_y = y + height / 2
        # A tank viewed on a steep slope can be nearly square, while an
        # unrotated tank remains wide.  Keep both orientations.
        if area < minimum_area or not 0.75 <= aspect_ratio <= 5.0:
            continue
        # Restrict detection to the top 1800 physical pixels.  This excludes
        # the bottom HUD while keeping tanks on low terrain detectable.
        if not 0 <= center_y < min(PLAYFIELD_BOTTOM, image_height):
            continue
        if width > 120 * scale or height > 80 * scale:
            continue
        confidence = min(0.99, 0.55 + area / max(minimum_area * 12, 1) * 0.4)
        detections.append(Detection(x + width // 2, y + height // 2, round(confidence, 3)))
    return sorted(detections, key=lambda item: item.x)


def _detect_green_template(bgr: np.ndarray) -> Detection | None:
    """Locate the green tank despite aiming/health UI overlapping its pixels."""
    template = cv2.imread(str(SELF_TEMPLATE_PATH), cv2.IMREAD_UNCHANGED)
    if template is None or template.shape[2] != 4:
        return None
    body, alpha = template[:, :, :3], template[:, :, 3]
    image_height, image_width = bgr.shape[:2]
    top = 0
    bottom = min(PLAYFIELD_BOTTOM, image_height)
    search = bgr[top:bottom]
    reference_scale = image_width / REFERENCE_WIDTH
    best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
    for multiplier in (0.85, 0.925, 1.0, 1.075, 1.15):
        scale = reference_scale * multiplier
        resized_body = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        resized_alpha = cv2.resize(alpha, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        height, width = resized_body.shape[:2]
        if height >= search.shape[0] or width >= search.shape[1]:
            continue
        match = cv2.matchTemplate(search, resized_body, cv2.TM_CCORR_NORMED, mask=resized_alpha)
        _, score, _, location = cv2.minMaxLoc(match)
        if best is None or score > best[0]:
            best = score, location, (width, height)
    if best is None or best[0] < 0.75:
        return None
    score, (x, y), (width, height) = best
    return Detection(x + width // 2, top + y + height // 2, round(float(score), 3))


def detect_tanks(bgr: np.ndarray) -> tuple[Detection | None, list[Detection]]:
    """Return the largest green tank candidate and every red tank candidate."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array((35, 90, 70)), np.array((90, 255, 255)))
    red_low = cv2.inRange(hsv, np.array((0, 90, 70)), np.array((10, 255, 255)))
    red_high = cv2.inRange(hsv, np.array((170, 90, 70)), np.array((179, 255, 255)))
    green_candidates = _detections_from_mask(green, bgr.shape[1], bgr.shape[0])
    if not green_candidates:
        scale = bgr.shape[1] / REFERENCE_WIDTH
        # The green tank body is split by its dark track details.  Join nearby
        # fragments only when ordinary colour detection did not find a tank.
        green_kernel = np.ones((max(3, round(8 * scale)), max(3, round(24 * scale))), np.uint8)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, green_kernel)
        green_candidates = _detections_from_mask(green, bgr.shape[1], bgr.shape[0])
    red_candidates = _detections_from_mask(cv2.bitwise_or(red_low, red_high), bgr.shape[1], bgr.shape[0])
    # Colour candidates are more reliable on uncomplicated screens.  Template
    # matching is a fallback for a selected tank that is merged with its UI.
    green_template = _detect_green_template(bgr)
    self_tank = max([*green_candidates, *([green_template] if green_template else [])], key=lambda item: item.confidence, default=None)
    if self_tank:
        scale = bgr.shape[1] / REFERENCE_WIDTH
        # Exclude the red aiming chevron drawn directly above the local tank.
        red_candidates = [
            enemy
            for enemy in red_candidates
            if not (
                abs(enemy.x - self_tank.x) <= 100 * scale
                and 15 * scale <= self_tank.y - enemy.y <= 210 * scale
            )
        ]
    return self_tank, red_candidates

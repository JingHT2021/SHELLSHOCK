"""Conservative chassis-rail localization inside a detector's self-tank ROI."""
from __future__ import annotations

import cv2
import numpy as np

from shellshock.domain.world import DetectionBox, Point


def fit_self_center(image: np.ndarray, box: DetectionBox) -> Point | None:
    """Return the track center only when two aligned bright green rails agree.

    The turret and barrel are deliberately excluded by requiring a separated,
    similarly wide lower pair with a darker wheel interior. Sloped, occluded,
    small or poorly colored tanks fall back to the caller's pose/box estimate.
    Input follows the perception pipeline's BGR convention.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    if not all(np.isfinite(v) for v in (box.x, box.y, box.width, box.height)):
        return None
    x0, y0 = max(0, int(np.floor(box.x))), max(0, int(np.floor(box.y)))
    x1 = min(image.shape[1], int(np.ceil(box.x + box.width)))
    y1 = min(image.shape[0], int(np.ceil(box.y + box.height)))
    w, h = x1 - x0, y1 - y0
    if w < 24 or h < 16 or not .5 < w / h < 3:
        return None
    blue, green, red = cv2.split(image[y0:y1, x0:x1, :3].astype(np.float32))
    mask = ((green > 140) & (green > 1.3 * red) & (green > 1.3 * blue)).astype(np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             np.ones((1, max(8, int(w * .28))), np.uint8))
    rows = []
    for y in range(int(h * .35), h):
        edges = np.flatnonzero(np.diff(np.r_[0, opened[y], 0]))
        spans = [(int(b - a), int(a), int(b - 1)) for a, b in zip(edges[::2], edges[1::2])]
        if spans:
            length, left, right = max(spans)
            if length >= w * .55:
                rows.append((y, length, left, right))
    groups: list[list[tuple[int, int, int, int]]] = []
    for row in rows:
        if not groups or row[0] > groups[-1][-1][0] + 1:
            groups.append([])
        groups[-1].append(row)
    rails = [max(group, key=lambda row: row[1]) for group in groups
             if group[-1][0] - group[0][0] <= h * .20]
    candidates = []
    for upper in rails:
        for lower in rails:
            gap = lower[0] - upper[0]
            if not w * .10 <= gap <= w * .35:
                continue
            if min(upper[1], lower[1]) / max(upper[1], lower[1]) < .75:
                continue
            cx1, cx2 = (upper[2] + upper[3]) / 2, (lower[2] + lower[3]) / 2
            if abs(cx1 - cx2) > w * .10:
                continue
            cx, cy = (cx1 + cx2) / 2, (upper[0] + lower[0]) / 2
            if not (.20 * w <= cx <= .80 * w and .45 * h <= cy <= .92 * h):
                continue
            interior = mask[upper[0] + 2:lower[0] - 1,
                            max(upper[2], lower[2]):min(upper[3], lower[3]) + 1]
            if not interior.size or interior.mean() > .55:
                continue
            candidates.append((cy, min(upper[1], lower[1]), cx))
    if not candidates:
        return None
    cy, _, cx = max(candidates)  # The lower coherent pair surrounds the wheels.
    return float(x0 + cx), float(y0 + cy)

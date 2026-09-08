"""Fixed-screen training capture and detector-derived crop assets."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab

from .obstacle_geometry import detect_portal_geometry
from .wind import detect_wind


def capture_region_for_width(width: int) -> tuple[int, int, int, int]:
    if width <= 0:
        raise ValueError("width must be positive")
    if width == 2560:
        return 0, 0, 2560, 1300
    if width == 3840:
        return 0, 0, 3840, 1850
    return 0, 0, width, round(width * 1850 / 3840)


def _crop(image: np.ndarray, box, padding: int = 12):
    if box is None:
        return None
    x, y, width, height = box
    left, top = max(0, round(x - padding)), max(0, round(y - padding))
    right, bottom = min(image.shape[1], round(x + width + padding)), min(image.shape[0], round(y + height + padding))
    return image[top:bottom, left:right].copy() if right > left and bottom > top else None


def save_capture_assets(image: np.ndarray, output_dir: Path, stem: str, wind_box=None, portal_boxes=(), padding: int = 12) -> dict[str, int]:
    output_dir = Path(output_dir)
    counts = {"full": 0, "wind": 0, "wormholes": 0}
    for directory in ("full", "wind", "wormholes"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_dir / "full" / f"{stem}.png"), image):
        raise RuntimeError("failed to save full capture")
    counts["full"] = 1
    wind = _crop(image, wind_box, padding)
    if wind is not None and cv2.imwrite(str(output_dir / "wind" / f"{stem}.png"), wind):
        counts["wind"] = 1
    for index, box in enumerate(portal_boxes):
        portal = _crop(image, box, padding)
        if portal is not None and cv2.imwrite(str(output_dir / "wormholes" / f"{stem}_{index:02d}.png"), portal):
            counts["wormholes"] += 1
    return counts


def capture_fixed_screen(output_dir: Path, *, width: int | None = None, region=None, now=None, stem: str | None = None):
    if width is None:
        import win32api
        width = win32api.GetSystemMetrics(0)
    region = region or capture_region_for_width(width)
    x, y, capture_width, capture_height = region
    rgb = np.array(ImageGrab.grab(bbox=(x, y, x + capture_width, y + capture_height), all_screens=True))
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, _, wind_box = detect_wind(image)
    portals = detect_portal_geometry(image)
    portal_boxes = [(item.center[0] - item.radius, item.center[1] - item.radius, item.radius * 2, item.radius * 2) for item in portals]
    stem = stem or (now() if now else datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    return save_capture_assets(image, output_dir, stem, wind_box, portal_boxes)


def save_shot_metadata(metadata_dir: Path, stem: str, *, direction: str, angle_degrees: float, power: float) -> Path:
    """Save the launch vector used to recover a manually marked muzzle point."""
    sign = 1.0 if direction == "right" else -1.0
    angle = np.deg2rad(float(angle_degrees))
    payload = {
        "direction": direction,
        "angle_degrees": float(angle_degrees),
        "power": float(power),
        "direction_vector": [float(sign * np.cos(angle)), float(-np.sin(angle))],
    }
    path = Path(metadata_dir) / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

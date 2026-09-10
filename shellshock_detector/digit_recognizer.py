"""Small offline digit recognizer for the game's fixed-font HUD glyphs."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_TRAINED_MODEL = None
_TRAINED_MODEL_PATH = Path(__file__).resolve().parents[1] / "train" / "runs" / "wind_digit_cnn_v2" / "wind_digit_cnn.pt"


_CANVAS = (48, 32)


def _normalize(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(_CANVAS, dtype=np.uint8)
    crop = mask[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2]
    scale = min((_CANVAS[0] - 4) / crop.shape[0], (_CANVAS[1] - 4) / crop.shape[1])
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros(_CANVAS, dtype=np.uint8)
    top = (_CANVAS[0] - resized.shape[0]) // 2
    left = (_CANVAS[1] - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


@lru_cache(maxsize=1)
def _templates() -> dict[str, tuple[np.ndarray, ...]]:
    templates: dict[str, list[np.ndarray]] = {str(i): [] for i in range(10)}
    fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX)
    for digit in range(10):
        for font in fonts:
            for scale in (1.2, 1.5, 1.8):
                for thickness in (2, 3, 4):
                    canvas = np.zeros((80, 60), dtype=np.uint8)
                    cv2.putText(canvas, str(digit), (8, 62), font, scale, 255, thickness, cv2.LINE_AA)
                    _, mask = cv2.threshold(canvas, 80, 255, cv2.THRESH_BINARY)
                    templates[str(digit)].append(_normalize(mask))
    return {key: tuple(value) for key, value in templates.items()}


def _score(candidate: np.ndarray, template: np.ndarray) -> float:
    a = candidate > 0
    b = template > 0
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.0


def _mask_candidates(gray: np.ndarray, foreground: str) -> tuple[np.ndarray, ...]:
    thresholds = (100, 130, 160, 190) if foreground == "bright" else (70, 100, 130, 160)
    masks = []
    for threshold in thresholds:
        mode = cv2.THRESH_BINARY if foreground == "bright" else cv2.THRESH_BINARY_INV
        _, mask = cv2.threshold(gray, threshold, 255, mode)
        masks.append(mask)
    return tuple(masks)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = []
    for x, y, box_width, box_height, area in stats[1:count]:
        if box_height < max(8, int(height * 0.28)):
            continue
        if box_width > max(40, int(width * 0.7)):
            continue
        if area < max(8, int(box_width * box_height * 0.08)):
            continue
        result.append((x, y, box_width, box_height))
    return sorted(result)


def _classify(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, float]:
    x, y, width, height = box
    crop = mask[max(0, y - 2):y + height + 2, max(0, x - 2):x + width + 2]
    candidate = _normalize(crop)
    scored = [
        (max(_score(candidate, template) for template in templates), digit)
        for digit, templates in _templates().items()
    ]
    score, digit = max(scored)
    return digit, score


def recognize_digits(image: np.ndarray, *, foreground: str = "bright") -> str | None:
    """Recognize one or more fixed-font digits from a small grayscale/BGR ROI."""
    if foreground not in {"bright", "dark"}:
        raise ValueError("foreground must be 'bright' or 'dark'")
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    best: tuple[float, str] | None = None
    for mask in _mask_candidates(gray, foreground):
        boxes = _components(mask)
        if not boxes:
            continue
        digits = [_classify(mask, box) for box in boxes]
        score = sum(item[1] for item in digits) / len(digits)
        value = "".join(item[0] for item in digits)
        candidate = (score, value)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best is not None and best[0] >= 0.25 else None


def _load_trained_model():
    global _TRAINED_MODEL
    if _TRAINED_MODEL is not None:
        return _TRAINED_MODEL
    if not _TRAINED_MODEL_PATH.exists():
        return None
    try:
        import torch
        from torch import nn

        class WindDigitNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
                )
                self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(48 * 12 * 8, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 10))

            def forward(self, x):
                return self.classifier(self.features(x))

        model = WindDigitNet()
        model.load_state_dict(torch.load(_TRAINED_MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _TRAINED_MODEL = (torch, model)
        return _TRAINED_MODEL
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def recognize_trained_digits(image: np.ndarray) -> tuple[str | None, float]:
    """Recognize a dark fixed-font digit string with the trained HUD model."""
    loaded = _load_trained_model()
    if loaded is None or image is None or image.size == 0:
        return None, 0.0
    torch, model = loaded
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = gray.shape[:2]
    boxes = sorted(
        (int(x), int(y), int(box_width), int(box_height))
        for x, y, box_width, box_height, area in stats[1:count]
        if box_height >= max(8, int(height * 0.28))
        and box_width <= max(40, width * 0.7)
        and area >= max(8, box_width * box_height * 0.08)
        and y >= height * 0.15
        and y + box_height <= height * 0.9
    )
    if not boxes:
        return None, 0.0
    tensors = []
    for x, y, box_width, box_height in boxes:
        digit_mask = mask[y:y + box_height, x:x + box_width]
        ys, xs = np.where(digit_mask > 0)
        if len(xs) == 0:
            return None, 0.0
        crop = digit_mask[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
        scale = min(44 / crop.shape[0], 28 / crop.shape[1])
        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((48, 32), dtype=np.float32)
        top, left = (48 - resized.shape[0]) // 2, (32 - resized.shape[1]) // 2
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized / 255.0
        tensors.append(canvas)
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(np.stack(tensors)[:, None, ...])), dim=1)
        confidence, predicted = probabilities.max(dim=1)
    return "".join(str(value) for value in predicted.tolist()), float(confidence.min().item())

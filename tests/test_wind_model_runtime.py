from pathlib import Path

import cv2

from shellshock_detector import digit_recognizer


def test_runtime_model_recognizes_known_wind_digit():
    path = Path("train/annotate_check/wind/20260908_142009_977608.png")
    value, confidence = digit_recognizer.recognize_trained_digits(cv2.imread(str(path)))
    assert value == "24"
    assert 0.0 <= confidence <= 1.0


def test_runtime_model_returns_unavailable_for_empty_input(monkeypatch):
    monkeypatch.setattr(digit_recognizer, "_TRAINED_MODEL", None)
    monkeypatch.setattr(digit_recognizer, "_TRAINED_MODEL_PATH", Path("missing-wind-model.pt"))
    value, confidence = digit_recognizer.recognize_trained_digits(None)
    assert value is None
    assert confidence == 0.0

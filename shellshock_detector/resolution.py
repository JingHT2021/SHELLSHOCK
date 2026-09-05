from __future__ import annotations

RESOLUTION_PRESETS = ("auto", "2560x1600", "3840x2160")


def resolution_warning(preset: str, width: int, height: int) -> str | None:
    """Return a non-blocking warning when a selected preset differs from capture."""
    if preset == "auto":
        return None
    expected_width, expected_height = (int(value) for value in preset.split("x", maxsplit=1))
    if (width, height) == (expected_width, expected_height):
        return None
    return f"configured resolution {preset} but captured {width}x{height}"

"""Windows DPI helpers for physical-pixel window capture."""

from __future__ import annotations

import ctypes
from typing import Tuple


CaptureBox = Tuple[int, int, int, int]


def enable_per_monitor_dpi_awareness() -> bool:
    """Ask Windows to return window coordinates in physical pixels.

    This is important on a 4K display using 125%, 150%, or 200% Windows
    scaling.  The calls are intentionally best-effort: Windows may reject a
    second attempt after another library has already configured the process.
    """
    if not hasattr(ctypes, "windll"):
        return False

    user32 = ctypes.windll.user32
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        set_context = user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = ctypes.c_bool
        if set_context(ctypes.c_void_p(-4)):
            return True
    except (AttributeError, OSError):
        pass

    try:
        shcore = ctypes.windll.shcore
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        set_awareness = shcore.SetProcessDpiAwareness
        set_awareness.argtypes = [ctypes.c_int]
        set_awareness.restype = ctypes.c_long
        if set_awareness(2) in (0, 5):  # S_OK or E_ACCESSDENIED (already set)
            return True
    except (AttributeError, OSError):
        pass

    try:
        set_aware = user32.SetProcessDPIAware
        set_aware.restype = ctypes.c_bool
        return bool(set_aware())
    except (AttributeError, OSError):
        return False


def client_capture_box(top_left: tuple[int, int], bottom_right: tuple[int, int]) -> CaptureBox:
    """Build a Pillow capture box from physical client-area screen points."""
    left, top = top_left
    right, bottom = bottom_right
    if right <= left or bottom <= top:
        raise ValueError("Client capture area must have a positive width and height")
    return left, top, right, bottom

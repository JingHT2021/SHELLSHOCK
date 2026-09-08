"""Small Windows-only desktop boundary for the YOLO aiming entrypoint."""
from __future__ import annotations

import ctypes

import cv2
import numpy as np
from PIL import ImageGrab
import win32api
import win32con
import win32gui
import win32process
import pywintypes

from .capture import GAME_CAPTURE_HEIGHT
from .dpi import client_capture_box, enable_per_monitor_dpi_awareness


enable_per_monitor_dpi_awareness()


def screen_to_client_point(screen_point, client_origin, client_size):
    x, y = screen_point[0] - client_origin[0], screen_point[1] - client_origin[1]
    if not (0 <= x < client_size[0] and 0 <= y < client_size[1]):
        raise ValueError("mouse target is outside the ShellShock Live client area")
    return x, y


def _process_name(process_id):
    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
    if not process:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_uint32(len(buffer))
        return buffer.value.rsplit("\\", 1)[-1] if ctypes.windll.kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)) else ""
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def find_game_window():
    matches = []
    def collect(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        title = win32gui.GetWindowText(hwnd).lower()
        if "shellshock live" in title or _process_name(pid).lower() == "shellshocklive.exe":
            rect = win32gui.GetClientRect(hwnd)
            if rect[2] - rect[0] >= 320 and rect[3] - rect[1] >= 240:
                matches.append(((rect[2] - rect[0]) * (rect[3] - rect[1]), hwnd))
    win32gui.EnumWindows(collect, None)
    if not matches:
        raise RuntimeError("ShellShock Live game window not found or has no usable client area")
    return max(matches)[1]


def capture_client_area(hwnd):
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    right, bottom = win32gui.ClientToScreen(hwnd, win32gui.GetClientRect(hwnd)[2:])
    if right <= left or bottom <= top:
        raise RuntimeError("ShellShock Live client area is empty")
    rgb = np.array(ImageGrab.grab(bbox=client_capture_box((left, top), (right, bottom)), all_screens=True))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:GAME_CAPTURE_HEIGHT].copy()


def client_screen_geometry(hwnd):
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    rect = win32gui.GetClientRect(hwnd)
    return (left, top), (rect[2], rect[3])


def ensure_game_window_is_active(hwnd):
    if not win32gui.IsWindow(hwnd):
        raise RuntimeError("aim skipped: ShellShock Live window was closed")
    try:
        win32gui.SetForegroundWindow(hwnd)
    except (OSError, pywintypes.error):
        pass


def click_screen_point(point):
    win32api.SetCursorPos(point)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

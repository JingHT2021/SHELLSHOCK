from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import ImageGrab
import win32gui
import win32process
import win32api
import win32con

from .aiming import disc_click_point
from .ballistics import refine_normal_integer_shot, solve_target
from .capture import GAME_CAPTURE_HEIGHT
from .dpi import client_capture_box, enable_per_monitor_dpi_awareness
from .models import DetectionResult
from .obstacle_geometry import ObstacleGeometry, detect_pink_obstacle_geometry
from .reflection import solve_single_reflection
from .resolution import resolution_warning
from .training_data import save_training_sample
from .wind import detect_wind

enable_per_monitor_dpi_awareness()

@dataclass(frozen=True)
class TrainingPaths:
    raw_path: Path
    label_path: Path
    annotated_path: Path
    result: DetectionResult


def screen_to_client_point(
    screen_point: tuple[int, int],
    client_origin: tuple[int, int],
    client_size: tuple[int, int],
) -> tuple[int, int]:
    """Convert a physical screen pixel to a game-client pixel."""
    x = screen_point[0] - client_origin[0]
    y = screen_point[1] - client_origin[1]
    if not (0 <= x < client_size[0] and 0 <= y < client_size[1]):
        raise ValueError("mouse target is outside the ShellShock Live client area")
    return x, y


def aim_click_for_target(
    result: DetectionResult,
    target_x: int,
    target_y: int,
    manual_self: tuple[int, int] | None = None,
    shot_mode: str = "normal",
    geometry: ObstacleGeometry | None = None,
) -> tuple[dict[str, object], tuple[int, int]]:
    """Return the shot solution and its client-relative aim-disc click point."""
    if manual_self is None and result.self_tank is None:
        raise RuntimeError("aim skipped: green self tank not found")
    self_x, self_y = manual_self if manual_self is not None else (result.self_tank.x, result.self_tank.y)
    wind_value = result.wind.value
    wind_direction = result.wind.direction
    if wind_value is None or (wind_direction is None and wind_value != 0):
        # Aiming remains usable when OCR/HUD detection loses the wind panel.
        # Treat unknown wind as calm rather than issuing no click at all.
        wind_value = 0
        wind_direction = "right"

    solution = solve_target(
        self_x,
        self_y,
        target_x,
        target_y,
        wind_value,
        wind_direction or "right",
        result.image_width,
    )
    fallback: str | None = None
    if shot_mode == "reflection" and geometry is not None and (geometry.circles or geometry.lines):
        selected = solve_single_reflection(
            (int(self_x), int(self_y)), (target_x, target_y), float(wind_value),
            wind_direction or "right", result.image_width, geometry,
        )
        if selected.get("status") not in {"reachable", "closest"}:
            raise RuntimeError(f"aim skipped: {selected.get('reason', 'no valid one-reflection solution')}")
    else:
        if shot_mode == "reflection":
            fallback = "normal:no-obstacle"
        normal_mode = "minimum" if shot_mode in {"normal", "minimum", "reflection"} else shot_mode
        if normal_mode == "minimum":
            selected_theory = solution["minimum_power"]
            if selected_theory.get("status") != "reachable":
                    raise RuntimeError("aim skipped: target has no usable -90--90 degree solution")
            if not selected_theory["within_power_limit"]:
                raise RuntimeError("aim skipped: minimum power is over 100")
        elif normal_mode == "maximum":
            maximum_solutions = solution["power_100"]["solutions"]
            if not maximum_solutions:
                raise RuntimeError("aim skipped: target has no reachable 100-power arc")
            # A target can have exactly one valid fixed-power trajectory;
            # when it does, that sole trajectory is also the highest arc.
            selected_theory = {**max(maximum_solutions, key=lambda item: float(item["angle_degrees"])), "power": 100.0}
        else:
            raise ValueError("shot_mode must be 'normal', 'minimum', 'maximum', or 'reflection'")
        selected = refine_normal_integer_shot(
            self_x, self_y, target_x, target_y, wind_value, wind_direction or "right",
            result.image_width, float(selected_theory["angle_degrees"]), float(selected_theory["power"]),
        )
        selected["mode"] = "normal"
        if fallback:
            selected["fallback"] = fallback
    solution["selected"] = selected
    return solution, disc_click_point(
        self_x,
        self_y,
        str(selected["direction"]),
        float(selected["angle_degrees"]),
        float(selected["power"]),
        result.image_width,
    )


def analyze_image(image: np.ndarray) -> tuple[DetectionResult, tuple[int, int, int, int] | None]:
    wind, wind_error, wind_box = detect_wind(image)
    errors: list[str] = []
    if wind_error:
        errors.append(wind_error)
    return DetectionResult(image.shape[1], image.shape[0], None, [], wind, errors), wind_box


def matches_game_window(title: str, process_name: str) -> bool:
    """Accept the normal title, or the Unity executable when the title is blank."""
    return "shellshock live" in title.lower() or process_name.lower() == "shellshocklive.exe"


def has_usable_client_area(rect: tuple[int, int, int, int]) -> bool:
    return rect[2] - rect[0] >= 320 and rect[3] - rect[1] >= 240


def choose_window_handle(
    foreground_handle: int,
    foreground_rect: tuple[int, int, int, int],
    matched_handles: list[tuple[int, int]],
    foreground_matches_game: bool,
) -> int | None:
    """Prefer the active game window that received the R hotkey."""
    if foreground_handle and foreground_matches_game and has_usable_client_area(foreground_rect):
        return foreground_handle
    return max(matched_handles)[1] if matched_handles else None


def _process_name(process_id: int) -> str:
    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
    if not process:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_uint32(len(buffer))
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return buffer.value.rsplit("\\", 1)[-1]
    finally:
        ctypes.windll.kernel32.CloseHandle(process)
    return ""


def _annotate(image: np.ndarray, result: DetectionResult, wind_box: tuple[int, int, int, int] | None) -> np.ndarray:
    annotated = image.copy()
    for label, item, color in [("SELF", result.self_tank, (0, 255, 0))]:
        if item:
            cv2.circle(annotated, (item.x, item.y), 24, color, 2)
            cv2.putText(annotated, f"{label} ({item.x}, {item.y})", (item.x - 50, item.y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for index, item in enumerate(result.enemies, start=1):
        cv2.circle(annotated, (item.x, item.y), 24, (0, 0, 255), 2)
        cv2.putText(annotated, f"ENEMY {index} ({item.x}, {item.y})", (item.x - 70, item.y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    if wind_box:
        x, y, width, height = wind_box
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (255, 220, 0), 2)
        cv2.putText(annotated, f"WIND {result.wind.value} {result.wind.direction}", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
    return annotated


def process_capture(
    image: np.ndarray,
    train_dir: Path,
    now: Callable[[], str] | None = None,
    resolution: str = "auto",
    yolo_annotations: list[tuple[int, tuple[int, int]]] | None = None,
) -> TrainingPaths:
    timestamp = now() if now else datetime.now().strftime("%Y%m%d_%H%M%S")
    result, wind_box = analyze_image(image)
    warning = resolution_warning(resolution, image.shape[1], image.shape[0])
    if resolution != "auto":
        expected_width, expected_height = (int(value) for value in resolution.split("x", maxsplit=1))
        if image.shape[1] == expected_width and image.shape[0] == min(expected_height, GAME_CAPTURE_HEIGHT):
            warning = None
    if warning:
        result.errors.append(warning)
    raw_path, label_path, annotated_path = save_training_sample(image, train_dir, timestamp, yolo_annotations or [])
    return TrainingPaths(raw_path, label_path, annotated_path, result)


def find_game_window() -> int:
    matches: list[tuple[int, int]] = []

    def collect(hwnd: int, _: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, process_id = win32process.GetWindowThreadProcessId(hwnd)
        if matches_game_window(win32gui.GetWindowText(hwnd), _process_name(process_id)):
            rect = win32gui.GetClientRect(hwnd)
            if has_usable_client_area(rect):
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                matches.append((area, hwnd))

    win32gui.EnumWindows(collect, None)
    foreground_handle = win32gui.GetForegroundWindow()
    foreground_rect = win32gui.GetClientRect(foreground_handle) if foreground_handle else (0, 0, 0, 0)
    foreground_matches_game = False
    if foreground_handle:
        _, foreground_process_id = win32process.GetWindowThreadProcessId(foreground_handle)
        foreground_matches_game = matches_game_window(
            win32gui.GetWindowText(foreground_handle), _process_name(foreground_process_id)
        )
    selected = choose_window_handle(foreground_handle, foreground_rect, matches, foreground_matches_game)
    if selected is None:
        raise RuntimeError("ShellShock Live game window not found or has no usable client area")
    return selected


def capture_client_area(hwnd: int) -> np.ndarray:
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    right, bottom = win32gui.ClientToScreen(hwnd, win32gui.GetClientRect(hwnd)[2:])
    if right <= left or bottom <= top:
        raise RuntimeError("ShellShock Live client area is empty")
    box = client_capture_box((left, top), (right, bottom))
    rgb = np.array(ImageGrab.grab(bbox=box, all_screens=True))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:GAME_CAPTURE_HEIGHT].copy()


def client_screen_geometry(hwnd: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return physical screen origin and size of a game client area."""
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    client_rect = win32gui.GetClientRect(hwnd)
    return (left, top), (client_rect[2], client_rect[3])


def ensure_game_window_is_active(
    hwnd: int,
    is_window: Callable[[int], bool] = win32gui.IsWindow,
    activate: Callable[[int], None] = win32gui.SetForegroundWindow,
) -> None:
    """Reactivate the captured game window before an injected mouse click."""
    if not is_window(hwnd):
        raise RuntimeError("aim skipped: ShellShock Live window was closed")
    try:
        activate(hwnd)
    except OSError:
        # Windows can reject foreground activation from the global keyboard
        # listener thread even while the already-captured game window is valid.
        # The caller still targets that verified window's screen coordinates.
        pass


def click_screen_point(point: tuple[int, int]) -> None:
    """Move to a physical screen point and issue one left click."""
    win32api.SetCursorPos(point)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def aim_at_screen_position(
    screen_target: tuple[int, int],
    resolution: str = "auto",
    click: Callable[[tuple[int, int]], None] = click_screen_point,
    manual_self: tuple[int, int] | None = None,
    shot_mode: str = "normal",
    train_dir: Path = Path("train"),
) -> tuple[TrainingPaths, dict[str, object], tuple[int, int] | None]:
    """Analyze a mouse target and click its calculated aim-disc point once.

    This is intentionally guarded: any invalid target or unavailable ballistic
    input raises before the pointer is moved.
    """
    hwnd = find_game_window()
    client_origin, client_size = client_screen_geometry(hwnd)
    target_x, target_y = screen_to_client_point(screen_target, client_origin, client_size)
    if manual_self is None:
        raise RuntimeError("aim skipped: press S to record self position first")
    if manual_self[1] >= GAME_CAPTURE_HEIGHT or target_y >= GAME_CAPTURE_HEIGHT:
        raise RuntimeError("aim skipped: self or target is below the saved 2000-pixel capture")
    image = capture_client_area(hwnd)
    paths = process_capture(
        image, train_dir, resolution=resolution,
        yolo_annotations=[(2, manual_self), (0, (target_x, target_y))],
    )
    solution, click_client = aim_click_for_target(
        paths.result, target_x, target_y, manual_self=manual_self, shot_mode=shot_mode,
        geometry=detect_pink_obstacle_geometry(image),
    )
    if not (0 <= click_client[0] < client_size[0] and 0 <= click_client[1] < client_size[1]):
        return paths, solution, None
    ensure_game_window_is_active(hwnd)
    click_screen = (client_origin[0] + click_client[0], client_origin[1] + click_client[1])
    click(click_screen)
    return paths, solution, click_screen


def capture_once(
    resolution: str = "auto",
    train_dir: Path = Path("train"),
    yolo_annotations: list[tuple[int, tuple[int, int]]] | None = None,
) -> TrainingPaths:
    return process_capture(
        capture_client_area(find_game_window()), train_dir, resolution=resolution,
        yolo_annotations=yolo_annotations,
    )

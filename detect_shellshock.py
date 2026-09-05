from __future__ import annotations

import argparse
from pathlib import Path

import keyboard
import win32api

from shellshock_detector.dpi import enable_per_monitor_dpi_awareness

enable_per_monitor_dpi_awareness()

from shellshock_detector.app import (
    GAME_CAPTURE_HEIGHT,
    aim_at_screen_position,
    capture_once,
    client_screen_geometry,
    find_game_window,
    screen_to_client_point,
)
from shellshock_detector.ballistics import format_ballistics
from shellshock_detector.resolution import RESOLUTION_PRESETS


MAXIMUM_SHOT_HOTKEYS = ("page up",)
MINIMUM_SHOT_HOTKEYS = ("page down",)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ShellShock Live on the R hotkey.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for raw, JSON, and annotated files.")
    parser.add_argument(
        "--resolution",
        choices=RESOLUTION_PRESETS,
        default="auto",
        help="Expected game-window resolution; auto uses the captured client size.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path("train")
    manual_self: tuple[int, int] | None = None
    manual_target: tuple[int, int] | None = None
    shot_mode = "minimum"

    def mouse_client_point() -> tuple[int, int]:
        hwnd = find_game_window()
        origin, size = client_screen_geometry(hwnd)
        point = screen_to_client_point(win32api.GetCursorPos(), origin, size)
        if point[1] >= GAME_CAPTURE_HEIGHT:
            raise RuntimeError("point is below the saved 1800-pixel capture")
        return point

    def current_annotations() -> list[tuple[int, tuple[int, int]]]:
        annotations: list[tuple[int, tuple[int, int]]] = []
        if manual_self is not None:
            annotations.append((2, manual_self))
        if manual_target is not None:
            annotations.append((0, manual_target))
        return annotations

    def run_capture() -> None:
        try:
            paths = capture_once(
                args.output_dir, args.resolution, train_dir=train_dir,
                yolo_annotations=current_annotations(),
            )
            print(f"Saved: {paths.raw_path}, {paths.json_path}, {paths.annotated_path}")
            print("Wind: " + (f"{paths.result.wind.value} {paths.result.wind.direction}" if paths.result.wind.value is not None else "not found (T uses 0 wind)"))
        except RuntimeError as error:
            print(f"Capture skipped: {error}")

    def set_self_position() -> None:
        nonlocal manual_self
        try:
            manual_self = mouse_client_point()
            print(f"Self position recorded: {manual_self}")
        except (RuntimeError, ValueError) as error:
            print(f"Self position skipped: {error}")

    def use_maximum_shot() -> None:
        nonlocal shot_mode
        shot_mode = "maximum"
        print("Shot mode: 100 power, highest reachable angle")

    def use_minimum_shot() -> None:
        nonlocal shot_mode
        shot_mode = "minimum"
        print("Shot mode: minimum power")

    def run_mouse_target_aim() -> None:
        nonlocal manual_target
        try:
            manual_target = mouse_client_point()
            paths, solution, click_point = aim_at_screen_position(
                win32api.GetCursorPos(), args.output_dir, args.resolution,
                manual_self=manual_self, shot_mode=shot_mode, train_dir=train_dir,
            )
            print(f"Saved: {paths.raw_path}, {paths.json_path}, {paths.annotated_path}")
            if shot_mode == "maximum":
                high_arc = solution["power_100"]["solutions"][-1]
                print(
                    f"Selected: 100 power high arc: {high_arc['direction']} "
                    f"{high_arc['angle_degrees']:.4f} degrees"
                )
            else:
                print(format_ballistics([solution]))
            print(f"Manual positions: self={manual_self}, target={manual_target}; mode={shot_mode}")
            print(f"Aim click: screen {click_point}")
        except (RuntimeError, ValueError) as error:
            print(f"Aim skipped: {error}")

    keyboard.add_hotkey("r", run_capture)
    keyboard.add_hotkey("q", set_self_position)
    keyboard.add_hotkey("e", run_mouse_target_aim)
    for hotkey in MAXIMUM_SHOT_HOTKEYS:
        keyboard.add_hotkey(hotkey, use_maximum_shot)
    for hotkey in MINIMUM_SHOT_HOTKEYS:
        keyboard.add_hotkey(hotkey, use_minimum_shot)
    print(
        f"Ready ({args.resolution}): Q=self, E=target and aim, R=capture, "
        "PageUp=100-power high arc, PageDown=minimum power, Esc=quit."
    )
    keyboard.wait("esc")


if __name__ == "__main__":
    main()

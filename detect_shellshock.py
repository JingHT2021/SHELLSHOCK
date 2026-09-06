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


def describe_selected_shot(selected: dict[str, object]) -> str:
    """Format only the playable, integer-refined controls for terminal output."""
    angle = int(selected["angle_degrees"])
    power = int(selected["power"])
    error = float(selected["target_error"])
    horizontal_error = float(selected.get("horizontal_error", 0.0))
    if abs(horizontal_error) < 0.005:
        landing = "landing=center"
    else:
        side = "right" if horizontal_error > 0 else "left"
        landing = f"landing={side} {abs(horizontal_error):.2f}px"
    if selected.get("mode") == "reflection":
        obstacle = selected["obstacle"]
        assert isinstance(obstacle, dict)
        theory = selected["theory"]
        assert isinstance(theory, dict)
        point = selected["reflection_point"]
        label = "Reflection (closest)" if selected.get("status") == "closest" else "Reflection"
        return (
            f"{label}: (power={power}, angle={angle})  target_error={error:.2f}px  {landing}; "
            f"{obstacle['kind']} #{obstacle['index']}; point={point}; "
            f"theory angle={float(theory['angle_degrees']):.3f}, "
            f"power={float(theory['power']):.3f}"
        )
    prefix = f"{selected['fallback']}; " if selected.get("fallback") else ""
    return f"{prefix}Normal: (power={power}, angle={angle})  target_error={error:.2f}px  {landing}"


def format_aim_report(
    self_position: tuple[int, int],
    aim_click: tuple[int, int] | None,
    selected: dict[str, object],
    wind_value: int | None,
    wind_direction: str | None,
) -> str:
    """Return a compact, terminal-readable summary of an executed shot."""
    wind = f"{wind_value} {wind_direction}" if wind_value is not None and wind_direction else "unknown (using calm)"
    mode = str(selected.get("mode", "normal"))
    if selected.get("fallback"):
        mode += " (fallback)"
    aim_label = f"AIM CLICK  {aim_click}" if aim_click is not None else "AIM CLICK OUTSIDE CLIENT"
    position_line = f"SELF  {self_position}  →  {aim_label}  |  WIND {wind}  |  MODE {mode}"
    # ANSI bold red is supported by Windows Terminal and most modern consoles;
    # reset immediately so paths and later logs keep their normal colors.
    shot_line = f"\x1b[1;31m{describe_selected_shot(selected)}\x1b[0m"
    return f"{position_line}\n{shot_line}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ShellShock Live on the R hotkey.")
    parser.add_argument(
        "--resolution",
        choices=RESOLUTION_PRESETS,
        default="auto",
        help="Expected game-window resolution; auto uses the captured client size.",
    )
    args = parser.parse_args()
    train_dir = Path("train")
    manual_self: tuple[int, int] | None = None
    manual_target: tuple[int, int] | None = None
    shot_mode = "normal"

    def mouse_client_point() -> tuple[int, int]:
        hwnd = find_game_window()
        origin, size = client_screen_geometry(hwnd)
        point = screen_to_client_point(win32api.GetCursorPos(), origin, size)
        if point[1] >= GAME_CAPTURE_HEIGHT:
            raise RuntimeError("point is below the saved 2000-pixel capture")
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
            paths = capture_once(args.resolution, train_dir=train_dir, yolo_annotations=current_annotations())
            print(f"Saved training data: {paths.raw_path}, {paths.label_path}, {paths.annotated_path}")
            print("Wind: " + (f"{paths.result.wind.value} {paths.result.wind.direction}" if paths.result.wind.value is not None else "not found (T uses 0 wind)"))
        except RuntimeError as error:
            print(f"Capture skipped: {error}")

    def set_self_position() -> None:
        nonlocal manual_self
        try:
            manual_self = mouse_client_point()
            print(f"Self position set: {manual_self} (shown with aim click after E)")
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

    def use_reflection_shot() -> None:
        nonlocal shot_mode
        shot_mode = "reflection"
        print("Shot mode: reflection (one obstacle bounce required)")

    def use_normal_shot() -> None:
        nonlocal shot_mode
        shot_mode = "normal"
        print("Shot mode: normal (integer-refined trajectory)")

    def run_mouse_target_aim() -> None:
        nonlocal manual_target
        try:
            manual_target = mouse_client_point()
            paths, solution, click_point = aim_at_screen_position(
                win32api.GetCursorPos(), args.resolution,
                manual_self=manual_self, shot_mode=shot_mode, train_dir=train_dir,
            )
            selected = solution["selected"]
            assert isinstance(selected, dict)
            assert manual_self is not None
            print(
                format_aim_report(
                    manual_self, click_point, selected,
                    paths.result.wind.value, paths.result.wind.direction,
                )
            )
            if click_point is None:
                print("Aim not clicked: calculated aim-disc point is outside the game client area")
            print(f"Target {manual_target}  |  mode={shot_mode}")
            print(f"Saved training data: {paths.raw_path}, {paths.label_path}, {paths.annotated_path}")
        except (RuntimeError, ValueError) as error:
            print(f"Aim skipped: {error}")

    keyboard.add_hotkey("q", set_self_position)
    keyboard.add_hotkey("e", run_mouse_target_aim)
    keyboard.add_hotkey("r", use_reflection_shot)
    keyboard.add_hotkey("t", use_normal_shot)
    for hotkey in MAXIMUM_SHOT_HOTKEYS:
        keyboard.add_hotkey(hotkey, use_maximum_shot)
    for hotkey in MINIMUM_SHOT_HOTKEYS:
        keyboard.add_hotkey(hotkey, use_minimum_shot)
    print(
        f"Ready ({args.resolution}): Q=self, E=target/recalculate/aim, "
        "R=reflection, T=normal, PageUp=100-power high arc, "
        "PageDown=minimum power, Esc=quit."
    )
    keyboard.wait("esc")


if __name__ == "__main__":
    main()

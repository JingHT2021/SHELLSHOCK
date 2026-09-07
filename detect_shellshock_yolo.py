"""Independent YOLO aim entrypoint; does not alter detect_shellshock.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import keyboard
import win32api

from shellshock_detector.aiming import disc_click_point
from shellshock_detector.app import (
    capture_client_area, client_screen_geometry, click_screen_point,
    ensure_game_window_is_active, find_game_window, screen_to_client_point,
)
from shellshock_detector.global_solver import solve_integer_shot
from shellshock_detector.yolo_runtime import YoloDetector
from shellshock_detector.world_geometry import build_world
from shellshock_detector.wind import detect_wind

DEFAULT_WEIGHTS = Path("train/runs/shellshock_yolo11n_cv65_final_all/weights/best.pt")
EXIT_HOTKEY = "delete"


def select_mode(key: str, current_mode: str = "normal") -> str:
    if key not in {"page up", "page down"}:
        return key
    arc = "high_arc" if key == "page up" else "low_arc"
    return f"wormhole_{arc}" if current_mode.startswith("wormhole") else arc


def format_aim_report(mode: str, solution: dict[str, object], wind_value: int | float | None, wind_direction: str | None, click: tuple[int, int]) -> str:
    wind = f"{wind_value if wind_value is not None else 0:>2} {wind_direction or 'right':<5}"
    controls = f"\x1b[31m({int(solution['power']):>3}, {int(solution['angle_degrees']):>3}°)\x1b[0m"
    portals = int(solution.get("portal_count", 0))
    events = "→".join(str(item) for item in solution.get("events", ()))
    return f"MODE {mode:<9} WIND {wind}  {controls}\nPORTALS {portals:>2}  EVENTS {events:<24} CLICK {click}"


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO ShellShock wormhole aim")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.6)
    args = parser.parse_args()
    detector = YoloDetector(str(args.weights), args.confidence)
    mode = "normal"

    def choose(value: str) -> None:
        nonlocal mode
        mode = value
        print(f"Mode: {mode}", flush=True)

    def aim() -> None:
        try:
            print(f"Searching {mode} shot...", flush=True)
            hwnd = find_game_window()
            origin, size = client_screen_geometry(hwnd)
            target = screen_to_client_point(win32api.GetCursorPos(), origin, size)
            image = capture_client_area(hwnd)
            detections = detector.detect(image)
            world = build_world(detections, image.shape[1], image.shape[0])
            orange = sum(box.name == "portal_orange" for box in detections)
            blue = sum(box.name == "portal_blue" for box in detections)
            print(f"World: self={'yes' if world.self_position else 'no'}; portals orange={orange}, blue={blue}, pairs={len(world.portal_pairs)}, unpaired={world.unpaired_portals}", flush=True)
            if world.self_position is None:
                raise RuntimeError("YOLO did not find one unambiguous self tank")
            wind, _, _ = detect_wind(image)
            value, direction = wind.value or 0, wind.direction or "right"
            solution = solve_integer_shot(world.self_position, target, world, value, direction, image.shape[1], mode)
            if solution.get("status") != "reachable":
                reason = str(solution.get("reason"))
                messages = {
                    "no-portal-pair": "YOLO did not produce a usable orange/blue portal pair",
                    "no-verified-wormhole-shot": "portals were found, but no collision-free one/two-hop shot was verified",
                }
                raise RuntimeError(f"no safe {mode} shot: {messages.get(reason, reason)}")
            click = disc_click_point(*world.self_position, str(solution["direction"]), float(solution["angle_degrees"]), float(solution["power"]), image.shape[1])
            if not (0 <= click[0] < size[0] and 0 <= click[1] < size[1]):
                raise RuntimeError("aim disc point is outside game client")
            ensure_game_window_is_active(hwnd)
            click_screen_point((origin[0] + click[0], origin[1] + click[1]))
            print(format_aim_report(mode, solution, value, direction, click), flush=True)
        except (RuntimeError, ValueError) as error:
            print(f"Aim skipped: {error}")

    keyboard.add_hotkey("e", aim)
    keyboard.add_hotkey("t", lambda: choose("normal"))
    keyboard.add_hotkey("h", lambda: choose("wormhole_low_arc"))
    keyboard.add_hotkey("r", lambda: choose("reflection"))
    keyboard.add_hotkey("page up", lambda: choose(select_mode("page up", mode)))
    keyboard.add_hotkey("page down", lambda: choose(select_mode("page down", mode)))
    keyboard.add_hotkey(EXIT_HOTKEY, lambda: print("Exiting YOLO aim...", flush=True))
    print("Ready: E=aim, T=normal, H=wormhole, R=reflection, PageUp=high arc, PageDown=low arc, Del=quit")
    keyboard.wait(EXIT_HOTKEY)


if __name__ == "__main__":
    main()

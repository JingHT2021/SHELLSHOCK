"""Independent YOLO aim entrypoint; does not alter detect_shellshock.py."""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import keyboard
import win32api

from shellshock_detector_yolo.aiming import disc_click_point
from shellshock_detector_yolo.desktop import (
    capture_client_area, client_screen_geometry, click_screen_point,
    ensure_game_window_is_active, find_game_window, screen_to_client_point,
)
from shellshock_detector.dataset_capture import capture_fixed_screen, save_shot_metadata
from shellshock_detector_yolo.global_solver import solve_integer_shot
from shellshock_detector_yolo.shot_modes import mode_parts, normalize_mode
from shellshock_detector_yolo.yolo_runtime import YoloDetector
from shellshock_detector_yolo.world_geometry import build_world_from_image
from shellshock_detector_yolo.wind import detect_wind
from shellshock_detector_yolo.reflection_layer_c import FinalResultManager, FinalReplayResult

DEFAULT_WEIGHTS = Path("train/runs/shellshock_yolo11n_cv65_final_all/weights/best.pt")
EXIT_HOTKEY = "delete"


def select_mode(key: str, current_mode: str = "normal_low") -> str:
    if key not in {"page up", "page down"}:
        return normalize_mode(key)
    family, _ = mode_parts(current_mode)
    return f"{family}_{'high' if key == 'page up' else 'low'}"


def format_aim_report(mode: str, solution: dict[str, object], wind_value: int | float | None, wind_direction: str | None, click: tuple[int, int]) -> str:
    wind = f"{wind_value if wind_value is not None else 0:>2} {wind_direction or 'right':<5}"
    controls = f"\x1b[31m({int(solution['power']):>3}, {int(solution['angle_degrees']):>3}°)\x1b[0m"
    portals = int(solution.get("portal_count", 0))
    reflections = int(solution.get("reflection_count", 0))
    reflection_point = solution.get("reflection_point")
    reflection_obstacle = solution.get("reflection_obstacle")
    events = "→".join(str(item) for item in solution.get("events", ()))
    reflection = (
        f"REFLECTIONS {reflections:>2} @ {reflection_point} {reflection_obstacle['kind']} #{reflection_obstacle['index']}"
        if reflections and isinstance(reflection_obstacle, dict) else
        f"REFLECTIONS {reflections:>2} @ {reflection_point}" if reflections else "REFLECTIONS  0"
    )
    lines = [f"MODE {mode:<9} WIND {wind}  {controls}",
             f"PORTALS {portals:>2}  {reflection}  EVENTS {events:<24} CLICK {click}"]
    metrics = []
    for field, label, precision, suffix in (
        ('miss_distance', 'MISS', 2, ' px'), ('clearance', 'CLEARANCE', 2, ' px'),
        ('incidence', 'INCIDENCE', 3, ''), ('theoretical_min_power', 'PMIN', 2, ''),
        ('low_start_power', 'LOW START', 0, ''), ('theory_parameter', 'THEORY PARAM', 6, ''),
    ):
        if solution.get(field) is not None:
            metrics.append(f"{label} {float(solution[field]):.{precision}f}{suffix}")
    if metrics:
        lines.append('  '.join(metrics))
    timing = solution.get('timing')
    if isinstance(timing, dict):
        lines.append(
            "TIME A {:.1f}ms B {:.1f}ms C1 {:.1f}ms C2 {:.1f}ms TOTAL {:.1f}ms".format(
                1000 * float(timing.get('layer_a_seconds', 0.0)),
                1000 * float(timing.get('layer_b_seconds', 0.0)),
                1000 * float(timing.get('layer_c1_seconds', 0.0)),
                1000 * float(timing.get('layer_c2_seconds', 0.0)),
                1000 * float(timing.get('total_seconds', 0.0)),
            )
        )
    for portal in solution.get('portal_radii', ()):
        lines.append(f"PORTAL {portal['id']} VISUAL {portal['visual']:.2f} TRIGGER {portal['trigger']:.2f} AVOID {portal['avoid']:.2f}")
    if solution.get('arc_fallback'):
        lines.append(f"single reflection solution: using {solution.get('selected_arc', 'available')} endpoint")
    return '\n'.join(lines)


def format_solver_diagnostics(diagnostics: dict[str, object], *, line_count: int = 0, circle_count: int = 0) -> str:
    timing = diagnostics.get('timing') if isinstance(diagnostics.get('timing'), dict) else {}
    a_reasons = diagnostics.get('layer_a_invalid_reasons') or {}
    b_reasons = diagnostics.get('layer_b_invalid_reasons') or {}
    b_soft_reasons = diagnostics.get('layer_b_soft_invalid_reasons') or {}
    reasons = [f"{key}={value}" for key, value in sorted({
        **a_reasons, **b_reasons, **{f"SOFT_{key}": value for key, value in b_soft_reasons.items()}
    }.items())]
    time_text = "TIME A {:.1f}ms B {:.1f}ms C1 {:.1f}ms C2 {:.1f}ms TOTAL {:.1f}ms".format(
        1000 * float(timing.get('layer_a_seconds', 0.0)),
        1000 * float(timing.get('layer_b_seconds', 0.0)),
        1000 * float(timing.get('layer_c1_seconds', 0.0)),
        1000 * float(timing.get('layer_c2_seconds', 0.0)),
        1000 * float(timing.get('total_seconds', diagnostics.get('total_seconds', 0.0))),
    )
    return '\n'.join((
        f"OBSTACLES lines={line_count} circles={circle_count}",
        "STAGES A_passed={} B_passed={} C_raw={} C_unique={} replays={} failed={}".format(
            diagnostics.get('layer_a_passed', 0), diagnostics.get('layer_b_passed', 0),
            diagnostics.get('integer_candidates_raw', 0), diagnostics.get('integer_candidates_unique', 0),
            diagnostics.get('integer_full_replays', diagnostics.get('full_replays', 0)),
            diagnostics.get('integer_replay_failed', 0),
        ),
        f"REASONS {' '.join(reasons) if reasons else 'none'}",
        time_text,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO ShellShock wormhole aim")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--capture-dir", type=Path, default=Path("train/yolo_captures"))
    parser.add_argument("--shot-metadata-dir", type=Path, default=Path("train/shot_metadata"))
    parser.add_argument("--capture-x", type=int, default=0)
    parser.add_argument("--capture-y", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=None)
    parser.add_argument("--capture-height", type=int, default=None)
    args = parser.parse_args()
    detector = YoloDetector(str(args.weights), args.confidence)
    mode = "normal_low"
    capture_mode = False
    final_manager = None

    def choose(value: str) -> None:
        nonlocal mode, final_manager
        mode = value
        final_manager = None
        print(f"Mode: {mode}", flush=True)

    def switch_final_candidate(delta: int) -> None:
        nonlocal final_manager
        if mode_parts(mode)[0] != "reflection" or final_manager is None:
            if mode_parts(mode)[0] == "reflection":
                print("No active reflection candidates; press E to calculate first", flush=True)
                return
            choose(select_mode("page up" if delta < 0 else "page down", mode))
            return
        result = final_manager.switch(delta)
        if result is not None:
            index = final_manager.current_final_index + 1
            print(f"ACTIVE {index}/{len(final_manager.results)} POWER={result.power} ANGLE={result.angle}", flush=True)

    def toggle_capture() -> None:
        nonlocal capture_mode
        capture_mode = not capture_mode
        print(f"Capture mode: {'ON' if capture_mode else 'OFF'}", flush=True)

    def aim() -> None:
        try:
            capture_stem = None
            if capture_mode:
                capture_stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                region = None
                if args.capture_width is not None or args.capture_height is not None:
                    if args.capture_width is None or args.capture_height is None:
                        raise ValueError("--capture-width and --capture-height must be provided together")
                    region = (args.capture_x, args.capture_y, args.capture_width, args.capture_height)
                try:
                    print(f"Capture: {capture_fixed_screen(args.capture_dir, region=region, stem=capture_stem)}", flush=True)
                except (OSError, RuntimeError, ValueError) as error:
                    print(f"Capture skipped: {error}", flush=True)
            print(f"Searching {mode} shot...", flush=True)
            hwnd = find_game_window()
            origin, size = client_screen_geometry(hwnd)
            target = screen_to_client_point(win32api.GetCursorPos(), origin, size)
            image = capture_client_area(hwnd)
            detections = detector.detect(image)
            world = build_world_from_image(detections, image)
            orange = sum(box.name == "portal_orange" for box in detections)
            blue = sum(box.name == "portal_blue" for box in detections)
            print(f"World: self={'yes' if world.self_position else 'no'}; obstacles lines={len(world.lines)}, circles={len(world.circles)}; portals orange={orange}, blue={blue}, pairs={len(world.portal_pairs)}, unpaired={world.unpaired_portals}", flush=True)
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
                details = format_solver_diagnostics(solution.get('diagnostics', {}), line_count=len(world.lines), circle_count=len(world.circles))
                raise RuntimeError(f"no safe {mode} shot: {messages.get(reason, reason)}\n{details}")
            reflection_results = []
            if mode_parts(mode)[0] == "reflection":
                for item in solution.get("diagnostics", {}).get("final_results", ()):
                    reflection_results.append(FinalReplayResult(
                        int(item["angle_degrees"]), int(item["power"]), True, None,
                        float(item.get("miss_distance", float("inf"))),
                        float(item.get("clearance", 0.0)), float(item.get("incidence", 0.0)),
                        float(item.get("timing", {}).get("total_seconds", 0.0)),
                        0.0, payload=dict(item),
                    ))

            def click_final(result):
                try:
                    latest_hwnd = find_game_window()
                    latest_origin, latest_size = client_screen_geometry(latest_hwnd)
                    latest_click = disc_click_point(
                        *world.self_position, str(result.payload.get("direction", solution.get("direction", "right"))),
                        float(result.angle), float(result.power), latest_size[0],
                    )
                    if not (0 <= latest_click[0] < latest_size[0] and 0 <= latest_click[1] < latest_size[1]):
                        print(f"CLICK_STATUS=OFFSCREEN CLICK_POS={latest_click} MANUAL CONTROL: POWER={result.power} ANGLE={result.angle} (power, angle)=({result.power}, {result.angle})", flush=True)
                        return "OFFSCREEN"
                    ensure_game_window_is_active(latest_hwnd)
                    click_screen_point((latest_origin[0] + latest_click[0], latest_origin[1] + latest_click[1]))
                    print(f"CLICK_STATUS=SUCCESS CLICK_POS={latest_click}", flush=True)
                    return "SUCCESS"
                except Exception as error:
                    print(f"CLICK_STATUS=FAILED reason={error} MANUAL CONTROL: POWER={result.power} ANGLE={result.angle} (power, angle)=({result.power}, {result.angle})", flush=True)
                    return "FAILED"

            if reflection_results:
                final_manager = FinalResultManager(reflection_results, click_final)
                final_manager.activate(0)
            click = disc_click_point(*world.self_position, str(solution["direction"]), float(solution["angle_degrees"]), float(solution["power"]), image.shape[1])
            if not (0 <= click[0] < size[0] and 0 <= click[1] < size[1]):
                raise RuntimeError("aim disc point is outside game client")
            if not reflection_results:
                ensure_game_window_is_active(hwnd)
                click_screen_point((origin[0] + click[0], origin[1] + click[1]))
            print(format_aim_report(mode, solution, value, direction, click), flush=True)
            if capture_stem:
                save_shot_metadata(args.shot_metadata_dir, capture_stem, direction=str(solution["direction"]), angle_degrees=float(solution["angle_degrees"]), power=float(solution["power"]))
        except (RuntimeError, ValueError) as error:
            print(f"Aim skipped: {error}")

    keyboard.add_hotkey("e", aim)
    keyboard.add_hotkey("caps lock", toggle_capture)
    keyboard.add_hotkey("t", lambda: choose("normal_low"))
    keyboard.add_hotkey("h", lambda: choose("wormhole_low"))
    keyboard.add_hotkey("r", lambda: choose("reflection_low"))
    keyboard.add_hotkey("page up", lambda: switch_final_candidate(-1))
    keyboard.add_hotkey("page down", lambda: switch_final_candidate(1))
    keyboard.add_hotkey(EXIT_HOTKEY, lambda: print("Exiting YOLO aim...", flush=True))
    print("Ready: E=aim, CapsLock=toggle capture, T=normal, H=wormhole, R=reflection, PageUp=high arc, PageDown=low arc, Del=quit")
    keyboard.wait(EXIT_HOTKEY)


if __name__ == "__main__":
    main()

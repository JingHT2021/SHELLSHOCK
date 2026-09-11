"""Independent YOLO aim entrypoint; does not alter detect_shellshock.py."""
from __future__ import annotations

from shellshock.config.paths import DATA_ROOT

import argparse
from pathlib import Path
from datetime import datetime

import keyboard
import win32api

from shellshock.interaction.aiming import disc_click_point
from shellshock.interaction.click_policy import click_client_point
from shellshock.adapters.windows import (
    capture_client_area, client_screen_geometry, click_screen_point,
    ensure_game_window_is_active, find_game_window, screen_to_client_point,
)
from shellshock.capture.storage import save_capture_assets, save_shot_metadata
from shellshock.application.solver import solve_integer_shot
from shellshock.application.scene import analyze_frame
from shellshock.config.paths import LOG_ROOT
from shellshock.planning.policies import mode_parts, normalize_mode
from shellshock.perception.yolo import YoloDetector
from shellshock.perception.world import build_world_from_image_with_diagnostics
from shellshock.perception.wind import detect_wind
from shellshock.interaction.results import FinalResultManager, FinalReplayResult

DEFAULT_WEIGHTS = (DATA_ROOT / 'runs/shellshock_yolo11n_pose_v1/weights/best.pt')
EXIT_HOTKEY = "delete"
REFLECTION_LOG_DIR = LOG_ROOT


from shellshock.planning.policies import select_mode as select_mode


def display_mode(mode: str) -> str:
    return "reflection" if mode_parts(mode)[0] == "reflection" else mode


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
    lines = [f"MODE {display_mode(mode):<9} WIND {wind}  {controls}",
             f"PORTALS {portals:>2}  {reflection}  EVENTS {events}"]
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
    if solution.get('relaxed_target_acceptance'):
        lines.append(f"RELAXED_TARGET radius={float(solution.get('target_accept_radius', 0.0)):.2f} px")
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
    replay_text = (
        f"RELAXED_REPLAY used={bool(diagnostics.get('relaxed_target_acceptance_used'))} "
        f"radius={diagnostics.get('relaxed_target_accept_radius', 0.0)} px"
        if diagnostics.get('relaxed_target_acceptance_used') else None
    )
    trace_lines = []
    portal_traces = diagnostics.get("layer_a_route_trace") or ()
    trace_lines.append(f"PORTAL_A_TRACE total={len(portal_traces)}")
    for route in portal_traces:
        if not isinstance(route, dict):
            continue
        route_name = ",".join(str(item) for item in route.get("route", ()))
        detail = route.get("reason") or ("PASS" if route.get("status") == "PASS" else "-")
        trace_lines.append(f"  route=[{route_name}] {route.get('status', '?')} {detail}")
        for segment in route.get("segments", ()):
            if isinstance(segment, dict):
                segment_detail = segment.get("reason") or segment.get("status") or "-"
                trace_lines.append(f"    {segment.get('name', '?')} {segment.get('status', '?')} {segment_detail}")
    for label, key in (("A", "layer_a_trace"), ("B", "layer_b_trace"), ("C1", "layer_c1_trace"),
                       ("C2", "layer_c2_trace"), ("FINAL", "final_trace")):
        entries = diagnostics.get(key) or ()
        trace_lines.append(f"{label}_TRACE total={len(entries)}")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            detail = entry.get("reason") or entry.get("decision") or "-"
            trace_lines.append(f"  {entry.get('id', '?')} {entry.get('status', '?')} {detail}")
            if label == "A":
                path = entry.get("path_diagnostics") or {}
                frame = path.get("reflection_frame")
                if isinstance(frame, dict):
                    trace_lines.append(
                        f"    REFLECTION_FRAME origin={frame.get('origin')} tangent={frame.get('tangent')} "
                        f"normal={frame.get('normal')} local_source={frame.get('local_source')} "
                        f"local_target={frame.get('local_target')} acceleration={frame.get('local_acceleration')}"
                    )
                    trace_lines.append(
                        f"    DIRECTIONS before={path.get('directions_before_reflection', ())} "
                        f"after={path.get('directions_after_reflection', ())}"
                    )
                    for phase in ("before_segments", "after_segments"):
                        for segment in path.get(phase, ()):
                            if isinstance(segment, dict):
                                trace_lines.append(
                                    f"    {phase.upper()} {segment.get('name', '?')} "
                                    f"{segment.get('status', '?')} {segment.get('reason') or '-'} "
                                    f"from={segment.get('from')} to={segment.get('to')} "
                                    f"directions={segment.get('directions_after', ())}"
                                )
    return '\n'.join(tuple(item for item in (
        f"OBSTACLES lines={line_count} circles={circle_count}",
        "STAGES A_passed={} B_passed={} C_raw={} C_unique={} replays={} failed={}".format(
            diagnostics.get('layer_a_passed', 0), diagnostics.get('layer_b_passed', 0),
            diagnostics.get('integer_candidates_raw', 0), diagnostics.get('integer_candidates_unique', 0),
            diagnostics.get('integer_full_replays', diagnostics.get('full_replays', 0)),
            diagnostics.get('integer_replay_failed', 0),
        ),
        f"REASONS {' '.join(reasons) if reasons else 'none'}",
        replay_text,
        *trace_lines,
        time_text,
    ) if item is not None))


def format_solver_summary(diagnostics: dict[str, object]) -> str:
    """Return the short reflection status suitable for the interactive terminal."""
    a_reasons = diagnostics.get('layer_a_invalid_reasons') or {}
    b_reasons = diagnostics.get('layer_b_invalid_reasons') or {}
    b_soft_reasons = diagnostics.get('layer_b_soft_invalid_reasons') or {}
    reasons = [f"{key}={value}" for key, value in sorted({
        **a_reasons, **b_reasons, **{f"SOFT_{key}": value for key, value in b_soft_reasons.items()}
    }.items())]
    timing = diagnostics.get('timing') if isinstance(diagnostics.get('timing'), dict) else {}
    lines = [
        "STAGES A_passed={} B_passed={} C_raw={} C_unique={} replays={} failed={}".format(
            diagnostics.get('layer_a_passed', 0), diagnostics.get('layer_b_passed', 0),
            diagnostics.get('integer_candidates_raw', 0), diagnostics.get('integer_candidates_unique', 0),
            diagnostics.get('integer_full_replays', diagnostics.get('full_replays', 0)),
            diagnostics.get('integer_replay_failed', 0),
        ),
        f"REASONS {' '.join(reasons) if reasons else 'none'}",
        "TIME {:.1f}ms".format(1000 * float(timing.get('total_seconds', diagnostics.get('total_seconds', 0.0)))),
    ]
    if diagnostics.get('relaxed_target_acceptance_used'):
        lines.append(
            f"RELAXED_REPLAY used=True radius={diagnostics.get('relaxed_target_accept_radius', 0.0)} px"
        )
    return '\n'.join(lines)


def write_reflection_diagnostics_log(
    diagnostics: dict[str, object], *, line_count: int = 0, circle_count: int = 0,
    log_dir: Path = REFLECTION_LOG_DIR,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = (log_dir / f"reflection_{stamp}.log").resolve()
    path.write_text(
        format_solver_diagnostics(diagnostics, line_count=line_count, circle_count=circle_count) + "\n",
        encoding="utf-8",
    )
    return path


def format_normal_diagnostics(diagnostics: dict[str, object]) -> str:
    target = diagnostics.get('target', {})
    rejected = diagnostics.get('rejected_reasons') or {}
    rejected_text = ' '.join(f'{key}={value}' for key, value in sorted(rejected.items())) or 'none'
    return '\n'.join((
        'TARGET=({:.1f},{:.1f}) WIND={} {}'.format(
            float(target.get('x', 0.0)), float(target.get('y', 0.0)),
            diagnostics.get('wind_value', 0), diagnostics.get('wind_direction', 'right'),
        ),
        'THEORY angle={:.2f} power={:.2f} CANDIDATES={} VERIFIED={}'.format(
            float(diagnostics.get('theory_angle', 0.0)), float(diagnostics.get('theory_power', 0.0)),
            diagnostics.get('candidate_count', 0), diagnostics.get('verified_count', 0),
        ),
        f'REJECTED {rejected_text}',
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO ShellShock wormhole aim")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--capture-dir", type=Path, default=(DATA_ROOT / 'yolo_captures'))
    parser.add_argument("--shot-metadata-dir", type=Path, default=(DATA_ROOT / 'yolo_captures/metadata'))
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
        print(f"Mode: {display_mode(mode)}", flush=True)

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
            print(f"Searching {display_mode(mode)} shot...", flush=True)
            hwnd = find_game_window()
            origin, size = client_screen_geometry(hwnd)
            target = screen_to_client_point(win32api.GetCursorPos(), origin, size)
            image = capture_client_area(hwnd)
            detections = detector.detect(image)
            analysis = analyze_frame(image, detections=detections)
            world, keypoint_errors = analysis.world, analysis.diagnostics
            for item in keypoint_errors:
                errors = [value for key, value in item.items() if key.endswith("error_px") and value is not None]
                if errors:
                    print(f"KEYPOINT_ERROR class={item['class']} source={item['source']} px={','.join(f'{value:.2f}' for value in errors)}", flush=True)
            orange = sum(box.name == "portal_orange" for box in detections)
            blue = sum(box.name == "portal_blue" for box in detections)
            print(f"World: self={'yes' if world.self_position else 'no'}; obstacles lines={len(world.lines)}, circles={len(world.circles)}; portals orange={orange}, blue={blue}, pairs={len(world.portal_pairs)}, unpaired={world.unpaired_portals}", flush=True)
            if world.self_position is None:
                raise RuntimeError("YOLO did not find one unambiguous self tank")
            value, direction = analysis.wind_value, analysis.wind_direction
            solution = solve_integer_shot(world.self_position, target, world, value, direction, image.shape[1], mode)
            if mode_parts(mode)[0] == "normal" and solution.get('diagnostics'):
                print(format_normal_diagnostics(solution['diagnostics']), flush=True)
            if solution.get("status") != "reachable":
                reason = str(solution.get("reason"))
                messages = {
                    "no-portal-pair": "YOLO did not produce a usable orange/blue portal pair",
                    "no-verified-wormhole-shot": "portals were found, but no collision-free one/two-hop shot was verified",
                }
                diagnostics = solution.get('diagnostics', {})
                if mode_parts(mode)[0] == "reflection":
                    try:
                        log_path = write_reflection_diagnostics_log(
                            diagnostics, line_count=len(world.lines), circle_count=len(world.circles),
                        )
                        log_status = f"REFLECTION_LOG {log_path}"
                    except OSError as error:
                        log_status = f"REFLECTION_LOG_FAILED {error}"
                    summary = format_solver_summary(diagnostics)
                    raise RuntimeError(
                        f"no safe {mode} shot: {messages.get(reason, reason)}\n{summary}\n{log_status}"
                    )
                details = format_solver_diagnostics(diagnostics, line_count=len(world.lines), circle_count=len(world.circles))
                raise RuntimeError(f"no safe {mode} shot: {messages.get(reason, reason)}\n{details}")
            if mode_parts(mode)[0] == "reflection":
                try:
                    reflection_log_path = write_reflection_diagnostics_log(
                        solution.get('diagnostics', {}), line_count=len(world.lines), circle_count=len(world.circles),
                    )
                    # Debug-only terminal dump; keep disabled during normal play.
                    # print(format_solver_diagnostics(solution.get('diagnostics', {}), line_count=len(world.lines), circle_count=len(world.circles)), flush=True)
                    print(f"REFLECTION_LOG {reflection_log_path}", flush=True)
                except OSError as error:
                    print(f"REFLECTION_LOG_FAILED {error}", flush=True)
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

            click = disc_click_point(*world.self_position, str(solution["direction"]), float(solution["angle_degrees"]), float(solution["power"]), image.shape[1])
            print(format_aim_report(mode, solution, value, direction, click), flush=True)

            def click_final(result):
                try:
                    latest_hwnd = find_game_window()
                    latest_origin, latest_size = client_screen_geometry(latest_hwnd)
                    latest_click = disc_click_point(
                        *world.self_position, str(result.payload.get("direction", solution.get("direction", "right"))),
                        float(result.angle), float(result.power), latest_size[0],
                    )
                    return click_client_point(
                        latest_click,
                        latest_origin,
                        latest_size,
                        activate=lambda: ensure_game_window_is_active(latest_hwnd),
                        click=click_screen_point,
                    )
                except Exception as error:
                    print(f"CLICK_STATUS=FAILED reason={error} MANUAL CONTROL: POWER={result.power} ANGLE={result.angle} (power, angle)=({result.power}, {result.angle})", flush=True)
                    return "FAILED"

            click_status = "FAILED"
            if reflection_results:
                final_manager = FinalResultManager(reflection_results, click_final)
                click_status = final_manager.activate(0) or "FAILED"
            if not reflection_results:
                click_status = click_client_point(
                    click,
                    origin,
                    size,
                    activate=lambda: ensure_game_window_is_active(hwnd),
                    click=click_screen_point,
                )
            if capture_mode and click_status == "SUCCESS":
                capture_stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                try:
                    post_click_hwnd = find_game_window()
                    post_click_image = capture_client_area(post_click_hwnd)
                    _, _, post_wind_box = detect_wind(post_click_image)
                    portal_boxes = [
                        (box.x, box.y, box.width, box.height)
                        for box in detections
                        if box.name in {"portal_orange", "portal_blue"}
                    ]
                    print(
                        f"Capture: {save_capture_assets(post_click_image, args.capture_dir, capture_stem, post_wind_box, portal_boxes)}",
                        flush=True,
                    )
                    save_shot_metadata(
                        args.shot_metadata_dir,
                        capture_stem,
                        direction=str(solution["direction"]),
                        angle_degrees=float(solution["angle_degrees"]),
                        power=float(solution["power"]),
                        wind_value=float(value),
                        wind_direction=str(direction),
                    )
                except Exception as error:
                    print(f"Capture skipped after click: {error}", flush=True)
        except (RuntimeError, ValueError) as error:
            print(f"Aim skipped: {error}")

    keyboard.add_hotkey("e", aim)
    keyboard.add_hotkey("f5", aim)
    keyboard.add_hotkey("caps lock", toggle_capture)
    keyboard.add_hotkey("t", lambda: choose("normal_low"))
    keyboard.add_hotkey("h", lambda: choose("wormhole_low"))
    keyboard.add_hotkey("r", lambda: choose("reflection"))
    keyboard.add_hotkey("page up", lambda: switch_final_candidate(-1))
    keyboard.add_hotkey("page down", lambda: switch_final_candidate(1))
    keyboard.add_hotkey(EXIT_HOTKEY, lambda: print("Exiting YOLO aim...", flush=True))
    print("Ready: E=aim, CapsLock=toggle capture, T=normal, H=wormhole, R=reflection, PageUp=high arc, PageDown=low arc, Del=quit")
    keyboard.wait(EXIT_HOTKEY)


if __name__ == "__main__":
    main()

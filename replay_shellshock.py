"""Offline ShellShock replay, annotation merge, trajectory overlay, and tuning UI."""
from __future__ import annotations

from shellshock.config.paths import DATA_ROOT

import argparse
import json
import sys
from dataclasses import replace
from math import cos, radians, sin
from pathlib import Path

import cv2

from shellshock.annotations.conversion import (
    AnnotationBox, SceneAnnotation, annotations_to_world, annotations_to_world_with_diagnostics, load_manual_scene, merge_annotations,
    save_manual_scene, scene_to_dict, yolo_detections_to_annotations,
)
from shellshock.perception.guide import GuideDetection
from shellshock.application.solver import solve_integer_shot
from shellshock.application.scene import analyze_frame
from shellshock.annotations.bundle import annotation_paths
from shellshock.rendering.overlay import draw_trajectory
from shellshock.physics.launch import muzzle_position
from shellshock.planning.policies import normalize_mode
from shellshock.rendering.trajectory import sample_solution_trajectory
from shellshock.perception.yolo import YoloDetector


DEFAULT_IMAGE = (DATA_ROOT / 'annotate/images')
DEFAULT_WEIGHTS = (DATA_ROOT / 'runs/shellshock_yolo11n_pose_v1/weights/best.pt')
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ARROW_DELTAS = {
    81: ("angle", -1), 2424832: ("angle", -1),
    83: ("angle", 1), 2555904: ("angle", 1),
    82: ("power", 1), 2490368: ("power", 1),
    84: ("power", -1), 2621440: ("power", -1),
}
HOME_KEY_CODES = {36, 2359296}


def annotated_image_paths(image_dir: Path = DEFAULT_IMAGE) -> list[Path]:
    image_dir = Path(image_dir)
    if image_dir.is_file():
        return [image_dir]
    return sorted(
        (path for path in image_dir.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def latest_annotated_image(image_path: Path = DEFAULT_IMAGE) -> Path:
    image_path = Path(image_path)
    images = annotated_image_paths(image_path)
    if not images:
        raise FileNotFoundError(f"no annotated images found in {image_path}")
    return images[-1]


def arrow_adjustment(raw_key):
    return ARROW_DELTAS.get(raw_key)


def wind_adjustment(key):
    if key in (ord("z"), ord("Z")):
        return -1.0
    if key in (ord("c"), ord("C")):
        return 1.0
    return None


def is_reset_view_key(raw_key, key):
    return raw_key in HOME_KEY_CODES


def derive_self_center(muzzle, direction, angle_degrees, extension):
    sign = 1.0 if direction == "right" else -1.0
    angle = radians(float(angle_degrees))
    return (float(muzzle[0]) - sign * float(extension) * cos(angle), float(muzzle[1]) + float(extension) * sin(angle))


def solution_launch_point(center, solution, image_width, extension=35.0):
    if solution.get('launch_point') is not None:
        return tuple(solution['launch_point'])
    return muzzle_position(center,solution['direction'],float(solution['angle_degrees']),image_width,barrel_length=extension)


def fit_to_canvas(image_width, image_height, canvas_width, canvas_height):
    scale = min(float(canvas_width) / image_width, float(canvas_height) / image_height)
    rendered_width, rendered_height = round(image_width * scale), round(image_height * scale)
    return scale, ((canvas_width - rendered_width) // 2, (canvas_height - rendered_height) // 2)


from shellshock.planning.policies import select_mode as select_replay_mode


def guide_from_report(report):
    payload = report.get("guide", {})
    return GuideDetection(
        str(payload.get("status", "uncertain")),
        tuple((float(x), float(y)) for x, y in payload.get("points", ())),
        float(payload.get("confidence", 0.0) or 0.0),
        payload.get("reason"),
        dict(payload.get("fit_parameters", {})),
    )


def manual_preview_solution(direction, angle_degrees, power, mode):
    # Match the solver's maximum replay horizon so high arcs include their descent.
    return {"status": "reachable", "mode": normalize_mode(mode), "direction": direction, "angle_degrees": float(angle_degrees), "power": int(max(1, min(100, power))), "flight_time_seconds": 12.0, "events": [], "manual_preview": True}


def make_calibration_sample(solution, predicted_points, actual_point):
    actual = (float(actual_point[0]), float(actual_point[1]))
    points = tuple((float(p[0]), float(p[1])) for p in predicted_points if p is not None)
    if points:
        index = min(range(len(points)), key=lambda i: (points[i][0] - actual[0]) ** 2 + (points[i][1] - actual[1]) ** 2)
        predicted = points[index]
        fraction = index / max(1, len(points) - 1)
    else:
        predicted, fraction = None, 0.0
    return {
        "angle_degrees": float(solution.get("angle_degrees", 0.0) or 0.0),
        "power": int(solution.get("power", 0) or 0),
        "actual_point": actual,
        "predicted_point": predicted,
        "trajectory_fraction": float(fraction),
    }


def nearest_calibration_index(samples, point, radius=50.0):
    hits = [(index, (float(item["actual_point"][0]) - point[0]) ** 2 + (float(item["actual_point"][1]) - point[1]) ** 2) for index, item in enumerate(samples)]
    if not hits:
        return None
    index, distance = min(hits, key=lambda item: item[1])
    return index if distance <= float(radius) ** 2 else None


def calibration_hint(samples):
    if not samples:
        return None
    angle = sum(float(item["angle_degrees"]) for item in samples) / len(samples)
    power = sum(float(item["power"]) for item in samples) / len(samples)
    dx = sum(float(item["actual_point"][0]) - float(item["predicted_point"][0]) for item in samples if item.get("predicted_point")) / max(1, sum(bool(item.get("predicted_point")) for item in samples))
    dy = sum(float(item["actual_point"][1]) - float(item["predicted_point"][1]) for item in samples if item.get("predicted_point")) / max(1, sum(bool(item.get("predicted_point")) for item in samples))
    # Local screen-space correction: rightward error needs more power; a
    # downward error needs a slightly lower elevation angle.
    suggested_angle = angle - dy * 0.05
    suggested_power = power + dx * 0.02
    return {"angle_degrees": round(max(0.0, min(90.0, suggested_angle)), 2), "power": round(max(1.0, min(100.0, suggested_power)), 2), "mean_dx": round(dx, 2), "mean_dy": round(dy, 2), "sample_count": len(samples)}


def compare_trajectories(predicted, observed):
    predicted = tuple(p for p in predicted if p is not None)
    if not predicted or not observed:
        return {"matched_point_count": 0, "mean_distance": None, "max_distance": None, "endpoint_distance": None}
    errors = []
    for x, y in observed:
        errors.append(min(((x-px)**2 + (y-py)**2) ** 0.5 for px, py in predicted))
    return {"matched_point_count": len(errors), "mean_distance": sum(errors) / len(errors), "max_distance": max(errors), "endpoint_distance": min(((observed[-1][0]-px)**2 + (observed[-1][1]-py)**2) ** 0.5 for px, py in predicted)}


def solver_log_lines(solution):
    diagnostics = solution.get("diagnostics", {}) if isinstance(solution, dict) else {}
    keys = ("layer_a_generated", "layer_a_rejected", "layer_b_seed_count",
            "candidate_count", "verified_count", "budget_exhausted")
    lines = [" ".join(f"{key}={diagnostics[key]}" for key in keys if key in diagnostics)]
    if solution.get("selected_route_id") is not None:
        lines.append(f"SELECTED route_id={solution.get('selected_route_id')} branch_id={solution.get('selected_branch_id')} candidate_id={solution.get('selected_candidate_id')}")
    for entry in diagnostics.get("route_trace", ()):
        layer_b = entry.get("layer_b", "PASS" if entry.get("continuous_seeds", 0) else "REJECT")
        lines.append(
            f"ROUTE {entry.get('route', [])} layer_a={entry.get('layer_a', '?')} "
            f"layer_b={layer_b} reason={entry.get('reason', '')} "
            f"seeds={entry.get('continuous_seeds', 0)}"
        )
    for entry in diagnostics.get("candidate_trace", ()):
        lines.append(
            f"CANDIDATE route={entry.get('route', [])} angle={entry.get('angle', '?')} "
            f"power={entry.get('power', '?')} layer_c1={entry.get('layer_c1', '?')} "
            f"layer_c2={entry.get('layer_c2', '?')} reason={entry.get('reason', '')}"
        )
    for reason, count in sorted(diagnostics.get("rejected_reasons", {}).items()):
        lines.append(f"{reason}={count}")
    selected = diagnostics.get("selected_route")
    branch = diagnostics.get("selected_branch")
    candidate = diagnostics.get("selected_candidate")
    if selected is not None:
        lines.append(f"A_CHAIN id={selected.get('id')} route={selected.get('route')} reason={selected.get('reason')}")
    if branch is not None:
        lines.append(f"B_BRANCH id={branch.get('id')} parent_route_id={branch.get('route_id')} seed_index={branch.get('seed_index')}")
    if candidate is not None:
        lines.append(f"C_SELECTED id={candidate.get('id')} parent_route_id={candidate.get('route_id')} parent_branch_id={candidate.get('branch_id')} angle={candidate.get('angle')} power={candidate.get('power')}")
    return [line for line in lines if line]


def build_final_trace(solution, predicted_points, world):
    """Return JSON-safe lineage and exact geometry for the selected replay."""
    from shellshock.math2d.geometry import trajectory_position
    from shellshock.physics.events.portal import portal_map, translate

    segments = []
    raw_segments = solution.get("segments", ())
    for index, segment in enumerate(raw_segments):
        start = tuple(float(value) for value in segment["start"])
        velocity = tuple(float(value) for value in segment["velocity"])
        acceleration = tuple(float(value) for value in segment["acceleration"])
        duration = float(segment["duration"])
        end = tuple(float(value) for value in trajectory_position(start, velocity, acceleration, duration))
        segments.append({"index": index, "start": start, "end": end, "velocity": velocity,
                         "acceleration": acceleration, "duration": duration,
                         "elapsed": float(segment.get("elapsed", 0.0))})
    transitions = []
    portals = portal_map(world)
    for event_index, event in enumerate(solution.get("event_trace", ())):
        if event.get("kind") != "portal":
            continue
        portal_id = str(event.get("id"))
        if portal_id not in portals:
            continue
        entry, exit_portal, exit_id = portals[portal_id]
        entry_point = tuple(float(value) for value in event["point"])
        exit_point = tuple(float(value) for value in translate(entry_point, entry, exit_portal))
        transitions.append({"event_index": event_index, "portal_id": portal_id,
                            "entry_portal": {"center": tuple(entry.center), "radius": float(entry.radius)},
                            "exit_portal": {"id": exit_id, "center": tuple(exit_portal.center), "radius": float(exit_portal.radius)},
                            "entry_point": entry_point, "exit_point": exit_point,
                            "displacement": tuple(exit_point[i] - entry_point[i] for i in (0, 1)),
                            "time": float(event.get("time", 0.0))})
    return {"selected_route_id": solution.get("selected_route_id"),
            "selected_branch_id": solution.get("selected_branch_id"),
            "selected_candidate_id": solution.get("selected_candidate_id"),
            "route": solution.get("route", ()),
            "events": list(solution.get("events", ())),
            "waypoints": [dict(item) for item in solution.get("event_trace", ())],
            "segments": segments, "portal_transitions": transitions,
            "sampled_points": [None if point is None else tuple(float(value) for value in point) for point in predicted_points]}


def manual_controls(scene, solution):
    """Carry the complete solved aim into interactive preview."""
    scene.metadata["direction"] = solution.get("direction", scene.metadata.get("direction", "right"))
    angle = solution.get("angle_degrees")
    power = solution.get("power")
    return float(45 if angle is None else angle), int(50 if power is None else power)


def save_calibration_samples(image_path, output_root, samples):
    destination = calibration_samples_path(image_path, output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"image": str(image_path), "samples": list(samples)}, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
    return destination


def calibration_samples_path(image_path, output_root):
    return Path(output_root) / "calibration" / f"{Path(image_path).stem}.json"


def load_calibration_samples(image_path, output_root):
    path = calibration_samples_path(image_path, output_root)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    return list(samples) if isinstance(samples, list) else []


def render_replay_visual(image, predicted, guide, scene, report, *, show_annotations=True, show_guide=False, line_width=1):
    overlay = image.copy()
    draw_trajectory(overlay, predicted, thickness=line_width)
    if show_annotations:
        for box in scene.boxes:
            color = (0, 0, 255) if box.name == "enemy" else (0, 200, 0)
            cv2.rectangle(overlay, (round(box.x), round(box.y)), (round(box.x + box.width), round(box.y + box.height)), color, 2)
            for index, point in enumerate(box.keypoints):
                if box.name == "self" and index == 0:
                    continue
                if point.visible <= 0:
                    continue
                p = (round(point.x), round(point.y))
                point_color = (255, 255, 255) if index == 0 else (255, 120, 255)
                cv2.circle(overlay, p, 7, point_color, -1)
                cv2.circle(overlay, p, 9, (0, 0, 0), 1)
                cv2.putText(overlay, f"{box.name}.kp{index}", (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .42, point_color, 1)
    if report.get("target") is not None:
        target = tuple(round(value) for value in report["target"])
        cv2.drawMarker(overlay, target, (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
        cv2.putText(overlay, "TARGET (E)", (target[0] + 12, target[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    if report.get("self_center") is not None:
        center = tuple(round(value) for value in report["self_center"])
        cv2.drawMarker(overlay, center, (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(overlay, "SELF CENTER", (center[0] + 12, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    for point, label in report.get("event_markers", ()):
        point = tuple(round(value) for value in point)
        cv2.drawMarker(overlay, point, (255, 0, 255), cv2.MARKER_DIAMOND, 18, 2)
        cv2.putText(overlay, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 2)
    solution = report.get("solution", {})
    wind_value = report.get("pending_wind_value", report.get("wind_value", 0))
    pending = "*" if report.get("pending_wind_value") is not None and report.get("pending_wind_value") != report.get("wind_value") else ""
    wind = f"WIND={wind_value:g}{pending} {str(report.get('wind_direction', 'right')).upper()}"
    controls = f"ANGLE={solution.get('angle_degrees', 'N/A')} POWER={solution.get('power', 'N/A')}"
    manual = " MANUAL=ON" if report.get("manual_adjust") else ""
    text = f"SOURCE={report.get('source', 'pending')} MODE={solution.get('mode', '')} {wind} {controls}{manual}"
    cv2.putText(overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
    if pending:
        cv2.putText(overlay, f"WIND PENDING={wind_value:g}  (ENTER=APPLY)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
    cv2.putText(overlay, f"TRAJECTORY=solver-red EVENTS={','.join(map(str, solution.get('events', ())))}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
    if report.get("debug_logs"):
        cv2.putText(overlay, f"LOG: {report['debug_logs'][0][:120]}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 220, 255), 1)
    for index, sample in enumerate(report.get("calibration_samples", ())):
        point = tuple(round(value) for value in sample["actual_point"])
        cv2.drawMarker(overlay, point, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 2)
        cv2.putText(overlay, f"CAL {index + 1}", (point[0] + 8, point[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
    hint = report.get("calibration_hint")
    if hint:
        cv2.putText(overlay, f"CALIBRATION n={hint['sample_count']} DX={hint['mean_dx']:+.1f} DY={hint['mean_dy']:+.1f} (1=mark)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
    if report.get("calibration_armed"):
        cv2.putText(overlay, "CALIBRATION ARMED: click an actual trajectory point", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 2)
    return overlay


def _paths(stem, root):
    root = Path(root)
    return (root / "annotate" / "labels" / f"{stem}.txt", root / "annotate" / "pose_geometry" / f"{stem}.json", root / "annotate" / "metadata" / f"{stem}.json")


def load_replay_scene(image, source, image_path, data_root, detector=None):
    stem = Path(image_path).stem
    label, geometry, metadata = annotation_paths(image_path, data_root)
    if detector is not None:
        detections = detector.detect(image)
        automatic = yolo_detections_to_annotations(detections, image.shape[1], image.shape[0])
    else:
        automatic = SceneAnnotation(image.shape[1], image.shape[0])
    manual = load_manual_scene(label, geometry, metadata, image.shape[1], image.shape[0])
    if source == "yolo":
        return automatic, (label, geometry, metadata)
    if source == "annotation":
        return manual, (label, geometry, metadata)
    return merge_annotations(automatic, manual), (label, geometry, metadata)


def _target(scene):
    enemies = [item for item in scene.boxes if item.name == "enemy"]
    return max(enemies, key=lambda item: item.confidence).center if enemies else None


def replay_image(image_path, *, source="hybrid", weights=None, mode="normal_low", root=DATA_ROOT, output_root=(DATA_ROOT / 'annotate/replay'), target=None, scene_override=None, barrel_extension=35.0, angle_override=None, power_override=None, wind_override=None, persist=True, candidate_index=0):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    detector = YoloDetector(str(weights or DEFAULT_WEIGHTS)) if source in {"yolo", "hybrid"} and scene_override is None else None
    if scene_override is None:
        scene, paths = load_replay_scene(image, source, image_path, root, detector)
    else:
        scene = scene_override
        paths = annotation_paths(image_path, root)
    analysis = analyze_frame(image, scene=scene, wind_override=wind_override)
    world, muzzle, keypoint_errors = analysis.world, analysis.muzzle, analysis.diagnostics
    meta = scene.metadata
    direction = str(meta.get("direction", "right")).lower()
    angle = float(meta.get("angle_degrees", 0.0) or 0.0)
    if scene.self_center is None and muzzle is not None:
        from dataclasses import replace
        world = replace(world, self_position=derive_self_center(muzzle, direction, angle, barrel_extension * image.shape[1] / 2560.0))
    target = target or _target(scene)
    if angle_override is not None and power_override is not None:
        solution = manual_preview_solution(direction, angle_override, power_override, mode)
    elif world.self_position is None or target is None:
        solution = {"status": "unreachable", "reason": "missing_self_or_target"}
    else:
        wind_value = float(meta.get("wind_value", 0.0) or 0.0)
        wind_direction = str(meta.get("wind_direction", "right")).lower()
        solution = solve_integer_shot(world.self_position, target, world, wind_value, wind_direction, image.shape[1], normalize_mode(mode), barrel_extension=barrel_extension)
    if str(mode).startswith("reflection_"):
        candidates = solution.get("diagnostics", {}).get("final_results", ())
        if candidates:
            selected_index = int(candidate_index) % len(candidates)
            diagnostics = solution.get("diagnostics", {})
            solution = {**candidates[selected_index], "mode": normalize_mode(mode), "status": "reachable", "diagnostics": diagnostics}
            solution["candidate_index"] = selected_index
            solution["candidate_count"] = len(candidates)
    if world.self_position is not None and solution.get("status") == "reachable":
        source_point = solution_launch_point(world.self_position, solution, image.shape[1], barrel_extension)
    else:
        source_point = world.self_position or (0.0, 0.0)
    wind_value = float(meta.get("wind_value", 0.0) or 0.0)
    predicted = sample_solution_trajectory(solution, source_point, image.shape[1], wind_value, str(meta.get("wind_direction", "right")), 120, world=world)
    solution["final_trace"] = build_final_trace(solution, predicted, world)
    guide = GuideDetection("disabled", reason="game_guide_detection_disabled")
    event_markers = []
    if solution.get("reflection_point") is not None:
        event_markers.append((solution["reflection_point"], "REFLECTION"))
    for index, event in enumerate(solution.get("portal_sequence", ())):
        try:
            portal_index, color = str(event).split(":", 1)
            pair = world.portal_pairs[int(portal_index)]
            portal = pair.orange if color == "orange" else pair.blue
            event_markers.append((portal.center, f"PORTAL {event}"))
        except (ValueError, IndexError):
            pass
    report = {"image": str(image_path), "source": source, "target": target, "self_center": world.self_position, "solution": solution, "wind_value": wind_value, "wind_direction": str(meta.get("wind_direction", "right")), "event_markers": event_markers, "debug_logs": solver_log_lines(solution), "keypoint_errors": keypoint_errors, "geometry_sources": {item["class"]: item["source"] for item in keypoint_errors}, "guide": {"status": guide.status, "confidence": 0.0, "reason": guide.reason, "points": (), "fit_parameters": {}}, "error": compare_trajectories(predicted, ())}
    report["predicted_points"] = predicted
    overlay = render_replay_visual(image, predicted, guide, scene, report)
    if persist:
        output_root = Path(output_root)
        (output_root / "overlays").mkdir(parents=True, exist_ok=True)
        (output_root / "reports").mkdir(parents=True, exist_ok=True)
        (output_root / "merged_annotations").mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(output_root / "overlays" / f"{stem}.png"), overlay)
        (output_root / "reports" / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
        keypoint_log = [
            "KEYPOINT_ERROR class={} source={} {}".format(
                item.get("class"), item.get("source"), ", ".join(f"{key}={value:.2f}px" for key, value in item.items() if key.endswith("error_px") and value is not None)
            )
            for item in keypoint_errors
        ]
        (output_root / "reports" / f"{stem}.log").write_text("\n".join([*report.get("debug_logs", ()), *keypoint_log]) + "\n", encoding="utf-8")
        (output_root / "merged_annotations" / f"{stem}.json").write_text(json.dumps(scene_to_dict(scene), ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
    return report, overlay, scene, paths


def interactive(image_path, args):
    report, overlay, scene, paths = replay_image(image_path, source=args.source, weights=args.weights, mode=args.mode, root=args.root, output_root=args.output_root, target=tuple(args.target) if args.target else None, barrel_extension=args.barrel_extension)
    image = cv2.imread(str(image_path))
    image_paths = annotated_image_paths(Path(image_path).parent)
    try:
        image_index = image_paths.index(Path(image_path))
    except ValueError:
        image_paths = [Path(image_path)] + [path for path in image_paths if path != Path(image_path)]
        image_index = 0
    selected = None
    cursor = [20.0, 20.0]
    target = tuple(args.target) if args.target else None
    mode = args.mode
    show_annotations = True
    show_guide = True
    manual_adjust = False
    calibration_armed = False
    calibration_samples = load_calibration_samples(image_path, args.output_root)
    visible_calibration_samples = list(calibration_samples)
    angle_override = None
    power_override = None
    wind_value_control = float(report.get("wind_value", 0.0) or 0.0) * (-1 if report.get("wind_direction") == "left" else 1)
    applied_wind_value = None
    zoom = 1.0
    view_origin = [0.0, 0.0]
    window = "ShellShock Replay"
    canvas_width, canvas_height = args.display_width, args.display_height
    report["pending_wind_value"] = wind_value_control
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, canvas_width, canvas_height)

    def image_point(x, y):
        base_scale, (base_ox, base_oy) = fit_to_canvas(image.shape[1], image.shape[0], canvas_width, canvas_height)
        scale = base_scale * zoom
        if view_origin == [0.0, 0.0] and zoom == 1.0:
            view_origin[0], view_origin[1] = base_ox, base_oy
        return ((x - view_origin[0]) / scale, (y - view_origin[1]) / scale)

    def mouse(event, x, y, _flags, _userdata):
        nonlocal selected, report, calibration_armed, visible_calibration_samples, zoom
        point = image_point(x, y)
        cursor[0], cursor[1] = point
        if event == cv2.EVENT_MOUSEWHEEL:
            old_scale = fit_to_canvas(image.shape[1], image.shape[0], canvas_width, canvas_height)[0] * zoom
            anchor = ((x - view_origin[0]) / old_scale, (y - view_origin[1]) / old_scale)
            direction = 1 if _flags > 0 else -1
            zoom = max(0.5, min(4.0, zoom * (1.15 if direction > 0 else 1 / 1.15)))
            new_scale = fit_to_canvas(image.shape[1], image.shape[0], canvas_width, canvas_height)[0] * zoom
            view_origin[0], view_origin[1] = x - anchor[0] * new_scale, y - anchor[1] * new_scale
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if manual_adjust and calibration_armed:
                sample = make_calibration_sample(report["solution"], report.get("predicted_points", ()), point)
                sample.update({"mode": mode, "wind_value": wind_value_control, "wind_direction": report.get("wind_direction", "right")})
                calibration_samples.append(sample)
                visible_calibration_samples.append(sample)
                calibration_armed = False
                report["calibration_samples"] = visible_calibration_samples
                report["calibration_hint"] = calibration_hint(calibration_samples)
                report["calibration_armed"] = False
                return
            hits = [(index, (item.center[0]-point[0])**2 + (item.center[1]-point[1])**2) for index, item in enumerate(scene.boxes)]
            selected = min(hits, key=lambda item: item[1])[0] if hits and min(hits, key=lambda item: item[1])[1] <= 50**2 else None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if calibration_armed:
                calibration_armed = False
                report["calibration_armed"] = False
                return
            calibration_index = nearest_calibration_index(visible_calibration_samples, point)
            if manual_adjust and calibration_index is not None:
                sample = visible_calibration_samples.pop(calibration_index)
                if sample in calibration_samples:
                    calibration_samples.remove(sample)
                report["calibration_samples"] = visible_calibration_samples
                report["calibration_hint"] = calibration_hint(calibration_samples)
                return
            hits = [(index, (item.center[0]-point[0])**2 + (item.center[1]-point[1])**2) for index, item in enumerate(scene.boxes)]
            if hits and min(hits, key=lambda item: item[1])[1] <= 50**2:
                index = min(hits, key=lambda item: item[1])[0]
                deleted_item = scene.boxes[index]
                scene.deleted.append({"name": deleted_item.name, "x": deleted_item.center[0], "y": deleted_item.center[1]})
                scene.boxes.pop(index)
                selected = None

    cv2.setMouseCallback(window, mouse)
    def recalculate():
        nonlocal report, overlay, paths
        report, overlay, _ignored, paths = replay_image(image_path, source=args.source, weights=args.weights, mode=mode, root=args.root, output_root=args.output_root, target=target, scene_override=scene, barrel_extension=args.barrel_extension, angle_override=angle_override, power_override=power_override, wind_override=applied_wind_value)
        report["manual_adjust"] = manual_adjust
        report["calibration_samples"] = visible_calibration_samples
        report["calibration_hint"] = calibration_hint(calibration_samples)
        report["pending_wind_value"] = wind_value_control

    def switch_image(step):
        nonlocal image, image_index, image_path, report, overlay, scene, paths, selected, target, angle_override, power_override, wind_value_control, applied_wind_value, zoom, view_origin, visible_calibration_samples, calibration_samples
        if len(image_paths) < 2:
            return
        image_index = (image_index + step) % len(image_paths)
        image_path = image_paths[image_index]
        image = cv2.imread(str(image_path))
        if image is None:
            return
        selected = None
        target = tuple(args.target) if args.target else None
        angle_override = power_override = None
        calibration_samples = load_calibration_samples(image_path, args.output_root)
        visible_calibration_samples = list(calibration_samples)
        applied_wind_value = None
        report, overlay, scene, paths = replay_image(image_path, source=args.source, weights=args.weights, mode=mode, root=args.root, output_root=args.output_root, target=target, barrel_extension=args.barrel_extension)
        wind_value_control = float(report.get("wind_value", 0.0) or 0.0) * (-1 if report.get("wind_direction") == "left" else 1)
        report["manual_adjust"] = manual_adjust
        report["calibration_samples"] = visible_calibration_samples
        report["calibration_hint"] = calibration_hint(calibration_samples)
        report["pending_wind_value"] = wind_value_control
        zoom = 1.0
        view_origin = [0.0, 0.0]
        cursor[0], cursor[1] = image.shape[1] / 2.0, image.shape[0] / 2.0

    while True:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        base_scale, _ = fit_to_canvas(image.shape[1], image.shape[0], canvas_width, canvas_height)
        scale = base_scale * zoom
        if view_origin == [0.0, 0.0] and zoom == 1.0:
            view_origin[0], view_origin[1] = fit_to_canvas(image.shape[1], image.shape[0], canvas_width, canvas_height)[1]
        ox, oy = round(view_origin[0]), round(view_origin[1])
        rendered = render_replay_visual(image, report.get("predicted_points", ()), guide_from_report(report), scene, report, show_annotations=show_annotations, show_guide=show_guide, line_width=1)
        size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        resized = cv2.resize(rendered, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        view = cv2.copyMakeBorder(resized, max(0, oy), max(0, canvas_height - oy - size[1]), max(0, ox), max(0, canvas_width - ox - size[0]), cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if oy < 0 or ox < 0 or oy + size[1] > canvas_height or ox + size[0] > canvas_width:
            canvas = view
            view = canvas[max(0, -oy):max(0, -oy) + canvas_height, max(0, -ox):max(0, -ox) + canvas_width]
            if view.shape[:2] != (canvas_height, canvas_width):
                padded = cv2.copyMakeBorder(view, 0, max(0, canvas_height - view.shape[0]), 0, max(0, canvas_width - view.shape[1]), cv2.BORDER_CONSTANT, value=(0, 0, 0))
                view = padded[:canvas_height, :canvas_width]
        if selected is not None:
            item = scene.boxes[selected]
            cv2.rectangle(view, (round(ox + item.x*scale-3), round(oy + item.y*scale-3)), (round(ox + (item.x+item.width)*scale+3), round(oy + (item.y+item.height)*scale+3)), (0, 255, 255), 3)
        cv2.imshow(window, view)
        raw = cv2.waitKeyEx(30)
        key = raw & 0xff
        if raw in (27,) or key in (ord("q"), ord("Q")): break
        if is_reset_view_key(raw, key):
            zoom = 1.0
            view_origin[0], view_origin[1] = 0.0, 0.0
            continue
        if key in (ord("e"), ord("E")):
            target = tuple(cursor)
            recalculate()
        elif key in (ord("<"), ord(",")):
            switch_image(-1)
        elif key in (ord(">"), ord(".")):
            switch_image(1)
        elif key in (9,):
            manual_adjust = not manual_adjust
            if manual_adjust:
                angle_override, power_override = manual_controls(scene, report["solution"])
            else:
                angle_override = power_override = None
            recalculate()
        elif manual_adjust and wind_adjustment(key) is not None:
            wind_value_control += wind_adjustment(key)
            report["pending_wind_value"] = wind_value_control
        elif manual_adjust and key in (13, 10):
            applied_wind_value = wind_value_control
            recalculate()
        elif manual_adjust and key == ord("1"):
            calibration_armed = True
            report["calibration_armed"] = True
        elif key == ord("1"):
            scene.self_center = tuple(cursor)
            recalculate()
        elif manual_adjust and key == ord("2"):
            # Reserved: calibration samples are recorded only; no auto-apply.
            pass
        elif manual_adjust and arrow_adjustment(raw) is not None:
            visible_calibration_samples = list(calibration_samples)
            field, delta = arrow_adjustment(raw)
            if field == "angle":
                angle_override = max(0.0, min(90.0, angle_override + delta))
            else:
                power_override = max(1, min(100, power_override + delta))
            recalculate()
        elif raw == 7602176:
            recalculate()
        elif key in (ord("r"), ord("R"), ord("h"), ord("H"), ord("t"), ord("T")):
            mode = select_replay_mode(chr(key).lower(), mode)
            recalculate()
        elif raw in (2162688, 2228224):
            mode = select_replay_mode("page up" if raw == 2162688 else "page down", mode)
            recalculate()
        elif key == ord("v"):
            show_annotations = not show_annotations
        elif key == ord("g"):
            show_guide = not show_guide
        elif key == ord("f"):
            full = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full > 0 else cv2.WINDOW_FULLSCREEN)
        elif key == ord("s"):
            save_manual_scene(scene, *paths)
            save_calibration_samples(image_path, args.output_root, calibration_samples)
        elif selected is not None and arrow_adjustment(raw) is not None:
            dx, dy = {81: (-1, 0), 82: (0, -1), 83: (1, 0), 84: (0, 1), 2424832: (-1, 0), 2490368: (0, -1), 2555904: (1, 0), 2621440: (0, 1)}[raw]
            item = scene.boxes[selected]
            shifted = tuple(replace(point, x=point.x + dx, y=point.y + dy) for point in item.keypoints)
            scene.boxes[selected] = replace(item, x=item.x + dx, y=item.y + dy, source="manual", keypoints=shifted)
            recalculate()
        elif 48 <= key <= 57 and key != ord("2"):
            from shellshock.datasets.yolo import CLASS_NAMES
            class_id = key - 48
            name = CLASS_NAMES[class_id]
            scene.boxes.append(AnnotationBox(name, cursor[0] - 15, cursor[1] - 15, 30, 30, 1.0, "manual"))
            selected = len(scene.boxes) - 1
    cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", choices=("hybrid", "yolo", "annotation"), default="hybrid")
    parser.add_argument("--mode", default="normal_low")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=(DATA_ROOT / 'annotate/replay'))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--target", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--barrel-extension", type=float, default=35.0)
    parser.add_argument("--display-width", type=int, default=1800)
    parser.add_argument("--display-height", type=int, default=1000)
    return parser


def should_launch_interactive(argv, explicit_interactive=False):
    """Open the viewer for a bare invocation, while keeping parameterized runs JSON-only."""
    return bool(explicit_interactive or not argv)


def main():
    args = build_parser().parse_args()
    args.image = latest_annotated_image(args.image)
    if should_launch_interactive(sys.argv[1:], explicit_interactive=args.interactive):
        interactive(args.image, args)
    else:
        target = tuple(args.target) if args.target else None
        report, _, _, _ = replay_image(args.image, source=args.source, weights=args.weights, mode=args.mode, root=args.root, output_root=args.output_root, target=target, barrel_extension=args.barrel_extension)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=list))


if __name__ == "__main__":
    main()

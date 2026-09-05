"""One-bounce projectile planning and integer replay for detected obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, isfinite, pi, sin, sqrt
from typing import Literal

from .ballistics import (
    GRAVITY_AT_REFERENCE,
    MAX_POWER,
    REFERENCE_WIDTH,
    SPEED_PER_POWER_AT_REFERENCE,
    WIND_ACCELERATION_PER_UNIT_AT_REFERENCE,
)
from .obstacle_geometry import CircleObstacle, LineObstacle, ObstacleGeometry


EVENT_TIME_EPSILON = 1e-4
CONTINUOUS_RESIDUAL = 2.0
CIRCLE_MIN_NORMAL_SPEED = 2.0
LINE_ENDPOINT_MARGIN_AT_REFERENCE = 12.0
TARGET_HIT_RADIUS_AT_REFERENCE = 24.0

Vector = tuple[float, float]


@dataclass(frozen=True)
class CollisionEvent:
    obstacle_index: int
    kind: Literal["circle", "line"]
    time: float
    point: Vector
    normal: Vector


def _add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1]


def _mul(a: Vector, scalar: float) -> Vector:
    return a[0] * scalar, a[1] * scalar


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _norm(a: Vector) -> float:
    return hypot(*a)


def _unit(a: Vector) -> Vector:
    size = _norm(a)
    if size == 0:
        raise ValueError("normal must not be zero")
    return a[0] / size, a[1] / size


def _screen_to_physical(point: tuple[int, int] | Vector) -> Vector:
    return float(point[0]), -float(point[1])


def _physical_to_screen(point: Vector) -> Vector:
    return point[0], -point[1]


def _position(start: Vector, velocity: Vector, acceleration: Vector, time: float) -> Vector:
    return _add(start, _add(_mul(velocity, time), _mul(acceleration, 0.5 * time * time)))


def _velocity(velocity: Vector, acceleration: Vector, time: float) -> Vector:
    return _add(velocity, _mul(acceleration, time))


def reflect_velocity(velocity: Vector, normal: Vector) -> Vector:
    """Return the mirror reflection across a unit surface normal."""
    n = _unit(normal)
    result = _sub(velocity, _mul(n, 2.0 * _dot(velocity, n)))
    return (round(result[0], 12), round(result[1], 12))


def _circle_events(
    start: Vector, velocity: Vector, acceleration: Vector, end_time: float,
    obstacle_index: int, circle: CircleObstacle,
) -> list[CollisionEvent]:
    center = _screen_to_physical(circle.center)
    radius = float(circle.radius)
    # A quartic intersection under acceleration is bracketed deterministically.
    steps = max(80, min(800, int(end_time * 160)))
    previous_time = 0.0
    previous_value = _norm(_sub(start, center)) ** 2 - radius ** 2
    events: list[CollisionEvent] = []
    for index in range(1, steps + 1):
        current_time = end_time * index / steps
        current_point = _position(start, velocity, acceleration, current_time)
        current_value = _norm(_sub(current_point, center)) ** 2 - radius ** 2
        if previous_value > 0 >= current_value:
            low, high = previous_time, current_time
            for _ in range(36):
                middle = (low + high) / 2
                value = _norm(_sub(_position(start, velocity, acceleration, middle), center)) ** 2 - radius ** 2
                if value > 0:
                    low = middle
                else:
                    high = middle
            time = high
            point = _position(start, velocity, acceleration, time)
            events.append(CollisionEvent(obstacle_index, "circle", time, point, _unit(_sub(point, center))))
            break
        previous_time, previous_value = current_time, current_value
    return events


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    if abs(a) <= 1e-12:
        return [] if abs(b) <= 1e-12 else [-c / b]
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []
    root = sqrt(max(0.0, discriminant))
    return [(-b - root) / (2 * a), (-b + root) / (2 * a)]


def _line_events(
    start: Vector, velocity: Vector, acceleration: Vector, end_time: float,
    obstacle_index: int, line: LineObstacle,
) -> list[CollisionEvent]:
    a = _screen_to_physical(line.start)
    b = _screen_to_physical(line.end)
    tangent = _sub(b, a)
    length_squared = _dot(tangent, tangent)
    if length_squared == 0:
        return []
    # cross(tangent, p(t)-a) is quadratic and zero on the infinite line.
    cross = lambda point: tangent[0] * point[1] - tangent[1] * point[0]
    roots = _quadratic_roots(
        0.5 * cross(acceleration),
        cross(_sub(velocity, (0.0, 0.0))),
        cross(_sub(start, a)),
    )
    events: list[CollisionEvent] = []
    for time in roots:
        if not EVENT_TIME_EPSILON < time <= end_time + EVENT_TIME_EPSILON:
            continue
        point = _position(start, velocity, acceleration, time)
        fraction = _dot(_sub(point, a), tangent) / length_squared
        if not 0 <= fraction <= 1:
            continue
        normal = _unit((-tangent[1], tangent[0]))
        if _dot(_velocity(velocity, acceleration, time), normal) > 0:
            normal = _mul(normal, -1.0)
        events.append(CollisionEvent(obstacle_index, "line", time, point, normal))
    return events


def first_collision(
    start: Vector, velocity: Vector, acceleration: Vector, end_time: float,
    geometry: ObstacleGeometry, ignored: tuple[str, int] | None = None,
) -> CollisionEvent | None:
    """Find the unique first contact within a finite trajectory interval."""
    events: list[CollisionEvent] = []
    for index, circle in enumerate(geometry.circles):
        if ignored != ("circle", index):
            events.extend(_circle_events(start, velocity, acceleration, end_time, index, circle))
    for index, line in enumerate(geometry.lines):
        if ignored != ("line", index):
            events.extend(_line_events(start, velocity, acceleration, end_time, index, line))
    if not events:
        return None
    events.sort(key=lambda event: event.time)
    if len(events) > 1 and abs(events[1].time - events[0].time) <= EVENT_TIME_EPSILON:
        return None
    return events[0]


def _acceleration(wind_value: float, wind_direction: str, image_width: float) -> Vector:
    scale = image_width / REFERENCE_WIDTH
    if wind_direction not in {"left", "right"}:
        raise ValueError("wind_direction must be 'left' or 'right'")
    wind = wind_value * WIND_ACCELERATION_PER_UNIT_AT_REFERENCE * scale
    return (wind if wind_direction == "right" else -wind, -GRAVITY_AT_REFERENCE * scale)


def _obstacle_point_normal(kind: str, obstacle: CircleObstacle | LineObstacle, parameter: float) -> tuple[Vector, Vector]:
    if kind == "circle":
        assert isinstance(obstacle, CircleObstacle)
        center = _screen_to_physical(obstacle.center)
        normal = cos(parameter), sin(parameter)
        return _add(center, _mul(normal, obstacle.radius)), normal
    assert isinstance(obstacle, LineObstacle)
    start, end = _screen_to_physical(obstacle.start), _screen_to_physical(obstacle.end)
    tangent = _sub(end, start)
    return _add(start, _mul(tangent, parameter)), _unit((-tangent[1], tangent[0]))


def _surface_samples(kind: str) -> list[float]:
    return [2 * pi * index / 72 for index in range(72)] if kind == "circle" else [index / 24 for index in range(1, 24)]


def _valid_contact(event: CollisionEvent, kind: str, obstacle: CircleObstacle | LineObstacle, scale: float) -> bool:
    hit_speed = _velocity((0.0, 0.0), (0.0, 0.0), 0.0)  # type anchor for physical-vector intent
    del hit_speed
    if kind == "circle":
        return True
    assert isinstance(obstacle, LineObstacle)
    start, end = _screen_to_physical(obstacle.start), _screen_to_physical(obstacle.end)
    return min(_norm(_sub(event.point, start)), _norm(_sub(event.point, end))) >= LINE_ENDPOINT_MARGIN_AT_REFERENCE * scale


def _candidate_for_surface(
    source: Vector, target: Vector, acceleration: Vector, kind: str,
    obstacle: CircleObstacle | LineObstacle, parameter: float, image_width: float,
) -> dict[str, object] | None:
    point, normal = _obstacle_point_normal(kind, obstacle, parameter)
    scale = image_width / REFERENCE_WIDTH
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE * scale
    # Deterministic coarse times; t2 is solved/refined by checking residual.
    best: tuple[float, float, float, Vector, Vector] | None = None
    for t1_index in range(2, 81):
        t1 = t1_index * 0.1
        v0 = _sub(_mul(_sub(point, source), 1 / t1), _mul(acceleration, 0.5 * t1))
        v1 = _add(_mul(_sub(point, source), 1 / t1), _mul(acceleration, 0.5 * t1))
        v2 = reflect_velocity(v1, normal)
        normal_speed = _dot(v1, normal)
        if (kind == "circle" and normal_speed >= -CIRCLE_MIN_NORMAL_SPEED) or (
            kind == "line" and abs(normal_speed) < CIRCLE_MIN_NORMAL_SPEED
        ):
            continue
        for t2_index in range(2, 81):
            t2 = t2_index * 0.1
            predicted = _add(point, _add(_mul(v2, t2), _mul(acceleration, 0.5 * t2 * t2)))
            error = _norm(_sub(predicted, target))
            if best is None or error < best[0]:
                best = error, t1, t2, v0, v2
    if best is None:
        return None
    residual, t1, t2, v0, v2 = best
    speed = _norm(v0)
    angle = degrees(atan2(v0[1], abs(v0[0])))
    if not -90 <= angle <= 90 or not 0 <= speed / speed_per_power <= MAX_POWER:
        return None
    return {
        "obstacle_kind": kind, "reflection_physical": point, "normal": normal,
        "t1": t1, "t2": t2, "v0": v0, "v2": v2,
        "continuous_error": residual,
        "theory": {"angle_degrees": angle, "power": speed / speed_per_power},
    }


def solve_single_reflection(
    self_position: tuple[int, int], target_position: tuple[int, int], wind_value: float,
    wind_direction: str, image_width: int, geometry: ObstacleGeometry, integer_radius: int = 2,
) -> dict[str, object]:
    """Find minimum-power valid circle/line one-bounce shot and integer-refine it."""
    if not geometry.circles and not geometry.lines:
        return {"status": "unreachable", "reason": "no-obstacle"}
    source, target = _screen_to_physical(self_position), _screen_to_physical(target_position)
    acceleration = _acceleration(wind_value, wind_direction, image_width)
    raw: list[dict[str, object]] = []
    for kind, obstacles in (("circle", geometry.circles), ("line", geometry.lines)):
        for index, obstacle in enumerate(obstacles):
            for parameter in _surface_samples(kind):
                candidate = _candidate_for_surface(source, target, acceleration, kind, obstacle, parameter, image_width)
                if candidate is None:
                    continue
                event = first_collision(source, candidate["v0"], acceleration, float(candidate["t1"]), geometry)
                if event is None or event.kind != kind or event.obstacle_index != index:
                    continue
                if kind == "circle" and abs(_dot(_velocity(candidate["v0"], acceleration, event.time), event.normal)) < CIRCLE_MIN_NORMAL_SPEED:
                    continue
                if not _valid_contact(event, kind, obstacle, image_width / REFERENCE_WIDTH):
                    continue
                post = first_collision(event.point, candidate["v2"], acceleration, float(candidate["t2"]), geometry, ignored=(kind, index))
                if post is not None:
                    continue
                candidate["obstacle"] = {"kind": kind, "index": index}
                raw.append(candidate)
    if not raw:
        return {"status": "unreachable", "reason": "no-valid-reflection"}
    # Exact continuous candidates retain the original minimum-power rule.
    # When the coarse numerical search cannot reach the strict residual, keep
    # the closest legal bounce instead of treating a visible obstacle as absent.
    exact = [item for item in raw if float(item["continuous_error"]) <= CONTINUOUS_RESIDUAL]
    theory = min(
        exact or raw,
        key=lambda item: (
            float(item["theory"]["power"]), float(item["theory"]["angle_degrees"])
        ) if exact else (
            float(item["continuous_error"]), float(item["theory"]["power"]), float(item["theory"]["angle_degrees"])
        ),
    )
    result = refine_reflection_integer_shot(
        theory, self_position, target_position, wind_value, wind_direction, image_width, geometry, integer_radius,
    )
    if result.get("status") == "reachable" and not exact:
        result["status"] = "closest"
    return result


def refine_reflection_integer_shot(
    theory: dict[str, object], self_position: tuple[int, int], target_position: tuple[int, int],
    wind_value: float, wind_direction: str, image_width: int, geometry: ObstacleGeometry, radius: int = 2,
) -> dict[str, object]:
    """Replay nearby integer controls and preserve exactly the planned one bounce."""
    source, target = _screen_to_physical(self_position), _screen_to_physical(target_position)
    acceleration = _acceleration(wind_value, wind_direction, image_width)
    scale = image_width / REFERENCE_WIDTH
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE * scale
    planned = theory["obstacle"]
    assert isinstance(planned, dict)
    allowed = (str(planned["kind"]), int(planned["index"]))
    theory_values = theory["theory"]
    assert isinstance(theory_values, dict)
    candidates: list[dict[str, object]] = []
    for angle in range(max(-90, round(float(theory_values["angle_degrees"])) - radius), min(90, round(float(theory_values["angle_degrees"])) + radius) + 1):
        for power in range(max(0, round(float(theory_values["power"])) - radius), min(100, round(float(theory_values["power"])) + radius) + 1):
            speed = speed_per_power * power
            direction = 1.0 if float(theory["v0"][0]) >= 0 else -1.0
            velocity = (direction * speed * cos(angle * pi / 180), speed * sin(angle * pi / 180))
            first = first_collision(source, velocity, acceleration, 16.0, geometry)
            if first is None or (first.kind, first.obstacle_index) != allowed:
                continue
            if first.kind == "circle" and abs(_dot(_velocity(velocity, acceleration, first.time), first.normal)) < CIRCLE_MIN_NORMAL_SPEED:
                continue
            reflected = reflect_velocity(_velocity(velocity, acceleration, first.time), first.normal)
            # Search post-bounce time closest to target; any second event rejects it.
            best_time, best_error = 0.0, float("inf")
            for step in range(1, 1601):
                time = step * 0.01
                point = _position(first.point, reflected, acceleration, time)
                error = _norm(_sub(point, target))
                if error < best_error:
                    best_time, best_error = time, error
            second = first_collision(first.point, reflected, acceleration, best_time, geometry, ignored=allowed)
            if second is not None:
                continue
            final_screen = _physical_to_screen(_position(first.point, reflected, acceleration, best_time))
            candidates.append({
                "status": "reachable", "mode": "reflection", "angle_degrees": int(angle), "power": int(power),
                "direction": "right" if direction > 0 else "left", "obstacle": planned,
                "reflection_point": _physical_to_screen(first.point),
                "theory": theory_values, "target_error": best_error,
                "horizontal_error": final_screen[0] - float(target_position[0]),
                "final_position": final_screen,
            })
    if not candidates:
        return {"status": "unreachable", "reason": "no-valid-integer-reflection"}
    return min(candidates, key=lambda item: (float(item["target_error"]), int(item["power"]), int(item["angle_degrees"])))

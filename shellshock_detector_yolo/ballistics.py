"""Pure, scale-aware projectile calculations for normal ShellShock shots."""

from __future__ import annotations

from math import atan2, copysign, cos, degrees, radians, sin, sqrt


REFERENCE_WIDTH = 1920.0
GRAVITY_AT_REFERENCE = 379.106
SPEED_PER_POWER_AT_REFERENCE = 9.836246
WIND_ACCELERATION_PER_UNIT_AT_REFERENCE = 0.510
MAX_POWER = 100.0


def _scale(image_width: int | float) -> float:
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    return image_width / REFERENCE_WIDTH


def _wind_acceleration(
    value: int | float, direction: str, image_width: int | float
) -> float:
    if direction not in {"left", "right"}:
        raise ValueError("wind_direction must be 'left' or 'right'")
    sign = 1.0 if direction == "right" else -1.0
    return sign * value * WIND_ACCELERATION_PER_UNIT_AT_REFERENCE * _scale(image_width)


def _solution(
    direction: str,
    distance: float,
    height: float,
    wind: float,
    gravity: float,
    time: float,
) -> dict[str, float | str]:
    """Restore an initial velocity vector from a known flight time."""
    velocity_x = (distance - 0.5 * wind * time * time) / time
    velocity_y = (height + 0.5 * gravity * time * time) / time
    if velocity_x <= 0:
        raise ValueError("solution requires a non-positive local horizontal velocity")
    return {
        "direction": direction,
        "angle_degrees": round(degrees(atan2(velocity_y, velocity_x)), 4),
        "flight_time_seconds": round(time, 4),
    }


def _fixed_power_solutions(
    distance: float,
    height: float,
    wind: float,
    gravity: float,
    speed: float,
    direction: str,
) -> list[dict[str, float | str]]:
    """Return valid fixed-speed arcs, ordered low-to-high by flight time."""
    quadratic_a = (wind * wind + gravity * gravity) / 4.0
    quadratic_b = gravity * height - wind * distance - speed * speed
    quadratic_c = distance * distance + height * height
    discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
    discriminant_tolerance = 1e-12 * max(
        1.0,
        abs(quadratic_b * quadratic_b),
        abs(4.0 * quadratic_a * quadratic_c),
    )
    if discriminant < -discriminant_tolerance:
        return []
    if abs(discriminant) <= discriminant_tolerance:
        discriminant = 0.0

    root = sqrt(discriminant)
    # q-method avoids cancellation when one root is much smaller than the
    # other.  A zero discriminant is a tangent arc, not two trajectories.
    if root == 0.0:
        time_squares = (-quadratic_b / (2.0 * quadratic_a),)
    else:
        q = -0.5 * (quadratic_b + copysign(root, quadratic_b))
        time_squares = (q / quadratic_a, quadratic_c / q)
    solutions: list[dict[str, float | str]] = []
    for time_squared in time_squares:
        if time_squared <= 0:
            continue
        try:
            solutions.append(
                _solution(direction, distance, height, wind, gravity, sqrt(time_squared))
            )
        except ValueError:
            continue
    return sorted(solutions, key=lambda solution: float(solution["flight_time_seconds"]))


def solve_target(
    self_x: int | float,
    self_y: int | float,
    target_x: int | float,
    target_y: int | float,
    wind_value: int | float,
    wind_direction: str,
    image_width: int | float,
) -> dict[str, object]:
    """Calculate normal-shell solutions for one screen-space target.

    Screen coordinates are converted to a local ballistic frame: horizontal
    distance is always positive in the firing direction and positive height is
    above the firing tank. Angles use the UI-ready -90--90 degree range.
    """
    horizontal = target_x - self_x
    direction = "right" if horizontal >= 0 else "left"
    direction_sign = 1.0 if direction == "right" else -1.0
    distance = abs(float(horizontal))
    height = float(self_y - target_y)
    scale = _scale(image_width)
    gravity = GRAVITY_AT_REFERENCE * scale
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE * scale
    local_wind = _wind_acceleration(wind_value, wind_direction, image_width) * direction_sign

    fixed_solutions = _fixed_power_solutions(
        distance,
        height,
        local_wind,
        gravity,
        speed_per_power * MAX_POWER,
        direction,
    )

    range_to_target = sqrt(distance * distance + height * height)
    force_magnitude = sqrt(local_wind * local_wind + gravity * gravity)
    if range_to_target == 0:
        minimum_solution: dict[str, object] = {
            "status": "reachable",
            "direction": direction,
            "angle_degrees": 0.0,
            "flight_time_seconds": 0.0,
            "power": 0.0,
            "within_power_limit": True,
        }
    else:
        optimal_time = sqrt(2.0 * range_to_target / force_magnitude)
        minimum_speed_squared = (
            gravity * height
            - local_wind * distance
            + range_to_target * force_magnitude
        )
        try:
            minimum_solution = _solution(
                direction, distance, height, local_wind, gravity, optimal_time
            )
        except ValueError:
            # The unconstrained analytic extremum can require firing away from
            # the target.  It is not usable by the 0--90 degree UI, so expose
            # an explicit JSON-safe unavailable result rather than raising.
            minimum_solution = {
                "status": "unreachable",
                "direction": direction,
                "angle_degrees": None,
                "flight_time_seconds": None,
                "power": None,
                "within_power_limit": False,
            }
        else:
            raw_power = sqrt(max(0.0, minimum_speed_squared)) / speed_per_power
            minimum_solution["status"] = "reachable"
            minimum_solution["power"] = round(raw_power, 4)
            minimum_solution["within_power_limit"] = raw_power <= MAX_POWER

    return {
        "target": {"x": target_x, "y": target_y},
        "dx": horizontal,
        "dy": height,
        "target_direction": direction,
        "wind_acceleration": round(local_wind, 4),
        "power_100": {
            "power": MAX_POWER,
            "status": "reachable" if fixed_solutions else "unreachable",
            "solutions": fixed_solutions,
        },
        "minimum_power": minimum_solution,
    }


def predicted_horizontal_displacement(
    power: float,
    angle_degrees: float,
    wind_value: int | float,
    wind_direction: str,
    firing_direction: str,
    image_width: int | float,
) -> float:
    """Predict level-ground horizontal displacement in screen-x coordinates."""
    if firing_direction not in {"left", "right"}:
        raise ValueError("firing_direction must be 'left' or 'right'")
    scale = _scale(image_width)
    gravity = GRAVITY_AT_REFERENCE * scale
    speed = SPEED_PER_POWER_AT_REFERENCE * scale * power
    angle = radians(angle_degrees)
    flight_time = 2.0 * speed * sin(angle) / gravity
    firing_sign = 1.0 if firing_direction == "right" else -1.0
    wind = _wind_acceleration(wind_value, wind_direction, image_width)
    return firing_sign * speed * cos(angle) * flight_time + 0.5 * wind * flight_time**2


def _integer_candidates(value: float, radius: int, minimum: int, maximum: int) -> range:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    center = round(value)
    return range(max(minimum, center - radius), min(maximum, center + radius) + 1)


def _position_at_time(
    velocity_x: float, velocity_y: float, acceleration_x: float, gravity: float, time: float,
) -> tuple[float, float]:
    """Return physical-frame displacement, where positive y is upwards."""
    return (
        velocity_x * time + 0.5 * acceleration_x * time * time,
        velocity_y * time - 0.5 * gravity * time * time,
    )


def _target_height_times(velocity_y: float, gravity: float, target_height: float) -> list[float]:
    """Return strictly-positive times at which a trajectory reaches target_height."""
    discriminant = velocity_y * velocity_y - 2.0 * gravity * target_height
    if discriminant < 0:
        return []
    root = sqrt(discriminant)
    return [time for time in ((velocity_y - root) / gravity, (velocity_y + root) / gravity) if time > 0]


def refine_normal_integer_shot(
    self_x: int | float,
    self_y: int | float,
    target_x: int | float,
    target_y: int | float,
    wind_value: int | float,
    wind_direction: str,
    image_width: int | float,
    theory_angle: float,
    theory_power: float,
    radius: int = 2,
) -> dict[str, object]:
    """Choose the nearest playable integer normal shot by re-simulating it.

    The analytic normal solver remains the source of the continuous candidate;
    this function only searches a small integer neighbourhood around it.
    """
    if not -90 <= theory_angle <= 90 or theory_power < 0:
        raise ValueError("theory controls must be within the aim disc")
    horizontal = float(target_x - self_x)
    direction = "right" if horizontal >= 0 else "left"
    direction_sign = 1.0 if direction == "right" else -1.0
    target_height = float(self_y - target_y)
    scale = _scale(image_width)
    gravity = GRAVITY_AT_REFERENCE * scale
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE * scale
    wind = _wind_acceleration(wind_value, wind_direction, image_width)
    candidates: list[dict[str, object]] = []
    for angle_degrees in _integer_candidates(theory_angle, radius, -90, 90):
        angle = radians(angle_degrees)
        for power in _integer_candidates(theory_power, radius, 0, int(MAX_POWER)):
            speed = speed_per_power * power
            velocity_x = direction_sign * speed * cos(angle)
            velocity_y = speed * sin(angle)
            for time in _target_height_times(velocity_y, gravity, target_height):
                dx, _ = _position_at_time(velocity_x, velocity_y, wind, gravity, time)
                error = abs(dx - horizontal)
                candidates.append({
                    "angle_degrees": angle_degrees,
                    "power": power,
                    "direction": direction,
                    "flight_time_seconds": time,
                    "final_position": {"x": float(self_x) + dx, "y": float(target_y)},
                    "target_error": error,
                    "horizontal_error": float(self_x) + dx - float(target_x),
                })
    if not candidates:
        raise RuntimeError("integer refinement found no playable normal trajectory")
    return min(candidates, key=lambda item: (
        float(item["target_error"]), int(item["power"]), int(item["angle_degrees"])
    ))


def format_ballistics(results: list[dict[str, object]]) -> str:
    """Format JSON-ready target solutions for the capture hotkey output."""
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        power_100 = result["power_100"]
        minimum = result["minimum_power"]
        lines.append(
            f"Target {index}: dx={result['dx']}, dy={result['dy']}, "
            f"direction={result['target_direction']}"
        )
        if power_100["status"] == "unreachable":
            lines.append("  100 power: unreachable")
        else:
            for label, solution in zip(("low arc", "high arc"), power_100["solutions"]):
                lines.append(
                    f"  100 power {label}: {solution['direction']} "
                    f"{solution['angle_degrees']:.4f} degrees, "
                    f"{solution['flight_time_seconds']:.4f}s"
                )
        if minimum.get("status") == "unreachable":
            lines.append("  minimum power: unreachable")
        else:
            limit = "within limit" if minimum["within_power_limit"] else "over 100 limit"
            lines.append(
                f"  minimum power: {minimum['power']:.4f}, {minimum['direction']} "
                f"{minimum['angle_degrees']:.4f} degrees ({limit})"
            )
    return "\n".join(lines)

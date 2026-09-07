"""Integer-shot search validated by global projectile event replay."""
from __future__ import annotations

from math import cos, radians, sin

from .ballistics import GRAVITY_AT_REFERENCE, REFERENCE_WIDTH, SPEED_PER_POWER_AT_REFERENCE, WIND_ACCELERATION_PER_UNIT_AT_REFERENCE, solve_target
from .projectile_events import advance_through_world
from .world_geometry import Point, World
from .wormhole_solver import solve_wormhole_integer_shot


def solve_integer_shot(source: Point, target: Point, world: World, wind_value: float, wind_direction: str, image_width: int, mode: str, force_power: int | None = None) -> dict[str, object]:
    if mode in {"wormhole", "wormhole_low_arc", "wormhole_high_arc"} and not world.portal_pairs:
        return {"status": "unreachable", "reason": "no-portal-pair"}
    if mode in {"wormhole", "wormhole_low_arc", "wormhole_high_arc"}:
        return solve_wormhole_integer_shot(source, target, world, wind_value, wind_direction, image_width,
                                           arc_preference="high" if mode == "wormhole_high_arc" else "low")
    scale = image_width / REFERENCE_WIDTH
    wind = wind_value * WIND_ACCELERATION_PER_UNIT_AT_REFERENCE * scale * (1 if wind_direction == "right" else -1)
    if mode in {"normal", "low_arc", "high_arc"}:
        theory = solve_target(*source, *target, wind_value, wind_direction, image_width)["minimum_power"]
        if theory.get("status") != "reachable":
            return {"status": "unreachable", "reason": "no-verified-shot"}
        directions = (str(theory["direction"]),)
        if mode == "high_arc":
            fixed = solve_target(*source, *target, wind_value, wind_direction, image_width)["power_100"]["solutions"]
            if not fixed:
                return {"status": "unreachable", "reason": "no-verified-shot"}
            high = max(fixed, key=lambda item: float(item["angle_degrees"]))
            angles = range(max(0, round(float(high["angle_degrees"])) - 3), min(90, round(float(high["angle_degrees"])) + 3) + 1)
            powers = (100,)
        else:
            angles = range(max(0, round(float(theory["angle_degrees"])) - 3), min(90, round(float(theory["angle_degrees"])) + 3) + 1)
            powers = (force_power,) if force_power is not None else range(max(1, round(float(theory["power"])) - 3), min(100, round(float(theory["power"])) + 3) + 1)
    else:
        directions = ("left", "right")
        angles = range(0, 91, 5)
        powers = (force_power,) if force_power is not None else range(5, 101, 5)
    legal: list[dict[str, object]] = []
    for direction in directions:
        sign = -1 if direction == "left" else 1
        for angle_degrees in angles:
            angle = radians(angle_degrees)
            for power in powers:
                speed = power * SPEED_PER_POWER_AT_REFERENCE * scale
                replay = advance_through_world(source, (sign * speed * cos(angle), -speed * sin(angle)), (wind, GRAVITY_AT_REFERENCE * scale), world, 12.0, target, 24 * scale)
                if replay.terminal_kind != "target":
                    continue
                if mode == "wormhole" and replay.portal_count < 1:
                    continue
                if mode == "reflection":
                    continue
                legal.append({"status": "reachable", "direction": direction, "angle_degrees": angle_degrees, "power": power, "portal_count": replay.portal_count, "events": [event.kind for event in replay.events]})
    if not legal:
        return {"status": "unreachable", "reason": "wormhole-required" if mode == "wormhole" else "no-verified-shot"}
    return min(legal, key=lambda item: (int(item["power"]), int(item["angle_degrees"]), str(item["direction"])))

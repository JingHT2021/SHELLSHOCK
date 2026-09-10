"""Screen-space sampling for solver results used by replay overlays."""

from __future__ import annotations

from math import cos, hypot, radians, sin

from .ballistics import GRAVITY_AT_REFERENCE, SPEED_PER_POWER_AT_REFERENCE, _scale, _wind_acceleration
from .collision import find_first_collision, trajectory_position, trajectory_velocity
from .wormhole_solver import WormholeTrajectory, first_circle_entry_time
from .solver_config import portal_trigger_radius


def _manual_reflected_trajectory(source, velocity, acceleration, duration, world, count):
    """Sample a manually controlled shot while applying the first obstacle bounces."""
    point = tuple(float(value) for value in source)
    current_velocity = tuple(float(value) for value in velocity)
    remaining = float(duration)
    points = [point]
    segment_count = max(2, count // 2)
    for _ in range(2):
        collision = find_first_collision(point, current_velocity, acceleration, world, remaining, avoid_portals=False, image_width=world.image_width)
        if collision is None:
            points.extend(tuple(float(value) for value in trajectory_position(point, current_velocity, acceleration, remaining * i / (segment_count - 1))) for i in range(1, segment_count))
            break
        before_count = max(2, round(segment_count * collision.time / max(duration, 1e-9)))
        points.extend(tuple(float(value) for value in trajectory_position(point, current_velocity, acceleration, collision.time * i / (before_count - 1))) for i in range(1, before_count))
        point = tuple(float(value) for value in collision.point)
        if collision.obstacle_kind == "line":
            line = world.lines[collision.obstacle_index]
            dx, dy = line.end[0] - line.start[0], line.end[1] - line.start[1]
            length = hypot(dx, dy)
            normal = (-dy / length, dx / length) if length else (0.0, 0.0)
        else:
            circle = world.circles[collision.obstacle_index]
            dx, dy = point[0] - circle.center[0], point[1] - circle.center[1]
            length = hypot(dx, dy)
            normal = (dx / length, dy / length) if length else (0.0, -1.0)
        incoming = trajectory_velocity(current_velocity, acceleration, collision.time)
        dot = incoming[0] * normal[0] + incoming[1] * normal[1]
        current_velocity = (incoming[0] - 2.0 * dot * normal[0], incoming[1] - 2.0 * dot * normal[1])
        remaining -= collision.time
        if remaining <= 1e-6:
            break
    return tuple(points)


def sample_solution_trajectory(solution, source, image_width, wind_value=0, wind_direction="right", samples=120, world=None):
    if solution.get("status") != "reachable":
        return ()
    if solution.get("trajectory_points"):
        return tuple((float(x), float(y)) for x, y in solution["trajectory_points"])
    duration = float(solution.get("flight_time_seconds", 0.0))
    if duration <= 0:
        return ()
    scale = _scale(image_width)
    direction = str(solution.get("direction", wind_direction))
    power = float(solution.get("power", 0.0))
    angle = radians(float(solution.get("angle_degrees", 0.0)))
    speed = power * SPEED_PER_POWER_AT_REFERENCE * scale
    velocity = ((1 if direction == "right" else -1) * speed * cos(angle), -speed * sin(angle))
    acceleration = (_wind_acceleration(float(wind_value), wind_direction, image_width), GRAVITY_AT_REFERENCE * scale)
    count = max(2, int(samples))
    if solution.get("manual_preview") and world is not None:
        return _manual_reflected_trajectory(source, velocity, acceleration, duration, world, count)
    portal_sequence = tuple(solution.get("portal_sequence", ()))
    if world is not None and portal_sequence and not solution.get("reflection_point"):
        portals = {}
        for index, pair in enumerate(world.portal_pairs):
            portals[f"{index}:orange"] = (pair.orange, pair.blue)
            portals[f"{index}:blue"] = (pair.blue, pair.orange)
        point = tuple(source)
        current_velocity = tuple(velocity)
        elapsed = 0.0
        remaining = duration
        route_points = [tuple(float(value) for value in point)]
        for portal_id in portal_sequence:
            if portal_id not in portals:
                break
            entry, exit_portal = portals[portal_id]
            trajectory = WormholeTrajectory(point, current_velocity, acceleration, remaining)
            entry_time = first_circle_entry_time(trajectory, entry.center, portal_trigger_radius(entry), 0.0, remaining)
            if entry_time is None:
                break
            segment_count = max(3, round(count / (len(portal_sequence) + 1)))
            route_points.extend(tuple(float(value) for value in trajectory_position(point, current_velocity, acceleration, entry_time * i / (segment_count - 1))) for i in range(1, segment_count))
            contact = trajectory_position(point, current_velocity, acceleration, entry_time)
            current_velocity = tuple(float(value) for value in (current_velocity[0] + acceleration[0] * entry_time, current_velocity[1] + acceleration[1] * entry_time))
            point = tuple(float(exit_portal.center[i] + contact[i] - entry.center[i]) for i in (0, 1))
            route_points.append(point)
            elapsed += entry_time
            remaining = max(0.0, duration - elapsed)
        if route_points[-1] != point:
            route_points.append(point)
        tail_count = max(2, count - len(route_points) + 1)
        route_points.extend(tuple(float(value) for value in trajectory_position(point, current_velocity, acceleration, remaining * i / (tail_count - 1))) for i in range(1, tail_count))
        return tuple(route_points)
    reflection = solution.get("reflection_point")
    outgoing = solution.get("actual_v_after")
    if reflection is not None and outgoing is not None:
        probe_count = max(20, count // 2)
        probe = [duration * i / (probe_count - 1) for i in range(probe_count)]
        collision_time = min(probe, key=lambda t: (trajectory_position(source, velocity, acceleration, t)[0] - reflection[0]) ** 2 + (trajectory_position(source, velocity, acceleration, t)[1] - reflection[1]) ** 2)
        before = [tuple(float(value) for value in trajectory_position(source, velocity, acceleration, t)) for t in probe if t < collision_time]
        after_count = max(2, count - len(before))
        after = [tuple(float(value) for value in trajectory_position(reflection, outgoing, acceleration, (duration-collision_time) * i / (after_count - 1))) for i in range(1, after_count)]
        return tuple(before + [tuple(float(value) for value in reflection)] + after)
    return tuple(tuple(float(value) for value in trajectory_position(source, velocity, acceleration, duration * i / (count - 1))) for i in range(count))

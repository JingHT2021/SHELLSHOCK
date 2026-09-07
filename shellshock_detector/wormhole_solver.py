"""Analytic candidate generation for shots that must traverse paired portals."""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, ceil, cos, degrees, hypot, pi, radians, sin, sqrt
from itertools import product
from typing import Iterable

from .ballistics import GRAVITY_AT_REFERENCE, REFERENCE_WIDTH, SPEED_PER_POWER_AT_REFERENCE, WIND_ACCELERATION_PER_UNIT_AT_REFERENCE
from .projectile_events import advance_through_world
from .world_geometry import Point, Portal, PortalPair, World

EPSILON = 1e-8


@dataclass(frozen=True)
class WormholeTrajectory:
    source: Point
    velocity: Point
    acceleration: Point
    flight_time: float = 0.0

    @property
    def time(self) -> float:
        return self.flight_time

    @property
    def angle_degrees(self) -> float:
        return degrees(atan2(-self.velocity[1], abs(self.velocity[0])))

    def position(self, time: float) -> Point:
        return (self.source[0] + self.velocity[0] * time + self.acceleration[0] * time * time / 2,
                self.source[1] + self.velocity[1] * time + self.acceleration[1] * time * time / 2)


@dataclass(frozen=True)
class PortalEvent:
    portal_id: str
    time: float


@dataclass(frozen=True)
class _Entry:
    portal_id: str
    entry: Portal
    exit: Portal

    @property
    def shift(self) -> Point:
        return self.exit.center[0] - self.entry.center[0], self.exit.center[1] - self.entry.center[1]


def minimum_ballistic_speed(source: Point, target: Point, acceleration: Point) -> float:
    dx, dy = target[0] - source[0], target[1] - source[1]
    magnitude = hypot(dx, dy)
    force = hypot(*acceleration)
    return sqrt(max(0.0, force * magnitude - (dx * acceleration[0] + dy * acceleration[1])))


def solve_ballistic_for_speed(source: Point, target: Point, acceleration: Point, speed: float) -> tuple[WormholeTrajectory, ...]:
    """Solve a constant-acceleration shot at one speed, low flight time first."""
    dx, dy = target[0] - source[0], target[1] - source[1]
    ax, ay = acceleration
    a = (ax * ax + ay * ay) / 4
    b = -(dx * ax + dy * ay + speed * speed)
    c = dx * dx + dy * dy
    if a <= EPSILON:
        return ()
    discriminant = b * b - 4 * a * c
    if discriminant < -EPSILON:
        return ()
    root = sqrt(max(0.0, discriminant))
    z_values = ((-b - root) / (2 * a), (-b + root) / (2 * a))
    trajectories: list[WormholeTrajectory] = []
    for z in z_values:
        if z <= EPSILON:
            continue
        time = sqrt(z)
        velocity = ((dx - ax * z / 2) / time, (dy - ay * z / 2) / time)
        trajectory = WormholeTrajectory(source, velocity, acceleration, time)
        if not any(abs(time - other.flight_time) < EPSILON for other in trajectories):
            trajectories.append(trajectory)
    return tuple(sorted(trajectories, key=lambda item: item.flight_time))
def _cubic_real_roots(a: float, b: float, c: float, d: float) -> tuple[float, ...]:
    """Return all real roots using the depressed-cubic form."""
    if abs(a) <= EPSILON:
        if abs(b) <= EPSILON:
            return () if abs(c) <= EPSILON else (-d / c,)
        disc = c * c - 4 * b * d
        if disc < -EPSILON:
            return ()
        root = sqrt(max(0.0, disc))
        return tuple(sorted({(-c - root) / (2 * b), (-c + root) / (2 * b)}))
    p = (3 * a * c - b * b) / (3 * a * a)
    q = (27 * a * a * d - 9 * a * b * c + 2 * b ** 3) / (27 * a ** 3)
    disc = (q / 2) ** 2 + (p / 3) ** 3
    offset = -b / (3 * a)
    if disc > EPSILON:
        return (offset + _cuberoot(-q / 2 + sqrt(disc)) + _cuberoot(-q / 2 - sqrt(disc)),)
    if abs(disc) <= EPSILON:
        u = _cuberoot(-q / 2)
        return tuple(sorted({offset + 2 * u, offset - u}))
    phi = acos(max(-1.0, min(1.0, -q / 2 / sqrt(-(p / 3) ** 3))))
    radius = 2 * sqrt(-p / 3)
    return tuple(sorted(radius * cos((phi + 2 * pi * k) / 3) + offset for k in range(3)))


def _cuberoot(value: float) -> float:
    return value ** (1 / 3) if value >= 0 else -(-value) ** (1 / 3)


def first_circle_entry_time(trajectory: WormholeTrajectory, center: Point, radius: float, start: float, end: float) -> float | None:
    """Find the earliest outside-to-inside contact via cubic stationary points."""
    if end <= start:
        return None
    sx, sy = trajectory.source[0] - center[0], trajectory.source[1] - center[1]
    vx, vy = trajectory.velocity
    ax, ay = trajectory.acceleration
    c4 = (ax * ax + ay * ay) / 4
    c3 = vx * ax + vy * ay
    c2 = vx * vx + vy * vy + sx * ax + sy * ay
    c1 = 2 * (sx * vx + sy * vy)
    c0 = sx * sx + sy * sy - radius * radius
    def value(t: float) -> float:
        return ((((c4 * t + c3) * t + c2) * t + c1) * t + c0)
    if value(start) <= 0:
        return None
    stationary = [root for root in _cubic_real_roots(4 * c4, 3 * c3, 2 * c2, c1) if start < root < end]
    points = [start, *sorted(stationary), end]
    for low, high in zip(points, points[1:]):
        if value(low) > 0 and value(high) <= 0:
            for _ in range(48):
                middle = (low + high) / 2
                if value(middle) > 0:
                    low = middle
                else:
                    high = middle
            return high
    return None


def _entries(pairs: Iterable[PortalPair]) -> tuple[_Entry, ...]:
    entries: list[_Entry] = []
    for index, pair in enumerate(pairs):
        entries += [_Entry(f"{index}:orange", pair.orange, pair.blue), _Entry(f"{index}:blue", pair.blue, pair.orange)]
    return tuple(entries)


def _portal_sequence(world: World, replay: object) -> tuple[str, ...]:
    sequence: list[str] = []
    for event in replay.events:  # type: ignore[attr-defined]
        if event.kind != "portal":
            continue
        choices = [(hypot(event.point[0] - portal.center[0], event.point[1] - portal.center[1]), f"{index}:{color}")
                   for index, pair in enumerate(world.portal_pairs) for color, portal in (("orange", pair.orange), ("blue", pair.blue))]
        sequence.append(min(choices)[1])
    return tuple(sequence)


def solve_wormhole_integer_shot(source: Point, target: Point, world: World, wind_value: float, wind_direction: str, image_width: int) -> dict[str, object]:
    if not world.portal_pairs:
        return {"status": "unreachable", "reason": "no-portal-pair"}
    scale = image_width / REFERENCE_WIDTH
    acceleration = (wind_value * WIND_ACCELERATION_PER_UNIT_AT_REFERENCE * scale * (1 if wind_direction == "right" else -1), GRAVITY_AT_REFERENCE * scale)
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE * scale
    entries = _entries(world.portal_pairs)
    best: dict[str, object] | None = None
    cache: dict[tuple[float, float, int], tuple[WormholeTrajectory, ...]] = {}
    for hops in (1, 2):
        for planned in product(entries, repeat=hops):
            shift = (sum(item.shift[0] for item in planned), sum(item.shift[1] for item in planned))
            virtual = (target[0] - shift[0], target[1] - shift[1])
            minimum = ceil(minimum_ballistic_speed(source, virtual, acceleration) / speed_per_power)
            # Integer controls only need a tight neighbourhood above the
            # continuous lower bound; larger powers cannot be lowest-power
            # candidates and make four-pair maps unnecessarily expensive.
            ceiling = min(100, minimum + 1, int(best["power"]) if best else 100)
            for power in range(max(1, minimum), ceiling + 1):
                trajectories = cache.setdefault((shift[0], shift[1], power), solve_ballistic_for_speed(source, virtual, acceleration, power * speed_per_power))  # type: ignore[arg-type]
                for trajectory in trajectories:
                    first_entry = first_circle_entry_time(trajectory, planned[0].entry.center, planned[0].entry.radius, 0.0, trajectory.flight_time)
                    if first_entry is None:
                        continue
                    raw_angle = degrees(atan2(-trajectory.velocity[1], abs(trajectory.velocity[0])))
                    if not 0 <= raw_angle <= 90:
                        continue
                    direction = "right" if trajectory.velocity[0] >= 0 else "left"
                    for angle in sorted({max(0, min(90, round(raw_angle) + delta)) for delta in (-1, 0, 1)}):
                        velocity = (power * speed_per_power * (1 if direction == "right" else -1) * cos(radians(angle)), -power * speed_per_power * sin(radians(angle)))
                        replay = advance_through_world(source, velocity, acceleration, world, 12.0, target, 24 * scale)
                        sequence = _portal_sequence(world, replay)
                        wanted = tuple(item.portal_id for item in planned)
                        if replay.terminal_kind != "target" or sequence != wanted:
                            continue
                        candidate = {"status": "reachable", "direction": direction, "angle_degrees": angle, "power": power, "portal_count": len(sequence), "portal_sequence": list(sequence), "events": [event.kind for event in replay.events]}
                        if best is None or (power, len(sequence), angle, direction) < (int(best["power"]), int(best["portal_count"]), int(best["angle_degrees"]), str(best["direction"])):
                            best = candidate
                if best and power >= int(best["power"]):
                    break
    return best or {"status": "unreachable", "reason": "no-verified-wormhole-shot"}

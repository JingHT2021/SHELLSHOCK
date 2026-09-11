"""Pure constant-acceleration shots at a given speed."""
from dataclasses import dataclass
from math import atan2,degrees,hypot,sqrt
from shellshock.math2d.geometry import trajectory_position
Point=tuple[float,float]
EPSILON=1e-8

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
        return trajectory_position(self.source,self.velocity,self.acceleration,time)


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

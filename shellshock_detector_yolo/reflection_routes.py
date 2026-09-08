"""Bounded portal event orders and translation unfolding for one reflection."""
from dataclasses import dataclass
from itertools import product

from .collision import trajectory_position, trajectory_velocity, closest_approach_to_target
from .portal_replay import _entry_time
from .solver_config import portal_trigger_radius


@dataclass(frozen=True)
class ReflectionRoute:
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()

    @property
    def has_portals(self):
        return bool(self.before or self.after)


def portal_entries(world):
    return {f'{i}:{color}': (p, partner, f'{i}:{other}')
            for i, pair in enumerate(world.portal_pairs)
            for color, p, other, partner in (('orange', pair.orange, 'blue', pair.blue),
                                            ('blue', pair.blue, 'orange', pair.orange))}


def reflection_routes(world):
    """Empty route plus every split of one/two planned transmissions."""
    yield ReflectionRoute()
    ids = tuple(portal_entries(world))
    for hops in (1, 2):
        for order in product(ids, repeat=hops):
            for split in range(hops+1):
                yield ReflectionRoute(order[:split], order[split:])


def unfolded_endpoints(source, target, world, route):
    # P(t) gains an instantaneous translation at each portal, while velocity
    # and acceleration stay unchanged. Before-bounce shifts move S forward;
    # after-bounce shifts move the virtual target backward. C stays physical.
    entries = portal_entries(world)
    def shift(sequence):
        return tuple(sum(entries[p][1].center[i]-entries[p][0].center[i] for p in sequence) for i in (0, 1))
    before, after = shift(route.before), shift(route.after)
    return tuple(source[i]+before[i] for i in (0, 1)), tuple(target[i]-after[i] for i in (0, 1))


def route_enters_planned_portals(source, acceleration, world, solution, route):
    """Cheap necessary condition before checking every physical obstacle.

    Walk just the planned trigger circles in each ballistic leg. Real replay
    still checks all unplanned portals, surface ordering and actual normals.
    """
    if not route.has_portals:
        return True
    entries = portal_entries(world)
    def leg(point, velocity, duration, sequence):
        exit_id = None
        for portal_id in sequence:
            portal, partner, partner_id = entries[portal_id]
            time = _entry_time(point, velocity, acceleration, portal, portal_trigger_radius(portal), duration,
                               allow_inside=portal_id == exit_id)
            if time is None or time >= duration:
                return False
            point = trajectory_position(point, velocity, acceleration, time)
            point = tuple(point[i]+partner.center[i]-portal.center[i] for i in (0, 1))
            velocity = trajectory_velocity(velocity, acceleration, time)
            duration -= time
            exit_id = partner_id
        return True
    if not leg(source, solution.velocity, solution.t1, route.before):
        return False
    incoming = trajectory_velocity(solution.velocity, acceleration, solution.t1)
    dot = sum(incoming[i]*solution.normal[i] for i in (0, 1))
    outgoing = tuple(incoming[i]-2*dot*solution.normal[i] for i in (0, 1))
    return leg(solution.contact, outgoing, solution.t2, route.after)


def route_approach_error(source, acceleration, world, solution, route):
    """Smooth proximity objective to locate narrow portal-compatible intervals.

    Coarse reflection samples alone can skip an entire trigger window. This
    objective provides continuous search seeds; it never accepts a final shot.
    """
    entries = portal_entries(world)
    def leg(point, velocity, duration, sequence):
        total = 0.
        exit_id = None
        for index, portal_id in enumerate(sequence):
            portal, partner, partner_id = entries[portal_id]
            radius = portal_trigger_radius(portal)
            miss, _, _ = closest_approach_to_target(point, velocity, acceleration, portal.center, duration)
            total += miss/max(radius, 1e-8)
            time = _entry_time(point, velocity, acceleration, portal, radius, duration, allow_inside=portal_id == exit_id)
            if time is None or time >= duration:
                return total+len(sequence)-index
            point = trajectory_position(point, velocity, acceleration, time)
            point = tuple(point[i]+partner.center[i]-portal.center[i] for i in (0, 1))
            velocity = trajectory_velocity(velocity, acceleration, time)
            duration -= time
            exit_id = partner_id
        return total
    incoming = trajectory_velocity(solution.velocity, acceleration, solution.t1)
    dot = sum(incoming[i]*solution.normal[i] for i in (0, 1))
    outgoing = tuple(incoming[i]-2*dot*solution.normal[i] for i in (0, 1))
    return leg(source, solution.velocity, solution.t1, route.before)+leg(solution.contact, outgoing, solution.t2, route.after)

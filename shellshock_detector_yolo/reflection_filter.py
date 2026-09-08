"""Reflection adapter for the reusable conservative Layer-A event filter."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import Callable, Iterable

from .coarse_path_filter import PortalHop, reflection_path_possible
from .reflection_proxy import ProxyCandidate, evaluate_layer_b
from .reflection_routes import ReflectionRoute, portal_entries
from .solver_config import (LAYER_A_CIRCLE_REPRESENTATIVE_COUNT,
                            LAYER_B_CIRCLE_PARAMETER_DELTA, LAYER_B_LINE_PARAMETERS)


@dataclass(frozen=True)
class CoarsePathCandidate:
    family: object
    route: ReflectionRoute
    q_seed: float
    contact: tuple[float, float]
    normal: tuple[float, float]
    events: tuple[tuple[float, float], ...]


def _surface(world, family, q):
    if family.kind == "line":
        line = world.lines[family.index]
        dx, dy = line.end[0] - line.start[0], line.end[1] - line.start[1]
        length = hypot(dx, dy)
        return (line.start[0] + q * dx, line.start[1] + q * dy), (-dy / length, dx / length), (dx / length, dy / length)
    circle = world.circles[family.index]
    normal = (cos(q), sin(q))
    return (circle.center[0] + circle.radius * normal[0], circle.center[1] + circle.radius * normal[1]), normal, (-normal[1], normal[0])


def _circle_seed(world, family, source, target):
    circle = world.circles[family.index]
    midpoint = ((source[0] + target[0]) / 2, (source[1] + target[1]) / 2)
    dx, dy = midpoint[0] - circle.center[0], midpoint[1] - circle.center[1]
    return atan2(dy, dx) % (2 * pi) if hypot(dx, dy) > 1e-9 else 0.0


def _circle_seeds(world, family, source, target):
    """Return evenly spaced Layer-A surface representatives around the circle."""
    base = _circle_seed(world, family, source, target)
    step = 2 * pi / LAYER_A_CIRCLE_REPRESENTATIVE_COUNT
    return tuple((base + index * step) % (2 * pi) for index in range(LAYER_A_CIRCLE_REPRESENTATIVE_COUNT))


def _hops(world, portal_ids):
    entries = portal_entries(world)
    return tuple(PortalHop(portal_id, entries[portal_id][0].center, entries[portal_id][1].center) for portal_id in portal_ids)


def build_coarse_candidates(source, target, world, families: Iterable[object], routes: Iterable[ReflectionRoute], acceleration):
    """Keep reflection routes unless no velocity or acceleration sign can reach an event.

    The only velocity assumptions are that a shot initially rises on screen and
    may travel left or right.  Magnitudes, collision times, scoring, virtual
    high points, and replay intentionally belong to later layers.
    """
    candidates: list[CoarsePathCandidate] = []
    reasons: dict[str, int] = {}
    directions = ((-1.0, -1.0), (1.0, -1.0))
    for family in families:
        for route in routes:
            seeds = (0.5,) if family.kind == "line" else _circle_seeds(world, family, source, target)
            last_reason = "A_SEGMENT_DIRECTION"
            passed_any = False
            for seed in seeds:
                contact, normal, tangent = _surface(world, family, seed)
                result = reflection_path_possible(source, contact, target, _hops(world, route.before), _hops(world, route.after),
                                                  directions, acceleration, tangent, normal)
                if result.valid:
                    candidates.append(CoarsePathCandidate(family, route, seed, contact, normal, result.points))
                    passed_any = True
                    continue
                last_reason = result.reason
            if not passed_any:
                reasons[last_reason] = reasons.get(last_reason, 0) + 1
    return candidates, {"layer_a_generated": sum(reasons.values()) + len(candidates), "layer_a_passed": len(candidates), "invalid_reasons": reasons}


def _nearby_parameters(candidate):
    family = candidate.family
    if family.kind == "line":
        return tuple(q for q in LAYER_B_LINE_PARAMETERS if family.lower <= q <= family.upper)
    return tuple(q % (2 * pi) for q in (candidate.q_seed - LAYER_B_CIRCLE_PARAMETER_DELTA, candidate.q_seed,
                                         candidate.q_seed + LAYER_B_CIRCLE_PARAMETER_DELTA))


def fair_candidate_prefix(candidates, limit):
    """Round-robin Layer-A seeds so enumeration order cannot monopolize B."""
    buckets = {}
    for candidate in candidates:
        family = candidate.family
        key = (family.kind, family.index, getattr(family, "side", "BOTH"), candidate.route)
        buckets.setdefault(key, []).append(candidate)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for values in buckets.values():
            if values and len(selected) < limit:
                selected.append(values.pop(0))
    return selected


def rank_proxy_candidates(candidates, source, target, world, acceleration, speed_per_power,
                          fixed_solver: Callable, score: Callable, *, top_k: int,
                          image_width: int | None = None):
    """Run the bounded analytic Layer-B proxy and event-order filter."""
    return evaluate_layer_b(
        candidates, source, target, world, acceleration, speed_per_power,
        fixed_solver, score, _nearby_parameters, _surface, top_k=top_k,
        image_width=image_width or world.image_width or 1920,
    )

"""Bounded continuous single-bounce shooting, in screen coordinates (y down)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import atan2, ceil, copysign, cos, degrees, hypot, isfinite, pi, radians, sin, sqrt
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .ballistics import GRAVITY_AT_REFERENCE, SPEED_PER_POWER_AT_REFERENCE, _scale, _wind_acceleration
from .muzzle_geometry import muzzle_position
from .world_geometry import Point, World
from .reflection_routes import reflection_routes, unfolded_endpoints, route_enters_planned_portals, route_approach_error
from .solver_config import (
    REFLECTION_LOW_POWER_MARGIN, REFLECTION_MIN_INCIDENCE, INTEGER_ANGLE_RADIUS,
    LINE_ENDPOINT_MARGIN_AT_REFERENCE, MISS_TIE_THRESHOLD_AT_REFERENCE,
    MAX_FLIGHT_TIME, N_LINE_SCAN, N_CIRCLE_SCAN, TOP_K_LAYER_A, TOP_K_LAYER_A_RESCUE,
    TOP_K_LAYER_B, TOP_K_LAYER_B_RESCUE, TOP_K_LAYER_C, MAX_FULL_REPLAYS,
    LAYER_C_LINE_WINDOW, LAYER_C_CIRCLE_WINDOW,
    PORTAL_ROUTE_PRIORITY_BONUS,
    REFLECTION_FALLBACK_TARGET_ACCEPT_RADIUS_AT_REFERENCE,
)
from .reflection_filter import build_coarse_candidates, fair_candidate_prefix, rank_proxy_candidates
from .reflection_layer_c import (
    ContinuousSurfaceSolution, FinalReplayResult, build_surface_interval,
    generate_integer_candidates, group_integer_candidates, replay_top_integer_candidates,
    sample_surface_interval,
)


def _muzzle_source_for_candidate(source: Point, direction: str, angle: int, image_width: int) -> Point:
    """Return the angle-specific projectile origin for a final integer candidate."""
    return muzzle_position(source, direction, angle, image_width)


def _c1_corrected_branch(source, target, world, acceleration, speed_per_power,
                         image_width, family, route, contact, normal, branch_index):
    """Re-solve a sampled contact after placing the barrel at its integer angle."""
    virtual_source, virtual_target = unfolded_endpoints(source, target, world, route)
    solutions = fixed_contact_solutions(
        virtual_source, virtual_target, contact, normal, acceleration, speed_per_power,
    )
    if family.kind == 'circle' and family.side != 'BOTH':
        solutions = [s for s in solutions if (
            np.dot(np.asarray(s.velocity) + np.asarray(acceleration) * s.t1, normal) > 0
        ) == (family.side == 'INNER')]
    branch = solutions[branch_index] if branch_index < len(solutions) else None
    previous_angle = None
    for _ in range(2):
        if branch is None:
            return None
        angle = round(branch.angle_degrees)
        corrected_source = _muzzle_source_for_candidate(
            source, getattr(branch, 'direction', 'right'), angle, image_width,
        )
        corrected_virtual_source, corrected_virtual_target = unfolded_endpoints(
            corrected_source, target, world, route,
        )
        solutions = fixed_contact_solutions(
            corrected_virtual_source, corrected_virtual_target,
            contact, normal, acceleration, speed_per_power,
        )
        if family.kind == 'circle' and family.side != 'BOTH':
            solutions = [s for s in solutions if (
                np.dot(np.asarray(s.velocity) + np.asarray(acceleration) * s.t1, normal) > 0
            ) == (family.side == 'INNER')]
        corrected = solutions[branch_index] if branch_index < len(solutions) else None
        if corrected is None:
            return None
        branch = corrected
        corrected_angle = round(branch.angle_degrees)
        if corrected_angle == angle or corrected_angle == previous_angle:
            break
        previous_angle = corrected_angle
    return branch


@dataclass(frozen=True)
class ContactSolution:
    contact: Point
    normal: Point
    velocity: Point
    t1: float
    t2: float
    power: float
    angle_degrees: float
    direction: str
    incidence: float
    parameter: float = 0.
    discriminant_norm: float = 1.0
    root_slope: float = 1.0


@dataclass(frozen=True)
class SurfaceFamily:
    kind: str
    index: int
    side: str
    lower: float
    upper: float


@dataclass
class ReflectionBundle:
    low_solution: dict | None = None
    high_solution: dict | None = None
    obstacle_kind: str | None = None
    obstacle_index: int | None = None
    theoretical_min_power: float | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def low(self):
        return self.low_solution

    @property
    def high(self):
        return self.high_solution


def reflection_proxy_score(solution, candidate):
    score = solution.power + 8.0 * (1.0 - solution.incidence)
    if candidate.route.has_portals:
        score -= PORTAL_ROUTE_PRIORITY_BONUS
    return score


def positive_polynomial_roots(a, b, c):
    """Stable quadratic/linear roots; an identity is handled by its caller."""
    tolerance = 1e-12 * max(abs(a), abs(b), abs(c), 1.)
    if abs(a) <= tolerance:
        return [-c/b] if abs(b) > tolerance and -c/b > 0 else []
    disc = b*b-4*a*c
    if disc < -1e-12*max(b*b, abs(4*a*c), 1.):
        return []
    root = sqrt(max(0., disc))
    q = -.5*(b+copysign(root, b))
    roots = [-b/(2*a)] if root == 0 else [q/a, c/q]
    return sorted({float(r) for r in roots if r > 0 and isfinite(r)})


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def fixed_contact_solutions(source, target, contact, normal, acceleration, speed_per_power, desired_power=None):
    """Solve flight-time ratio analytically for a fixed point and normal.

    Parallel/identity constraints have a continuous time family. Representative
    ratios include its minimum-power point and admissible interval endpoints;
    the nondegenerate case has at most two algebraic branches.
    """
    source, target, contact, normal, acceleration = map(lambda x: np.asarray(x, dtype=float), (source, target, contact, normal, acceleration))
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return []
    normal = normal/norm
    d, e, b = contact-source, target-contact, acceleration/2
    reflect = lambda x: x-2*np.dot(x, normal)*normal
    p, q = reflect(e), reflect(b)
    coefficients = (-_cross(p, b), _cross(d, b)-_cross(p, q), _cross(d, q))
    ratios = positive_polynomial_roots(*coefficients)
    coefficient_scale = max(*(abs(value) for value in coefficients), 1e-12)
    normalized = tuple(value / coefficient_scale for value in coefficients)
    discriminant_norm = max(0.0, normalized[1] * normalized[1] - 4 * normalized[0] * normalized[2])
    identity = max(abs(c) for c in coefficients) <= 1e-12*max(np.linalg.norm(d)*np.linalg.norm(b), np.linalg.norm(p)*np.linalg.norm(q), 1.)

    def reconstruct(u, explicit_z=None):
        h = u*b+q
        denominator = u*float(np.dot(h, h))
        if denominator <= 1e-18:
            if np.linalg.norm(d-u*p) > 1e-7 or np.linalg.norm(b) <= 1e-12:
                return None
            z = float(np.linalg.norm(d)/np.linalg.norm(b))/(u*u)
        else:
            z = -float(np.dot(d-u*p, h))/denominator
        if explicit_z is not None:
            z = explicit_z
        if z <= 1e-12 or not isfinite(z):
            return None
        t2, t1 = sqrt(z), u*sqrt(z)
        if t1 <= 1e-5 or t2 <= 1e-5 or t1+t2 > MAX_FLIGHT_TIME:
            return None
        velocity = d/t1-b*t1
        incoming = velocity+acceleration*t1
        outgoing = reflect(incoming)
        residual = contact+outgoing*t2+b*t2*t2-target
        if np.linalg.norm(residual) > 1e-6*max(1., np.linalg.norm(d), np.linalg.norm(e)):
            return None
        speed = float(np.linalg.norm(velocity))
        incidence = abs(float(np.dot(incoming, normal)))/max(float(np.linalg.norm(incoming)), 1e-12)
        angle = degrees(atan2(-velocity[1], abs(velocity[0])))
        if speed <= 0 or not 0 <= angle <= 90 or incidence < REFLECTION_MIN_INCIDENCE:
            return None
        root_slope = abs(2 * normalized[0] * u + normalized[1])
        return ContactSolution(
            tuple(contact), tuple(normal), tuple(velocity), t1, t2,
            speed/speed_per_power, angle, 'right' if velocity[0] >= 0 else 'left', incidence,
            discriminant_norm=discriminant_norm, root_slope=root_slope,
        )

    if identity:
        # Solve the remaining one-dimensional family in log time-ratio space.
        grid = np.linspace(-8., 8., 65)
        for lo, hi in zip(grid[:-1], grid[1:]):
            fit = minimize_scalar(lambda x: (s.power if (s := reconstruct(float(np.exp(x)))) else 1e9), bounds=(lo, hi), method='bounded')
            ratios.extend([float(np.exp(lo)), float(np.exp(fit.x)), float(np.exp(hi))])
    # h=0 can vanish at an isolated ratio even when its quadratic constraint
    # is identically zero. The travel-time equation then leaves speed free.
    if np.dot(b, b) > 1e-18:
        singular_ratio = -float(np.dot(q, b))/float(np.dot(b, b))
        if singular_ratio > 0 and np.linalg.norm(singular_ratio*b+q) < 1e-9 and np.linalg.norm(d-singular_ratio*p) < 1e-7:
            ratios.append(singular_ratio)
            if desired_power is not None:
                squares = positive_polynomial_roots(float(np.dot(b,b)), -2*float(np.dot(d,b))-(desired_power*speed_per_power)**2, float(np.dot(d,d)))
                return [s for w in squares if (s := reconstruct(singular_ratio, w/singular_ratio**2)) is not None]
    return sorted((s for u in set(ratios) if (s := reconstruct(u)) is not None), key=lambda s: s.t1/s.t2)


def _families(world, scale):
    for index, line in enumerate(world.lines):
        length = hypot(line.end[0]-line.start[0], line.end[1]-line.start[1])
        if length > 2*LINE_ENDPOINT_MARGIN_AT_REFERENCE*scale:
            yield SurfaceFamily('line', index, 'BOTH', LINE_ENDPOINT_MARGIN_AT_REFERENCE*scale/length, 1-LINE_ENDPOINT_MARGIN_AT_REFERENCE*scale/length)
    for index, circle in enumerate(world.circles):
        if circle.radius > 0:
            for side in ('INNER', 'OUTER'):
                yield SurfaceFamily('circle', index, side, 0., 2*pi)


def _surface(world, family, parameter):
    if family.kind == 'line':
        line = world.lines[family.index]
        d = np.asarray(line.end)-line.start
        n = np.array((-d[1], d[0]), dtype=float)/np.linalg.norm(d)
        return np.asarray(line.start)+parameter*d, n
    circle = world.circles[family.index]
    n = np.array((cos(parameter), sin(parameter)))
    return np.asarray(circle.center)+circle.radius*n, n


def _branch_intervals(evaluate, family, sample_count=None):
    """Coarse samples locate branches; all optima and roots are refined."""
    count = sample_count if sample_count is not None else (N_LINE_SCAN if family.kind == 'line' else N_CIRCLE_SCAN)
    grid = np.linspace(family.lower, family.upper, count+1)
    if hasattr(evaluate, 'seed_parameters'):
        grid = np.array(sorted(set((*grid, *evaluate.seed_parameters(grid)))))
    values = [evaluate(float(x)) for x in grid]
    intervals = []
    for branch in range(max((len(v) for v in values), default=0)):
        for i in range(len(grid)-1):
            if branch < len(values[i]) or branch < len(values[i+1]):
                lower, upper = float(grid[i]), float(grid[i+1])
                left_valid, right_valid = branch < len(values[i]), branch < len(values[i+1])
                if left_valid != right_valid:
                    valid, invalid = (lower, upper) if left_valid else (upper, lower)
                    for _ in range(40):
                        midpoint = (valid+invalid)/2
                        if branch < len(evaluate(midpoint)):
                            valid = midpoint
                        else:
                            invalid = midpoint
                    if left_valid:
                        upper = valid
                    else:
                        lower = valid
                intervals.append((branch, lower, upper))
    return intervals


def _fixed_power_contacts(evaluate, intervals, power):
    if hasattr(evaluate, 'at_power'):
        evaluate = evaluate.at_power(power)
    contacts = []
    for branch, lower, upper in intervals:
        def value(x):
            solutions = evaluate(float(x))
            return solutions[branch].power-power if branch < len(solutions) else float('nan')
        # Include the local minimum to bracket both roots around a tangent.
        fit = minimize_scalar(lambda x: abs(value(x)) if isfinite(value(x)) else 1e9, bounds=(lower, upper), method='bounded')
        extremum = minimize_scalar(lambda x: value(x) if isfinite(value(x)) else 1e9, bounds=(lower, upper), method='bounded')
        points = sorted(set((lower, upper, float(fit.x), float(extremum.x))))
        roots = [x for x in points if isfinite(value(x)) and abs(value(x)) < 1e-9]
        for a, b in zip(points[:-1], points[1:]):
            va, vb = value(a), value(b)
            if isfinite(va) and isfinite(vb) and va*vb < 0:
                try:
                    roots.append(brentq(value, a, b, xtol=1e-10))
                except ValueError:
                    pass
        for x in roots:
            solutions = evaluate(x)
            if branch < len(solutions) and abs(solutions[branch].power-power) < 1e-4:
                candidate = solutions[branch]
                if not any(abs(candidate.parameter-s.parameter) < 1e-7 and abs(candidate.t1-s.t1) < 1e-6 for s in contacts):
                    contacts.append(candidate)
    return contacts


def _route_evaluator(source, target, world, acceleration, image_width, speed_per_power, family, route):
    """Create an isolated cache per geometric surface and planned event order."""
    from .reflection_replay import verify_continuous_contact
    virtual_source, virtual_target = unfolded_endpoints(source, target, world, route)
    cache = {}
    raw_cache = {}
    def raw(x, desired_power=None):
        key = (x, desired_power)
        if key not in raw_cache:
            contact, normal = _surface(world, family, x)
            raw_cache[key] = fixed_contact_solutions(virtual_source, virtual_target, contact, normal,
                                                    acceleration, speed_per_power, desired_power)
        return raw_cache[key]
    def evaluate(x, desired_power=None):
        key = (x, desired_power)
        if key not in cache:
            contact, normal = _surface(world, family, x)
            solutions = raw(x, desired_power)
            if family.kind == 'circle' and family.side != 'BOTH':
                solutions = [s for s in solutions if (np.dot(np.asarray(s.velocity)+np.asarray(acceleration)*s.t1, normal) > 0) == (family.side == 'INNER')]
            if route.has_portals:
                from .combined_replay import replay_combined_shot
                solutions = [s for s in solutions
                             if route_enters_planned_portals(source, acceleration, world, s, route)
                             and replay_combined_shot(source, s.velocity, acceleration, world, target, image_width,
                                                     family, route.before, route.after, max_time=s.t1+s.t2,
                                                     expected_contact=s.contact, expected_time=s.t1,
                                                     compute_clearance=False, require_exact_target=True) is not None]
            else:
                solutions = [s for s in solutions if verify_continuous_contact(source, world, acceleration, image_width, family, s)]
            cache[key] = [replace(s, parameter=x) for s in solutions]
        return cache[key]
    evaluate.at_power = lambda power: lambda x: evaluate(x, power)
    if route.has_portals:
        def seed_parameters(grid):
            def proximity(x):
                return min((route_approach_error(source, acceleration, world, s, route) for s in raw(float(x))), default=1e9)
            seeds = []
            for lo, hi in zip(grid[:-1], grid[1:]):
                fit = minimize_scalar(proximity, bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
                if fit.fun < 1e9:
                    seeds.append(float(fit.x))
            return seeds
        evaluate.seed_parameters = seed_parameters
    return evaluate


def _legacy_solve_reflection_bundle(source: Point, target: Point, world: World, wind_value: float, wind_direction: str, image_width: int):
    from .reflection_replay import replay_integer_contact, source_matches_circle_side
    scale = _scale(image_width)
    acceleration = (_wind_acceleration(wind_value, wind_direction, image_width), GRAVITY_AT_REFERENCE*scale)
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE*scale
    started = perf_counter()
    diagnostics = {'families': 0, 'route_families': 0, 'fixed_power_roots': 0, 'integer_replays': 0,
                   'layer_c_evaluations': 0, 'full_replays': 0, 'rescue_used': False}
    bundles = []
    routes = tuple(reflection_routes(world))
    families = tuple(_families(world, scale))
    coarse, coarse_diagnostics = build_coarse_candidates(source, target, world, families, routes, acceleration)
    coarse_diagnostics = dict(coarse_diagnostics)
    coarse_invalid_reasons = coarse_diagnostics.pop('invalid_reasons', {})
    diagnostics.update(coarse_diagnostics)
    diagnostics['layer_a_invalid_reasons'] = coarse_invalid_reasons
    def proxy_score(solution, candidate):
        return reflection_proxy_score(solution, candidate)
    proxies, proxy_diagnostics = rank_proxy_candidates(fair_candidate_prefix(coarse, TOP_K_LAYER_A), source, target, world, acceleration,
                                                        speed_per_power, fixed_contact_solutions, proxy_score,
                                                        top_k=TOP_K_LAYER_B, image_width=image_width)
    if not proxies:
        diagnostics['rescue_used'] = True
        proxies, proxy_diagnostics = rank_proxy_candidates(fair_candidate_prefix(coarse, TOP_K_LAYER_A_RESCUE), source, target, world, acceleration,
                                                            speed_per_power, fixed_contact_solutions, proxy_score,
                                                            top_k=TOP_K_LAYER_B_RESCUE, image_width=image_width)
    proxy_diagnostics = dict(proxy_diagnostics)
    proxy_invalid_reasons = proxy_diagnostics.pop('invalid_reasons', {})
    diagnostics.update(proxy_diagnostics)
    diagnostics['layer_b_invalid_reasons'] = proxy_invalid_reasons
    selected = proxies[:TOP_K_LAYER_C]
    visited = set()
    for family in families:
        board = (family.kind, family.index)
        if board in visited:
            continue
        visited.add(board)
        if family.kind == 'circle':
            family = replace(family, side='INNER' if source_matches_circle_side(source, world, replace(family, side='INNER')) else 'OUTER')
        diagnostics['families'] += 1
        matching = [proxy for proxy in selected if (proxy.coarse.family.kind, proxy.coarse.family.index) == board]
        if not matching:
            continue
        minima = []
        route_families = []
        for proxy in matching:
            route = proxy.coarse.route
            route_family = replace(family, side='BOTH') if family.kind == 'circle' and route.before else family
            if not route.before and not source_matches_circle_side(source, world, route_family):
                continue
            diagnostics['route_families'] += 1
            window = LAYER_C_LINE_WINDOW if route_family.kind == 'line' else LAYER_C_CIRCLE_WINDOW
            lower = max(route_family.lower, proxy.coarse.q_seed-window)
            upper = min(route_family.upper, proxy.coarse.q_seed+window)
            if upper <= lower:
                continue
            route_family = replace(route_family, lower=lower, upper=upper)
            evaluate = _route_evaluator(source, target, world, acceleration, image_width, speed_per_power, route_family, route)
            intervals = _branch_intervals(evaluate, route_family, sample_count=4)
            diagnostics['layer_c_evaluations'] += 5
            if not intervals:
                continue
            route_families.append((route, route_family, evaluate, intervals))
            for branch, lo, hi in intervals:
                def objective(x):
                    values = evaluate(float(x))
                    return values[branch].power if branch < len(values) else 1e9
                fit = minimize_scalar(objective, bounds=(lo, hi), method='bounded', options={'xatol': 1e-8})
                minima.extend((objective(lo), objective(hi), objective(fit.x)))
        if not minima:
            continue
        minimum = min(minima)
        start = max(1, ceil(minimum+REFLECTION_LOW_POWER_MARGIN))
        high_floor = max(1, ceil(minimum))
        if minimum > 100:
            continue
        power_cache = {}
        def replay_power(power):
            if power not in power_cache:
                valid = []
                seen = set()
                for route, route_family, evaluate, intervals in route_families:
                    roots = _fixed_power_contacts(evaluate, intervals, power)
                    diagnostics['fixed_power_roots'] += len(roots)
                    for root in roots:
                        for angle in range(max(0, round(root.angle_degrees)-INTEGER_ANGLE_RADIUS), min(90, round(root.angle_degrees)+INTEGER_ANGLE_RADIUS)+1):
                            key = (route, angle, root.direction)
                            if key in seen:
                                continue
                            seen.add(key)
                            if diagnostics['full_replays'] >= MAX_FULL_REPLAYS:
                                break
                            diagnostics['integer_replays'] += 1
                            diagnostics['full_replays'] += 1
                            if route.has_portals:
                                from .combined_replay import replay_combined_shot
                                speed = power*speed_per_power
                                radians = angle*pi/180
                                velocity = ((-1 if root.direction == 'left' else 1)*speed*cos(radians), -speed*sin(radians))
                                shot = replay_combined_shot(source, velocity, acceleration, world, target, image_width,
                                                            route_family, route.before, route.after)
                                if shot is not None:
                                    shot.update(power=power, angle_degrees=angle, direction=root.direction)
                            else:
                                shot = replay_integer_contact(source, target, world, acceleration, image_width, route_family, power, angle, root.direction)
                            if shot is not None:
                                shot['planned_portals_before'] = list(route.before)
                                shot['planned_portals_after'] = list(route.after)
                                shot['theoretical_min_power'] = minimum
                                shot['theory_parameter'] = root.parameter
                                shot['low_start_power'] = start
                                valid.append(shot)
                power_cache[power] = valid
            return power_cache[power]
        low = high = None
        for power in range(start, 101):
            valid = replay_power(power)
            if valid:
                low = select_integer_candidate(valid, scale, high=False)
                break
        for power in range(100, high_floor-1, -1):
            valid = replay_power(power)
            if valid:
                high = select_integer_candidate(valid, scale, high=True)
                break
        if low or high:
            bundles.append(ReflectionBundle(low, high, family.kind, family.index, minimum, {'surface_side': family.side, 'low_start_power': start, 'high_min_power': high_floor}))
    if not bundles:
        diagnostics['total_seconds'] = perf_counter()-started
        return ReflectionBundle(diagnostics=diagnostics)
    # Bundle priority is evaluated before numerical tie breaking.
    bundle = select_reflection_bundle(bundles, scale)
    bundle.diagnostics.update(diagnostics)
    bundle.diagnostics['total_seconds'] = perf_counter()-started
    return bundle


def solve_reflection_bundle(source: Point, target: Point, world: World, wind_value: float, wind_direction: str, image_width: int):
    """Run Layer A/B/C and retain a full per-item diagnostic trace for every decision."""
    from .reflection_replay import replay_integer_contact, source_matches_circle_side
    from .combined_replay import replay_combined_shot

    scale = _scale(image_width)
    acceleration = (_wind_acceleration(wind_value, wind_direction, image_width), GRAVITY_AT_REFERENCE * scale)
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE * scale
    started = perf_counter()
    diagnostics = {'integer_candidates_raw': 0, 'integer_candidates_unique': 0,
                   'integer_full_replays': 0, 'full_replays': 0, 'rescue_used': False}
    routes = tuple(reflection_routes(world))
    families = tuple(_families(world, scale))
    layer_a_started = perf_counter()
    coarse, coarse_diag = build_coarse_candidates(source, target, world, families, routes, acceleration)
    layer_a_seconds = perf_counter() - layer_a_started
    coarse_diag = dict(coarse_diag)
    diagnostics.update({k: v for k, v in coarse_diag.items() if k != 'invalid_reasons'})
    diagnostics['layer_a_invalid_reasons'] = coarse_diag.get('invalid_reasons', {})
    diagnostics['layer_a_trace'] = coarse_diag.get('layer_a_trace', [])

    def proxy_score(solution, candidate):
        return reflection_proxy_score(solution, candidate)

    layer_b_started = perf_counter()
    proxies, proxy_diag = rank_proxy_candidates(
        coarse, source, target, world, acceleration,
        speed_per_power, fixed_contact_solutions, proxy_score, top_k=TOP_K_LAYER_B,
        image_width=image_width,
    )
    if not proxies:
        diagnostics['rescue_used'] = True
        proxies, proxy_diag = rank_proxy_candidates(
            coarse, source, target, world, acceleration,
            speed_per_power, fixed_contact_solutions, proxy_score, top_k=TOP_K_LAYER_B_RESCUE,
            image_width=image_width,
        )
    layer_b_seconds = perf_counter() - layer_b_started
    proxy_diag = dict(proxy_diag)
    diagnostics.update({k: v for k, v in proxy_diag.items() if k != 'invalid_reasons'})
    diagnostics['layer_b_invalid_reasons'] = proxy_diag.get('invalid_reasons', {})
    diagnostics['layer_b_soft_invalid_reasons'] = proxy_diag.get('soft_invalid_reasons', {})
    diagnostics['layer_b_trace'] = proxy_diag.get('layer_b_trace', [])

    layer_c1_started = perf_counter()
    continuous = []
    c1_trace = []
    for proxy in proxies:
        family = proxy.coarse.family
        route = proxy.coarse.route
        route_family = replace(family, side='BOTH') if family.kind == 'circle' and route.before else family
        if family.kind == 'circle' and not route.before and not source_matches_circle_side(source, world, route_family):
            c1_trace.append({"id": f"C1:proxy={id(proxy)}", "status": "FAIL", "decision": "SKIPPED", "reason": "C1_CIRCLE_SIDE_POLICY"})
            continue
        lower, upper = build_surface_interval(proxy)
        for parameter in sample_surface_interval(lower, upper):
            contact, normal = _surface(world, route_family, parameter)
            branch = _c1_corrected_branch(
                source, target, world, acceleration, speed_per_power, image_width,
                route_family, route, contact, normal, proxy.branch,
            )
            diagnostics['layer_c_continuous_solves'] = diagnostics.get('layer_c_continuous_solves', 0) + 1
            diagnostics['layer_c_surface_samples'] = diagnostics.get('layer_c_surface_samples', 0) + 1
            if branch is not None:
                c1_trace.append({"id": f"C1:proxy={id(proxy)}:q={parameter:.12g}", "status": "PASS", "decision": "CONTINUOUS_SOLUTION",
                                 "parameter": float(parameter), "branch": proxy.branch, "angle": float(branch.angle_degrees), "power": float(branch.power)})
                continuous.append(ContinuousSurfaceSolution(
                    source_b=proxy, surface_param=parameter, branch_id=proxy.branch,
                    angle_cont=branch.angle_degrees, power_cont=branch.power,
                    collision_point=tuple(branch.contact), normal=tuple(branch.normal),
                    incidence=branch.incidence, valid_math=True,
                ))
            else:
                c1_trace.append({"id": f"C1:proxy={id(proxy)}:q={parameter:.12g}", "status": "FAIL", "decision": "REJECTED",
                                 "parameter": float(parameter), "branch": proxy.branch, "reason": "C1_BRANCH_UNAVAILABLE"})

    raw = generate_integer_candidates(continuous)
    groups = group_integer_candidates(raw)
    diagnostics['integer_candidates_raw'] = len(raw)
    diagnostics['integer_candidates_unique'] = len({(c.angle, c.power) for c in raw})
    c1_trace.extend({"id": f"C1:integer=({candidate.angle},{candidate.power}):source={candidate.source_type}",
                     "status": "PASS", "decision": "GENERATED", "angle": candidate.angle,
                     "power": candidate.power, "source_type": candidate.source_type}
                    for candidate in raw)
    c1_trace.extend({"id": f"C1:group=({group.angle},{group.power})", "status": "PASS",
                     "decision": "ENTERED_C2_POOL", "angle": group.angle, "power": group.power,
                     "source_count": len(group.sources)} for group in groups)
    diagnostics['layer_c1_trace'] = c1_trace
    layer_c1_seconds = perf_counter() - layer_c1_started

    def replay_group(group, *, target_accept_radius=None, relaxed=False):
        for candidate in group.sources:
            continuous_source = candidate.source
            proxy = continuous_source.source_b
            family = proxy.coarse.family
            route = proxy.coarse.route
            direction = getattr(proxy.solution, 'direction', 'right')
            muzzle_source = _muzzle_source_for_candidate(source, direction, group.angle, image_width)
            route_family = replace(family, side='BOTH') if family.kind == 'circle' and route.before else family
            provenance = {
                'reflection_obstacle': {'kind': family.kind, 'index': family.index},
                'reflection_side': family.side,
                'layer_b_proxy': {
                    'branch': proxy.branch,
                    'surface_parameter': float(getattr(proxy.coarse, 'q_seed', 0.0)),
                    'score_b': float(proxy.score_b),
                },
                'route': {'before': list(route.before), 'after': list(route.after)},
            }
            if route.has_portals:
                speed = group.power * speed_per_power
                angle = radians(group.angle)
                velocity = ((-1 if direction == 'left' else 1) * speed * cos(angle), -speed * sin(angle))
                shot = replay_combined_shot(muzzle_source, velocity, acceleration, world, target, image_width,
                                             route_family, route.before, route.after,
                                             target_accept_radius=target_accept_radius)
            else:
                shot = replay_integer_contact(muzzle_source, target, world, acceleration, image_width,
                                               route_family, group.power, group.angle, direction,
                                               target_accept_radius=target_accept_radius)
            if shot is not None:
                shot = dict(shot)
                shot['c2_provenance'] = provenance
                if relaxed:
                    shot['relaxed_target_acceptance'] = True
                return FinalReplayResult(
                    group.angle, group.power, True, None,
                    float(shot.get('miss_distance', float('inf'))),
                    float(shot.get('clearance', 0.0)), float(shot.get('incidence', 0.0)),
                    candidate.grid_error, float(proxy.score_b), candidate,
                    dict(shot),
                )
        return FinalReplayResult(group.angle, group.power, False, 'C_INTEGER_REPLAY')

    layer_c2_started = perf_counter()
    observed_replays = []
    def traced_replay(group):
        result = replay_group(group)
        observed_replays.append((group, result))
        return result
    final, replay_diag = replay_top_integer_candidates(groups, traced_replay)
    final_keys = {(result.angle, result.power) for result in final}
    diagnostics['layer_c2_trace'] = [
        {"id": f"C2:integer=({group.angle},{group.power})", "status": "PASS" if result.valid else "FAIL",
         "decision": ("FINAL" if (result.valid and (group.angle, group.power) in final_keys)
                      else "DROPPED_FINAL_RESULT_LIMIT" if result.valid else "REJECTED"),
         "reason": result.invalid_reason,
         "angle": group.angle, "power": group.power,
         "c2_provenance": (result.payload or {}).get('c2_provenance')}
        for group, result in observed_replays
    ]
    if not final and groups:
        relaxed_replay = lambda group: replay_group(
            group,
            target_accept_radius=REFLECTION_FALLBACK_TARGET_ACCEPT_RADIUS_AT_REFERENCE,
            relaxed=True,
        )
        final, relaxed_diag = replay_top_integer_candidates(groups, relaxed_replay)
        replay_diag['integer_full_replays'] += relaxed_diag['integer_full_replays']
        replay_diag['integer_replay_passed'] += relaxed_diag['integer_replay_passed']
        replay_diag['integer_replay_failed'] += relaxed_diag['integer_replay_failed']
        replay_diag.update({
            'integer_relaxed_replays': relaxed_diag['integer_full_replays'],
            'integer_relaxed_replay_passed': relaxed_diag['integer_replay_passed'],
            'integer_relaxed_replay_failed': relaxed_diag['integer_replay_failed'],
            'relaxed_target_acceptance_used': bool(final),
            'relaxed_target_accept_radius': REFLECTION_FALLBACK_TARGET_ACCEPT_RADIUS_AT_REFERENCE,
        })
    layer_c2_seconds = perf_counter() - layer_c2_started
    diagnostics.update(replay_diag)
    diagnostics['final_results'] = [_result_to_shot(result) for result in final]
    diagnostics['final_trace'] = [{"id": f"FINAL:integer=({result.angle},{result.power})", "status": "PASS", "decision": "FINAL",
                                   "angle": result.angle, "power": result.power,
                                   "c2_provenance": (result.payload or {}).get('c2_provenance')}
                                  for result in final]
    diagnostics['full_replays'] = diagnostics['integer_full_replays']
    diagnostics['total_seconds'] = perf_counter() - started
    diagnostics['timing'] = {
        'layer_a_seconds': layer_a_seconds,
        'layer_b_seconds': layer_b_seconds,
        'layer_c1_seconds': layer_c1_seconds,
        'layer_c2_seconds': layer_c2_seconds,
        'total_seconds': diagnostics['total_seconds'],
    }
    if not final:
        return ReflectionBundle(diagnostics=diagnostics)
    low = _result_to_shot(final[0])
    high = _result_to_shot(final[-1])
    low['timing'] = diagnostics['timing']
    high['timing'] = diagnostics['timing']
    diagnostics['final_results'] = [dict(result, timing=diagnostics['timing']) for result in diagnostics['final_results']]
    return ReflectionBundle(low, high, None, None, final[0].actual_miss_px, diagnostics)


def _result_to_shot(result):
    shot = dict(result.payload or {})
    shot.update({
        'status': 'reachable', 'mode': 'reflection', 'angle_degrees': result.angle,
        'power': result.power, 'miss_distance': result.actual_miss_px,
        'clearance': result.min_clearance_px, 'incidence': result.actual_incidence,
        'manual_control': result.manual_control,
    })
    return shot


def select_integer_candidate(valid, scale, *, high=False):
    if high:
        return max(valid, key=lambda s: (s['angle_degrees'], -s['miss_distance'], s['clearance'], s['incidence'], s['endpoint_margin']))
    best_miss = min(s['miss_distance'] for s in valid)
    reliable = [s for s in valid if s['miss_distance'] <= best_miss+MISS_TIE_THRESHOLD_AT_REFERENCE*scale]
    return max(reliable, key=lambda s: (s['clearance'], s['incidence'], s['endpoint_margin'], -s['miss_distance']))


def select_reflection_bundle(bundles, scale):
    def category(b):
        return 0 if b.low and b.high and b.obstacle_kind == 'line' else 1 if b.low and b.high else 2 if b.obstacle_kind == 'line' else 3
    best_category = min(map(category, bundles))
    preferred = [b for b in bundles if category(b) == best_category]
    shots = lambda b: [s for s in (b.low, b.high) if s is not None]
    aggregate_miss = lambda b: sum(s['miss_distance'] for s in shots(b))
    best_miss = min(map(aggregate_miss, preferred))
    tied = [b for b in preferred if aggregate_miss(b) <= best_miss+MISS_TIE_THRESHOLD_AT_REFERENCE*scale]
    return max(tied, key=lambda b: (
        min(s['clearance'] for s in shots(b)), min(s['incidence'] for s in shots(b)),
        min(s['endpoint_margin'] for s in shots(b)), (b.high or b.low)['power'],
        -aggregate_miss(b), -(b.low or b.high)['power'], -b.obstacle_index,
    ))


def solve_reflection_integer_shot(source, target, world, wind_value, wind_direction, image_width, arc_preference='low'):
    if arc_preference not in {'low', 'high'}:
        raise ValueError("arc_preference must be 'low' or 'high'")
    bundle = solve_reflection_bundle(source, target, world, wind_value, wind_direction, image_width)
    shot = bundle.low if arc_preference == 'low' else bundle.high
    if shot is None:
        shot = bundle.high if arc_preference == 'low' else bundle.low
        if shot is None:
            return {'status': 'unreachable', 'reason': 'no-verified-reflection-shot', 'diagnostics': bundle.diagnostics}
        return dict(shot, arc_preference=arc_preference, arc_fallback=True, selected_arc='high' if arc_preference == 'low' else 'low', diagnostics=bundle.diagnostics)
    return dict(shot, arc_preference=arc_preference, arc_fallback=False, selected_arc=arc_preference, diagnostics=bundle.diagnostics)

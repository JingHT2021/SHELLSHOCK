"""Layer-B fixed-contact proxy validation and stability ranking."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, isfinite
from typing import Callable, Iterable

import numpy as np

from .proxy_events import ProxyEvent, validate_proxy_event_sequence
from .reflection_routes import unfolded_endpoints
from .solver_config import (
    LAYER_B_ANGLE_SPAN_WEIGHT, LAYER_B_CIRCLE_COVERAGE_WEIGHT,
    LAYER_B_CLEARANCE_WEIGHT, LAYER_B_INCIDENCE_WEIGHT,
    LAYER_B_LINE_COVERAGE_WEIGHT, LAYER_B_PORTAL_DEPTH_WEIGHT,
    LAYER_B_POWER_SPAN_WEIGHT, LAYER_B_ROOT_WEIGHT,
    REFLECTION_LOW_POWER_MARGIN, REFLECTION_MIN_INCIDENCE,
)


@dataclass(frozen=True)
class LayerBProxyCandidate:
    coarse: object
    solution: object
    branch: int
    score_b: float
    root_stability: float = 1.0
    discriminant_norm: float = 1.0
    root_slope: float = 1.0
    incidence: float = 0.0
    planned_portal_depth: float = 0.0
    min_unplanned_clearance: float = inf
    neighborhood_valid_count: int = 1
    neighborhood_sample_count: int = 1
    neighborhood_angle_span: float = 0.0
    neighborhood_power_span: float = 0.0
    proxy_events: tuple[ProxyEvent, ...] = ()
    valid: bool = True
    invalid_reason: str | None = None


# Backward-compatible name used by the first Layer-B implementation.
ProxyCandidate = LayerBProxyCandidate


def clearance_penalty(clearance):
    """Map a normalized safety margin to a bounded lower-is-better cost."""
    if not isfinite(clearance):
        return 0.0
    return LAYER_B_CLEARANCE_WEIGHT / (1.0 + max(0.0, clearance))


def _has_physical_solution_shape(solution):
    return all(hasattr(solution, name) for name in (
        "contact", "normal", "velocity", "t1", "t2", "power", "angle_degrees", "incidence"
    ))


def _hard_invalid_reason(solution, family, acceleration):
    if not isfinite(solution.power) or solution.power <= 0 or solution.power > 100 + REFLECTION_LOW_POWER_MARGIN:
        return "B_POWER_RANGE"
    if not isfinite(solution.angle_degrees) or not 0 <= solution.angle_degrees <= 90:
        return "B_ANGLE_RANGE"
    if not isfinite(solution.incidence) or solution.incidence < REFLECTION_MIN_INCIDENCE:
        return "B_GRAZING"
    if family.kind == "circle" and family.side in {"INNER", "OUTER"}:
        incoming = np.asarray(solution.velocity, dtype=float) + np.asarray(acceleration, dtype=float) * solution.t1
        dot = float(np.dot(incoming, np.asarray(solution.normal, dtype=float)))
        if (dot > 0) != (family.side == "INNER"):
            return "B_CIRCLE_SIDE"
    return None


def _root_stabilities(solutions):
    if all(hasattr(solution, "root_slope") for solution in solutions):
        return [max(0.0, solution.root_slope) / (1.0 + max(0.0, solution.root_slope))
                for solution in solutions]
    ratios = [solution.t1 / solution.t2 for solution in solutions]
    result = []
    for index, ratio in enumerate(ratios):
        neighbors = [abs(ratio - other) / max(1.0, abs(ratio), abs(other))
                     for other_index, other in enumerate(ratios) if other_index != index]
        result.append(min(neighbors, default=1.0))
    return result


def evaluate_layer_b(candidates: Iterable[object], source, target, world, acceleration,
                     speed_per_power, fixed_solver: Callable, base_score: Callable,
                     parameter_samples: Callable, surface: Callable, *, top_k: int,
                     image_width: int):
    """Evaluate bounded samples and reject hard-invalid event sequences."""
    raw: list[dict] = []
    invalid_reasons: dict[str, int] = {}
    soft_invalid_reasons: dict[str, int] = {}
    sample_totals: dict[int, int] = {}
    sample_count = root_count = 0

    for group_id, candidate in enumerate(candidates):
        virtual_source, virtual_target = unfolded_endpoints(source, target, world, candidate.route)
        seen_parameters = set()
        for parameter in parameter_samples(candidate):
            key = round(float(parameter), 12)
            if key in seen_parameters:
                continue
            seen_parameters.add(key)
            sample_totals[group_id] = sample_totals.get(group_id, 0) + 1
            contact, normal, _ = surface(world, candidate.family, parameter)
            sampled = replace(candidate, q_seed=parameter, contact=contact, normal=normal)
            sample_count += 1
            solutions = list(fixed_solver(virtual_source, virtual_target, contact, normal,
                                          acceleration, speed_per_power))
            stabilities = _root_stabilities(solutions) if solutions and all(_has_physical_solution_shape(s) for s in solutions) else [1.0] * len(solutions)
            for branch, (solution, root_stability) in enumerate(zip(solutions, stabilities)):
                root_count += 1
                validation = None
                if _has_physical_solution_shape(solution):
                    reason = _hard_invalid_reason(solution, candidate.family, acceleration)
                    hard_reasons = {"B_POWER_RANGE", "B_ANGLE_RANGE", "B_GRAZING"}
                    if reason in hard_reasons:
                        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                        continue
                    if reason is None:
                        validation = validate_proxy_event_sequence(
                            source, target, solution, candidate.family, candidate.route,
                            world, acceleration, image_width,
                        )
                        reason = validation.invalid_reason
                    if reason is not None:
                        soft_invalid_reasons[reason] = soft_invalid_reasons.get(reason, 0) + 1
                raw.append({
                    "group_id": group_id,
                    "parameter": key,
                    "coarse": sampled,
                    "solution": solution,
                    "branch": branch,
                    "root_stability": root_stability,
                    "validation": validation,
                })

    grouped: dict[tuple[int, int], list[dict]] = {}
    for item in raw:
        grouped.setdefault((item["group_id"], item["branch"]), []).append(item)

    proxies: list[LayerBProxyCandidate] = []
    for items in grouped.values():
        group_id = items[0]["group_id"]
        sample_total = sample_totals[group_id]
        valid_count = len({item["parameter"] for item in items})
        physical = [item for item in items if _has_physical_solution_shape(item["solution"])]
        angles = [item["solution"].angle_degrees for item in physical]
        powers = [item["solution"].power for item in physical]
        angle_span = max(angles) - min(angles) if angles else 0.0
        power_span = max(powers) - min(powers) if powers else 0.0
        branch_candidates = []
        for item in items:
            solution = item["solution"]
            validation = item["validation"]
            score = float(base_score(solution, item["coarse"]))
            if _has_physical_solution_shape(solution):
                score += LAYER_B_INCIDENCE_WEIGHT * (1.0 - min(1.0, solution.incidence))
                score += LAYER_B_POWER_SPAN_WEIGHT * max(0.0, power_span / 100.0)
                score += LAYER_B_ANGLE_SPAN_WEIGHT * angle_span
                score += LAYER_B_ROOT_WEIGHT * (1.0 - min(1.0, item["root_stability"]))
                coverage_weight = (LAYER_B_CIRCLE_COVERAGE_WEIGHT if item["coarse"].family.kind == "circle"
                                   else LAYER_B_LINE_COVERAGE_WEIGHT)
                score += coverage_weight * (1.0 - valid_count / max(1, sample_total))
                if item["coarse"].route.has_portals:
                    score += LAYER_B_PORTAL_DEPTH_WEIGHT * (1.0 - min(1.0, validation.planned_portal_depth))
                score += clearance_penalty(validation.min_unplanned_clearance)
            branch_candidates.append(LayerBProxyCandidate(
                coarse=item["coarse"], solution=solution, branch=item["branch"], score_b=score,
                root_stability=item["root_stability"],
                discriminant_norm=getattr(solution, "discriminant_norm", 1.0),
                root_slope=getattr(solution, "root_slope", 1.0),
                incidence=getattr(solution, "incidence", 0.0),
                planned_portal_depth=validation.planned_portal_depth if validation else 0.0,
                min_unplanned_clearance=validation.min_unplanned_clearance if validation else inf,
                neighborhood_valid_count=valid_count,
                neighborhood_sample_count=sample_total,
                neighborhood_angle_span=angle_span,
                neighborhood_power_span=power_span,
                proxy_events=validation.events if validation else (),
            ))
        proxies.append(min(branch_candidates, key=lambda candidate: candidate.score_b))

    buckets: dict[tuple, list[LayerBProxyCandidate]] = {}
    for proxy in sorted(proxies, key=lambda candidate: candidate.score_b):
        family = proxy.coarse.family
        key = (family.kind, family.index, getattr(family, "side", "BOTH"), proxy.coarse.route)
        buckets.setdefault(key, []).append(proxy)
    ordered_keys = sorted(buckets, key=lambda key: buckets[key][0].score_b)
    selected = []
    while len(selected) < top_k and any(buckets.values()):
        for key in ordered_keys:
            if buckets[key] and len(selected) < top_k:
                selected.append(buckets[key].pop(0))
    return selected, {
        "layer_b_evaluated": sample_count,
        "layer_b_samples": sample_count,
        "layer_b_roots": root_count,
        "layer_b_passed": len(selected),
        "invalid_reasons": invalid_reasons,
        "soft_invalid_reasons": soft_invalid_reasons,
    }

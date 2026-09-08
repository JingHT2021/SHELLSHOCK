"""Layer-C integer candidate generation, bounded replay, and activation state."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor, hypot, isfinite, pi

from .solver_config import (
    FINAL_INTEGER_CANDIDATE_POOL,
    FINAL_RESULT_MAX,
    LAYER_B_CIRCLE_PARAMETER_DELTA,
    LAYER_B_LINE_PARAMETERS,
)


@dataclass(frozen=True)
class ContinuousSurfaceSolution:
    source_b: object
    surface_param: float
    branch_id: int | str
    angle_cont: float
    power_cont: float
    collision_point: tuple[float, float] = (0.0, 0.0)
    normal: tuple[float, float] = (0.0, 0.0)
    incidence: float = 0.0
    residual: float = 0.0
    valid_math: bool = True


@dataclass(frozen=True)
class IntegerCandidate:
    angle: int
    power: int
    source: object
    source_type: str = "NEAREST_SAMPLE"
    grid_error: float = 0.0

    @property
    def source_b(self):
        return getattr(self.source, "source_b", self.source)


@dataclass
class IntegerCandidateGroup:
    angle: int
    power: int
    sources: list[IntegerCandidate] = field(default_factory=list)

    @property
    def best_source(self):
        return min(self.sources, key=pre_sort_key)

    @property
    def best_pre_score(self):
        return pre_sort_key(self.best_source)


@dataclass(frozen=True)
class FinalReplayResult:
    angle: int
    power: int
    valid: bool
    invalid_reason: str | None = None
    actual_miss_px: float = float("inf")
    min_clearance_px: float = 0.0
    actual_incidence: float = 0.0
    pre_grid_error: float = float("inf")
    source_score_b: float = float("inf")
    matched_source: object | None = None
    payload: dict | None = None

    @property
    def manual_control(self):
        return self.power, self.angle


def build_surface_interval(proxy, *, line_parameters=LAYER_B_LINE_PARAMETERS):
    """Return the local Layer-C interval around one Layer-B surface sample."""
    family = proxy.coarse.family
    parameter = float(getattr(proxy.coarse, "q_seed", getattr(proxy.solution, "parameter", 0.0)))
    if family.kind == "line":
        points = sorted(float(p) for p in line_parameters if family.lower <= p <= family.upper)
        if not points:
            return float(family.lower), float(family.upper)
        index = min(range(len(points)), key=lambda i: abs(points[i] - parameter))
        lower = family.lower if index == 0 else (points[index - 1] + points[index]) / 2
        upper = family.upper if index == len(points) - 1 else (points[index] + points[index + 1]) / 2
        return lower, upper
    half = LAYER_B_CIRCLE_PARAMETER_DELTA / 2
    return parameter - half, parameter + half


def sample_surface_interval(lower: float, upper: float, count: int = 7) -> list[float]:
    if count < 2 or upper < lower:
        raise ValueError("surface sample count must be at least two and interval must be ordered")
    step = (upper - lower) / (count - 1)
    return [lower + i * step for i in range(count)]


def calculate_grid_error(angle: float, power: float, angle_weight: float = 1.0, power_weight: float = 1.0) -> float:
    da = angle - round(angle)
    dp = power - round(power)
    return (angle_weight * da * da + power_weight * dp * dp) ** 0.5


def _source_values(source):
    return (float(getattr(source, "angle_cont", getattr(source, "angle_degrees", 0.0))),
            float(getattr(source, "power_cont", getattr(source, "power", 0.0))))


def _candidate(source, angle, power, source_type):
    angle_cont, power_cont = _source_values(source)
    grid_error = ((angle_cont - int(angle)) ** 2 + (power_cont - int(power)) ** 2) ** 0.5
    return IntegerCandidate(int(angle), int(power), source, source_type, grid_error)


def _integer_range_between(left, right, minimum, maximum):
    return range(max(minimum, ceil(min(left, right))), min(maximum, floor(max(left, right))) + 1)


def generate_integer_candidates(samples, *, angle_range=(0, 90), power_range=(1, 100)):
    valid = [s for s in samples if getattr(s, "valid_math", False)]
    result = []
    for sample in valid:
        angle, power = _source_values(sample)
        ai, pi_ = round(angle), round(power)
        if angle_range[0] <= ai <= angle_range[1] and power_range[0] <= pi_ <= power_range[1]:
            result.append(_candidate(sample, ai, pi_, "NEAREST_SAMPLE"))
    for left, right in zip(valid, valid[1:]):
        if getattr(left, "branch_id", None) != getattr(right, "branch_id", None):
            continue
        s0, s1 = float(left.surface_param), float(right.surface_param)
        a0, p0 = _source_values(left)
        a1, p1 = _source_values(right)
        if p0 != p1:
            for power in _integer_range_between(p0, p1, *power_range):
                fraction = (power - p0) / (p1 - p0)
                angle = a0 + fraction * (a1 - a0)
                for integer_angle in {round(angle), floor(angle), ceil(angle)}:
                    if angle_range[0] <= integer_angle <= angle_range[1]:
                        result.append(_candidate(left, integer_angle, power, "POWER_CROSS"))
        if a0 != a1:
            for angle in _integer_range_between(a0, a1, *angle_range):
                fraction = (angle - a0) / (a1 - a0)
                power = p0 + fraction * (p1 - p0)
                for integer_power in {round(power), floor(power), ceil(power)}:
                    if power_range[0] <= integer_power <= power_range[1]:
                        result.append(_candidate(left, angle, integer_power, "ANGLE_CROSS"))
    return result


def pre_sort_key(candidate):
    source = candidate.source_b
    return (candidate.grid_error,
            -float(getattr(source, "incidence", 0.0)),
            float(getattr(source, "score_b", float("inf"))))


def group_integer_candidates(candidates, limit=FINAL_INTEGER_CANDIDATE_POOL):
    groups = {}
    for candidate in candidates:
        if candidate.angle < 0 or candidate.angle > 90 or candidate.power < 1 or candidate.power > 100:
            continue
        groups.setdefault((candidate.angle, candidate.power), IntegerCandidateGroup(candidate.angle, candidate.power)).sources.append(candidate)
    return sorted(groups.values(), key=lambda group: group.best_pre_score)[:limit]


def final_sort_key(result):
    return (float(result.actual_miss_px), -float(result.min_clearance_px),
            -float(result.actual_incidence), float(result.pre_grid_error),
            float(result.source_score_b))


def replay_top_integer_candidates(groups, replay, *, max_results=FINAL_RESULT_MAX):
    results = [replay(group) for group in groups]
    valid = sorted((result for result in results if result.valid), key=final_sort_key)
    return valid[:max_results], {
        "integer_candidate_pool_size": len(groups),
        "integer_full_replays": len(groups),
        "integer_replay_passed": len(valid),
        "integer_replay_failed": len(results) - len(valid),
        "final_result_count": min(len(valid), max_results),
    }


class FinalResultManager:
    def __init__(self, results, click):
        self.results = list(results)
        self._click = click
        self.current_final_index = 0

    def current(self):
        return self.results[self.current_final_index]

    def activate(self, index=None):
        if not self.results:
            return None
        if index is not None:
            self.current_final_index = max(0, min(int(index), len(self.results) - 1))
        return self._click(self.current())

    def switch(self, delta):
        if not self.results:
            return None
        old_index = self.current_final_index
        self.current_final_index = max(0, min(old_index + int(delta), len(self.results) - 1))
        if self.current_final_index != old_index:
            self.activate()
        return self.current()

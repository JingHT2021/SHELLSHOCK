"""Presentation state for selecting verified candidate results."""
from dataclasses import dataclass,field

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
        self.current_final_index = (old_index + int(delta)) % len(self.results)
        if self.current_final_index != old_index:
            self.activate()
        return self.current()

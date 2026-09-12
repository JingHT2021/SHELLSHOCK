"""Transient aiming state used by the live keyboard controller."""

from __future__ import annotations


class AimState:
    """Hold a manually marked tank center for exactly one calculation."""

    def __init__(self) -> None:
        self._temporary_self: tuple[float, float] | None = None

    def mark_temporary_self(self, point: tuple[float, float]) -> None:
        self._temporary_self = (float(point[0]), float(point[1]))

    def source_for_shift(self, detected_self: tuple[float, float] | None) -> tuple[float, float] | None:
        source = self._temporary_self if self._temporary_self is not None else detected_self
        self._temporary_self = None
        return source

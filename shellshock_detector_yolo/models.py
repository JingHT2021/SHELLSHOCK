from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    confidence: float


@dataclass(frozen=True)
class Wind:
    value: int | None
    direction: str | None
    confidence: float


@dataclass
class DetectionResult:
    image_width: int
    image_height: int
    self_tank: Detection | None
    enemies: list[Detection]
    wind: Wind
    errors: list[str]
    ballistics: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "image": {"width": self.image_width, "height": self.image_height},
            "self": asdict(self.self_tank) if self.self_tank else None,
            "enemies": [asdict(enemy) for enemy in self.enemies],
            "wind": asdict(self.wind),
            "errors": self.errors,
            "ballistics": self.ballistics,
        }

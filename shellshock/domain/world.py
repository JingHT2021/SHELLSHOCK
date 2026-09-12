"""Immutable geometry independent of recognition and UI dependencies."""
from dataclasses import dataclass
from typing import Literal

Point = tuple[float, float]


@dataclass(frozen=True)
class PoseKeypoint:
    x: float
    y: float
    visible: int = 0
    confidence: float = 0.0


@dataclass(frozen=True)
class DetectionBox:
    name: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    keypoints: tuple[PoseKeypoint, ...] = ()
    source: str = "yolo"
    refined_center: Point | None = None


@dataclass(frozen=True)
class CircleObstacle:
    center: Point
    radius: float


@dataclass(frozen=True)
class RewardZone:
    center: Point
    radius: float
    multiplier: float
    object_id: str


@dataclass(frozen=True)
class LineObstacle:
    start: Point
    end: Point


@dataclass(frozen=True)
class Portal:
    color: Literal["orange", "blue"]
    center: Point
    radius: float
    number: int | None = None


@dataclass(frozen=True)
class PortalPair:
    orange: Portal
    blue: Portal


@dataclass(frozen=True)
class World:
    image_width: int = 0
    image_height: int = 0
    self_position: Point | None = None
    circles: tuple[CircleObstacle, ...] = ()
    lines: tuple[LineObstacle, ...] = ()
    portal_pairs: tuple[PortalPair, ...] = ()
    unpaired_portals: int = 0
    blackholes: tuple[CircleObstacle, ...] = ()
    rewards: tuple[RewardZone, ...] = ()
    unpaired_portal_regions: tuple[Portal, ...] = ()

    def __post_init__(self) -> None:
        """Defend frozen-world semantics when callers supply ordinary lists."""
        object.__setattr__(self, "circles", tuple(self.circles))
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "portal_pairs", tuple(self.portal_pairs))
        object.__setattr__(self, "blackholes", tuple(self.blackholes))
        object.__setattr__(self, "rewards", tuple(self.rewards))
        object.__setattr__(self, "unpaired_portal_regions", tuple(self.unpaired_portal_regions))


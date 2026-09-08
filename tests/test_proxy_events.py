from dataclasses import dataclass

import pytest

from shellshock_detector_yolo.reflection_routes import ReflectionRoute
from shellshock_detector_yolo.world_geometry import CircleObstacle, LineObstacle, Portal, PortalPair, World


@dataclass(frozen=True)
class Family:
    kind: str
    index: int
    side: str = "BOTH"


@dataclass(frozen=True)
class Solution:
    contact: tuple[float, float]
    normal: tuple[float, float]
    velocity: tuple[float, float]
    t1: float
    t2: float
    power: float = 50.0
    angle_degrees: float = 0.0
    incidence: float = 1.0


def test_collect_segment_events_orders_wrong_line_before_planned_portal():
    from shellshock_detector_yolo.proxy_events import collect_segment_events

    world = World(
        lines=(LineObstacle((4, -5), (4, 5)),),
        portal_pairs=(PortalPair(Portal("orange", (8, 0), 1), Portal("blue", (20, 0), 1)),),
    )

    events = collect_segment_events(
        (0, 0), (1, 0), (0, 0), 10, world, 1920, planned_portal="0:orange"
    )

    assert [(event.event_type, event.object_id) for event in events[:2]] == [
        ("line_collision", "0"),
        ("portal", "0:orange"),
    ]


def test_collect_segment_events_finds_wrong_circle_before_target_time():
    from shellshock_detector_yolo.proxy_events import collect_segment_events

    world = World(circles=(CircleObstacle((5, 0), 1),))
    events = collect_segment_events((0, 0), (1, 0), (0, 0), 10, world, 1920)

    assert events[0].event_type == "circle_collision"
    assert events[0].object_id == "0"
    assert events[0].time == pytest.approx(4.0)


def test_validate_proxy_sequence_accepts_target_before_late_obstacle():
    from shellshock_detector_yolo.proxy_events import validate_proxy_event_sequence

    world = World(
        lines=(
            LineObstacle((5, -5), (5, 5)),
            LineObstacle((20, -5), (20, 5)),
        )
    )
    solution = Solution(contact=(5, 0), normal=(1, 0), velocity=(1, 0), t1=5, t2=5)

    result = validate_proxy_event_sequence(
        (0, 0), (0, 0), solution, Family("line", 0), ReflectionRoute(), world, (0, 0), 1920
    )

    assert result.valid
    assert [event.event_type for event in result.events] == ["reflection", "target"]


def test_validate_proxy_sequence_rejects_wrong_reflector_first():
    from shellshock_detector_yolo.proxy_events import validate_proxy_event_sequence

    world = World(
        lines=(
            LineObstacle((5, -5), (5, 5)),
            LineObstacle((3, -5), (3, 5)),
        )
    )
    solution = Solution(contact=(5, 0), normal=(1, 0), velocity=(1, 0), t1=5, t2=5)

    result = validate_proxy_event_sequence(
        (0, 0), (0, 0), solution, Family("line", 0), ReflectionRoute(), world, (0, 0), 1920
    )

    assert not result.valid
    assert result.invalid_reason == "B_UNPLANNED_LINE_COLLISION"


def test_validate_proxy_sequence_rejects_unplanned_portal_before_reflector():
    from shellshock_detector_yolo.proxy_events import validate_proxy_event_sequence

    world = World(
        lines=(LineObstacle((10, -5), (10, 5)),),
        portal_pairs=(PortalPair(Portal("orange", (7, 0), 1), Portal("blue", (30, 0), 1)),),
    )
    solution = Solution(contact=(10, 0), normal=(1, 0), velocity=(1, 0), t1=10, t2=10)

    result = validate_proxy_event_sequence(
        (0, 0), (0, 0), solution, Family("line", 0), ReflectionRoute(), world, (0, 0), 1920
    )

    assert not result.valid
    assert result.invalid_reason == "B_UNPLANNED_PORTAL"


def test_validate_proxy_sequence_reports_finite_unplanned_circle_clearance():
    from shellshock_detector_yolo.proxy_events import validate_proxy_event_sequence

    world = World(
        lines=(LineObstacle((5, -5), (5, 5)),),
        circles=(CircleObstacle((2.5, 3), 1),),
    )
    solution = Solution(contact=(5, 0), normal=(1, 0), velocity=(1, 0), t1=5, t2=5)

    result = validate_proxy_event_sequence(
        (0, 0), (0, 0), solution, Family("line", 0), ReflectionRoute(), world, (0, 0), 1920
    )

    assert result.valid
    assert result.min_unplanned_clearance == pytest.approx(8.0)


def test_validate_proxy_sequence_walks_planned_portal_before_reflection():
    from shellshock_detector_yolo.proxy_events import validate_proxy_event_sequence

    world = World(
        lines=(LineObstacle((45, -5), (45, 5)),),
        portal_pairs=(PortalPair(Portal("orange", (4, 0), 1), Portal("blue", (30, 0), 1)),),
    )
    # Enter at x=3.1, emerge around x=29.1, then reach the board at x=45
    # after 19 seconds of total unshifted travel. Reflection returns to x=38.
    solution = Solution(contact=(45, 0), normal=(1, 0), velocity=(1, 0), t1=19, t2=7)

    result = validate_proxy_event_sequence(
        (0, 0), (38, 0), solution, Family("line", 0), ReflectionRoute(("0:orange",), ()),
        world, (0, 0), 1920
    )

    assert result.valid
    assert [(event.event_type, event.object_id) for event in result.events] == [
        ("portal", "0:orange"),
        ("reflection", "line:0"),
        ("target", "target"),
    ]


def test_validate_proxy_sequence_rejects_planned_portal_miss():
    from shellshock_detector_yolo.proxy_events import validate_proxy_event_sequence

    world = World(
        lines=(LineObstacle((9, -5), (9, 5)),),
        portal_pairs=(PortalPair(Portal("orange", (4, 4), 1), Portal("blue", (10, 4), 1)),),
    )
    solution = Solution(contact=(9, 0), normal=(1, 0), velocity=(1, 0), t1=9, t2=5)

    result = validate_proxy_event_sequence(
        (0, 0), (4, 0), solution, Family("line", 0), ReflectionRoute(("0:orange",), ()),
        world, (0, 0), 1920
    )

    assert not result.valid
    assert result.invalid_reason == "B_PLANNED_PORTAL_MISS"


def test_validate_proxy_sequence_walks_planned_portal_after_reflection():
    from shellshock_detector_yolo.proxy_events import validate_proxy_event_sequence

    world = World(
        lines=(LineObstacle((5, 10), (15, 10)),),
        portal_pairs=(PortalPair(Portal("orange", (15, 5), 1), Portal("blue", (30, 5), 1)),),
    )
    solution = Solution(contact=(10, 10), normal=(0, 1), velocity=(1, 1), t1=10, t2=7)

    result = validate_proxy_event_sequence(
        (0, 0), (32, 3), solution, Family("line", 0), ReflectionRoute((), ("0:orange",)),
        world, (0, 0), 1920
    )

    assert result.valid
    assert [(event.event_type, event.object_id) for event in result.events] == [
        ("reflection", "line:0"),
        ("portal", "0:orange"),
        ("target", "target"),
    ]

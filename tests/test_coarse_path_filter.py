from shellshock_detector_yolo.coarse_path_filter import (
    PortalHop,
    axis_maybe_reachable,
    path_possible,
    reflection_path_possible,
    segment_possible,
)


def test_axis_reachability_keeps_direct_motion_or_acceleration_recovery():
    assert axis_maybe_reachable(8.0, 1.0, -1.0)
    assert axis_maybe_reachable(8.0, -1.0, 1.0)
    assert not axis_maybe_reachable(8.0, -1.0, -1.0)


def test_segment_rejects_only_an_axis_opposed_by_velocity_and_acceleration():
    assert segment_possible((0.0, 0.0), (10.0, 10.0), ((1.0, -1.0),), (0.0, 1.0))
    assert not segment_possible((0.0, 0.0), (10.0, 10.0), ((1.0, -1.0),), (0.0, -1.0))


def test_portal_to_portal_path_uses_each_exit_as_the_next_segment_start():
    hops = (
        PortalHop("0:orange", (10.0, 0.0), (100.0, 0.0)),
        PortalHop("1:orange", (110.0, 0.0), (200.0, 0.0)),
    )

    result = path_possible((0.0, 0.0), (250.0, 0.0), hops, ((1.0, 0.0),), (0.0, 0.0))

    assert result.valid
    assert result.points == ((0.0, 0.0), (10.0, 0.0), (100.0, 0.0), (110.0, 0.0), (200.0, 0.0), (250.0, 0.0))


def test_portal_to_portal_path_rejects_the_first_definitely_impossible_leg():
    hops = (PortalHop("0:orange", (10.0, 0.0), (100.0, 0.0)),)

    result = path_possible((0.0, 0.0), (50.0, 0.0), hops, ((1.0, 0.0),), (0.0, 0.0))

    assert not result.valid
    assert result.reason == "A_SEGMENT_DIRECTION"


def test_reflection_path_preserves_order_across_portals_on_both_sides():
    before = (PortalHop("0:orange", (-15.0, 0.0), (-10.0, 0.0)),)
    after = (PortalHop("1:orange", (-10.0, 0.0), (-5.0, 0.0)),)

    result = reflection_path_possible(
        (-20.0, 0.0), (0.0, 0.0), (-20.0, 0.0), before, after,
        ((1.0, 0.0),), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0),
    )

    assert result.valid
    assert result.points == ((-20.0, 0.0), (-15.0, 0.0), (-10.0, 0.0), (0.0, 0.0),
                             (-10.0, 0.0), (-5.0, 0.0), (-20.0, 0.0))

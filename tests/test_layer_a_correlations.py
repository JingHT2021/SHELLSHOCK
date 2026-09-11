from math import cos, sin
import numpy as np
import pytest
from shellshock.domain.world import World, RewardZone, LineObstacle, CircleObstacle, Portal, PortalPair
from shellshock.physics.engine import RouteEvent
from shellshock.math2d.geometry import trajectory_position
from shellshock.math2d.continuous import reflect_velocity
from shellshock.planning.layer_a import _route_possible


def test_acceleration_axis_down_then_up_is_impossible_without_reflection():
    world = World(rewards=(RewardZone((10, 10), .1, 2, 'r'),))
    assert not _route_possible((0, 0), (20, 0), world, (RouteEvent('reward', 0),), (0, 10), 0, 0)


def test_vertical_wall_cannot_reverse_acceleration_axis_velocity():
    world = World(lines=(LineObstacle((10, 9), (10, 11)),))
    assert not _route_possible((0, 0), (0, 0), world, (RouteEvent('line', 0),), (0, 10), 0, 0)


def test_horizontal_wall_reflection_preserves_reachable_path():
    a=(0, 10);v=(10, 5);p=(0,0);q=trajectory_position(p,v,a,1)
    reflected=reflect_velocity((10,15),(0,1));target=trajectory_position(q,reflected,a,1.5)
    world=World(lines=(LineObstacle((q[0]-2,q[1]),(q[0]+2,q[1])),))
    assert _route_possible(p,target,world,(RouteEvent('line',0),),a,0,0)


@pytest.mark.parametrize('duration', [.0001, 1., 10000.])
def test_no_finite_time_horizon_is_imposed(duration):
    p=(0.,0.);v=(12.,-15.);a=(2.,9.)
    waypoint=trajectory_position(p,v,a,duration)
    target=trajectory_position(p,v,a,duration*2)
    world=World(rewards=(RewardZone(waypoint,.01,2,'r'),))
    assert _route_possible(p,target,world,(RouteEvent('reward',0),),a,0,0)


def test_portal_translation_preserves_arrival_velocity():
    p=(0.,0.);v=(12.,-15.);a=(2.,9.);q=trajectory_position(p,v,a,.7)
    offset=(100.,50.);exit=tuple(np.add(q,offset));target=tuple(np.add(trajectory_position(p,v,a,1.8),offset))
    world=World(portal_pairs=(PortalPair(Portal('orange',q,.1),Portal('blue',exit,.1)),))
    assert _route_possible(p,target,world,(RouteEvent('portal','0:orange'),),a,0,0)


def test_circle_normal_uses_the_same_contact_point():
    a=(0.,10.);p=(0.,0.);v=(10.,-5.);t=1.;q=trajectory_position(p,v,a,t)
    center=(q[0]+3,q[1]+4);normal=(-.6,-.8)
    outgoing=reflect_velocity((10.,5.),normal);target=trajectory_position(q,outgoing,a,.7)
    world=World(circles=(CircleObstacle(center,5),))
    assert _route_possible(p,target,world,(RouteEvent('circle',0),),a,0,0)


def test_overlapping_rewards_allow_zero_time_passive_events():
    world=World(rewards=(RewardZone((0,0),2,2,'a'),RewardZone((0,0),2,3,'b')))
    assert _route_possible((0,0),(10,0),world,(RouteEvent('reward',0),RouteEvent('reward',1)),(0,10),0,0)


def test_same_contact_prevents_different_incoming_and_outgoing_line_points():
    # Zero force: horizontal reflection must preserve vx and invert vy. At y=10,
    # from (0,0) back to y=0, the only contact for target x=30 is x=15.
    # The segment's central contact range ends before that point.
    world=World(lines=(LineObstacle((0,10),(10,10)),))
    assert not _route_possible((0,0),(30,0),world,(RouteEvent('line',0),),(0,0),0,0)


def test_known_random_reward_and_circle_routes_are_never_rejected():
    rng=np.random.default_rng(20260911)
    for _ in range(30):
        p=tuple(rng.uniform(-10,10,2));v=tuple(rng.uniform(-20,20,2));a=tuple(rng.uniform(-10,10,2))
        t1,t2,t3=rng.uniform(.1,3,3)
        reward=trajectory_position(p,v,a,t1)
        contact=trajectory_position(p,v,a,t1+t2)
        normal=np.array((cos(t3),sin(t3)))
        center=tuple(np.subtract(contact,5*normal))
        outgoing=reflect_velocity(tuple(np.add(v,np.multiply(a,t1+t2))),normal)
        target=trajectory_position(contact,outgoing,a,t3)
        world=World(rewards=(RewardZone(reward,.05,2,'r'),),circles=(CircleObstacle(center,5),))
        assert _route_possible(p,target,world,(RouteEvent('reward',0),RouteEvent('circle',0)),a,0,0)


def test_zero_budget_is_unknown_not_impossible():
    from shellshock.planning.layer_a import analyze_route
    world=World(rewards=(RewardZone((10,-10),2,2,'r'),))
    result=analyze_route((0,0),(20,0),world,(RouteEvent('reward',0),),(0,10),0,0,box_budget=0)
    assert result.possible and result.reason == 'continuous-budget-exhausted'


@pytest.mark.parametrize('kind', ['line','circle'])
def test_narrow_time_velocity_and_contact_witnesses_survive_contraction(kind):
    from shellshock.math2d.reachability import ContactRegion,FlightState,_contract
    from shellshock.math2d.intervals import Interval
    rng=np.random.default_rng(9876)
    def bound(v):
        epsilon=1e-7*max(1.,abs(v))
        return Interval(v-epsilon,v+epsilon)
    for _ in range(30):
        p=tuple(rng.uniform(-100,100,2));v=rng.uniform(-30,30,2);a=rng.uniform(-10,10,2)
        t1,t2=rng.uniform(.01,5,2);q=trajectory_position(p,v,a,t1)
        angle=rng.uniform(-3.14,3.14);normal=np.array((cos(angle),sin(angle)))
        incoming=v+a*t1;outgoing=np.array(reflect_velocity(incoming,normal))
        target=trajectory_position(q,outgoing,a,t2)
        if kind=='circle':
            shape=ContactRegion(tuple(np.subtract(q,7*normal)),7,'circle')
        else:
            tangent=np.array((-normal[1],normal[0]))
            shape=ContactRegion(q,7,'line',(tuple(np.subtract(q,7*tangent)),tuple(np.add(q,7*tangent))))
        regions=[ContactRegion(p),shape,ContactRegion(target)]
        points=[p,q,target];times=[t1,t2];outs=[v,outgoing];ins=[incoming,outgoing+a*t2]
        state=FlightState([[bound(x) for x in point] for point in points],
                          [bound(t) for t in times],[[bound(x) for x in vel] for vel in outs],
                          [[bound(x) for x in vel] for vel in ins])
        _contract(state,regions,a,8)
        assert all(t.contains(w) for t,w in zip(state.times,times))
        assert all(axis.contains(w) for point,witness in zip(state.positions,points) for axis,w in zip(point,witness))


def test_two_axes_cannot_use_different_flight_times():
    from shellshock.math2d.reachability import ContactRegion,FlightState,_contract
    from shellshock.math2d.intervals import Interval,EmptyInterval
    point=lambda x,y:[Interval(x,x),Interval(y,y)]
    whole=Interval(-float('inf'),float('inf'))
    regions=[ContactRegion((0,0)),ContactRegion((10,10))]
    state=FlightState([point(0,0),point(10,10)],[Interval(0,float('inf'))],
                      [point(10,5)],[[whole,whole]])
    with pytest.raises(EmptyInterval):
        _contract(state,regions,(0,0),5)


def test_circle_contact_normal_cannot_be_replaced_by_another_normal():
    from shellshock.math2d.reachability import ContactRegion,FlightState,_contract
    from shellshock.math2d.intervals import Interval,EmptyInterval
    point=lambda x,y:[Interval(x,x),Interval(y,y)]
    whole=Interval(-float('inf'),float('inf'))
    regions=[ContactRegion((-1,2)),ContactRegion((0,0),1,'circle'),ContactRegion((1,0))]
    state=FlightState([point(-1,2),point(0,1),point(1,0)],[Interval(0,float('inf'))]*2,
                      [[whole,whole] for _ in range(2)],[[whole,whole] for _ in range(2)])
    with pytest.raises(EmptyInterval):
        _contract(state,regions,(0,0),5)

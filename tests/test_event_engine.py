from types import SimpleNamespace

def world(**kw):
    base=dict(image_width=1920,circles=(),lines=(),portal_pairs=(),blackholes=(),rewards=())
    base.update(kw); return SimpleNamespace(**base)

def test_blackhole_margin_blocks_near_miss():
    from shellshock.physics.engine import replay_route
    w=world(blackholes=(SimpleNamespace(center=(50,15),radius=6),))
    r=replay_route((0,0),(100,0),(0,0),w,(100,0),())
    assert not r['valid'] and r['reason']=='unexpected-blackhole'

def test_reward_shrink_and_stacking():
    from shellshock.physics.engine import replay_route
    rewards=tuple(SimpleNamespace(center=(x,0),radius=10,multiplier=m,object_id=str(x)) for x,m in [(30,2),(60,3)])
    r=replay_route((0,0),(100,0),(0,0),world(rewards=rewards),(100,0),())
    assert r['valid'] and r['damage_multiplier']==6
    assert len(r['reward_ids'])==2
    assert r['segments']

def test_portal_velocity_and_offset_preserved():
    from shellshock.physics.engine import replay_route,RouteEvent
    p=SimpleNamespace(orange=SimpleNamespace(center=(30,0),radius=10),blue=SimpleNamespace(center=(130,0),radius=10))
    r=replay_route((0,0),(100,0),(0,0),world(portal_pairs=(p,)),(200,0),(RouteEvent('portal','0:orange'),))
    assert r['valid'] and r['portal_count']==1
    assert abs(r['flight_time_seconds']-1)<1e-7
    assert r['segments'][1]['start'][0]-r['event_trace'][0]['point'][0]==100

def test_line_only_central_ninety_percent_reflects():
    from shellshock.physics.engine import replay_route,RouteEvent
    line=SimpleNamespace(start=(0,50),end=(100,50))
    r=replay_route((2,0),(0,100),(0,0),world(lines=(line,)),(2,-20),(RouteEvent('line',0),))
    assert not r['valid'] and r['reason']=='line-endpoint-margin'

def test_circle_root_at_interval_endpoint_is_retained():
    from shellshock.math2d.geometry import parabola_circle_roots
    assert parabola_circle_roots((0,0),(1,0),(0,0),(2,0),1,1)==(1.,)

def test_passive_reward_does_not_turn_next_planned_portal_into_obstacle():
    from shellshock.physics.engine import replay_route,RouteEvent
    p=SimpleNamespace(orange=SimpleNamespace(center=(100,0),radius=20),blue=SimpleNamespace(center=(300,0),radius=20))
    reward=SimpleNamespace(center=(80,0),radius=5,multiplier=2,object_id='r')
    w=world(portal_pairs=(p,),rewards=(reward,))
    route=(RouteEvent('reward',0),RouteEvent('portal','0:orange'))
    r=replay_route((0,0),(10,0),(0,0),w,(350,0),route,max_time=20)
    assert r['valid'],r

def test_unpaired_portal_is_still_a_forbidden_region():
    from shellshock.domain.world import World,Portal
    from shellshock.physics.engine import replay_route
    from shellshock.perception.world import build_world,DetectionBox
    w=build_world([DetectionBox('portal_orange',40,5,20,20,.9)],200,100)
    r=replay_route((0,0),(100,0),(0,0),w,(100,0),())
    assert not r['valid'] and r['reason']=='unexpected-portal'

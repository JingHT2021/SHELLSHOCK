from math import isclose
import numpy as np
from shellshock.math2d.geometry import trajectory_position

def test_three_points_reconstruct_known_accelerated_path():
    from shellshock.math2d.continuous import through_waypoint
    p=(0.,0.); v=(12.,-15.); a=(2.,9.)
    w=trajectory_position(p,v,a,.7); t=trajectory_position(p,v,a,1.8)
    solutions=through_waypoint(p,w,t,a)
    assert any(np.allclose(s.velocity,v) and isclose(s.duration,1.8) for s in solutions)

def test_impossible_waypoint_order_is_rejected():
    from shellshock.math2d.continuous import through_waypoint
    assert through_waypoint((0,0),(10,0),(-10,0),(0,10)) == ()

def test_continuous_a_conservative_for_regions():
    from shellshock.math2d.continuous import regions_maybe_reachable
    # Region overlap permits monotone travel even when center ordering reverses.
    assert regions_maybe_reachable([((0,0),0),((10,0),10),((5,100),0)],(0,10))

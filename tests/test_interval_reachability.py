import pytest
from shellshock.math2d.intervals import Interval, velocity_range, time_ranges


def test_velocity_range_includes_interior_minimum():
    result=velocity_range(Interval(1,1),1,Interval(.5,2))
    assert result.lo <= 2 <= result.hi
    assert result.lo == pytest.approx(2)


def test_time_constraint_keeps_both_quadratic_branches():
    result=time_ranges(Interval(1,1),1,Interval(3,4),Interval(0,float('inf')))
    assert len(result)==2
    assert result[0].lo <= .3 <= result[0].hi
    assert result[1].lo <= 3 <= result[1].hi


def test_zero_force_unbounded_time_limits():
    result=velocity_range(Interval(10,10),0,Interval(0,float('inf')))
    assert result.lo <= 0 and result.hi == float('inf')
    result=velocity_range(Interval(0,0),0,Interval(0,float('inf')))
    assert result.lo <= 0 <= result.hi



def test_random_continuous_witnesses_remain_inside_analytic_ranges():
    import numpy as np
    from shellshock.math2d.intervals import displacement_range
    rng=np.random.default_rng(324)
    for _ in range(1000):
        d,k=rng.uniform(-100,100,2);t=float(10**rng.uniform(-5,5))
        time=Interval(t*.8,t*1.2);displacement=Interval(d-1,d+1)
        velocity=d/t+k*t
        assert velocity_range(displacement,k,time).contains(velocity)
        pieces=time_ranges(displacement,k,Interval(velocity-.1,velocity+.1),time)
        assert any(piece.contains(t) for piece in pieces)
        assert displacement_range(Interval(velocity-.1,velocity+.1),-k,time).contains(d)

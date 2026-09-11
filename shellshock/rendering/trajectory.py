"""Sample validated flight segments; None separates instantaneous portal jumps."""
from math import cos,radians,sin
from shellshock.math2d.ballistics import GRAVITY_AT_REFERENCE,SPEED_PER_POWER_AT_REFERENCE,_wind_acceleration
from shellshock.math2d.geometry import trajectory_position
from shellshock.physics.engine import replay_route

def sample_solution_trajectory(solution,source,image_width,wind_value=0,wind_direction="right",samples=120,world=None):
    segments=solution.get('segments',())
    if not segments and solution.get('manual_preview') and world is not None:
        speed=solution['power']*SPEED_PER_POWER_AT_REFERENCE*image_width/1920
        angle=radians(solution['angle_degrees'])
        velocity=((1 if solution['direction']=='right' else -1)*speed*cos(angle),-speed*sin(angle))
        acceleration=(_wind_acceleration(wind_value,wind_direction,image_width),GRAVITY_AT_REFERENCE*image_width/1920)
        replay=replay_route(source,velocity,acceleration,world,source,preview=True,max_time=solution.get('flight_time_seconds',6))
        solution.update(replay)
        segments=replay['segments']
    if not segments:
        return ()
    duration=sum(s['duration'] for s in segments)
    points=[]
    for segment in segments:
        if points and points[-1]!=tuple(segment['start']):
            points.append(None)
        count=max(2,round(samples*segment['duration']/max(duration,1e-12)))
        points.extend(tuple(float(x) for x in trajectory_position(segment['start'],segment['velocity'],segment['acceleration'],segment['duration']*i/(count-1))) for i in range(count))
    return tuple(points)

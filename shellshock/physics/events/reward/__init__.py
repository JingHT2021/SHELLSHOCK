"""Passive damage regions: each object contributes its multiplier once."""
from shellshock.config.solver import WAYPOINT_RADIUS_SCALE

def trigger_radius(region):
    return region.radius*WAYPOINT_RADIUS_SCALE

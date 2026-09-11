"""Black holes are forbidden regions until an attraction model is calibrated."""
from shellshock.config.solver import UNEXPECTED_MARGIN_PIXELS

def forbidden_radius(blackhole):
    return blackhole.radius+UNEXPECTED_MARGIN_PIXELS

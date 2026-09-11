"""Physical reflection rules; surface radius is never shrunk."""
from shellshock.math2d.continuous import reflect_velocity
from shellshock.config.solver import LINE_CONTACT_FRACTION

def contact_allowed(fraction):
    margin=(1-LINE_CONTACT_FRACTION)/2
    return margin-1e-8<=fraction<=1-margin+1e-8

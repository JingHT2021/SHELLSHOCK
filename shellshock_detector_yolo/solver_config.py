"""Solver tolerances and pixel dimensions at the calibration resolution."""
from math import hypot

from .ballistics import REFERENCE_WIDTH

REFLECTION_LOW_POWER_MARGIN = 2
REFLECTION_MIN_INCIDENCE = 0.08
INTEGER_ANGLE_RADIUS = 2
PORTAL_TRIGGER_RADIUS_SCALE = 0.80
PORTAL_AVOID_MARGIN_AT_REFERENCE = 5.0
CIRCLE_INSIDE_MARGIN_AT_REFERENCE = 20.0
CIRCLE_OUTSIDE_MARGIN_AT_REFERENCE = 40.0
TARGET_ACCEPT_RADIUS_AT_REFERENCE = 24.0
REFLECTION_FALLBACK_TARGET_ACCEPT_RADIUS_AT_REFERENCE = 48.0
LINE_ENDPOINT_MARGIN_AT_REFERENCE = 6.0
MISS_TIE_THRESHOLD_AT_REFERENCE = 2.0
COLLISION_TIME_EPS = 1e-5
ROOT_IMAG_EPS = 1e-7
EQUATION_RESIDUAL_TOL = 1e-5
CIRCLE_SIDE_EPS = 1e-5
MAX_FLIGHT_TIME = 12.0
N_LINE_SCAN = 32
N_CIRCLE_SCAN = 64

# Three-layer reflection search.  The normal limits optimize common scenes;
# the rescue limits are used once only when the normal pass finds no replayed
# integer shot.
# Layer A is intentionally unbounded; this legacy value is retained only for
# compatibility with the superseded solver implementation below.
TOP_K_LAYER_A = 12
TOP_K_LAYER_A_RESCUE = 24
TOP_K_LAYER_B = 20
TOP_K_LAYER_B_RESCUE = 20
TOP_K_LAYER_C = 4
FINAL_INTEGER_CANDIDATE_POOL = 20
FINAL_RESULT_MAX = 5
MAX_FULL_REPLAYS = FINAL_INTEGER_CANDIDATE_POOL
LAYER_A_CIRCLE_REPRESENTATIVE_COUNT = 8
LAYER_B_LINE_PARAMETERS = (1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6)
LAYER_B_CIRCLE_PARAMETER_DELTA = 0.2617993877991494  # pi / 12
LAYER_B_ROOT_WEIGHT = 2.0
LAYER_B_INCIDENCE_WEIGHT = 6.0
LAYER_B_PORTAL_DEPTH_WEIGHT = 2.0
LAYER_B_CLEARANCE_WEIGHT = 2.0
LAYER_B_LINE_COVERAGE_WEIGHT = 2.0
LAYER_B_CIRCLE_COVERAGE_WEIGHT = 4.0
LAYER_B_ANGLE_SPAN_WEIGHT = 0.05
LAYER_B_POWER_SPAN_WEIGHT = 2.0
PORTAL_ROUTE_PRIORITY_BONUS = 100.0
LAYER_C_LINE_WINDOW = 0.16
LAYER_C_CIRCLE_WINDOW = 0.35


def portal_trigger_radius(portal):
    return portal.radius * PORTAL_TRIGGER_RADIUS_SCALE


def portal_avoid_radius(portal, scale):
    return portal.radius + PORTAL_AVOID_MARGIN_AT_REFERENCE * scale


def allowed_circle_sides(source, circle, scale):
    """Return reflection sides compatible with the tank's tolerant circle side."""
    distance = hypot(source[0] - circle.center[0], source[1] - circle.center[1])
    if distance > circle.radius + CIRCLE_OUTSIDE_MARGIN_AT_REFERENCE * scale:
        return ("OUTER",)
    if distance < circle.radius + CIRCLE_INSIDE_MARGIN_AT_REFERENCE * scale:
        return ("INNER",)
    return ("INNER", "OUTER")

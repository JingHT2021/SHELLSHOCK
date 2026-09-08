"""Mode dispatch and a common result envelope; search lives in each solver."""
from math import isfinite
from .normal_solver import solve_normal_integer_shot
from .shot_modes import mode_parts, normalize_mode
from .wormhole_solver import solve_wormhole_integer_shot


def solve_integer_shot(source, target, world, wind_value, wind_direction, image_width,
                       mode, force_power=None):
    if not isfinite(image_width) or image_width <= 0:
        raise ValueError('image_width must be positive')
    if wind_direction not in {'left', 'right'} or not isfinite(wind_value):
        raise ValueError('invalid wind')
    if not all(isfinite(v) for point in (source, target) for v in point):
        raise ValueError('source and target must be finite')
    if force_power is not None and (isinstance(force_power, bool) or not isinstance(force_power, int) or not 1 <= force_power <= 100):
        raise ValueError('force_power must be an integer in 1..100')
    family, variant = mode_parts(mode)
    if force_power is not None and family != 'normal':
        raise ValueError('force_power is supported only for normal mode')
    if family == 'normal':
        result = solve_normal_integer_shot(source, target, world, wind_value, wind_direction, image_width,
                                          arc_preference=variant, force_power=force_power)
    elif family == 'wormhole':
        result = solve_wormhole_integer_shot(source, target, world, wind_value, wind_direction, image_width, arc_preference=variant)
    else:
        from .reflection_solver import solve_reflection_integer_shot
        result = solve_reflection_integer_shot(source, target, world, wind_value, wind_direction, image_width, arc_preference=variant)
    return {**result, 'mode': normalize_mode(mode), 'mode_family': family, 'mode_variant': variant}

"""Mode dispatch and a common result envelope; search lives in each solver."""
from math import isfinite
from .normal_solver import solve_normal_integer_shot
from .muzzle_geometry import muzzle_position
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
        solve = lambda origin: solve_normal_integer_shot(
            origin, target, world, wind_value, wind_direction, image_width,
            arc_preference=variant, force_power=force_power,
        )
    elif family == 'wormhole':
        solve = lambda origin: solve_wormhole_integer_shot(
            origin, target, world, wind_value, wind_direction, image_width,
            arc_preference=variant,
        )
    else:
        from .reflection_solver import solve_reflection_integer_shot
        solve = lambda origin: solve_reflection_integer_shot(
            origin, target, world, wind_value, wind_direction, image_width,
            arc_preference=variant,
        )
    result = solve(source)
    if family in {'normal', 'wormhole'}:
        # The projectile starts at the barrel tip, whose location depends on
        # the selected angle. Re-solve until the angle used to place the
        # muzzle agrees with the returned angle (bounded for safety).
        muzzle_source = source
        for _ in range(2):
            if result.get('status') != 'reachable':
                break
            corrected_source = muzzle_position(
                source,
                str(result['direction']),
                float(result['angle_degrees']),
                image_width,
            )
            if corrected_source == muzzle_source:
                break
            muzzle_source = corrected_source
            result = solve(muzzle_source)
    return {**result, 'mode': normalize_mode(mode), 'mode_family': family, 'mode_variant': variant}

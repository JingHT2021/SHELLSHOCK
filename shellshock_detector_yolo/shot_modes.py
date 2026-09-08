"""Canonical mode families, with aliases for existing scripts."""
ALIASES = {
    'normal': 'normal_low', 'low_arc': 'normal_low', 'high_arc': 'normal_high',
    'wormhole': 'wormhole_low', 'wormhole_low_arc': 'wormhole_low',
    'wormhole_high_arc': 'wormhole_high', 'reflection': 'reflection_low',
}


def normalize_mode(mode):
    mode = ALIASES.get(mode, mode)
    if mode not in {f'{family}_{variant}' for family in ('normal', 'wormhole', 'reflection') for variant in ('low', 'high')}:
        raise ValueError(f'unknown shot mode: {mode}')
    return mode


def mode_parts(mode):
    return tuple(normalize_mode(mode).split('_'))

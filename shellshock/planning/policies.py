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


def select_mode(key,current_mode="normal_low"):
    key=key.lower()
    if key in {'page up','page down'}:
        family,_=mode_parts(current_mode)
        return f"{family}_{'high' if key=='page up' else 'low'}"
    aliases={'r':'reflection_low','h':'wormhole_low','t':'normal_low'}
    return normalize_mode(aliases.get(key,key if key in {"normal_low","normal_high","wormhole_low","wormhole_high","reflection_low","reflection_high","reflection"} else current_mode))

"""Paired portals preserve velocity and entry offset."""
def portal_map(world):
    return {f'{i}:{color}':(p,partner,f'{i}:{other}')
            for i,pair in enumerate(world.portal_pairs)
            for color,p,other,partner in (('orange',pair.orange,'blue',pair.blue),('blue',pair.blue,'orange',pair.orange))}

def translate(contact,entry,exit):
    return tuple(exit.center[i]+contact[i]-entry.center[i] for i in (0,1))

"""Rank only physically verified shots by rewards and accuracy."""
def priority(result):
    p=bool(result.get('portal_count',0));r=bool(result.get('reflection_count',0))
    return (float(result.get('damage_multiplier',1)),int(p)+int(r),int(p))


def select_result(results, arc_preference):
    best_priority=max(map(priority,results))
    pool=[r for r in results if priority(r)==best_priority]
    best_miss=min(r['miss_distance'] for r in pool)
    if any(r.get('reflection_obstacle',{}).get('kind')=='circle' for r in pool):
        reliable=[r for r in pool if r['miss_distance']<=best_miss+2.]
        return min(reliable,key=lambda r:(-r['power'] if r.get('reflection_obstacle',{}).get('kind')=='circle' else 0,
                                         r['miss_distance'],-r['clearance']))
    return min(pool,key=lambda r:(r['miss_distance'], -r['angle_degrees'] if arc_preference=='high' else r['power'],
                                 -r['clearance']))


"""Generate bounded event routes with reward and simple-route coverage."""
from itertools import permutations
from shellshock.physics.engine import RouteEvent
from shellshock.physics.events.portal import portal_map
from shellshock.planning.iteration import _round_robin
def candidate_routes(world, family, limit=None):
    """Fair bounded event topologies; reserve simpler fallbacks at every tier."""
    from heapq import heappush,heappop
    from math import prod
    portals=[RouteEvent('portal',key) for key in portal_map(world)]
    surfaces=[RouteEvent(kind,i) for kind,items in (('circle',world.circles),('line',world.lines)) for i in range(len(items))]
    both=[(p,s) for p in portals for s in surfaces]
    double=[(p,q) for p in portals for q in portals if p.index!=q.index]
    categories=[both,[(*pair,s) for pair in double for s in surfaces],[(p,) for p in portals],double,[(s,) for s in surfaces],[()]]
    def permitted(base):
        if family=='wormhole':return any(e.kind=='portal' for e in base)
        if family=='reflection':return any(e.kind in {'circle','line'} for e in base)
        return bool(world.rewards) or not base
    categories=[[base for base in category if permitted(base)] for category in categories]
    # Best products first, with empty subset explicitly reserved as a fallback.
    full=tuple(range(len(world.rewards)));queue=[(-prod(world.rewards[i].multiplier for i in full),full)]
    subsets=[];seen_subsets={full}
    while queue and len(subsets)<16:
        _,subset=heappop(queue)
        if subset:subsets.append(subset)
        for i in subset:
            child=tuple(j for j in subset if j!=i)
            if child not in seen_subsets:
                seen_subsets.add(child);heappush(queue,(-prod(world.rewards[j].multiplier for j in child),child))
    subsets.append(())
    def subset_routes(subset):
        rewards=tuple(RouteEvent('reward',i) for i in subset)
        return _round_robin(_round_robin(permutations((*rewards,*base)) for base in category) for category in categories)
    routes=[];seen=set()
    for route in _round_robin(subset_routes(subset) for subset in subsets):
        if route in seen:continue
        if limit is not None and len(routes)>=limit:return routes,True
        seen.add(route);routes.append(route)
    return routes,bool(queue)


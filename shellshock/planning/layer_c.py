"""Local integer candidate generation, interleaved across continuous branches."""
from itertools import product
from shellshock.planning.iteration import _round_robin

def integer_candidates(route,legal,family,variant):
    def neighborhood(seed):
        angle,power,direction,*lineage=seed
        branch_id=lineage[0] if lineage else None
        offsets=sorted(product(range(-2,3),repeat=2),key=lambda d:d[0]**2+d[1]**2)
        for da,dp in offsets:
            ia,ip=round(angle)+da,round(power)+dp
            if 0<=ia<=90 and 1<=ip<=100 and not (family=='normal' and variant=='high' and ip<90):
                yield route,direction,ia,ip,branch_id
    return _round_robin(neighborhood(seed) for seed in legal)

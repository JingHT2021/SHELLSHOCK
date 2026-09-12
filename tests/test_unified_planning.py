from types import SimpleNamespace


def test_b_selection_keeps_three_per_route_then_best_forty_global():
    from shellshock.planning.unified import select_b_seeds

    seeds = [
        {'route_id': f'A-{route}', 'seed_id': f'{route}-{index}', 'score': score}
        for route in range(1, 5)
        for index, score in enumerate((100 - route - index for index in range(5)))
    ]
    selected = select_b_seeds(seeds, per_route=3, global_limit=40)

    assert len(selected) == 12
    assert all(sum(item['route_id'] == route for item in selected) <= 3
               for route in {item['route_id'] for item in selected})


def test_c2_selection_prefers_distinct_routes_and_refills_after_failure():
    from shellshock.planning.unified import select_c2_candidates

    candidates = [
        {'route_id': 'A-1', 'candidate_id': 'C-1', 'score': 100},
        {'route_id': 'A-1', 'candidate_id': 'C-2', 'score': 99},
        {'route_id': 'A-2', 'candidate_id': 'C-3', 'score': 98},
        {'route_id': 'A-3', 'candidate_id': 'C-4', 'score': 97},
        {'route_id': 'A-4', 'candidate_id': 'C-5', 'score': 96},
        {'route_id': 'A-5', 'candidate_id': 'C-6', 'score': 95},
        {'route_id': 'A-6', 'candidate_id': 'C-7', 'score': 94},
    ]

    selected = select_c2_candidates(candidates, limit=5, failed={'C-3'})

    assert [item['candidate_id'] for item in selected] == ['C-1', 'C-4', 'C-5', 'C-6', 'C-7']
    assert len({item['route_id'] for item in selected}) == 5

def test_reward_portal_reflection_priority():
    from shellshock.planning.unified import select_result
    def shot(mult,p,r,miss=1,power=90):
        return dict(damage_multiplier=mult,portal_count=p,reflection_count=r,miss_distance=miss,power=power,clearance=20,angle_degrees=60)
    a=shot(2,1,1); b=shot(2,0,1); c=shot(2,0,0); d=shot(1,1,1)
    assert select_result([d,c,b,a],'low') is a

def test_circle_prefers_more_power_for_comparable_error():
    from shellshock.planning.unified import select_result
    a=dict(damage_multiplier=1,portal_count=0,reflection_count=1,miss_distance=1,power=70,clearance=20,angle_degrees=60,reflection_obstacle={'kind':'circle'})
    b={**a,'power':95,'miss_distance':1.5}
    assert select_result([a,b],'low') is b

def test_portal_mode_generates_reflection_combinations():
    from shellshock.planning.unified import candidate_routes
    from shellshock.perception.world import World,Portal,PortalPair,LineObstacle
    w=World(portal_pairs=(PortalPair(Portal('orange',(20,20),10),Portal('blue',(50,20),10)),),lines=(LineObstacle((10,30),(80,30)),))
    routes,truncated=candidate_routes(w,'wormhole',200)
    assert any(any(e.kind=='portal' for e in r) and any(e.kind=='line' for e in r) for r in routes)

def test_route_budget_preserves_simple_and_opposite_portal_fallbacks():
    from shellshock.planning.unified import candidate_routes
    from shellshock.domain.world import World,Portal,PortalPair,LineObstacle,RewardZone
    w=World(portal_pairs=(PortalPair(Portal('orange',(20,20),10),Portal('blue',(50,20),10)),),lines=(LineObstacle((10,30),(80,30)),),rewards=tuple(RewardZone((i*20,10),5,2,str(i)) for i in range(4)))
    routes,_=candidate_routes(w,'wormhole',192)
    simple=[r for r in routes if len(r)==1 and r[0].kind=='portal']
    assert {r[0].index for r in simple}=={'0:orange','0:blue'}


def test_route_limit_none_enumerates_without_the_legacy_default_cap():
    from shellshock.planning.unified import candidate_routes
    from shellshock.domain.world import World, Portal, PortalPair, LineObstacle, RewardZone

    w = World(
        portal_pairs=(PortalPair(Portal('orange', (20, 20), 10), Portal('blue', (50, 20), 10)),),
        lines=tuple(LineObstacle((10 + i, 30), (80 + i, 30)) for i in range(4)),
        rewards=tuple(RewardZone((i * 20, 10), 5, 2, str(i)) for i in range(4)),
    )
    routes, truncated = candidate_routes(w, 'wormhole', None)

    assert routes
    assert truncated is False

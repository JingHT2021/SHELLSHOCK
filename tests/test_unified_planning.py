from types import SimpleNamespace

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

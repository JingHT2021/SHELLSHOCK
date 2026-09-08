# Layer B Proxy Event Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Layer B into a bounded constant-acceleration proxy that rejects wrong portal/reflector event orders and ranks stable candidates before Layer C.

**Architecture:** Keep the existing exact fixed-contact quadratic solver as the constant-acceleration mathematical kernel, because it covers the midpoint-parabola model without an `x`-monotonicity singularity. Add a focused `proxy_events.py` module for analytic segment event collection and a focused `reflection_proxy.py` module for candidate recovery, validation, neighborhood scoring, and Top-K selection. `reflection_filter.py` remains the Layer A adapter and delegates Layer B work; `reflection_solver.py` supplies real wind-plus-gravity acceleration.

**Tech Stack:** Python 3.13, NumPy, SciPy polynomial/Brent helpers already used by the project, pytest.

---

### Task 1: Analytic proxy-event primitives

**Files:**
- Create: `shellshock_detector_yolo/proxy_events.py`
- Create: `tests/test_proxy_events.py`

- [ ] **Step 1: Write failing tests** for planned portal entry, an earlier unplanned portal, an earlier wrong line, an earlier wrong circle, and target-before-late-obstacle ordering. Use real `World`, `PortalPair`, `LineObstacle`, and `CircleObstacle` values and assert `ProxyEvent` time/type/id ordering.
- [ ] **Step 2: Run tests and verify RED** with `python -m pytest -q tests/test_proxy_events.py`; expected failure is the missing module/API.
- [ ] **Step 3: Implement minimal primitives:**

```python
@dataclass(frozen=True)
class ProxyEvent:
    event_type: str
    object_id: str
    time: float
    planned: bool
    point: Point
    margin: float

def collect_segment_events(start, velocity, acceleration, duration, world, image_width,
                           *, planned_portal=None, planned_reflector=None,
                           ignored_portal=None) -> tuple[ProxyEvent, ...]: ...
```

Use the existing analytic `parabola_circle_roots`, `parabola_segment_roots`, and `_entry_time` helpers. Trigger radius applies only to the planned portal; every other portal uses the avoidance radius. Skip only the just-emerged portal at segment time zero.
- [ ] **Step 4: Run the new tests and verify GREEN.**

### Task 2: Planned route state machine

**Files:**
- Modify: `shellshock_detector_yolo/proxy_events.py`
- Modify: `tests/test_proxy_events.py`

- [ ] **Step 1: Write failing tests** for `Self -> portal -> reflector -> portal -> target`, planned-portal miss, wrong-first-portal, and wrong-reflector-first.
- [ ] **Step 2: Verify RED** with the focused test names.
- [ ] **Step 3: Implement:**

```python
@dataclass(frozen=True)
class ProxyEventValidation:
    valid: bool
    events: tuple[ProxyEvent, ...]
    invalid_reason: str | None
    planned_portal_depth: float
    min_unplanned_clearance: float

def validate_proxy_event_sequence(source, target, solution, family, route, world,
                                  acceleration, image_width) -> ProxyEventValidation: ...
```

Advance analytically through each planned portal, carry velocity and remaining duration, apply the physical portal translation, require the selected reflector to be the next solid event at `solution.t1`, reflect using the planned normal, then validate post-reflection portals and target ordering. Map failures to the plan's `B_*` reasons.
- [ ] **Step 4: Verify GREEN** for all proxy-event tests.

### Task 3: Layer B candidate model and stability ranking

**Files:**
- Create: `shellshock_detector_yolo/reflection_proxy.py`
- Modify: `tests/test_reflection_filter.py`

- [ ] **Step 1: Write failing tests** proving that invalid event sequences never reach scoring, all valid algebraic branches are preserved, circle seed neighborhoods aggregate `valid_count/angle_span/power_span`, and hard-invalid candidates cannot be rescued by score.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement the candidate model and evaluator:**

```python
@dataclass(frozen=True)
class LayerBProxyCandidate:
    coarse: CoarsePathCandidate
    solution: object
    branch: int
    score_b: float
    root_stability: float
    incidence: float
    planned_portal_depth: float
    min_unplanned_clearance: float
    neighborhood_valid_count: int
    neighborhood_angle_span: float
    neighborhood_power_span: float
    proxy_events: tuple[ProxyEvent, ...]
    valid: bool = True
    invalid_reason: str | None = None
```

Evaluate exactly three line parameters or the three-point neighborhood of every Layer A circle seed. Reject non-finite/non-positive/out-of-range power, invalid angle, weak incidence, excessive equation residual, invalid circle side, and invalid proxy event sequence. Rank remaining candidates by normalized power margin, incidence, portal depth, clearance, root separation, and neighborhood spans; group/deduplicate equivalent roots before Top-K.
- [ ] **Step 4: Verify GREEN** for focused Layer B tests.

### Task 4: Integrate real acceleration and diagnostics

**Files:**
- Modify: `shellshock_detector_yolo/reflection_filter.py`
- Modify: `shellshock_detector_yolo/reflection_solver.py`
- Modify: `shellshock_detector_yolo/solver_config.py`
- Modify: `tests/test_yolo_reflection_solver.py`

- [ ] **Step 1: Write failing integration tests** asserting Layer B receives horizontal wind, reports event counters and invalid reasons, remains bounded by Top-K, and avoids strict replay during proxy ranking.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Delegate `rank_proxy_candidates` to `reflection_proxy.evaluate_layer_b`; pass the real acceleration instead of `(0, gravity)`. Add explicit Layer B tolerances and score weights to `solver_config.py` without changing Layer C's strict replay authority.
- [ ] **Step 4: Verify GREEN** with the Layer A/B/solver integration suite.

### Task 5: Full verification and performance guard

**Files:**
- Modify only if a regression is exposed by tests.

- [ ] **Step 1: Run** `python -m pytest -q tests/test_proxy_events.py tests/test_reflection_filter.py tests/test_coarse_path_filter.py tests/test_wormhole_solver.py tests/test_yolo_reflection_solver.py tests/test_yolo_solver_package.py` and require zero failures.
- [ ] **Step 2: Run the performance test three times** to distinguish a stable regression from scheduler noise; require each solver-reported `total_seconds` to remain below its existing 1.5-second gate.
- [ ] **Step 3: Run** `git diff --check` and inspect `git diff --stat` plus the complete diff for accidental Layer A regressions or unrelated edits.


# Reflection Shot Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add R/T-selectable reflection and normal ShellShock Live aiming, automatically falling back to normal mode when no obstacles are detected and always executing integer angle and power controls selected by a local trajectory search.

**Architecture:** Keep the calibrated normal-shot analytic solver as the source of continuous candidates. Add a pure reflection module that represents trajectory segments, finds the first obstacle contact, performs one mirror reflection, and numerically searches circle/line surface parameters and segment times. Shared integer refinement re-simulates nearby integer controls; app.py supplies detected obstacle geometry and converts only the chosen integer solution into an aim-disc click.

**Tech Stack:** Python 3, NumPy, OpenCV obstacle detection, standard-library math/unittest, existing keyboard and Win32 click integration.

---

## Locked interaction contract

- Q records the self position. It is the only action that changes the source coordinate.
- R selects reflection mode; it does not capture a screenshot.
- T selects normal mode.
- Each E update records the current mouse position as target, captures a fresh frame, recalculates, and executes the click whenever Q has already recorded a self position.
- In reflection mode, absent circle/line geometry automatically selects the normal-mode solver for that E update. Terminal output explicitly says that this fallback occurred.
- PageUp/PageDown retain their normal-mode arc preferences. Reflection mode always chooses the minimum-power valid one-bounce result.
- Every externally selected and printed angle_degrees and power is a Python int in the game range: 0..90 and 0..100.

## File structure

| File | Responsibility |
| --- | --- |
| shellshock_detector/reflection.py | Pure vector math, normals, numerical continuous reflection candidates, collision/event simulation, and integer refinement. |
| shellshock_detector/ballistics.py | Normal-shot integer refinement while retaining solve_target and all calibrated equations. |
| shellshock_detector/app.py | Detect geometry on the fresh frame, route normal/reflection selection, return selected integer solution and click point. |
| detect_shellshock.py | Register Q, R, T, E and print mode-specific results/fallbacks. |
| tests/test_reflection.py | Synthetic geometry tests for surfaces, collisions, ranking, and integer correction. |
| tests/test_ballistics.py | Normal-mode integer-local-search regression tests. |
| tests/test_app.py | Mode routing, geometry passing, and integer click tests. |
| README.md | Updated hotkeys, modes, and integer-control behavior. |

## Numeric conventions and rejection rules

- Convert screen coordinates to an internal physical frame (x right, y up); convert to screen coordinates only for output.
- Use existing calibrated acceleration (wind_acceleration, -gravity), speed-per-power and image-width scaling.
- Treat each pink circle as a solid disk. Treat each detected line as a finite reflective segment with a named collision radius in reflection.py.
- Reject circle contacts with abs(dot(v_hit, normal)) below CIRCLE_MIN_NORMAL_SPEED. Reject line contacts inside LINE_ENDPOINT_MARGIN of either end.
- Use named, image-width-scaled TARGET_HIT_RADIUS_AT_REFERENCE to classify an integer trajectory as a target hit. If legal nearby integers miss it, retain the least-error legal result as closest; never click an invalid reflection sequence.
- Valid one-bounce sequence: no collision before t1; the planned obstacle is first at t1; no collision after reflection through target-evaluation time t1+t2. Ambiguous simultaneous obstacle contacts are invalid.

### Task 1: Normal integer refinement

**Files:** Modify tests/test_ballistics.py; modify shellshock_detector/ballistics.py.

- [ ] **Step 1: Write failing tests**

~~~python
from shellshock_detector.ballistics import refine_normal_integer_shot

def test_normal_refinement_returns_integer_best_nearby_controls():
    shot = refine_normal_integer_shot(
        0, 0, 1000, 0, 0, "right", 1920,
        theory_angle=45.4, theory_power=63.7, radius=2,
    )
    assert shot["angle_degrees"] == 45
    assert shot["power"] == 64
    assert type(shot["angle_degrees"]) is int
    assert type(shot["power"]) is int
    assert shot["target_error"] >= 0

def test_normal_refinement_clips_neighborhood_to_game_limits():
    shot = refine_normal_integer_shot(0, 0, 1, 0, 0, "right", 1920, 0.1, 0.1, radius=3)
    assert 0 <= shot["angle_degrees"] <= 90
    assert 0 <= shot["power"] <= 100
~~~

- [ ] **Step 2: Run the tests and verify red**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_ballistics.py -q

Expected: import failure for refine_normal_integer_shot.

- [ ] **Step 3: Implement the minimal pure API**

~~~python
def refine_normal_integer_shot(..., theory_angle: float, theory_power: float, radius: int = 2) -> dict[str, object]:
    # Enumerate clipped Cartesian product of nearby integer angle/power values.
    # Recalculate each real trajectory, find positive target-height crossing
    # nearest target x, rank by (target_error, power, angle).
    # Return integer controls only.
~~~

Reuse current left/right conversion. Do not change solve_target continuous return values or fixed-power arc math.

- [ ] **Step 4: Run focused tests**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_ballistics.py -q

Expected: all ballistics tests pass.

### Task 2: Obstacle and first-contact primitives

**Files:** Create tests/test_reflection.py; create shellshock_detector/reflection.py.

- [ ] **Step 1: Write failing geometry/collision tests**

~~~python
from shellshock_detector.obstacle_geometry import CircleObstacle, LineObstacle, ObstacleGeometry
from shellshock_detector.reflection import first_collision, reflect_velocity

def test_circle_collision_returns_outward_normal_at_first_entry():
    geometry = ObstacleGeometry([CircleObstacle((50, 0), 10)], [])
    event = first_collision((0, 0), (100, 0), (0, 0), 1.0, geometry)
    assert event.obstacle_index == 0
    assert event.kind == "circle"
    assert event.point == pytest.approx((40, 0))
    assert event.normal == pytest.approx((-1, 0))

def test_line_reflection_reverses_only_normal_velocity():
    assert reflect_velocity((4, -3), (0, 1)) == pytest.approx((4, 3))
~~~

Also add failing cases for near-tangent circle rejection, line endpoint margin, and an unrelated obstacle reached before a planned one.

- [ ] **Step 2: Run tests and verify red**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_reflection.py -q

Expected: missing reflection module.

- [ ] **Step 3: Implement bounded event roots**

~~~python
@dataclass(frozen=True)
class CollisionEvent:
    obstacle_index: int
    kind: Literal["circle", "line"]
    time: float
    point: tuple[float, float]
    normal: tuple[float, float]

def reflect_velocity(v, n):
    return subtract(v, scale(n, 2 * dot(v, n)))

def first_collision(start, velocity, acceleration, end_time, geometry, ignored=None):
    # Solve circle quadratic and line signed-distance roots; retain only
    # bounded, positive-time intersections and return the unique earliest event.
~~~

Represent all values in the physical frame. Convert detected obstacle endpoints/centers at the public boundary. Simultaneous events within EVENT_TIME_EPSILON are rejected as ambiguous.

- [ ] **Step 4: Run focused tests**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_reflection.py -q

Expected: all primitive tests pass.

### Task 3: Continuous one-reflection solver

**Files:** Modify tests/test_reflection.py; modify shellshock_detector/reflection.py.

- [ ] **Step 1: Add failing synthetic-scene tests**

~~~python
def test_solver_finds_one_valid_line_bounce_with_theory_diagnostics():
    result = solve_single_reflection(source, target, acceleration, geometry, limits, 1920)
    assert result["status"] == "reachable"
    assert result["obstacle"]["kind"] == "line"
    assert result["reflection_point"] is not None
    assert result["theory"]["power"] > 0

def test_solver_ranks_valid_obstacles_by_lower_continuous_power():
    result = solve_single_reflection(source, target, acceleration, two_obstacle_geometry, limits, 1920)
    assert result["obstacle"]["index"] == 1
~~~

Add tests that reject another obstacle on either segment and require exactly one event in every reachable result.

- [ ] **Step 2: Run tests and verify red**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_reflection.py -q

Expected: missing solve_single_reflection API.

- [ ] **Step 3: Implement deterministic numeric search**

~~~python
def solve_single_reflection(source, target, acceleration, geometry, limits, image_width):
    candidates = []
    for obstacle in obstacle_descriptors(geometry):
        for parameter, t1, t2 in optimize_surface_times(obstacle, source, target, acceleration):
            point, normal = obstacle.point_and_normal(parameter)
            v0 = (point - source) / t1 - 0.5 * acceleration * t1
            v1 = (point - source) / t1 + 0.5 * acceleration * t1
            v2 = reflect_velocity(v1, normal)
            if residual(point + v2 * t2 + 0.5 * acceleration * t2**2, target) > CONTINUOUS_RESIDUAL:
                continue
            candidates.append(validate_single_bounce(...))
    return min(valid_candidates, key=lambda x: (x["theory"]["power"], x["theory"]["angle_degrees"]))
~~~

For a line search (lambda, t1, t2), with lambda in 0..1. For a circle search (phi, t1, t2), with phi in 0..2*pi. Use deterministic coarse sampling then bounded coordinate descent; do not add SciPy. Reject non-finite values, invalid time/range, tangent/endpoint contact, wrong collision event sequence, and residual mismatch before ranking.

- [ ] **Step 4: Run focused tests**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_reflection.py -q

Expected: all reflection tests pass.

### Task 4: Reflection integer correction

**Files:** Modify tests/test_reflection.py; modify shellshock_detector/reflection.py.

- [ ] **Step 1: Write failing re-simulation tests**

~~~python
def test_integer_refinement_replays_exactly_one_bounce_on_selected_obstacle():
    result = refine_reflection_integer_shot(theory_result, source, target, acceleration, geometry, limits)
    assert type(result["angle_degrees"]) is int
    assert type(result["power"]) is int
    assert result["events"] == [result["obstacle"]]
    assert result["target_error"] >= 0

def test_integer_refinement_rejects_a_nearby_control_hitting_another_obstacle():
    result = refine_reflection_integer_shot(theory_result, source, target, acceleration, geometry, limits, radius=3)
    assert result["obstacle"] == theory_result["obstacle"]
~~~

- [ ] **Step 2: Run tests and verify red**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_reflection.py -q

Expected: missing refine_reflection_integer_shot API.

- [ ] **Step 3: Implement full event replay**

~~~python
def refine_reflection_integer_shot(theory, source, target, acceleration, geometry, limits, radius=2):
    legal = []
    for angle in range(round(theory_angle) - radius, round(theory_angle) + radius + 1):
        for power in range(round(theory_power) - radius, round(theory_power) + radius + 1):
            replay = simulate_until_target_evaluation(angle, power, ...)
            if replay.has_exactly_one_event_on(theory["obstacle"]):
                legal.append(replay)
    return min(legal, key=lambda x: (x["target_error"], x["power"], x["angle_degrees"]))
~~~

Use first_collision from Task 2, not an analytic shortcut. Reject zero/two-plus events, a different obstacle, tangent/endpoint event, or non-positive target time. Return JSON-safe unreachable status if no legal integer candidate exists.

- [ ] **Step 4: Run focused tests**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_reflection.py -q

Expected: all reflection tests pass.

### Task 5: App routing and fresh geometry capture

**Files:** Modify tests/test_app.py; modify shellshock_detector/app.py.

- [ ] **Step 1: Write failing application tests**

~~~python
def test_normal_mode_click_uses_integer_refined_controls():
    solution, _ = aim_click_for_target(result, 2000, 800, shot_mode="normal")
    assert type(solution["selected"]["angle_degrees"]) is int
    assert type(solution["selected"]["power"]) is int

def test_reflection_mode_receives_geometry_and_returns_diagnostics():
    solution, _ = aim_click_for_target(result, 2000, 800, shot_mode="reflection", geometry=geometry)
    assert solution["selected"]["obstacle"] is not None
    assert solution["selected"]["reflection_point"] is not None
~~~

Add a test that reflection mode with empty geometry executes the normal integer solver and returns selected["fallback"] == "normal:no-obstacle". Keep legacy minimum/maximum as aliases for normal selections.

- [ ] **Step 2: Run tests and verify red**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_app.py -q

Expected: failure naming geometry or selected result.

- [ ] **Step 3: Implement routing**

~~~python
def aim_click_for_target(result, target_x, target_y, manual_self=None, shot_mode="normal", geometry=None):
    if shot_mode == "reflection" and geometry and (geometry.circles or geometry.lines):
        selected = solve_and_refine_reflection(..., geometry=geometry)
    else:
        theory = existing_normal_selection(...)
        selected = refine_normal_integer_shot(..., theory["angle_degrees"], theory["power"])
    return {**solution, "selected": selected}, disc_click_point(...selected integer controls...)
~~~

Capture once in aim_at_screen_position; pass that same frame both to process_capture and detect_pink_obstacle_geometry. When selected mode is reflection but no geometry is available, add fallback="normal:no-obstacle" to the selected result and use the normal integer solver. Do not alter wind fallback, coordinate conversion, training labels, or click safety checks.

- [ ] **Step 4: Run focused tests**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_app.py -q

Expected: all app tests pass.

### Task 6: Hotkeys, terminal output, and README

**Files:** Create tests/test_hotkeys.py; modify detect_shellshock.py; modify README.md.

- [ ] **Step 1: Write a failing output test**

~~~python
from detect_shellshock import describe_selected_shot

def test_reflection_description_has_all_required_fields():
    text = describe_selected_shot({
        "mode": "reflection", "angle_degrees": 42, "power": 61,
        "obstacle": {"kind": "circle", "index": 0}, "reflection_point": (500.0, 400.0),
        "theory": {"angle_degrees": 42.3, "power": 60.7}, "target_error": 3.2,
    })
    assert "angle=42" in text and "power=61" in text and "circle #0" in text
~~~

- [ ] **Step 2: Run test and verify red**

Run: ./.venv/Scripts/python.exe -m pytest tests/test_hotkeys.py -q

Expected: missing describe_selected_shot.

- [ ] **Step 3: Implement state, bindings, and output**

~~~python
def use_reflection_shot():
    nonlocal shot_mode
    shot_mode = "reflection"

def use_normal_shot():
    nonlocal shot_mode
    shot_mode = "normal"

keyboard.add_hotkey("r", use_reflection_shot)
keyboard.add_hotkey("t", use_normal_shot)
~~~

Reflection output must state integer angle/power, circle/line index, reflection point, continuous theory angle/power, and final target error. A no-obstacle reflection request must state fallback=normal:no-obstacle followed by the normal integer angle/power and corrected error. Normal output must state integer angle/power and corrected error. Update readiness text and README key table; do not retain the former screenshot-only R binding.

- [ ] **Step 4: Run the full suite**

Run: ./.venv/Scripts/python.exe -m pytest -q

Expected: all tests pass.

### Task 7: Manual in-game verification

**Files:** No source changes unless a reproducible calibration issue is found.

- [ ] **Step 1: Launch and check hotkey state**

Run: ./.venv/Scripts/python.exe detect_shellshock.py --output-dir output

Expected: readiness output lists Q=self, E=target/recalculate/execute, R=reflection, T=normal.

- [ ] **Step 2: Verify normal integer output**

Press T, record self with Q, point to a target and press E.

Expected: printed final angle and power have no decimal portion and corrected target error is finite.

- [ ] **Step 3: Verify circle and line bounces**

For a reachable target behind each type, press R then E.

Expected: output identifies exactly one obstacle and a reflection point. No click occurs if no valid one-bounce integer solution exists.

- [ ] **Step 4: Record any calibration observations**

Save capture and diagnostics under output. Change only named tolerance/calibration constants, with a new regression test; do not globally loosen collision rules.

## Plan self-review

- Coverage: Tasks 1/5 retain normal logic plus integer search. Tasks 2–4 cover circle/line reflection formulas, required filters, minimum continuous power, and integer revalidation. Task 6 covers R/T/C and required output.
- Resolution: E executes the current mode; C has the former R capture behavior. An integer reflection solution must retain the planned obstacle and exactly one collision event, otherwise it is rejected.
- Repository note: git status currently reports that this workspace is not a Git repository. The plan therefore omits commit steps rather than prescribing failing repository mutations.

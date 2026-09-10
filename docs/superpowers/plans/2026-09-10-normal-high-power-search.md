# Normal High Local Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search power 94--100 and angle ±3° around the power-100 theoretical high arc, returning the replay candidate with minimum miss distance.

**Architecture:** Keep the bounded search in `normal_solver.py`. Continue using `solve_target` for the power-100 high-arc seed and `replay_portal_shot` as the authority for candidate validity and error.

**Tech Stack:** Python 3, pytest, existing ShellShock ballistic and replay modules.

---

### Task 1: Define the behavior with regression tests

**Files:**
- Create: `tests/test_normal_solver.py`

- [x] **Step 1: Add a failing bounded-search test**

Patch `solve_target` to return a 70° high-arc seed, patch
`solve_ballistic_for_speed` and assert it is not called, and record the power
and angle passed to replay. Assert powers `94..100`, angles `67..73` for each
power, 49 candidates, and selection of the mocked minimum-error power.

- [x] **Step 2: Verify the old implementation fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_normal_solver.py -v`

Expected: failure because the old solver uses a different power range and
recomputes an angle for each power.

- [x] **Step 3: Add selection and forced-power coverage**

Assert accuracy-first ordering with high-angle, clearance, and power
tie-breakers. Assert `force_power=95` replays only that power over angles
`67..73`, and `force_power=100` uses the same theoretical neighborhood.

### Task 2: Implement the local search

**Files:**
- Modify: `shellshock_detector_yolo/normal_solver.py`
- Test: `tests/test_normal_solver.py`

- [x] **Step 1: Add explicit high-mode search constants**

```python
NORMAL_HIGH_POWER_DEVIATION = 6
NORMAL_HIGH_ANGLE_RADIUS = 3
```

- [x] **Step 2: Use the power-100 seed and bounded power range**

```python
seed = max(arcs, key=lambda a: a['angle_degrees'])
powers = range(100-NORMAL_HIGH_POWER_DEVIATION, 101)
diagnostics['theory_angle'] = float(seed['angle_degrees'])
diagnostics['theory_power'] = 100.0
```

- [x] **Step 3: Keep one high-arc angle center and apply ±3°**

```python
centers = [angle_center]
angle_radius = NORMAL_HIGH_ANGLE_RADIUS if arc_preference == 'high' else INTEGER_ANGLE_RADIUS
```

Only the low-arc path continues to call `solve_ballistic_for_speed` per power.

- [x] **Step 4: Rank high-mode candidates by replay error first**

```python
return min(candidates, key=lambda c: (
    c['miss_distance'], -c['angle_degrees'], -c['clearance'], -c['power'],
))
```

- [x] **Step 5: Run focused and complete verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_normal_solver.py -v
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m compileall -q detect_shellshock_yolo.py shellshock_detector_yolo tests
git diff --check -- shellshock_detector_yolo/normal_solver.py tests/test_normal_solver.py
```

Expected: all tests pass, compilation exits 0, and diff check has no output.

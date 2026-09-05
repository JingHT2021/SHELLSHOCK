# ShellShock Live Ballistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute and print wind-aware normal-shell firing solutions for every detected enemy: fixed-power-100 arcs and the minimum-power extremum.

**Architecture:** Keep projectile math pure in a new `shellshock_detector/ballistics.py` module. It consumes screen-space tank detections plus wind, converts coordinates into the game’s upward-positive ballistic frame, and returns JSON-ready dictionaries. `app.py` calls that module after image detection, attaches results to `DetectionResult`, and the hotkey entry point formats the saved result for the terminal.

**Tech Stack:** Python 3.13, standard-library `math` and `dataclasses`, existing `unittest`, OpenCV only for existing fixture loading.

---

## File structure

- Create: `shellshock_detector/ballistics.py` — calibrated physics constants, pure solver, JSON conversion, terminal formatting.
- Create: `tests/test_ballistics.py` — pure numerical tests independent of OCR, OpenCV, desktop capture, or keyboard hooks.
- Modify: `shellshock_detector/models.py` — add JSON-ready `ballistics` results to `DetectionResult`.
- Modify: `shellshock_detector/app.py` — calculate ballistics after tank/wind detection and return the computed result from `process_capture`.
- Modify: `detect_shellshock.py` — print the formatted firing guidance after a successful `R` capture.
- Modify: `tests/test_models.py` — verify `ballistics` is serialized.
- Modify: `tests/test_app.py` — verify complete input creates JSON ballistics and incomplete input records a skip reason.
- Modify: `README.md` — document output fields, angle convention, calibration provenance, and its current accuracy limit.

The project directory is not a Git repository, so no commit command is included; retain the normal test checkpoints after every task.

### Task 1: Pure model contract and solver tests

**Files:**
- Create: `tests/test_ballistics.py`

- [ ] **Step 1: Write failing tests for no-wind fixed-power and minimum-power solutions.**

```python
import unittest

from shellshock_detector.ballistics import solve_target


class BallisticsTests(unittest.TestCase):
    def test_no_wind_same_height_has_45_degree_minimum_power_solution(self):
        result = solve_target(
            self_x=0, self_y=0, target_x=1000, target_y=0,
            wind_value=0, wind_direction="right", image_width=1920,
        )

        self.assertEqual(result["target_direction"], "right")
        self.assertAlmostEqual(result["minimum_power"]["angle_degrees"], 45.0, places=5)
        self.assertLess(result["minimum_power"]["power"], 100.0)
        self.assertEqual(len(result["power_100"]["solutions"]), 2)

    def test_left_target_mirrors_right_target_angle(self):
        right = solve_target(0, 0, 1000, 0, 0, "right", 1920)
        left = solve_target(0, 0, -1000, 0, 0, "right", 1920)

        self.assertEqual(left["target_direction"], "left")
        self.assertAlmostEqual(
            right["minimum_power"]["angle_degrees"],
            left["minimum_power"]["angle_degrees"],
            places=5,
        )

    def test_power_100_can_be_unreachable_while_minimum_power_is_reported(self):
        result = solve_target(0, 0, 5000, 0, 0, "right", 1920)

        self.assertEqual(result["power_100"]["status"], "unreachable")
        self.assertEqual(result["power_100"]["solutions"], [])
        self.assertGreater(result["minimum_power"]["power"], 100.0)
        self.assertFalse(result["minimum_power"]["within_power_limit"])
```

- [ ] **Step 2: Run the new tests and verify RED.**

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_ballistics -v`

Expected: import failure because `shellshock_detector.ballistics` does not exist.

### Task 2: Implement the calibrated pure solver

**Files:**
- Create: `shellshock_detector/ballistics.py`
- Test: `tests/test_ballistics.py`

- [ ] **Step 1: Add scale-aware constants and helpers.**

```python
from __future__ import annotations

from math import atan2, degrees, sqrt


REFERENCE_WIDTH = 1920
GRAVITY_AT_REFERENCE = 379.106
SPEED_PER_POWER_AT_REFERENCE = 9.836246
WIND_ACCELERATION_PER_UNIT_AT_REFERENCE = 0.495
MAX_POWER = 100.0


def _scale(image_width: int) -> float:
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    return image_width / REFERENCE_WIDTH


def _wind_acceleration(value: int, direction: str, image_width: int) -> float:
    if direction not in {"left", "right"}:
        raise ValueError("wind direction must be left or right")
    sign = 1.0 if direction == "right" else -1.0
    return sign * value * WIND_ACCELERATION_PER_UNIT_AT_REFERENCE * _scale(image_width)
```

- [ ] **Step 2: Implement fixed-power positive-time roots and their firing solutions.**

```python
def _solution(direction: str, distance: float, height: float, wind: float, gravity: float, time: float) -> dict[str, float | str]:
    velocity_x = (distance - 0.5 * wind * time * time) / time
    velocity_y = (height + 0.5 * gravity * time * time) / time
    if velocity_x <= 0 or velocity_y < 0:
        raise ValueError("solution requires an invalid firing direction")
    return {
        "direction": direction,
        "angle_degrees": round(degrees(atan2(velocity_y, velocity_x)), 4),
        "flight_time_seconds": round(time, 4),
    }


def _fixed_power_solutions(distance: float, height: float, wind: float, gravity: float, speed: float, direction: str) -> list[dict[str, float | str]]:
    quadratic_a = (wind * wind + gravity * gravity) / 4.0
    quadratic_b = gravity * height - wind * distance - speed * speed
    quadratic_c = distance * distance + height * height
    discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
    if discriminant < 0:
        return []
    roots = [(-quadratic_b - sqrt(discriminant)) / (2.0 * quadratic_a), (-quadratic_b + sqrt(discriminant)) / (2.0 * quadratic_a)]
    solutions = []
    for time_squared in roots:
        if time_squared <= 0:
            continue
        try:
            solutions.append(_solution(direction, distance, height, wind, gravity, sqrt(time_squared)))
        except ValueError:
            continue
    return sorted(solutions, key=lambda item: float(item["flight_time_seconds"]))
```

- [ ] **Step 3: Implement `solve_target` and the analytic minimum-power result.**

```python
def solve_target(self_x: int, self_y: int, target_x: int, target_y: int, wind_value: int, wind_direction: str, image_width: int) -> dict[str, object]:
    horizontal = target_x - self_x
    direction = "right" if horizontal >= 0 else "left"
    direction_sign = 1.0 if direction == "right" else -1.0
    distance = abs(float(horizontal))
    height = float(self_y - target_y)
    scale = _scale(image_width)
    gravity = GRAVITY_AT_REFERENCE * scale
    speed_per_power = SPEED_PER_POWER_AT_REFERENCE * scale
    wind = _wind_acceleration(wind_value, wind_direction, image_width) * direction_sign
    fixed = _fixed_power_solutions(distance, height, wind, gravity, speed_per_power * MAX_POWER, direction)
    range_to_target = sqrt(distance * distance + height * height)
    force_magnitude = sqrt(wind * wind + gravity * gravity)
    optimal_time = sqrt(2.0 * range_to_target / force_magnitude)
    minimum_speed_squared = gravity * height - wind * distance + range_to_target * force_magnitude
    minimum_speed = sqrt(max(0.0, minimum_speed_squared))
    minimum_solution = _solution(direction, distance, height, wind, gravity, optimal_time)
    minimum_solution["power"] = round(minimum_speed / speed_per_power, 4)
    minimum_solution["within_power_limit"] = minimum_solution["power"] <= MAX_POWER
    return {
        "target": {"x": target_x, "y": target_y},
        "dx": horizontal,
        "dy": int(height),
        "target_direction": direction,
        "wind_acceleration": round(wind, 4),
        "power_100": {"power": MAX_POWER, "status": "reachable" if fixed else "unreachable", "solutions": fixed},
        "minimum_power": minimum_solution,
    }
```

- [ ] **Step 4: Run the Task 1 tests and verify GREEN.**

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_ballistics -v`

Expected: all three tests pass.

- [ ] **Step 5: Add and verify the supplied opposite-wind calibration regression.**

Append this test to `tests/test_ballistics.py`:

```python
    def test_94_opposite_wind_nearly_cancels_83_degree_power_100_horizontal_motion(self):
        from shellshock_detector.ballistics import predicted_horizontal_displacement

        displacement = predicted_horizontal_displacement(
            power=100, angle_degrees=83, wind_value=94,
            wind_direction="left", firing_direction="right", image_width=3840,
        )

        self.assertLess(abs(displacement), 10.0)
```

Implement the test helper in `ballistics.py`:

```python
def predicted_horizontal_displacement(power: float, angle_degrees: float, wind_value: int, wind_direction: str, firing_direction: str, image_width: int) -> float:
    from math import cos, radians, sin

    if firing_direction not in {"left", "right"}:
        raise ValueError("firing direction must be left or right")
    scale = _scale(image_width)
    gravity = GRAVITY_AT_REFERENCE * scale
    speed = SPEED_PER_POWER_AT_REFERENCE * scale * power
    time = 2.0 * speed * sin(radians(angle_degrees)) / gravity
    initial_horizontal = speed * cos(radians(angle_degrees))
    firing_sign = 1.0 if firing_direction == "right" else -1.0
    wind = _wind_acceleration(wind_value, wind_direction, image_width)
    return firing_sign * initial_horizontal * time + 0.5 * wind * time * time
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_ballistics.BallisticsTests.test_94_opposite_wind_nearly_cancels_83_degree_power_100_horizontal_motion -v`

Expected: PASS with an absolute predicted displacement under 10 pixels.

### Task 3: Extend the result contract and application orchestration

**Files:**
- Modify: `shellshock_detector/models.py`
- Modify: `shellshock_detector/app.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write the failing result-serialization test.**

Append to `tests/test_models.py`:

```python
    def test_result_serializes_ballistics(self):
        result = DetectionResult(
            image_width=1920, image_height=1080,
            self_tank=Detection(100, 400, 0.9),
            enemies=[], wind=Wind(0, "right", 0.9), errors=[],
            ballistics=[{"target_direction": "right", "power_100": {"status": "reachable"}}],
        )

        self.assertEqual(result.to_dict()["ballistics"][0]["target_direction"], "right")
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_models.ResultModelTests.test_result_serializes_ballistics -v`

Expected: ERROR because `DetectionResult` does not accept `ballistics`.

- [ ] **Step 2: Add the defaulted `ballistics` field and JSON key.**

In `shellshock_detector/models.py`, import `field` and append this field after `errors`:

```python
    ballistics: list[dict[str, object]] = field(default_factory=list)
```

Add this key to `to_dict()`:

```python
            "ballistics": self.ballistics,
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_models -v`

Expected: all model tests pass.

- [ ] **Step 3: Write the failing integration tests for calculation and missing wind.**

Append to `tests/test_app.py`:

```python
    def test_analyze_image_adds_ballistics_for_complete_detection(self):
        image = cv2.imdecode(np.frombuffer((Path("output") / "20260905_192849_raw.png").read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)

        result, _ = analyze_image(image)

        self.assertEqual(len(result.ballistics), 3)
        self.assertEqual(result.ballistics[0]["target_direction"], "left")
        self.assertIn("minimum_power", result.ballistics[0])

    def test_analyze_image_skips_ballistics_without_wind_value(self):
        image = np.zeros((1600, 2560, 3), dtype=np.uint8)
        cv2.rectangle(image, (1650, 1200), (1700, 1230), (0, 255, 0), -1)
        cv2.rectangle(image, (1100, 1180), (1150, 1210), (0, 0, 255), -1)

        result, _ = analyze_image(image)

        self.assertEqual(result.ballistics, [])
        self.assertIn("ballistics skipped: wind value not found", result.errors)
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_app.AppTests.test_analyze_image_adds_ballistics_for_complete_detection tests.test_app.AppTests.test_analyze_image_skips_ballistics_without_wind_value -v`

Expected: first test fails because ballistics is empty and second fails because no skip error exists.

- [ ] **Step 4: Integrate the pure solver in `analyze_image`.**

Add this import in `shellshock_detector/app.py`:

```python
from .ballistics import solve_target
```

Immediately before returning from `analyze_image`, add:

```python
    if self_tank is None:
        errors.append("ballistics skipped: green self tank not found")
    elif not enemies:
        errors.append("ballistics skipped: red enemy tank not found")
    elif wind.value is None:
        errors.append("ballistics skipped: wind value not found")
    elif wind.direction is None:
        errors.append("ballistics skipped: wind direction not found")
    else:
        ballistics = [
            solve_target(self_tank.x, self_tank.y, enemy.x, enemy.y, wind.value, wind.direction, image.shape[1])
            for enemy in enemies
        ]
        return DetectionResult(image.shape[1], image.shape[0], self_tank, enemies, wind, errors, ballistics), wind_box
    return DetectionResult(image.shape[1], image.shape[0], self_tank, enemies, wind, errors), wind_box
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_app tests.test_models -v`

Expected: all application and model tests pass. If the pre-existing temporary-directory permission failure recurs, run only the two new `test_analyze_image_*` methods plus `tests.test_models` and report the unrelated environment failure separately.

### Task 4: Print firing guidance after successful capture

**Files:**
- Modify: `shellshock_detector/ballistics.py`
- Modify: `shellshock_detector/app.py`
- Modify: `detect_shellshock.py`
- Modify: `tests/test_ballistics.py`

- [ ] **Step 1: Write a failing terminal-format test.**

Append to `tests/test_ballistics.py`:

```python
    def test_format_ballistics_mentions_target_direction_and_minimum_power(self):
        from shellshock_detector.ballistics import format_ballistics

        text = format_ballistics([{
            "target": {"x": 900, "y": 500}, "dx": 800, "dy": 0,
            "target_direction": "right",
            "power_100": {"power": 100.0, "status": "reachable", "solutions": [
                {"direction": "right", "angle_degrees": 20.0, "flight_time_seconds": 1.0},
                {"direction": "right", "angle_degrees": 70.0, "flight_time_seconds": 2.0},
            ]},
            "minimum_power": {"direction": "right", "angle_degrees": 45.0, "power": 62.0, "within_power_limit": True},
        }])

        self.assertIn("Target 1: dx=800, dy=0, direction=right", text)
        self.assertIn("100 power low arc: right 20.0000 degrees", text)
        self.assertIn("minimum power: 62.0000, right 45.0000 degrees", text)
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_ballistics.BallisticsTests.test_format_ballistics_mentions_target_direction_and_minimum_power -v`

Expected: import failure because `format_ballistics` does not exist.

- [ ] **Step 2: Implement deterministic terminal formatting.**

Add this function to `shellshock_detector/ballistics.py`:

```python
def format_ballistics(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        power_100 = result["power_100"]
        minimum = result["minimum_power"]
        lines.append(f"Target {index}: dx={result['dx']}, dy={result['dy']}, direction={result['target_direction']}")
        if power_100["status"] == "unreachable":
            lines.append("  100 power: unreachable")
        else:
            labels = ("low arc", "high arc")
            for label, solution in zip(labels, power_100["solutions"]):
                lines.append(f"  100 power {label}: {solution['direction']} {solution['angle_degrees']:.4f} degrees, {solution['flight_time_seconds']:.4f}s")
        limit = "within limit" if minimum["within_power_limit"] else "over 100 limit"
        lines.append(f"  minimum power: {minimum['power']:.4f}, {minimum['direction']} {minimum['angle_degrees']:.4f} degrees ({limit})")
    return "\n".join(lines)
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_ballistics -v`

Expected: all ballistics tests pass.

- [ ] **Step 3: Return the computed result with each saved capture.**

Modify `OutputPaths` in `shellshock_detector/app.py`:

```python
class OutputPaths:
    raw_path: Path
    json_path: Path
    annotated_path: Path
    result: DetectionResult
```

Modify the return in `process_capture`:

```python
    return OutputPaths(raw_path, json_path, annotated_path, result)
```

Add this assertion inside `test_process_capture_writes_json_and_annotation`:

```python
            self.assertEqual(paths.result.ballistics, [])
```

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_app.AppTests.test_process_capture_writes_json_and_annotation -v`

Expected: PASS, or document the known Windows temporary-directory permission error if it prevents this unrelated file-write test from running.

- [ ] **Step 4: Print results in the hotkey callback.**

In `detect_shellshock.py`, import `format_ballistics` and add this after the existing `Saved:` print:

```python
            if paths.result.ballistics:
                print(format_ballistics(paths.result.ballistics))
            else:
                print("Ballistics unavailable: " + "; ".join(paths.result.errors))
```

Run: `.\\.venv\\Scripts\\python.exe detect_shellshock.py --help`

Expected: command exits 0 and still lists `--output-dir` and `--resolution`.

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_ballistics.py`, `tests/test_models.py`, `tests/test_app.py`, `tests/test_tanks.py`, `tests/test_wind.py`

- [ ] **Step 1: Add a concise firing-guidance section to `README.md`.**

Insert after the JSON example:

```markdown
## 弹道建议

每次按 `R` 后，终端会为每个敌方坦克输出：100 力度的低/高两条可命中弹道，以及理论最低力度和对应角度。角度始终为 0–90° 仰角；`left`/`right` 表示炮管方向。

计算使用 1920×1080 标定并按截图宽度缩放；当前风力系数来自一条 94 风实测，结果应作为首发建议。不同武器、弹跳、地形碰撞及版本差异会偏离普通炮弹模型；用多次实际落点可再校准风系数。
```

- [ ] **Step 2: Run the full non-desktop regression suite.**

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_ballistics tests.test_models tests.test_tanks tests.test_wind -v`

Expected: all tests pass.

- [ ] **Step 3: Run application tests separately and record environment-only failures accurately.**

Run: `.\\.venv\\Scripts\\python.exe -m unittest tests.test_app -v`

Expected: all logic tests pass. If `test_process_capture_writes_json_and_annotation` fails only with `PermissionError` inside a newly created `TemporaryDirectory`, report it as the existing Windows workspace permission issue; do not alter trajectory code to hide it.

- [ ] **Step 4: Inspect an existing complete 4K capture with the new pure solver.**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; from shellshock_detector.app import analyze_image; image=cv2.imread('output/20260905_192849_raw.png'); result,_=analyze_image(image); print(result.to_dict()['ballistics'])"
```

Expected: three ballistics entries, with a left-direction entry for the rotated enemy at approximately `(778, 687)`.

## Plan self-review

- Spec coverage: Tasks 1–2 implement scaled wind physics, fixed-power 100 roots, the minimum-power extremum, mirror directions, and the supplied 94-wind calibration; Task 3 makes it part of detection JSON with explicit missing-input handling; Task 4 prints it after `R`; Task 5 documents and verifies it.
- Placeholder scan: no deferred implementation or unspecified handling remains; all error paths and commands are explicit.
- Type consistency: solver returns JSON-ready dictionaries; `DetectionResult.ballistics`, `OutputPaths.result`, `format_ballistics`, application integration, and tests use the same names.

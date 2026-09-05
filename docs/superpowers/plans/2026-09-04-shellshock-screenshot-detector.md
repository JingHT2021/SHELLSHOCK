# ShellShock Live Screenshot Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows Python utility that captures only the ShellShock Live window when `R` is pressed, detects tanks and wind, and writes JSON plus an annotated image.

**Architecture:** A small `shellshock_detector` package separates pure image detection from Windows/window and keyboard adapters. Detection works in current image pixels and scales thresholds from the 2560×1600 reference size. Adapters are dependency-injected so tests never capture the desktop or install keyboard hooks.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, Pillow, pywin32, keyboard, pytesseract, pytest.

---

## File structure

- `requirements.txt` — pinned runtime/test dependencies.
- `shellshock_detector/models.py` — typed detection result data classes and JSON conversion.
- `shellshock_detector/tanks.py` — HSV tank candidates, filtering, annotation helpers.
- `shellshock_detector/wind.py` — wind panel location, arrow direction and OCR preparation.
- `shellshock_detector/app.py` — capture orchestration, output creation and hotkey loop.
- `detect_shellshock.py` — command-line entry point.
- `tests/test_models.py`, `tests/test_tanks.py`, `tests/test_wind.py`, `tests/test_app.py` — isolated tests.
- `tests/fixtures/` — copied user-provided reference images and a 2560×1600 full screenshot.

### Task 0: Isolated Python environment

**Files:**
- Create: `.venv/` (untracked local environment), `requirements.txt`

- [ ] **Step 1: Create the virtual environment.**

Run: `py -3 -m venv .venv`  
Expected: `.venv\\Scripts\\python.exe` exists.

- [ ] **Step 2: Add the test/runtime dependency list.**

Create `requirements.txt` with:

```
opencv-python>=4.10,<5
numpy>=2.0,<3
Pillow>=10,<12
pywin32>=306
keyboard>=0.13,<1
pytesseract>=0.3,<1
pytest>=8,<9
```

- [ ] **Step 3: Install and verify dependencies.**

Run: `.venv\\Scripts\\python.exe -m pip install -r requirements.txt` then `.venv\\Scripts\\python.exe -c "import cv2, keyboard, pytest; print(cv2.__version__)"`  
Expected: pip succeeds and the second command prints an OpenCV version.

### Task 1: Results contract and tank detection

**Files:**
- Create: `shellshock_detector/__init__.py`, `shellshock_detector/models.py`, `shellshock_detector/tanks.py`
- Create: `tests/test_models.py`, `tests/test_tanks.py`

- [ ] **Step 1: Write the failing model contract test.**

```python
from shellshock_detector.models import Detection, DetectionResult, Wind

def test_result_serializes_image_tanks_and_wind():
    result = DetectionResult(
        image_width=2560, image_height=1600,
        self_tank=Detection(1672, 763, 0.95),
        enemies=[Detection(1145, 755, 0.93)],
        wind=Wind(value=20, direction="right", confidence=0.9), errors=[])
    assert result.to_dict()["image"] == {"width": 2560, "height": 1600}
    assert result.to_dict()["wind"]["direction"] == "right"
```

- [ ] **Step 2: Run the test and verify RED.**

Run: `python -m pytest tests/test_models.py -v`  
Expected: import failure because `shellshock_detector` does not exist.

- [ ] **Step 3: Implement the minimal typed result model.**

```python
@dataclass(frozen=True)
class Detection: x: int; y: int; confidence: float
@dataclass(frozen=True)
class Wind: value: int | None; direction: str | None; confidence: float
```

Implement `DetectionResult.to_dict()` with `image`, `self`, `enemies`, `wind`, and `errors`; serialize absent objects as `None`.

- [ ] **Step 4: Run the model test and verify GREEN.**

Run: `python -m pytest tests/test_models.py -v`  
Expected: `1 passed`.

- [ ] **Step 5: Write failing color-detection tests.**

```python
def test_detect_tanks_returns_green_self_and_red_enemies():
    image = np.zeros((1600, 2560, 3), dtype=np.uint8)
    cv2.rectangle(image, (1650, 740), (1700, 770), (0, 255, 0), -1)
    cv2.rectangle(image, (1100, 730), (1150, 760), (0, 0, 255), -1)
    self_tank, enemies = detect_tanks(image)
    assert self_tank.x == 1675
    assert [(item.x, item.y) for item in enemies] == [(1125, 745)]
```

- [ ] **Step 6: Run tank test and verify RED.**

Run: `python -m pytest tests/test_tanks.py -v`  
Expected: import failure because `detect_tanks` is absent.

- [ ] **Step 7: Implement HSV masks and scale-aware contour filtering.**

Implement `detect_tanks(bgr: np.ndarray) -> tuple[Detection | None, list[Detection]]`: convert to HSV; threshold hue 35–90 for green and both 0–10/170–179 for red; dilate 3×3; filter contours using `area >= 80 * (width / 2560)**2`, aspect ratio 1.1–3.5, and center y in lower 65% of image. Return contour centers, sorted by x, with confidence derived from normalized area.

- [ ] **Step 8: Run all Task 1 tests and verify GREEN.**

Run: `python -m pytest tests/test_models.py tests/test_tanks.py -v`  
Expected: all tests pass.

### Task 2: Wind panel and OCR

**Files:**
- Create: `shellshock_detector/wind.py`, `tests/test_wind.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing arrow-direction tests.**

```python
def test_direction_is_right_when_triangle_points_right():
    panel = make_panel_with_triangle(points=[(75, 30), (55, 15), (55, 45)])
    assert detect_direction(panel) == "right"
```

- [ ] **Step 2: Run and verify RED.**

Run: `python -m pytest tests/test_wind.py::test_direction_is_right_when_triangle_points_right -v`  
Expected: import failure for `detect_direction`.

- [ ] **Step 3: Implement wind detection.**

Implement `find_wind_panel(bgr)` by searching the upper 35% for pale low-saturation pixels, grouping components, and returning the cloud/arrow bounding box. Implement `detect_direction(panel)` by thresholding the arrow separately and comparing its largest contour’s left/right x-extreme around its centroid. Implement `read_wind_value(panel)` using 4× upscaling, grayscale thresholding and `pytesseract.image_to_data(..., config="--psm 7 -c tessedit_char_whitelist=0123456789")`; return `None` on missing executable, no valid numeric token, or a value outside 0–100.

- [ ] **Step 4: Add a stable OCR-free test and verify GREEN.**

```python
def test_wind_failure_is_explicit_when_panel_is_missing():
    wind, error = detect_wind(np.zeros((1600, 2560, 3), dtype=np.uint8))
    assert wind.value is None
    assert error == "wind panel not found"
```

Run: `python -m pytest tests/test_wind.py -v`  
Expected: all tests pass without requiring Tesseract to be installed.

### Task 3: Window capture, outputs and hotkey loop

**Files:**
- Create: `shellshock_detector/app.py`, `detect_shellshock.py`, `tests/test_app.py`

- [ ] **Step 1: Write a failing injected-capture test.**

```python
def test_process_capture_writes_json_and_annotation(tmp_path):
    image = make_green_and_red_scene()
    paths = process_capture(image, tmp_path, now=lambda: "20260904_120000")
    assert paths.json_path.name == "20260904_120000_result.json"
    assert paths.annotated_path.exists()
```

- [ ] **Step 2: Run and verify RED.**

Run: `python -m pytest tests/test_app.py -v`  
Expected: import failure for `process_capture`.

- [ ] **Step 3: Implement capture orchestration.**

Implement `find_game_window()` with `win32gui.EnumWindows`, accepting visible titles containing `ShellShock Live`. Implement `capture_client_area(hwnd)` with `ImageGrab.grab(bbox=client_rect)` and return BGR. Implement `process_capture` to run detectors, call `cv2.imwrite`, and write UTF-8 JSON. All filenames use one `YYYYMMDD_HHMMSS` timestamp. If no game window is found, raise `RuntimeError("ShellShock Live window not found")` before any grab.

- [ ] **Step 4: Implement hotkey CLI.**

```python
keyboard.add_hotkey("r", lambda: capture_once(output_dir))
print("Ready: press R to analyze ShellShock Live; press Esc to quit.")
keyboard.wait("esc")
```

`detect_shellshock.py` must parse `--output-dir` (default `output`) and create it. It must print each saved result path or the caught `RuntimeError`.

- [ ] **Step 5: Run tests and verify GREEN.**

Run: `python -m pytest -v`  
Expected: all tests pass; tests do not require a live game window.

- [ ] **Step 6: Install dependencies and run a manual smoke test.**

Run: `python -m pip install -r requirements.txt` then `python detect_shellshock.py --output-dir output`  
Expected: terminal prints the hotkey instructions. With the game window visible, pressing `R` produces three timestamped files; `Esc` exits.

### Task 4: Real screenshot calibration and documentation

**Files:**
- Create: `README.md`, `tests/fixtures/full_scene.png`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add the provided full 2560×1600 screenshot as a fixture and write the failing integration assertion.**

```python
def test_full_scene_detects_one_green_and_two_red_tanks():
    result = analyze_image(load_fixture("full_scene.png"))
    assert result.self_tank is not None
    assert len(result.enemies) >= 2
    assert result.wind.direction == "right"
```

- [ ] **Step 2: Run integration test and verify whether it is RED or exposes calibration error.**

Run: `python -m pytest tests/test_app.py::test_full_scene_detects_one_green_and_two_red_tanks -v`

- [ ] **Step 3: Adjust only documented HSV/area constants until the fixture test passes.**

Keep values in named constants in `tanks.py`/`wind.py`; do not add special-case coordinates for this screenshot.

- [ ] **Step 4: Write usage and limits in README.**

Document Windows-only window capture, `R` and `Esc`, optional Tesseract installation/path configuration, output schema, the 2560×1600 reference resolution, and known limits: alternate UI themes, recolored tank skins, occlusion and nonstandard projectiles.

- [ ] **Step 5: Run final verification.**

Run: `python -m pytest -v`  
Expected: all tests pass. Then run `python detect_shellshock.py --help` and confirm `--output-dir` appears.

## Plan self-review

- Spec coverage: Tasks 1–2 cover static visual detection and OCR; Task 3 covers R/Esc and only-game-window capture; Task 4 covers 2560×1600 calibration, output verification and documentation.
- No placeholder scan: all behavior, test commands and dependencies are explicit.
- Type consistency: all modules exchange `Detection`, `Wind` and `DetectionResult`; `process_capture` owns file generation.

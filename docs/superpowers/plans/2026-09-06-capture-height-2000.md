# 2000-Pixel Capture Height Implementation Plan

**Goal:** Support the top 2000 physical pixels of the game client for captures, manual aiming, and tank detection.

**Architecture:** Set the capture limit to 2000 and apply it consistently in capture, aiming validation, and tank-search boundaries.

**Tech Stack:** Python 3, OpenCV, NumPy, unittest, pytest.

### Task 1: Add regression tests

**Files:** `tests/test_tanks.py`, `tests/test_app.py`

- [ ] Write a failing tank fixture containing green and red tanks centered at y=1975 and y=1965, plus a red tank centered at y=2025. Assert only the first two are found.
- [ ] Run `python -m pytest tests/test_tanks.py::TankDetectionTests::test_detects_tanks_up_to_fixed_2000_pixel_bottom_boundary -q` and verify that it fails at the current 1800-pixel limit.
- [ ] Add `test_game_capture_height_is_2000_physical_pixels` to assert `GAME_CAPTURE_HEIGHT == 2000`.
- [ ] Run `python -m pytest tests/test_app.py::AppTests::test_game_capture_height_is_2000_physical_pixels -q` and verify `1800 != 2000`.

### Task 2: Implement and verify the unified boundary

**Files:** `shellshock_detector/app.py`, `shellshock_detector/tanks.py`, `detect_shellshock.py`, `README.md`

- [ ] Set `GAME_CAPTURE_HEIGHT` and `PLAYFIELD_BOTTOM` to 2000; update every user-facing 1800-pixel capture message and the README crop range.
- [ ] Run both focused regression tests and expect 2 passed.
- [ ] Run `python -m pytest -q` and expect the full suite to pass.
- [ ] Commit the source, tests, docs, and this plan as `Extend capture height to 2000 pixels`, then push to `origin/main`.

# Left-Wind Panel Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect the wind HUD when its arrow is displayed either left or right of the cloud.

**Architecture:** Keep HSV component detection unchanged and make the pair-selection loop order-independent. The returned panel rectangle spans both adjacent components, preserving existing direction and OCR code.

**Tech Stack:** Python 3, OpenCV, NumPy, unittest, pytest.

---

### Task 1: Add a left-arrow regression test

**Files:**
- Modify: `tests/test_wind.py`

- [ ] Add a synthetic top-centre panel whose 85×55 cloud is at `(1879, 141)` and whose 23×31 arrow is at `(1843, 153)`. Assert `_find_panel_box` spans both and `detect_direction` returns `left`.
- [ ] Run `python -m pytest tests/test_wind.py::WindDetectionTests::test_panel_search_supports_a_left_wind_arrow -q` and verify it fails because the current pairing loop rejects a component to the left.

### Task 2: Make pairing order-independent

**Files:**
- Modify: `shellshock_detector/wind.py:116-132`
- Test: `tests/test_wind.py`

- [ ] Replace the directional `second[0] < first[0]` rejection with ordered pair bounds: `left, right = sorted((first, second), key=lambda item: item[0])`; calculate gap and output rectangle from `left` and `right`.
- [ ] Run the new test and `python -m pytest tests/test_wind.py -q`; expect all synthetic and fixture-independent wind tests to pass.
- [ ] Confirm the existing no-panel test still returns `Wind(value=0, direction=None, confidence=0.0)` and does not block the zero-wind fallback.
- [ ] Commit `shellshock_detector/wind.py`, `tests/test_wind.py`, and this plan as `Support left wind panel arrows`.

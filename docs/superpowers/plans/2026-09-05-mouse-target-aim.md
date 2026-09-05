# Mouse Target Aim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `T` aim at the current mouse target by calculating the wind-adjusted minimum-power shot and clicking its point on the game aim disc.

**Architecture:** Keep pixel-to-disc mapping pure in `shellshock_detector/aiming.py`, so it is unit-testable. Keep Windows capture, coordinate conversion and one guarded mouse click in `app.py`; the CLI hotkey merely orchestrates that operation.

**Tech Stack:** Python 3.13, OpenCV, pywin32, keyboard, unittest.

---

### Task 1: Pure aim-disc mapping

**Files:**
- Create: `shellshock_detector/aiming.py`
- Create: `tests/test_aiming.py`

- [ ] **Step 1: Write failing mapping tests** for right and left 45-degree shots, plus rejected invalid input.
- [ ] **Step 2: Run `python -m unittest tests.test_aiming -v`** and confirm import failure.
- [ ] **Step 3: Implement `disc_click_point`** with a 1920-reference radius, scale-aware radius, 0--100 power validation and left/right UI-angle mapping.
- [ ] **Step 4: Re-run the focused tests** and confirm all pass.

### Task 2: Game-window targeting and T hotkey

**Files:**
- Modify: `shellshock_detector/app.py`
- Modify: `detect_shellshock.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing tests** for screen-to-client conversion, a successful selected minimum-power solution, and no click when analysis lacks a usable solution.
- [ ] **Step 2: Run the focused tests** and confirm they fail because the API is missing.
- [ ] **Step 3: Implement `aim_at_screen_point`**: capture, analyze, solve target from the mouse target, map through `disc_click_point`, verify foreground handle, then call the injected click callback once.
- [ ] **Step 4: Add `T`** to the CLI using `keyboard` for the current mouse position and `win32api.SetCursorPos`/`mouse_event` for the final left click; preserve `R`.
- [ ] **Step 5: Run focused tests** and confirm the new suite passes.

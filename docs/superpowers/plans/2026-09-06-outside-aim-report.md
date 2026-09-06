# Outside-Client Aim Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Print a calculated angle and power when a safe click cannot be made because the aim point is outside the game client.

**Architecture:** `aim_at_screen_position` returns `None` for its screen-click point after a successful calculation whose client click point is out of bounds. The CLI formats that result with an explicit no-click label and never calls the click callback.

**Tech Stack:** Python 3, unittest, pytest.

---

### Task 1: Capture the no-click report contract in tests

**Files:**
- Modify: `tests/test_hotkeys.py`
- Modify: `tests/test_app.py`

- [ ] Add a report-format test passing `None` as the click point and asserting the output contains `AIM CLICK OUTSIDE CLIENT`, `power=28`, and `angle=35`.
- [ ] Run the report test and verify it fails because `format_aim_report` currently requires a coordinate tuple.
- [ ] Add an app-level test with mocked capture and click dependencies: an out-of-client `click_client` returns a solution and `None`, and the injected click list remains empty.
- [ ] Run the app test and verify it fails because the function currently raises `RuntimeError`.

### Task 2: Return and display safe, unclicked solutions

**Files:**
- Modify: `shellshock_detector/app.py:290-325`
- Modify: `detect_shellshock.py:57-67,151-167`
- Test: `tests/test_app.py`, `tests/test_hotkeys.py`

- [ ] Change the third `aim_at_screen_position` return value to `tuple[int, int] | None`; return `None` after calculating an out-of-client point and before activation or clicking.
- [ ] Make `format_aim_report` accept an optional click point; render `AIM CLICK OUTSIDE CLIENT` for `None` while retaining the selected power and angle text.
- [ ] In the E-hotkey handler, always print the aim report; print `Aim not clicked: calculated aim-disc point is outside the game client area` when the screen click is `None`.
- [ ] Run the new tests plus `python -m pytest tests/test_app.py tests/test_hotkeys.py -q -p no:cacheprovider`; expect all pass.
- [ ] Commit the implementation, tests, and this plan as `Report outside-client aim solutions`.

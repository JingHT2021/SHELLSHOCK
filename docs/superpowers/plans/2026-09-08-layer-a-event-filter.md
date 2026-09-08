# Layer A Event Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Layer A a conservative event-path filter reusable by reflection and wormhole modes.

**Architecture:** A generic module evaluates directed waypoint segments and portal teleports. Reflection contributes a small adapter for midpoint-local velocity reflection. Layer B retains ranking ownership.

**Tech Stack:** Python, NumPy, pytest.

---

### Task 1: Generic reachability primitives

**Files:**
- Create: `shellshock_detector_yolo/coarse_path_filter.py`
- Test: `tests/test_coarse_path_filter.py`

- [x] Write failing tests for direct velocity reachability, acceleration-only recovery, and a fully opposed axis.
- [x] Run `python -m pytest -q tests/test_coarse_path_filter.py` and confirm failure.
- [x] Implement `axis_maybe_reachable` and `segment_possible` with no ballistic or replay calls.
- [x] Re-run the focused test and confirm success.

### Task 2: Portal event traversal

**Files:**
- Modify: `shellshock_detector_yolo/coarse_path_filter.py`
- Modify: `tests/test_coarse_path_filter.py`

- [x] Write a failing portal-to-portal route test that verifies exit position replacement and rejection at the first impossible segment.
- [x] Run the focused test and confirm failure.
- [x] Implement ordered portal traversal with unchanged velocity directions.
- [x] Re-run the focused test and confirm success.

### Task 3: Reflection adapter and migration

**Files:**
- Modify: `shellshock_detector_yolo/reflection_filter.py`
- Modify: `shellshock_detector_yolo/reflection_solver.py`
- Modify: `tests/test_reflection_filter.py`

- [x] Write failing tests for pre/post portal reflection routes and removal of virtual-height scoring.
- [x] Run the focused test and confirm failure.
- [x] Replace reflection-only Layer A geometry/scoring with the generic event filter and midpoint reflection adapter.
- [x] Re-run focused reflection tests and confirm success.

### Task 4: Wormhole integration and regression

**Files:**
- Modify: `shellshock_detector_yolo/wormhole_solver.py`
- Modify: `tests/test_wormhole_solver.py`

- [x] Write a failing test proving the wormhole solver invokes the conservative filter for a portal-to-portal candidate.
- [x] Run the focused test and confirm failure.
- [x] Integrate the filter without changing strict replay validation.
- [x] Run `python -m pytest -q tests/test_coarse_path_filter.py tests/test_reflection_filter.py tests/test_yolo_reflection_solver.py tests/test_yolo_solver_package.py tests/test_wormhole_solver.py`.

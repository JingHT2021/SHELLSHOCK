# Layer C Integer Replay Implementation Plan

**Goal:** Replace the provisional Layer C power sweep with local continuous surface sampling, globally deduplicated integer candidates, a hard Top 10 replay budget, actual-miss ranking, and final candidate activation helpers.

**Architecture:** `reflection_layer_c.py` owns data models and pure candidate-pool mechanics. The existing reflection solver supplies Layer B proxy branches and a replay callback, so integer replay remains a true fresh-velocity operation. UI activation is kept in `aiming.py`/a small manager-facing helper and never changes solver validity.

**Tech Stack:** Python, dataclasses, NumPy/SciPy, pytest/unittest-compatible tests.

### Task 1: Define Layer C models and grid mechanics

**Files:**
- Create: `shellshock_detector_yolo/reflection_layer_c.py`
- Test: `tests/test_reflection_layer_c.py`

- [x] Add failing tests for seven-point local intervals, power/angle crossings, nearest-grid candidates, grid error, and `(angle, power)` grouping.
- [x] Implement only the pure mechanics needed by those tests: interval construction, linear crossing interpolation, nearest samples, grid-error calculation, stable grouping and pre-sort keys.
- [x] Run the focused tests and keep the API independent of world replay.

### Task 2: Build the bounded Top 10 pool

**Files:**
- Modify: `shellshock_detector_yolo/reflection_layer_c.py`
- Modify: `shellshock_detector_yolo/solver_config.py`
- Test: `tests/test_reflection_layer_c.py`

- [x] Add failing tests proving duplicate controls replay once, raw candidates are pre-sorted, and the pool is capped at 10.
- [x] Implement grouped candidates with all sources retained and deterministic cheap ordering.

### Task 3: Exact replay and final sorting

**Files:**
- Modify: `shellshock_detector_yolo/reflection_layer_c.py`
- Modify: `shellshock_detector_yolo/reflection_solver.py`
- Test: `tests/test_reflection_layer_c.py`, `tests/test_yolo_reflection_solver.py`

- [x] Add failing tests for replaying every Top 10 control, retaining fewer than five valid results, and sorting valid results by actual miss before tie-breakers.
- [x] Replace the old per-family power sweep with Layer C pool generation and callback-based fresh integer replay.
- [x] Preserve diagnostics and expose `full_replays <= 10`.

### Task 4: Candidate activation and hotkey-safe UI behavior

**Files:**
- Create or modify: `shellshock_detector_yolo/reflection_layer_c.py`, `shellshock_detector_yolo/aiming.py`
- Test: `tests/test_reflection_layer_c.py`, `tests/test_aiming.py`

- [x] Add failing tests for clamped PgUp/PgDn navigation and click failures that preserve `(power, angle)`.
- [x] Implement an index manager and click-result separation; click failures remain UI status only.

### Task 5: Verification

- [x] Run focused Layer C tests.
- [x] Run existing reflection and portal replay regression tests.
- [x] Run compilation and diff checks, then review the final diff for unrelated changes.

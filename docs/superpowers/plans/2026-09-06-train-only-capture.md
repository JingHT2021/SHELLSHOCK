# Train-Only Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save every new capture only under `train/`, never under `output/`.

**Architecture:** `process_capture` persists a training image, YOLO label, and annotated preview through `save_training_sample`, and returns those paths with analysis results. The command-line entry point removes output-directory configuration and reports only training artifacts.

**Tech Stack:** Python 3, OpenCV, unittest, pytest.

---

### Task 1: Define train-only persistence in tests

**Files:**
- Modify: `tests/test_app.py`

- [ ] Replace the runtime-output test with a train-only test that calls `process_capture(image, train_dir, now=...)`, asserts `<train>/raw_cropped/<timestamp>.png`, `<train>/raw_cropped/<timestamp>.txt`, and `<train>/annotated/<timestamp>.png` exist, and asserts no `output` path is created.
- [ ] Run the test and verify it fails because the current function writes `_raw.png`, `_result.json`, and `_annotated.png` to its output argument.

### Task 2: Remove runtime output persistence

**Files:**
- Modify: `shellshock_detector/app.py:31-36,192-218,290-336`
- Modify: `detect_shellshock.py:74-115,151-170`
- Modify: `README.md:11-20`
- Test: `tests/test_app.py`

- [ ] Replace output-path fields with training raw, label, and preview paths. Make `process_capture` require `train_dir` and return the paths from `save_training_sample`; remove PNG and JSON runtime writes.
- [ ] Remove `output_dir` parameters from `capture_once` and `aim_at_screen_position`, remove the CLI `--output-dir` option and directory creation, and print only the three training paths.
- [ ] Update README language to state that captures are saved only beneath `train/`.
- [ ] Run `python -m pytest tests/test_app.py tests/test_hotkeys.py -q -p no:cacheprovider`; expect all pass.

### Task 3: Remove old runtime artifacts

**Files:**
- Delete: `C:\Users\jingh\Desktop\game\shellshock\output`

- [ ] Verify the exact target is the project-local `output` directory and record its size.
- [ ] Delete that directory recursively only after the tests pass.
- [ ] Confirm the path no longer exists and `train/` remains intact.
- [ ] Commit code, tests, docs, and this plan as `Store captures only as training data`.

# ShellShock YOLO11n Dataset and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated, sparse-ID YOLO dataset and train a first reproducible YOLO11n detector without altering raw annotations.

**Architecture:** A standalone preparation module owns parsing, crop geometry, validation, candidate review, stratified splitting, derived file writes, reports, YAML and previews. A CLI calls that module and a separate training CLI consumes only the resulting YAML. Tests use temporary synthetic images and labels, so all geometric and split behavior is deterministic without touching real data.

**Tech Stack:** Python 3, OpenCV, PyYAML, pytest, Ultralytics YOLO.

---

### Task 1: Define the pure dataset preparation API and its tests

**Files:**
- Create: `shellshock_detector/yolo_dataset.py`
- Create: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Write failing crop-geometry tests**

```python
from shellshock_detector.yolo_dataset import transform_yolo_box_for_crop

def test_crop_clips_and_renormalizes_bottom_intersection():
    result = transform_yolo_box_for_crop(0, 0.5, 0.95, 0.1, 0.2, 3840, 2000, 1850)
    assert result == (0, 0.5, 1800 / 1850, 0.1, 100 / 1850)

def test_crop_discards_box_entirely_below_boundary():
    assert transform_yolo_box_for_crop(0, 0.5, 0.975, 0.1, 0.05, 3840, 2000, 1850) is None
```

- [ ] **Step 2: Run the crop tests to verify they fail because the module is missing**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_yolo_dataset.py -q`

Expected: collection error importing `shellshock_detector.yolo_dataset`.

- [ ] **Step 3: Implement `transform_yolo_box_for_crop` using pixel-space left/top/right/bottom clipping**

```python
def transform_yolo_box_for_crop(class_id, xc, yc, width, height, image_width, image_height, crop_height):
    left = (xc - width / 2) * image_width
    top = (yc - height / 2) * image_height
    right = (xc + width / 2) * image_width
    bottom = (yc + height / 2) * image_height
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(image_width), right), min(float(image_height), bottom, float(crop_height))
    if right <= left or bottom <= top:
        return None
    return class_id, (left + right) / 2 / image_width, (top + bottom) / 2 / crop_height, (right - left) / image_width, (bottom - top) / crop_height
```

- [ ] **Step 4: Run the crop tests and make them pass**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_yolo_dataset.py -q`

Expected: PASS.

### Task 2: Add strict parse/validation, sparse class statistics, and deterministic multilabel split

**Files:**
- Modify: `shellshock_detector/yolo_dataset.py`
- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Write failing tests for invalid-row rejection and preserving class 9**

```python
from shellshock_detector.yolo_dataset import parse_yolo_label_text

def test_parser_accepts_class_nine_and_rejects_out_of_range_identifier():
    valid, errors = parse_yolo_label_text('9 0.5 0.5 0.1 0.1\\n10 0.5 0.5 0.1 0.1\\n')
    assert [box.class_id for box in valid] == [9]
    assert 'class_id_out_of_range' in errors[0]
```

- [ ] **Step 2: Run that test and verify failure for missing parser**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_yolo_dataset.py -q`

Expected: import failure for `parse_yolo_label_text`.

- [ ] **Step 3: Implement parser, per-class statistics, and a seed-42 greedy multi-label split**

The parser must reject non-five-field rows, non-integers, non-finite floats, IDs outside `0..9`, non-positive dimensions, and boxes whose pixel extent is outside source dimensions. The splitter must accept eligible sample records, use a supplied RNG seed, target 20% validation count, and return class-coverage warnings rather than silently hide rare classes.

- [ ] **Step 4: Run the module tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_yolo_dataset.py -q`

Expected: PASS.

### Task 3: Implement derived dataset writes, reports, previews, candidate review, and YAML

**Files:**
- Modify: `shellshock_detector/yolo_dataset.py`
- Create: `prepare_yolo_dataset.py`
- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Write a failing temporary-directory integration test**

```python
def test_prepare_creates_sparse_id_yaml_and_never_modifies_raw(tmp_path):
    # create a 3840x2000 PNG and matching class-9 YOLO label
    # run prepare_dataset(raw_dir, output_dir, config)
    # assert source bytes unchanged, output image is 3840x1850,
    # YAML names map covers exactly 0..9, and output label retains class 9
```

- [ ] **Step 2: Run it to verify it fails because `prepare_dataset` is absent**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_yolo_dataset.py -q`

Expected: import failure for `prepare_dataset`.

- [ ] **Step 3: Implement `prepare_dataset` and CLI**

Copy/crop only valid paired samples into `train/yolo_dataset/{images,labels}/{train,val}`; preserve source extensions and same stems; emit `dataset.yaml`, `excluded_samples.csv`, `summary.json`, `class_distribution.csv`, `enemy_candidates.csv`, and `previews/*.jpg`. Candidate detection is conservative red HSV connected-component discovery; candidate frames are excluded only when a candidate box does not overlap any labelled enemy at IoU >= 0.35. Existing supplemental labels under `train/supplemental_labels/<stem>.txt` are validated and merged only into derived labels.

- [ ] **Step 4: Run the complete preparation test file**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_yolo_dataset.py -q`

Expected: PASS.

### Task 4: Add and test the YOLO11n training entry point

**Files:**
- Create: `train_yolo.py`
- Create: `tests/test_train_yolo.py`

- [ ] **Step 1: Write a failing configuration test**

```python
from train_yolo import TrainingConfig

def test_default_training_config_preserves_requested_reproducible_defaults():
    config = TrainingConfig()
    assert config.model == 'yolo11n.pt'
    assert config.imgsz == 960
    assert config.epochs == 200
    assert config.seed == 42
```

- [ ] **Step 2: Run it and verify import failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_train_yolo.py -q`

Expected: collection error importing `train_yolo`.

- [ ] **Step 3: Implement a lazy-import Ultralytics trainer CLI**

Use `YOLO(config.model)`, automatically select device `0` when CUDA is available otherwise `cpu`, use `batch=-1`, pass the fixed project/name output location, and write `training_summary.json` from `results.csv` plus validation metrics if available.

- [ ] **Step 4: Run the training configuration test**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_train_yolo.py -q`

Expected: PASS without requiring Ultralytics installed.

### Task 5: Run operational validation and train

**Files:**
- Read: `train/raw_cropped/*`
- Create: `train/yolo_dataset/*`
- Create: `train/runs/shellshock_yolo11n_v1/*`

- [ ] **Step 1: Run full unit tests**

Run: `\.venv\Scripts\python.exe -m pytest -q`

Expected: PASS or report pre-existing failures separately.

- [ ] **Step 2: Run dataset preparation**

Run: `\.venv\Scripts\python.exe prepare_yolo_dataset.py --raw-dir train/raw_cropped --output-dir train/yolo_dataset --seed 42 --preview-count 12`

Expected: summary with total/train/val/excluded counts, all ten classes, warnings for zero/rare classes, and reports/previews.

- [ ] **Step 3: Inspect generated previews and generated YAML**

Verify previews visually and verify `names` contains every ID 0 through 9. If unmatched enemy candidates are present, report their count and stop before training for manual review or explicit user override.

- [ ] **Step 4: Start first training only after candidate review produces no unresolved exclusions or user accepts their exclusion**

Run: `\.venv\Scripts\python.exe train_yolo.py --data train/yolo_dataset/dataset.yaml --project train/runs --name shellshock_yolo11n_v1`

Expected: Ultralytics writes weights, results CSV, curves, and validation artifacts under the fixed run directory.

- [ ] **Step 5: Parse final metrics**

Read the final validation metrics and report best weight path, mAP50, mAP50-95, precision, recall, per-class AP, and data imbalance observations.

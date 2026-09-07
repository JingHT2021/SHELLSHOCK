# ShellShock Five-Fold and Final Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a de-duplicated 65-image corpus from the enemy and portal-review batches, run deterministic five-fold YOLO fine-tuning from the first model, and train a final prediction model on all 65 images.

**Architecture:** A new corpus builder chooses `review_portal_mismatch` labels over raw labels for the two overlapping stems, merges class-0 supplemental labels, applies the existing safe crop transform, and emits a manifest. A deterministic multilabel five-fold splitter writes five derived YOLO datasets; each fold and the final all-data model start from the preserved v1 `best.pt` and write into separate run directories.

**Tech Stack:** Python 3, OpenCV, NumPy, Ultralytics YOLO11, PyTorch CUDA, pytest.

---

### Task 1: Test and build the merged corpus

**Files:**
- Modify: `shellshock_detector/yolo_dataset.py`
- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Write a failing source-priority test**

```python
def test_build_merged_corpus_prefers_review_labels_and_merges_enemy_supplements(tmp_path):
    raw, review, supplemental = tmp_path / "raw", tmp_path / "review", tmp_path / "supplemental"
    for directory in (raw, review, supplemental): directory.mkdir()
    assert cv2.imwrite(str(raw / "scene.png"), np.zeros((1800, 3840, 3), dtype=np.uint8))
    assert cv2.imwrite(str(review / "scene.png"), np.zeros((1800, 3840, 3), dtype=np.uint8))
    (raw / "scene.txt").write_text("5 .5 .5 .1 .1\n", encoding="utf-8")
    (review / "scene.txt").write_text("6 .5 .5 .1 .1\n", encoding="utf-8")
    (supplemental / "scene.txt").write_text("0 .2 .2 .1 .1\n", encoding="utf-8")
    summary = build_merged_corpus(raw, review, supplemental, tmp_path / "corpus")
    assert summary["unique_images"] == 1
    assert (tmp_path / "corpus" / "labels" / "scene.txt").read_text(encoding="utf-8").splitlines()[0].startswith("6 ")
    assert "0 0.200000" in (tmp_path / "corpus" / "labels" / "scene.txt").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run and verify the test fails because `build_merged_corpus` is missing**

Run: `pytest tests/test_yolo_dataset.py::test_build_merged_corpus_prefers_review_labels_and_merges_enemy_supplements -v`

Expected: FAIL with import error.

- [ ] **Step 3: Implement corpus construction**

```python
def build_merged_corpus(raw_dir, review_dir, supplemental_dir, output_dir, *, overwrite=False):
    # Index images by stem; review image/label wins on overlap.
    # Parse labels using source dimensions, append nonduplicate class-0 supplements,
    # crop 3840x2000 images to y=0..1850, then write images/, labels/, manifest.csv.
    # Refuse to replace corpus output unless overwrite=True.
```

Write `source=review` or `source=raw` in the manifest and return class statistics.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `pytest tests/test_yolo_dataset.py::test_build_merged_corpus_prefers_review_labels_and_merges_enemy_supplements -v`

Expected: PASS.

### Task 2: Deterministic five-fold datasets

**Files:**
- Modify: `shellshock_detector/yolo_dataset.py`
- Modify: `tests/test_yolo_dataset.py`
- Create: `prepare_yolo_fivefold.py`

- [ ] **Step 1: Write a failing fold-coverage test**

```python
def test_make_fivefold_splits_assigns_every_stem_to_exactly_one_validation_fold():
    folds = make_multilabel_folds({"a": {0}, "b": {0, 5}, "c": {2}, "d": {6}, "e": {3}, "f": {4}}, n_splits=5, seed=42)
    assert sorted(stem for fold in folds for stem in fold) == ["a", "b", "c", "d", "e", "f"]
    assert len({stem for fold in folds for stem in fold}) == 6
```

- [ ] **Step 2: Run and verify the test fails because `make_multilabel_folds` is missing**

Run: `pytest tests/test_yolo_dataset.py::test_make_fivefold_splits_assigns_every_stem_to_exactly_one_validation_fold -v`

Expected: FAIL with import error.

- [ ] **Step 3: Implement deterministic greedy multilabel folds and fold writers**

```python
def make_multilabel_folds(stem_classes, n_splits=5, seed=42):
    # Shuffle deterministically, allocate label-rich and rare-class samples to the
    # currently least represented fold, then balance image counts.

def write_fold_datasets(corpus_dir, output_dir, folds):
    # Hard-link source images when possible; copy labels; write one dataset.yaml,
    # train.txt and val.txt per fold; emit folds.csv and per-fold class counts.
```

- [ ] **Step 4: Run focused corpus/fold tests and create the real corpus**

Run: `pytest tests/test_yolo_dataset.py -v`

Then: `python prepare_yolo_fivefold.py --overwrite`

Expected: tests PASS; corpus has 65 samples and five fold directories.

### Task 3: Five-fold and final training runner

**Files:**
- Create: `train_yolo_fivefold.py`
- Create: `tests/test_train_yolo_fivefold.py`

- [ ] **Step 1: Write a failing plan-generation test**

```python
def test_fold_run_names_are_unique_and_final_run_is_separate():
    names = make_run_names("shellshock_yolo11n_cv65")
    assert names["folds"] == ["shellshock_yolo11n_cv65_fold1", "shellshock_yolo11n_cv65_fold2", "shellshock_yolo11n_cv65_fold3", "shellshock_yolo11n_cv65_fold4", "shellshock_yolo11n_cv65_fold5"]
    assert names["final"] == "shellshock_yolo11n_cv65_final_all"
```

- [ ] **Step 2: Run and verify it fails because the new runner is missing**

Run: `pytest tests/test_train_yolo_fivefold.py -v`

Expected: FAIL with module import error.

- [ ] **Step 3: Implement sequential fold training and aggregate report**

```python
def run_fivefold(corpus_root, base_weights, project, prefix, config):
    # Call existing train_yolo.train once per fold with its own dataset yaml.
    # Save fold_metrics.csv, mean/std metrics, and a JSON report.

def run_final_all_data(corpus_root, base_weights, project, prefix, config):
    # Train from base_weights using all 65 labels and save final_all data yaml/run.
```

Set `epochs=40`, `imgsz=960`, `batch=8`, `lr0=0.001`, seed 42. Do not use `resume`, and do not overwrite prior runs.

- [ ] **Step 4: Run runner tests and launch five folds then final all-data fine-tune**

Run: `pytest tests/test_train_yolo_fivefold.py tests/test_train_yolo.py -v`

Then: `python train_yolo_fivefold.py --epochs 40 --batch 8 --lr0 0.001`

Expected: unit tests PASS; five fold directories and one final all-data run are created.

### Task 4: Verification and report

**Files:**
- Generated: `train/fivefold/fold_metrics.csv`
- Generated: `train/fivefold/cross_validation_summary.json`
- Generated: `train/runs/shellshock_yolo11n_cv65_final_all/weights/best.pt`

- [ ] **Step 1: Verify every fold has best.pt, last.pt, results.csv, and 13 validation samples**

Run: inspect five fold manifests and run directories.

Expected: every source stem appears in exactly one fold validation manifest; every run contains weights and metrics.

- [ ] **Step 2: Report mean ± standard deviation for Precision, Recall, mAP50, and mAP50-95**

Use only the five fold validation metrics for cross-validation statistics. List sparse class warnings separately.

- [ ] **Step 3: Verify the final all-data model and state its intended use**

Run: inspect final `training_summary.json` and weights.

Expected: final model exists for practical inference; its training metrics are not presented as an unbiased validation estimate.

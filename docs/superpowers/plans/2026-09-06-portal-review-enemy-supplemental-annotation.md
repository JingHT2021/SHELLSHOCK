# Portal Review and Enemy Supplemental Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export every image whose orange and blue portal-label counts differ, and provide a safe batch UI for adding missing enemy labels without modifying raw annotations.

**Architecture:** Add pure, testable portal-audit and supplemental-label helpers to `shellshock_detector/yolo_dataset.py`. Keep OpenCV window code in a new `annotate_enemies.py` command, which maps display clicks to full-resolution coordinates and writes only additional class-0 boxes under `train/supplemental_labels`. The existing preparation process consumes these additive labels while removing near-duplicate boxes.

**Tech Stack:** Python 3, OpenCV, NumPy, pytest, existing YOLO label utilities.

---

## File structure

- Modify: `shellshock_detector/yolo_dataset.py` — portal count audit/export, class-0 supplemental merge de-duplication, shared box conversion helpers.
- Create: `export_portal_mismatches.py` — non-destructive CLI exporting review copies and manifest.
- Create: `annotate_enemies.py` — OpenCV click-to-add batch annotation UI.
- Modify: `prepare_yolo_dataset.py` — report supplemental-label source and duplicate removal in output summary.
- Modify: `tests/test_yolo_dataset.py` — portal export and supplemental merge regression tests.
- Create: `tests/test_annotate_enemies.py` — coordinate scaling, box-size, range selection, and label persistence tests.

### Task 1: Portal-count audit and non-destructive export

**Files:**
- Modify: `shellshock_detector/yolo_dataset.py`
- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Write failing tests for count mismatch detection and export**

```python
def test_export_portal_mismatches_copies_only_unequal_pairs_and_writes_manifest(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    for stem, labels in {"unequal": "5 .5 .5 .1 .1\n", "equal": "5 .5 .5 .1 .1\n6 .6 .5 .1 .1\n"}.items():
        image = raw / f"{stem}.png"
        assert cv2.imwrite(str(image), np.zeros((20, 20, 3), dtype=np.uint8))
        (raw / f"{stem}.txt").write_text(labels, encoding="utf-8")
    summary = export_portal_mismatches(raw, tmp_path / "review")
    assert summary["mismatch_images"] == 1
    assert (tmp_path / "review" / "unequal.png").exists()
    assert not (tmp_path / "review" / "equal.png").exists()
    assert "unequal.png,1,0,orange_blue_count_mismatch" in (tmp_path / "review" / "manifest.csv").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the new test and verify it fails because `export_portal_mismatches` does not exist**

Run: `pytest tests/test_yolo_dataset.py::test_export_portal_mismatches_copies_only_unequal_pairs_and_writes_manifest -v`

Expected: FAIL with an import error for `export_portal_mismatches`.

- [ ] **Step 3: Implement the export helper**

```python
def portal_counts(boxes: list[YoloBox]) -> tuple[int, int]:
    return (sum(box.class_id == 5 for box in boxes), sum(box.class_id == 6 for box in boxes))

def export_portal_mismatches(raw_dir: Path, review_dir: Path) -> dict[str, int]:
    review_dir.mkdir(parents=True, exist_ok=False)
    rows = [("image", "portal_orange", "portal_blue", "reason")]
    for image_path in _image_paths(raw_dir):
        boxes, errors = parse_yolo_label_text((raw_dir / f"{image_path.stem}.txt").read_text(encoding="utf-8"), 1, 1)
        orange, blue = portal_counts(boxes)
        if orange != blue:
            shutil.copy2(image_path, review_dir / image_path.name)
            shutil.copy2(raw_dir / f"{image_path.stem}.txt", review_dir / f"{image_path.stem}.txt")
            rows.append((image_path.name, orange, blue, "orange_blue_count_mismatch"))
    with (review_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return {"mismatch_images": len(rows) - 1}
```

Use the actual image dimensions when parsing labels, reject an existing review directory unless an explicit overwrite argument is supplied, and copy same-stem JSON from `train/geometry` when it exists.

- [ ] **Step 4: Run the portal export test and verify it passes**

Run: `pytest tests/test_yolo_dataset.py::test_export_portal_mismatches_copies_only_unequal_pairs_and_writes_manifest -v`

Expected: PASS.

### Task 2: Supplemental enemy-label data contract

**Files:**
- Modify: `shellshock_detector/yolo_dataset.py`
- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Write a failing duplicate-removal test**

```python
def test_supplemental_enemy_boxes_append_new_boxes_but_not_duplicate_existing_box(tmp_path):
    supplemental = tmp_path / "scene.txt"
    supplemental.write_text("0 0.5 0.5 0.1 0.1\n0 0.8 0.5 0.1 0.1\n", encoding="utf-8")
    existing = [YoloBox(0, .5, .5, .1, .1)]
    merged, errors, added, duplicates = merge_supplemental_boxes(existing, supplemental, 3840, 1800)
    assert errors == [] and added == 1 and duplicates == 1
    assert [(box.center_x, box.center_y) for box in merged] == [(.5, .5), (.8, .5)]
```

- [ ] **Step 2: Run the test and verify it fails because the public merge helper does not exist**

Run: `pytest tests/test_yolo_dataset.py::test_supplemental_enemy_boxes_append_new_boxes_but_not_duplicate_existing_box -v`

Expected: FAIL with an import error for `merge_supplemental_boxes`.

- [ ] **Step 3: Implement IoU-based class-0 merge accounting**

```python
def merge_supplemental_boxes(existing, supplemental_path, width, height, iou_threshold=0.90):
    supplemental, errors = parse_yolo_label_text(supplemental_path.read_text(encoding="utf-8"), width, height)
    merged, added, duplicates = list(existing), 0, 0
    for candidate in supplemental:
        if candidate.class_id != 0:
            errors.append("supplemental:class_id_must_be_enemy")
        elif any(box.class_id == 0 and _box_iou(box, candidate) >= iou_threshold for box in merged):
            duplicates += 1
        else:
            merged.append(candidate); added += 1
    return merged, errors, added, duplicates
```

Make `prepare_dataset` call this helper, and add `supplemental_enemy_boxes_added` and `supplemental_enemy_duplicates_skipped` to its summary.

- [ ] **Step 4: Run the focused data tests and verify they pass**

Run: `pytest tests/test_yolo_dataset.py -v`

Expected: PASS.

### Task 3: Portal export CLI

**Files:**
- Create: `export_portal_mismatches.py`
- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Write a failing CLI argument test**

```python
def test_portal_export_parser_defaults_to_raw_and_review_locations():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.raw_dir == Path("train/raw_cropped")
    assert args.review_dir == Path("train/review_portal_mismatch")
```

- [ ] **Step 2: Run it and verify `build_parser` is unavailable**

Run: `pytest tests/test_yolo_dataset.py::test_portal_export_parser_defaults_to_raw_and_review_locations -v`

Expected: FAIL with an import error from `export_portal_mismatches`.

- [ ] **Step 3: Implement the small CLI and summary output**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export orange/blue portal-count mismatches for review.")
    parser.add_argument("--raw-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--review-dir", type=Path, default=Path("train/review_portal_mismatch"))
    parser.add_argument("--overwrite", action="store_true")
    return parser
```

Call `export_portal_mismatches`, print the destination and mismatch count, and never delete a review directory; explicit overwrite may only replace that derived review directory.

- [ ] **Step 4: Run the CLI test and then export the real review copy**

Run: `pytest tests/test_yolo_dataset.py::test_portal_export_parser_defaults_to_raw_and_review_locations -v`

Then: `python export_portal_mismatches.py`

Expected: PASS, then a new `train/review_portal_mismatch/manifest.csv` with the actual mismatch count.

### Task 4: Testable click coordinate and supplemental persistence helpers

**Files:**
- Create: `annotate_enemies.py`
- Create: `tests/test_annotate_enemies.py`

- [ ] **Step 1: Write failing tests for range selection, scaled click mapping, and label writing**

```python
def test_select_images_includes_timestamp_boundaries(tmp_path):
    for stem in ("20260906_161533", "20260906_161534", "20260906_171759", "20260906_171800"):
        assert cv2.imwrite(str(tmp_path / f"{stem}.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    assert [p.stem for p in select_images(tmp_path, "20260906_161534", "20260906_171759")] == ["20260906_161534", "20260906_171759"]

def test_display_point_maps_back_to_original_coordinates():
    assert display_to_image_point((480, 225), (960, 450), (3840, 1800)) == (1920, 900)

def test_write_supplemental_enemy_label_uses_reference_box_size(tmp_path):
    write_enemy_supplement(tmp_path, "scene", [(1920, 900)], 3840, 1800)
    assert (tmp_path / "scene.txt").read_text(encoding="utf-8").startswith("0 0.500000 0.500000")
```

- [ ] **Step 2: Run tests and verify they fail because the module is missing**

Run: `pytest tests/test_annotate_enemies.py -v`

Expected: FAIL with `ModuleNotFoundError: annotate_enemies`.

- [ ] **Step 3: Implement pure helpers using existing training-data box semantics**

```python
def display_to_image_point(point, display_size, image_size):
    return (round(point[0] * image_size[0] / display_size[0]), round(point[1] * image_size[1] / display_size[1]))

def write_enemy_supplement(directory, stem, points, width, height):
    directory.mkdir(parents=True, exist_ok=True)
    labels = [yolo_label_line(0, point, width, height) for point in points]
    (directory / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
```

Import the existing `DEFAULT_BOX_SIZE_AT_REFERENCE` and `yolo_label_line` rather than duplicating fixed box dimensions.

- [ ] **Step 4: Run helper tests and verify they pass**

Run: `pytest tests/test_annotate_enemies.py -v`

Expected: PASS.

### Task 5: OpenCV batch annotation UI and real-data dry run

**Files:**
- Modify: `annotate_enemies.py`
- Modify: `tests/test_annotate_enemies.py`

- [ ] **Step 1: Write a failing key-action test**

```python
@pytest.mark.parametrize(("key", "action"), [(81, "previous"), (83, "next"), (27, "quit")])
def test_key_action_maps_arrow_and_escape_keys(key, action):
    assert key_action(key) == action
```

- [ ] **Step 2: Run it and verify it fails because `key_action` is absent**

Run: `pytest tests/test_annotate_enemies.py::test_key_action_maps_arrow_and_escape_keys -v`

Expected: FAIL with an import error for `key_action`.

- [ ] **Step 3: Implement the interactive loop**

```python
def run_annotation(images, supplemental_dir):
    # Load original labels and existing supplemental enemy boxes for the current image.
    # Draw existing labels in gray and newly-added enemy boxes in red after every mouse event.
    # EVENT_LBUTTONDOWN appends a mapped image point; EVENT_RBUTTONDOWN pops one point.
    # Right arrow writes the current supplemental file and advances; left arrow saves then goes back.
    # Esc saves and exits; window title lists index, filename, and control hints.
```

Use a maximum display width of 1600 pixels while preserving aspect ratio. Only save class-0 additions, and write a matching feedback preview JPEG under `train/enemy_annotation_previews/`.

- [ ] **Step 4: Run UI-independent tests and manually dry-run the selected 30-image range**

Run: `pytest tests/test_annotate_enemies.py -v`

Then: `python annotate_enemies.py --start 20260906_161534 --end 20260906_171759`

Expected: tests PASS; the window opens on the first of 30 selected images, left click draws an immediately visible fixed-size enemy box, and arrow navigation never edits `train/raw_cropped`.

### Task 6: Full verification and user-facing operating instructions

**Files:**
- Modify: `README.md` only if it already documents local data preparation commands.

- [ ] **Step 1: Run all new and existing test suites**

Run: `pytest tests/test_yolo_dataset.py tests/test_annotate_enemies.py tests/test_train_yolo.py -v`

Expected: all targeted tests PASS. Separately record any pre-existing unrelated failures from the full repository suite rather than changing unrelated code.

- [ ] **Step 2: Inspect generated artifacts without touching raw data**

Run: `Get-ChildItem train/review_portal_mismatch -File; Get-Content train/review_portal_mismatch/manifest.csv -TotalCount 5`

Expected: copied review images/labels and manifest are present; raw directory file hashes for a spot-check remain unchanged.

- [ ] **Step 3: Document exact operator commands**

```text
python export_portal_mismatches.py
python annotate_enemies.py --start 20260906_161534 --end 20260906_171759
python prepare_yolo_dataset.py --overwrite
python train_yolo.py --resume train/runs/shellshock_yolo11n_v1/weights/last.pt
```

Explain that portal review corrections must be brought into the active source-label workflow before dataset preparation, while enemy supplements are automatically merged.


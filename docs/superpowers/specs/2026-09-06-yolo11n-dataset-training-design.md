# ShellShock YOLO11n Dataset and Training Design

## Goal

Build a repeatable, non-destructive pipeline that derives a validated ShellShock YOLO Detection dataset from `train/raw_cropped`, preserves class identifiers `0` through `9`, creates review artifacts, and trains a first YOLO11n model only after validation succeeds.

## Dataset contract

- Raw images and label files remain untouched.
- `3840x2000` images are cropped to pixel rows `[0, 1850)`. Every affected YOLO box is denormalized to pixels, clipped to the cropped image, discarded only when its clipped area is empty, and then normalized against `3840x1850`.
- Other image dimensions are copied without geometric change.
- Invalid or unpaired samples are excluded from the derived dataset and recorded with a reason in a CSV report; processing continues for valid samples.
- The class space is permanently `0..9`: enemy, placeholder_1, self, obstacle_circle, obstacle_line, portal_orange, portal_blue, placeholder_7, placeholder_8, Triple_damage. Placeholder classes are valid but have zero samples until future annotations arrive.

## Partially annotated enemy images

YOLO cannot mark an unlabelled object as "ignore". Training on a frame where another enemy is present but unlabelled would teach the model that this enemy appearance is background.

The preparation pipeline therefore creates red-enemy candidate review images and a machine-readable candidate report. Before a candidate is manually confirmed and added in a separate supplemental-label layer, a frame containing an unlabelled candidate is excluded from the training dataset. Supplemental labels are merged only into the derived dataset; raw labels are never changed.

## Outputs

`train/yolo_dataset/` contains the train/validation image and label trees, `dataset.yaml`, reports, and rendered label previews. `train/runs/shellshock_yolo11n_v1/` is the fixed training output location. A preparation command prints total, train, validation, excluded, empty-label, and per-class image/box counts.

## Split and validation

The pipeline uses a deterministic seed of 42 and a multi-label greedy stratification that targets an 80/20 image split while attempting to keep every observed class in both splits. It warns if a class has fewer than two eligible images or cannot occur in both sets. It rejects malformed YOLO rows, out-of-range identifiers, non-finite values, out-of-image boxes, invalid dimensions, unreadable images, and unmatched image/label files from the derived set.

## Training

The training entry point uses Ultralytics `YOLO("yolo11n.pt")` for detection, uses CUDA when available, and centralizes parameters. Initial defaults are `imgsz=960`, `epochs=200`, `batch=-1`, and `seed=42`. Ultralytics resizes and letterboxes source images during training; source images are not physically resized.

## Verification

Unit tests cover crop/clip/renormalize behavior, invalid-sample exclusion, sparse class validation, and deterministic stratification. The operational validation runs preparation, checks reports and YAML, verifies rendered previews, then starts the long training run. Training results are parsed and reported with model artifact paths and standard validation metrics.

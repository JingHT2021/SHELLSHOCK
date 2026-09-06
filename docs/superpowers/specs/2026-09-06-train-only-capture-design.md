# Train-only capture storage

## Goal

Store new captures only as training data under `train/` and remove the
historical runtime `output/` directory.

## Design

Capture processing retains analysis, wind detection, ballistic calculation,
and YOLO training-sample persistence. It no longer creates raw screenshots,
JSON result files, or annotated screenshots in an output directory. The CLI
has no output-directory option and reports the saved training image, label,
and preview paths.

## Safety

The existing `output/` directory is deleted only after the code and tests
confirm that new captures no longer use it. The target is the exact project
path `C:\Users\jingh\Desktop\game\shellshock\output`.

## Verification

Tests assert that capture processing writes the three training artifacts and
does not create an output directory or runtime files.

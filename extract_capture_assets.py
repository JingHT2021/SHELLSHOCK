"""Generate wind and wormhole crops from completed full-frame captures."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from shellshock_detector.dataset_capture import save_detected_capture


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def extract_assets(full_dir: Path, output_dir: Path, start: str = "", end: str = "\U0010ffff") -> int:
    full_dir = Path(full_dir)
    output_dir = Path(output_dir)
    processed = 0
    for path in sorted(full_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not (start <= path.stem <= end):
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipped unreadable image: {path}")
            continue
        result = save_detected_capture(image, output_dir, path.stem)
        processed += 1
        print(f"Processed {path.name}: {result}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=Path("train/yolo_captures"))
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="\U0010ffff")
    args = parser.parse_args()
    count = extract_assets(args.capture_dir / "full", args.capture_dir, args.start, args.end)
    print(f"Finished: {count} full captures processed")


if __name__ == "__main__":
    main()

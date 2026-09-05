"""Batch-add strict pink obstacle labels to existing ShellShock training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shellshock_detector.training_data import annotate_pink_obstacle_sample


def annotate_directory(
    annotated_dir: Path,
    labels_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Process same-stem image/label pairs and return counts by obstacle class."""
    summary = {"images": 0, "added_circle": 0, "added_line": 0, "missing_labels": 0}
    for image_path in sorted(annotated_dir.glob("*.png")):
        summary["images"] += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            summary["missing_labels"] += 1
            continue
        added_classes = annotate_pink_obstacle_sample(
            image_path,
            label_path,
            image_path,
            dry_run=dry_run,
        )
        summary["added_circle"] += added_classes.count(3)
        summary["added_line"] += added_classes.count(4)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated-dir", type=Path, default=Path("train/annotated"))
    parser.add_argument("--labels-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--dry-run", action="store_true", help="detect and report without changing files")
    args = parser.parse_args()
    print(json.dumps(annotate_directory(args.annotated_dir, args.labels_dir, dry_run=args.dry_run), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

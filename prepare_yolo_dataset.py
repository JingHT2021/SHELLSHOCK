"""Build the derived ShellShock YOLO Detection dataset without editing raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

from shellshock_detector.yolo_dataset import CLASS_NAMES, PreparationConfig, prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/yolo_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--portal-review-dir",
        type=Path,
        default=None,
        help="Optional reviewed portal-label directory; its same-stem .txt files override raw labels without changing raw data.",
    )
    parser.add_argument("--annotation-override-dir", type=Path, default=None)
    parser.add_argument("--exclude-unreviewed-enemy-candidates", action="store_true", help="Exclude frames with red enemy candidates. This is opt-in because color candidates require human review.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing derived output directory; raw data is never modified.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        args.raw_dir,
        args.output_dir,
        PreparationConfig(
            seed=args.seed,
            preview_count=args.preview_count,
            exclude_unlabelled_enemy_candidates=args.exclude_unreviewed_enemy_candidates,
            portal_review_dir=args.portal_review_dir,
            annotation_override_dir=args.annotation_override_dir,
        ),
        overwrite=args.overwrite,
    )
    print(f"Total images: {summary['total_images']}")
    print(f"Eligible images: {summary['eligible_images']}")
    print(f"Train images: {summary['train_images']}")
    print(f"Val images: {summary['val_images']}")
    print(f"Excluded images/files: {summary['excluded_images']}")
    print(f"Empty-label images: {summary['empty_label_images']}")
    print(f"Unlabelled enemy candidates: {summary['unlabelled_enemy_candidates']}")
    print(f"Supplemental enemy boxes added: {summary['supplemental_enemy_boxes_added']}")
    print(f"Supplemental enemy duplicates skipped: {summary['supplemental_enemy_duplicates_skipped']}")
    print(f"Portal review label overrides: {summary['portal_review_label_overrides']}")
    for class_id, class_name in CLASS_NAMES.items():
        counts = summary["all"][class_name]
        train_counts = summary["train"][class_name]
        val_counts = summary["val"][class_name]
        print(f"\n{class_id} {class_name}:")
        print(f"  all:   images = {counts['images']}, boxes = {counts['boxes']}")
        print(f"  train: images = {train_counts['images']}, boxes = {train_counts['boxes']}")
        print(f"  val:   images = {val_counts['images']}, boxes = {val_counts['boxes']}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(f"\nDataset YAML: {args.output_dir / 'dataset.yaml'}")
    print(f"Reports and previews: {args.output_dir}")


if __name__ == "__main__":
    main()

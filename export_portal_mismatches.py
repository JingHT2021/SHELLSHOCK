"""Export samples whose orange and blue portal-label counts differ for manual review."""

from __future__ import annotations

import argparse
from pathlib import Path

from shellshock_detector.yolo_dataset import export_portal_mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--review-dir", type=Path, default=Path("train/review_portal_mismatch"))
    parser.add_argument("--overwrite", action="store_true", help="Replace only the derived review directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = export_portal_mismatches(args.raw_dir, args.review_dir, overwrite=args.overwrite)
    print(f"Portal-count mismatch images: {summary['mismatch_images']}")
    print(f"Invalid-label images skipped: {summary['invalid_label_images']}")
    print(f"Review copies and manifest: {args.review_dir}")


if __name__ == "__main__":
    main()

"""Merge ShellShock enemy and portal-review labels into deterministic five-fold datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from shellshock_detector.yolo_dataset import (
    build_merged_corpus,
    make_multilabel_folds,
    parse_yolo_label_text,
    write_fivefold_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--review-dir", type=Path, default=Path("train/review_portal_mismatch"))
    parser.add_argument("--supplemental-dir", type=Path, default=Path("train/supplemental_labels"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("train/yolo_cv_corpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/yolo_fivefold"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-stem-start",
        help="inclusive raw image stem lower bound; does not filter reviewed portal samples",
    )
    parser.add_argument(
        "--raw-stem-end",
        help="inclusive raw image stem upper bound; does not filter reviewed portal samples",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if bool(args.raw_stem_start) != bool(args.raw_stem_end):
        parser.error("--raw-stem-start and --raw-stem-end must be used together")
    raw_stems = None
    if args.raw_stem_start is not None:
        raw_stems = {
            path.stem
            for path in args.raw_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            and args.raw_stem_start <= path.stem <= args.raw_stem_end
        }
        print(
            f"Raw selection: {args.raw_stem_start}..{args.raw_stem_end} "
            f"({len(raw_stems)} image(s))"
        )
    summary = build_merged_corpus(
        args.raw_dir,
        args.review_dir,
        args.supplemental_dir,
        args.corpus_dir,
        raw_stems=raw_stems,
        overwrite=args.overwrite,
    )
    classes = {}
    for label_path in (args.corpus_dir / "labels").glob("*.txt"):
        boxes, errors = parse_yolo_label_text(label_path.read_text(encoding="utf-8"), 1, 1)
        if errors:
            raise ValueError(f"invalid corpus label {label_path}: {errors}")
        classes[label_path.stem] = {box.class_id for box in boxes}
    folds = make_multilabel_folds(classes, seed=args.seed)
    fold_summary = write_fivefold_datasets(args.corpus_dir, args.output_dir, folds, overwrite=args.overwrite)
    print(f"Merged images: {summary['unique_images']} (raw={summary['raw_source_images']}, review={summary['review_source_images']})")
    print(f"Supplemental enemy boxes added: {summary['supplemental_enemy_boxes_added']}")
    print(f"Five folds written: {fold_summary['folds']}; final all-data dataset: {fold_summary['final_dataset']}")


if __name__ == "__main__":
    main()

"""Prepare a single-class cropped YOLO Pose dataset for self-center refinement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

import cv2

from shellshock.datasets.yolo import parse_yolo_label_text
from shellshock.domain.world import DetectionBox
from shellshock.perception.self_center_model import crop_self_roi, make_self_center_label

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _group_key(stem: str) -> str:
    return stem[:13]


def _read_center(path: Path) -> tuple[float, float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    center = payload.get("self_center") if isinstance(payload, dict) else None
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    try:
        return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None


def _find_self_box(label_path: Path, width: int, height: int) -> DetectionBox | None:
    try:
        boxes, errors = parse_yolo_label_text(label_path.read_text(encoding="utf-8"), width, height)
    except OSError:
        return None
    if errors:
        raise ValueError(f"invalid labels in {label_path}: {' | '.join(errors)}")
    candidates = [box for box in boxes if box.class_id == 2]
    if not candidates:
        return None
    box = candidates[0]
    return DetectionBox(
        "self", (box.center_x - box.width / 2) * width,
        (box.center_y - box.height / 2) * height,
        box.width * width, box.height * height, 1.0,
    )


def prepare(
    raw_dir: Path = Path("train/annotate_check"),
    output_dir: Path = Path("train/self_center_dataset"),
    *, padding: float = 0.15,
    val_fraction: float = 0.20,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    image_root = raw_dir / "images"
    label_root = raw_dir / "labels"
    geometry_root = raw_dir / "pose_geometry"
    samples: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for image_path in sorted(image_root.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        label_path = label_root / f"{image_path.stem}.txt"
        geometry_path = geometry_root / f"{image_path.stem}.json"
        if image is None or not label_path.exists():
            skipped.append({"stem": image_path.stem, "reason": "missing_image_or_label"})
            continue
        center = _read_center(geometry_path)
        box = _find_self_box(label_path, image.shape[1], image.shape[0])
        if center is None or box is None:
            skipped.append({"stem": image_path.stem, "reason": "missing_self_box_or_center"})
            continue
        roi, origin = crop_self_roi(image, box, padding=padding, required_point=center)
        label = make_self_center_label(center=center, origin=origin, size=(roi.shape[1], roi.shape[0]), box=box)
        samples.append({"stem": image_path.stem, "image": image_path, "roi": roi, "label": label,
                        "origin": list(origin), "size": [int(roi.shape[1]), int(roi.shape[0])],
                        "center": list(center), "group": _group_key(image_path.stem)})

    groups: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        groups.setdefault(str(sample["group"]), []).append(sample)
    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    val_count = 0 if len(group_names) < 2 else max(1, min(len(group_names) - 1, round(len(group_names) * val_fraction)))
    val_groups = set(group_names[:val_count])
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    audit = []
    for sample in samples:
        split = "val" if sample["group"] in val_groups else "train"
        image_path = output_dir / "images" / split / f"{sample['stem']}.png"
        label_path = output_dir / "labels" / split / f"{sample['stem']}.txt"
        cv2.imwrite(str(image_path), sample["roi"])
        label_path.write_text(str(sample["label"]) + "\n", encoding="utf-8")
        audit.append({key: sample[key] for key in ("stem", "origin", "size", "center", "group")} | {"split": split})
    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [1, 3]\nnames:\n  0: self_center\n",
        encoding="utf-8",
    )
    (output_dir / "audit.json").write_text(json.dumps({"samples": audit, "skipped": skipped}, indent=2) + "\n", encoding="utf-8")
    return {"total_images": len(list(image_root.glob("*"))), "samples": len(samples), "train": sum(item["split"] == "train" for item in audit),
            "val": sum(item["split"] == "val" for item in audit), "skipped": skipped, "output_dir": str(output_dir.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("train/annotate_check"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/self_center_dataset"))
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

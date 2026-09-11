"""Fill missing self_center/self_muzzle fields in saved manual annotations.

The older annotator stored the barrel tip as ``self_muzzle``.  This migration
keeps that point and derives the tank center using the same geometry used by
replay_shellshock.py.  It can be run repeatedly; existing values are kept.
"""

from __future__ import annotations

from shellshock.config.paths import DATA_ROOT

import argparse
import json
import math
from pathlib import Path

import cv2


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def derive_self_center(muzzle, direction, angle_degrees, image_width, barrel_extension=35.0):
    """Convert a barrel-tip pixel point into the tank-center pixel point."""
    sign = 1.0 if str(direction).lower() == "right" else -1.0
    length = float(barrel_extension) * float(image_width) / 2560.0
    angle = math.radians(float(angle_degrees))
    return [
        float(muzzle[0]) - sign * length * math.cos(angle),
        float(muzzle[1]) + length * math.sin(angle),
    ]


def _find_image(root: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = root / "images" / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_metadata(root: Path, stem: str) -> dict:
    for directory in ("metadata", "shot_metadata"):
        path = root / directory / f"{stem}.json"
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                return value
    return {}


def _image_size(root: Path, stem: str) -> tuple[int, int]:
    image_path = _find_image(root, stem)
    if image_path is None:
        return 2560, 1440
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return 2560, 1440
    return int(image.shape[1]), int(image.shape[0])


def _self_box_center(root: Path, stem: str, width: int, height: int):
    label_path = root / "labels" / f"{stem}.txt"
    if not label_path.exists():
        return None
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    # Class 2 is the self vehicle in the current label schema.  Class 1 was
    # used by an older keypoint-oriented schema, so accept it as a fallback.
    candidates = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, box_w, box_h = (float(item) for item in parts[1:5])
        except ValueError:
            continue
        if class_id in (2, 1):
            candidates.append((class_id, x, y, box_w, box_h))
    if not candidates:
        return None
    _, x, y, box_w, box_h = sorted(candidates, key=lambda item: item[0] == 2, reverse=True)[0]
    return [x * width, y * height]


def _center_label_line(center, width, height, radius=8.0):
    left = max(0.0, float(center[0]) - radius)
    top = max(0.0, float(center[1]) - radius)
    right = min(float(width), float(center[0]) + radius)
    bottom = min(float(height), float(center[1]) + radius)
    return f"1 {(left + right) / 2 / width:.6f} {(top + bottom) / 2 / height:.6f} {(right - left) / width:.6f} {(bottom - top) / height:.6f}"


def _sync_center_label(root: Path, stem: str, center, width: int, height: int) -> bool:
    label_path = root / "labels" / f"{stem}.txt"
    if not label_path.exists():
        return False
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    if any(line.split()[:1] == ["1"] for line in lines if line.strip()):
        return False
    lines.append(_center_label_line(center, width, height))
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def migrate_root(root: Path, *, dry_run=False, barrel_extension=35.0) -> dict[str, int]:
    geometry_dir = root / "pose_geometry"
    stats = {"files": 0, "center_added": 0, "muzzle_added": 0, "label_center_added": 0, "unchanged": 0, "unresolved": 0}
    if not geometry_dir.exists():
        return stats

    for geometry_path in sorted(geometry_dir.glob("*.json")):
        stats["files"] += 1
        try:
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["unresolved"] += 1
            continue
        if not isinstance(geometry, dict):
            stats["unresolved"] += 1
            continue

        changed = False
        if "self_center" not in geometry:
            geometry["self_center"] = None
            changed = True
        if "self_muzzle" not in geometry:
            geometry["self_muzzle"] = None
            changed = True
            stats["muzzle_added"] += 1

        width, height = _image_size(root, geometry_path.stem)
        metadata = _load_metadata(root, geometry_path.stem)
        muzzle = geometry.get("self_muzzle")
        center = geometry.get("self_center")
        if center is None and isinstance(muzzle, (list, tuple)) and len(muzzle) == 2:
            direction = metadata.get("direction", "right")
            angle = metadata.get("angle_degrees", 90.0)
            geometry["self_center"] = derive_self_center(muzzle, direction, angle, width, barrel_extension)
            changed = True
            stats["center_added"] += 1
        elif center is None:
            fallback = _self_box_center(root, geometry_path.stem, width, height)
            if fallback is not None:
                geometry["self_center"] = fallback
                changed = True
                stats["center_added"] += 1

        if geometry.get("self_center") is None:
            stats["unresolved"] += 1

        if geometry.get("self_center") is not None and not dry_run:
            if _sync_center_label(root, geometry_path.stem, geometry["self_center"], width, height):
                stats["label_center_added"] += 1

        if changed and not dry_run:
            geometry_path.write_text(
                json.dumps(geometry, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        elif not changed:
            stats["unchanged"] += 1
    return stats


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[(DATA_ROOT / 'annotate_check'), (DATA_ROOT / 'annotate')],
        help="annotation roots to migrate",
    )
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing files")
    parser.add_argument("--barrel-extension", type=float, default=35.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for root in args.roots:
        stats = migrate_root(root, dry_run=args.dry_run, barrel_extension=args.barrel_extension)
        print(f"{root}: {stats}")


if __name__ == "__main__":
    main()

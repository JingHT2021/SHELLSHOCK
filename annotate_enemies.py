"""Add missing enemy boxes by mouse click without changing raw YOLO labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from shellshock_detector.training_data import DEFAULT_BOX_SIZE_AT_REFERENCE, yolo_label_line
from shellshock_detector.yolo_dataset import CLASS_NAMES, YoloBox, parse_yolo_label_text


def select_images(raw_dir: Path, start: str, end: str) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        (path for path in Path(raw_dir).iterdir() if path.suffix.lower() in extensions and start <= path.stem <= end),
        key=lambda path: path.name,
    )


def display_to_image_point(
    point: tuple[int, int], display_size: tuple[int, int], image_size: tuple[int, int]
) -> tuple[int, int]:
    return (
        round(point[0] * image_size[0] / display_size[0]),
        round(point[1] * image_size[1] / display_size[1]),
    )


def write_enemy_supplement(
    supplemental_dir: Path, stem: str, points: list[tuple[int, int]], image_width: int, image_height: int
) -> Path:
    supplemental_dir = Path(supplemental_dir)
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    label_path = supplemental_dir / f"{stem}.txt"
    lines = [yolo_label_line(0, point, image_width, image_height) for point in points]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return label_path


def key_action(key: int) -> str | None:
    if key in (81, 2424832):
        return "previous"
    if key in (83, 2555904):
        return "next"
    if key == 27:
        return "quit"
    return None


def _display_size(image_width: int, image_height: int, maximum_width: int = 1600, maximum_height: int = 1000) -> tuple[int, int]:
    scale = min(1.0, maximum_width / image_width, maximum_height / image_height)
    return max(1, round(image_width * scale)), max(1, round(image_height * scale))


def _box_points(box: YoloBox, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (round((box.center_x - box.width / 2) * width), round((box.center_y - box.height / 2) * height)),
        (round((box.center_x + box.width / 2) * width), round((box.center_y + box.height / 2) * height)),
    )


def _draw_annotations(image, original_boxes: list[YoloBox], added_points: list[tuple[int, int]]):
    preview = image.copy()
    height, width = preview.shape[:2]
    for box in original_boxes:
        left_top, right_bottom = _box_points(box, width, height)
        cv2.rectangle(preview, left_top, right_bottom, (150, 150, 150), 2)
        cv2.putText(preview, CLASS_NAMES[box.class_id], (left_top[0], max(18, left_top[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 2)
    scale = width / 1920.0
    half_width = round(DEFAULT_BOX_SIZE_AT_REFERENCE[0] * scale / 2)
    half_height = round(DEFAULT_BOX_SIZE_AT_REFERENCE[1] * scale / 2)
    for point in added_points:
        cv2.rectangle(preview, (point[0] - half_width, point[1] - half_height), (point[0] + half_width, point[1] + half_height), (0, 0, 255), 3)
        cv2.putText(preview, "enemy +", (point[0] - half_width, max(18, point[1] - half_height - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return preview


def _load_boxes(label_path: Path, width: int, height: int) -> list[YoloBox]:
    if not label_path.exists():
        return []
    boxes, errors = parse_yolo_label_text(label_path.read_text(encoding="utf-8"), width, height)
    if errors:
        raise ValueError(f"invalid labels in {label_path}: {' | '.join(errors)}")
    return boxes


def _supplemental_points(label_path: Path, width: int, height: int) -> list[tuple[int, int]]:
    return [
        (round(box.center_x * width), round(box.center_y * height))
        for box in _load_boxes(label_path, width, height)
        if box.class_id == 0
    ]


def run_annotation(images: list[Path], raw_dir: Path, supplemental_dir: Path, preview_dir: Path) -> None:
    if not images:
        raise ValueError("no images matched the selected timestamp range")
    window_name = "ShellShock enemy supplement"
    index = 0
    while 0 <= index < len(images):
        image_path = images[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read {image_path}")
        height, width = image.shape[:2]
        original_boxes = _load_boxes(Path(raw_dir) / f"{image_path.stem}.txt", width, height)
        added_points = _supplemental_points(Path(supplemental_dir) / f"{image_path.stem}.txt", width, height)
        display_width, display_height = _display_size(width, height)

        def redraw() -> None:
            rendered = _draw_annotations(image, original_boxes, added_points)
            rendered = cv2.resize(rendered, (display_width, display_height), interpolation=cv2.INTER_AREA)
            cv2.putText(rendered, f"{index + 1}/{len(images)} {image_path.name} | Left:add Right:undo arrows:save/navigate Esc:save/quit", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(window_name, rendered)

        def on_mouse(event: int, x: int, y: int, _flags: int, _userdata) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                added_points.append(display_to_image_point((x, y), (display_width, display_height), (width, height)))
                redraw()
            elif event == cv2.EVENT_RBUTTONDOWN and added_points:
                added_points.pop()
                redraw()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
        redraw()
        while True:
            action = key_action(cv2.waitKeyEx(0))
            if action is None:
                continue
            write_enemy_supplement(supplemental_dir, image_path.stem, added_points, width, height)
            preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_dir / f"{image_path.stem}.jpg"), _draw_annotations(image, original_boxes, added_points))
            if action == "quit":
                cv2.destroyAllWindows()
                return
            index += -1 if action == "previous" else 1
            break
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("train/raw_cropped"))
    parser.add_argument("--supplemental-dir", type=Path, default=Path("train/supplemental_labels"))
    parser.add_argument("--preview-dir", type=Path, default=Path("train/enemy_annotation_previews"))
    parser.add_argument("--start", default="20260906_161534")
    parser.add_argument("--end", default="20260906_171759")
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Ignore --start/--end and annotate every supported image in --raw-dir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.all_images:
        images = select_images(args.raw_dir, "", "\U0010ffff")
        print(f"Opening all {len(images)} images from {args.raw_dir}.")
    else:
        images = select_images(args.raw_dir, args.start, args.end)
        print(f"Opening {len(images)} images from {args.start} through {args.end}.")
    run_annotation(images, args.raw_dir, args.supplemental_dir, args.preview_dir)


if __name__ == "__main__":
    main()

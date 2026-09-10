"""Convert YOLO detections and editor annotations into one replay scene."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import hypot
from pathlib import Path
from typing import Iterable

from shellshock_detector.yolo_dataset import CLASS_NAMES, YoloBox, parse_yolo_label_text
from .world_geometry import DetectionBox, PoseKeypoint, World, build_world_from_image, build_world_from_image_with_diagnostics


@dataclass(frozen=True)
class AnnotationBox:
    name: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    source: str = "manual"
    keypoints: tuple[PoseKeypoint, ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass(frozen=True)
class LineAnnotation:
    start: tuple[float, float]
    end: tuple[float, float]
    source: str = "manual"


@dataclass
class SceneAnnotation:
    image_width: int
    image_height: int
    boxes: list[AnnotationBox] = field(default_factory=list)
    lines: list[LineAnnotation] = field(default_factory=list)
    self_center: tuple[float, float] | None = None
    self_muzzle: tuple[float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    deleted: list[dict[str, object]] = field(default_factory=list)


def _box_from_yolo(box: YoloBox, width: int, height: int, *, source: str, confidence: float = 1.0) -> AnnotationBox:
    return AnnotationBox(
        CLASS_NAMES[box.class_id],
        (box.center_x - box.width / 2) * width,
        (box.center_y - box.height / 2) * height,
        box.width * width,
        box.height * height,
        confidence,
        source,
    )


def _yolo_box_from_detection(box: DetectionBox) -> AnnotationBox:
    return AnnotationBox(box.name, box.x, box.y, box.width, box.height, box.confidence, "yolo", getattr(box, "keypoints", ()))


def yolo_detections_to_annotations(
    detections: Iterable[DetectionBox], image_width: int, image_height: int
) -> SceneAnnotation:
    return SceneAnnotation(
        image_width,
        image_height,
        [_yolo_box_from_detection(item) for item in detections],
    )


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None or not Path(path).exists():
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_manual_scene(
    label_path: Path | None,
    geometry_path: Path | None,
    metadata_path: Path | None,
    image_width: int,
    image_height: int,
) -> SceneAnnotation:
    boxes: list[AnnotationBox] = []
    if label_path is not None and Path(label_path).exists():
        text = Path(label_path).read_text(encoding="utf-8")
        parsed, errors = parse_yolo_label_text(text, image_width, image_height)
        if errors:
            raise ValueError(f"invalid labels in {label_path}: {' | '.join(errors)}")
        boxes = [_box_from_yolo(item, image_width, image_height, source="manual") for item in parsed]

    geometry = _read_json(geometry_path)
    lines = []
    for item in geometry.get("lines", ()):
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            continue
        start, end = item["start"], item["end"]
        if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == len(end) == 2:
            lines.append(LineAnnotation((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))))

    center = geometry.get("self_center")
    self_center = (float(center[0]), float(center[1])) if isinstance(center, (list, tuple)) and len(center) == 2 else None
    muzzle = geometry.get("self_muzzle")
    self_muzzle = (float(muzzle[0]), float(muzzle[1])) if isinstance(muzzle, (list, tuple)) and len(muzzle) == 2 else None
    metadata = _read_json(metadata_path)
    deleted = geometry.get("deleted", [])
    return SceneAnnotation(image_width, image_height, boxes, lines, self_center, self_muzzle, metadata, list(deleted) if isinstance(deleted, list) else [])


def _same_object(left: AnnotationBox, right: AnnotationBox) -> bool:
    if left.name != right.name:
        return False
    lx, ly = left.center
    rx, ry = right.center
    distance = hypot(lx - rx, ly - ry)
    scale = max(8.0, min(max(left.width, left.height), max(right.width, right.height)))
    intersection_width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    intersection_height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return distance <= scale * 0.75 or (union > 0 and intersection / union >= 0.10)


def _is_deleted(box: AnnotationBox, deleted: list[dict[str, object]]) -> bool:
    for item in deleted:
        if not isinstance(item, dict) or item.get("name") != box.name:
            continue
        try:
            point = (float(item["x"]), float(item["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if hypot(box.center[0] - point[0], box.center[1] - point[1]) <= max(box.width, box.height, 8.0):
            return True
    return False


def merge_annotations(yolo: SceneAnnotation, manual: SceneAnnotation) -> SceneAnnotation:
    """Overlay manual objects on YOLO objects while retaining unmatched YOLO objects."""
    result = SceneAnnotation(yolo.image_width, yolo.image_height, list(yolo.boxes), list(yolo.lines), yolo.self_center, yolo.self_muzzle, dict(yolo.metadata), list(manual.deleted))
    for manual_box in manual.boxes:
        matches = [index for index, item in enumerate(result.boxes) if _same_object(item, manual_box)]
        if matches:
            result.boxes[matches[0]] = manual_box
        else:
            result.boxes.append(manual_box)
    result.boxes = [item for item in result.boxes if not _is_deleted(item, result.deleted)]
    if manual.lines:
        result.lines = list(manual.lines)
    if manual.self_center is not None:
        result.self_center = manual.self_center
    if manual.self_muzzle is not None:
        result.self_muzzle = manual.self_muzzle
    result.metadata.update(manual.metadata)
    return result


def annotations_to_world(scene: SceneAnnotation, image) -> tuple[World, tuple[float, float] | None]:
    world, muzzle, _ = annotations_to_world_with_diagnostics(scene, image)
    return world, muzzle


def annotations_to_world_with_diagnostics(scene: SceneAnnotation, image) -> tuple[World, tuple[float, float] | None, list[dict[str, object]]]:
    detections = [DetectionBox(item.name, item.x, item.y, item.width, item.height, item.confidence, item.keypoints) for item in scene.boxes]
    world, diagnostics = build_world_from_image_with_diagnostics(detections, image)
    if scene.self_center is not None:
        from dataclasses import replace
        world = replace(world, self_position=scene.self_center)
    if scene.lines:
        from dataclasses import replace
        from .world_geometry import LineObstacle
        world = replace(world, lines=tuple(LineObstacle(item.start, item.end) for item in scene.lines))
    return world, scene.self_muzzle, diagnostics


def scene_to_dict(scene: SceneAnnotation) -> dict[str, object]:
    return {
        "image_width": scene.image_width,
        "image_height": scene.image_height,
        "boxes": [{**item.__dict__, "keypoints": [{"x": point.x, "y": point.y, "visible": point.visible, "confidence": point.confidence} for point in item.keypoints]} for item in scene.boxes],
        "lines": [{"start": list(item.start), "end": list(item.end), "source": item.source} for item in scene.lines],
        "self_center": list(scene.self_center) if scene.self_center else None,
        "metadata": scene.metadata,
        "deleted": scene.deleted,
    }


def save_manual_scene(scene: SceneAnnotation, label_path: Path, geometry_path: Path, metadata_path: Path, *, backup=True) -> None:
    """Persist an edited scene in the formats consumed by annotate_enemies.py."""
    for path in (label_path, geometry_path, metadata_path):
        path = Path(path)
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            backup_path.write_bytes(path.read_bytes())
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in scene.boxes:
        left = max(0.0, item.x)
        top = max(0.0, item.y)
        right = min(float(scene.image_width), item.x + item.width)
        bottom = min(float(scene.image_height), item.y + item.height)
        if right > left and bottom > top:
            class_id = next((key for key, value in CLASS_NAMES.items() if value == item.name), None)
            if class_id is not None:
                lines.append(f"{class_id} {(left + right) / 2 / scene.image_width:.6f} {(top + bottom) / 2 / scene.image_height:.6f} {(right-left) / scene.image_width:.6f} {(bottom-top) / scene.image_height:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text(json.dumps({"lines": [{"start": list(item.start), "end": list(item.end)} for item in scene.lines], "self_center": list(scene.self_center) if scene.self_center else None, "deleted": scene.deleted}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(scene.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

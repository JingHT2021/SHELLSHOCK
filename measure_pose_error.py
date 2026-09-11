"""Measure validation keypoint errors in pixels for a YOLO Pose model."""

from __future__ import annotations

from shellshock.config.paths import DATA_ROOT

from pathlib import Path
import numpy as np
from ultralytics import YOLO


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union else 0.0


def main():
    root = (DATA_ROOT / 'yolo_pose_dataset_v2')
    model = YOLO(str(DATA_ROOT / "runs/shellshock_yolo11n_pose_v1/weights/best.pt"))
    errors = []
    by_class = {}
    matched = 0
    for image_path in sorted((root / "images/val").glob("*")):
        image = __import__("cv2").imread(str(image_path))
        height, width = image.shape[:2]
        gt = []
        for line in (root / "labels/val" / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) != 11:
                continue
            cls = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
            box = ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)
            points = [(float(t[i]) * width, float(t[i + 1]) * height, int(float(t[i + 2]))) for i in (5, 8)]
            gt.append((cls, box, points))
        result = model.predict(str(image_path), imgsz=640, device="0", conf=0.25, verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            continue
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy().astype(int)
        pred_points = result.keypoints.xy.cpu().numpy()
        used = set()
        for cls, box, points in gt:
            candidates = sorted(((-iou(box, pred_boxes[j]), j) for j in range(len(pred_boxes)) if j not in used and pred_cls[j] == cls))
            if not candidates or -candidates[0][0] < 0.2:
                continue
            j = candidates[0][1]; used.add(j); matched += 1
            for k, (gx, gy, visible) in enumerate(points):
                if not visible:
                    continue
                px, py = pred_points[j][k]
                dx, dy = float(px - gx), float(py - gy)
                row = {"class": cls, "dx": dx, "dy": dy, "distance": float(np.hypot(dx, dy))}
                errors.append(row); by_class.setdefault(cls, []).append(row)
    if not errors:
        raise RuntimeError("No visible keypoints were matched")
    distances = np.array([x["distance"] for x in errors])
    dx = np.array([abs(x["dx"]) for x in errors]); dy = np.array([abs(x["dy"]) for x in errors])
    summary = {
        "validation_images": len(list((root / "images/val").glob("*"))),
        "matched_objects": matched,
        "visible_keypoints": len(errors),
        "mean_euclidean_px": float(distances.mean()),
        "median_euclidean_px": float(np.median(distances)),
        "p90_euclidean_px": float(np.percentile(distances, 90)),
        "mean_abs_x_px": float(dx.mean()),
        "mean_abs_y_px": float(dy.mean()),
        "by_class": {},
    }
    for cls, rows in sorted(by_class.items()):
        d = np.array([x["distance"] for x in rows])
        summary["by_class"][str(cls)] = {"points": len(rows), "mean_px": float(d.mean()), "median_px": float(np.median(d))}
    print(summary)


if __name__ == "__main__":
    main()

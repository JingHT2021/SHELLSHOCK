"""Conservative standard-circle fitting for detector-guided image regions."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class CircleFit:
    center: tuple[float, float]
    radius: float


def _fit(points, expected, min_radius, max_radius):
    if len(points) < 16:
        return None
    origin = points.mean(axis=0)
    local = points - origin
    matrix = np.column_stack((2 * local, np.ones(len(local))))
    try:
        solution, _, rank, _ = np.linalg.lstsq(matrix, (local ** 2).sum(axis=1), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center = solution[:2] + origin
    radius_sq = solution[2] + (solution[:2] ** 2).sum()
    if radius_sq <= 0:
        return None
    radius = float(np.sqrt(radius_sq))
    if not min_radius <= radius <= max_radius or np.linalg.norm(center - expected) > max_radius * .75:
        return None
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    tolerance = max(2.0, radius * .065)
    inliers = residual <= tolerance
    if inliers.mean() < .85 or np.sqrt(np.mean(residual ** 2)) > tolerance:
        return None
    angles = np.sort(np.arctan2(*(points[inliers] - center)[:, ::-1].T))
    covered = 2 * np.pi - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()
    # Tiny arcs, scattered pixels, and dense filled blobs do not constrain a circle.
    if covered < np.deg2rad(100) or len(points[inliers]) < radius * covered * .45:
        return None
    return CircleFit((float(center[0]), float(center[1])), radius)


def fit_standard_circle(roi, class_name, expected, min_radius, max_radius, *, color_only=False):
    """Fit colored arcs then edges; every candidate must pass residual/arc checks.

    Coordinates are local to ``roi``. Missing or ambiguous evidence returns None.
    Bounds originate in the detector box, permitting partially clipped circles.
    """
    if roi.size == 0 or min(roi.shape[:2]) < 3:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ranges = {'obstacle_circle': ((0, 0, 230), (179, 20, 255)),
              'portal_orange': ((0, 65, 90), (30, 255, 255)),
              'portal_blue': ((85, 65, 90), (130, 255, 255))}
    if color_only and class_name not in ranges:
        return None
    masks = []
    if class_name in ranges:
        lower, upper = ranges[class_name]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masks.append(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    if class_name == 'obstacle_circle' and not color_only:
        masks.append(cv2.inRange(hsv, np.array((125, 45, 100)), np.array((175, 255, 255))))
    edges = cv2.Canny(gray, 50, 150)
    if not (color_only and class_name in ranges):
        masks.append(edges)
    for mask in masks:
        points = np.column_stack(np.where(mask > 0)[::-1]).astype(float)
        result = _fit(points, expected, min_radius, max_radius)
        if result is not None:
            return result
    if color_only and class_name in ranges:
        return None
    # Avoid finding a coincidental contour inside dense texture/noise.
    if np.count_nonzero(edges) / edges.size > .22:
        return None
    candidates = []
    for contour in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
        result = _fit(contour.reshape(-1, 2).astype(float), expected, min_radius, max_radius)
        if result is not None:
            candidates.append(result)
    # Preserve the prior Hough fallback, but only accept independently supported arcs.
    if not candidates and np.count_nonzero(edges) >= 16:
        hough = cv2.HoughCircles(cv2.GaussianBlur(gray, (9, 9), 2), cv2.HOUGH_GRADIENT,
                                 1.2, max(12, int(min_radius)), param1=50, param2=8,
                                 minRadius=int(min_radius), maxRadius=int(max_radius))
        if hough is not None:
            points = np.column_stack(np.where(edges > 0)[::-1]).astype(float)
            for x, y, radius in hough[0]:
                support = points[np.abs(np.linalg.norm(points - (x, y), axis=1) - radius) <= 2]
                result = _fit(support, expected, min_radius, max_radius)
                if result is not None:
                    candidates.append(result)
    return min(candidates, key=lambda c: np.linalg.norm(np.array(c.center) - expected)) if candidates else None

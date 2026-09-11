"""Common drawing boundary; portal translations are not projectile segments."""
import cv2


def draw_trajectory(image, points, color=(255,100,0), thickness=1):
    for a,b in zip(points,points[1:]):
        if a is not None and b is not None:
            cv2.line(image,tuple(round(v) for v in a),tuple(round(v) for v in b),color,thickness)
    return image

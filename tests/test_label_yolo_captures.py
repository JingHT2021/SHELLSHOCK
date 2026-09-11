from types import SimpleNamespace

import numpy as np

from label_yolo_captures import _draw


def test_self_pose_center_does_not_draw_duplicate_white_dot():
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    self_detection = SimpleNamespace(
        name="self",
        keypoints=(SimpleNamespace(x=30, y=30, visible=2),),
    )

    rendered = _draw(
        image,
        circles=[],
        lines=[],
        center=(30, 30),
        muzzle=None,
        pose_detections=(self_detection,),
    )

    assert tuple(rendered[34, 34]) != (255, 255, 255)
    assert tuple(rendered[30, 30]) == (255, 0, 255)

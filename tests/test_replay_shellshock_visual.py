from types import SimpleNamespace

import numpy as np

from replay_shellshock import render_replay_visual


def test_replay_hides_duplicate_white_self_pose_point():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    scene = SimpleNamespace(
        boxes=(
            SimpleNamespace(
                name="self",
                x=20,
                y=20,
                width=20,
                height=20,
                keypoints=(SimpleNamespace(x=30, y=30, visible=2),),
            ),
        )
    )
    report = {
        "source": "test",
        "target": None,
        "self_center": (30, 30),
        "solution": {"mode": "normal", "events": ()},
        "error": {},
    }

    rendered = render_replay_visual(image, (), None, scene, report)

    assert tuple(rendered[34, 34]) != (255, 255, 255)

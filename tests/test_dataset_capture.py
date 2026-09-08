from pathlib import Path

import cv2
import numpy as np

from shellshock_detector.dataset_capture import (
    capture_region_for_width,
    save_shot_metadata,
    save_capture_assets,
)


def test_3840_capture_region_uses_top_1850_pixels():
    assert capture_region_for_width(3840) == (0, 0, 3840, 1850)


def test_other_capture_regions_scale_from_3840_reference():
    assert capture_region_for_width(1920) == (0, 0, 1920, 925)


def test_2560_capture_region_keeps_the_lower_game_area():
    assert capture_region_for_width(2560) == (0, 0, 2560, 1300)


def test_capture_assets_save_full_wind_and_each_portal(tmp_path: Path):
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    result = save_capture_assets(
        image,
        tmp_path,
        "scene",
        wind_box=(10, 20, 20, 10),
        portal_boxes=[(50, 30, 10, 12), (100, 40, 8, 8)],
    )
    assert result == {"full": 1, "wind": 1, "wormholes": 2}
    assert (tmp_path / "full" / "scene.png").exists()
    assert (tmp_path / "wind" / "scene.png").exists()
    assert len(list((tmp_path / "wormholes").glob("scene_*.png"))) == 2


def test_shot_metadata_saves_launch_direction_vector(tmp_path: Path):
    path = save_shot_metadata(tmp_path, "scene", direction="right", angle_degrees=0, power=50)
    assert path.exists()
    assert '"direction_vector": [1.0, -0.0]' in path.read_text(encoding="utf-8")

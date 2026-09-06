from pathlib import Path

import cv2

from shellshock_detector.wind import detect_wind

latest = max(Path("train/raw_cropped").glob("*.png"), key=lambda path: path.stat().st_mtime)
wind, error, box = detect_wind(cv2.imread(str(latest)))
print(f"{latest}: wind={wind.value}/{wind.direction}; box={box}; error={error}")

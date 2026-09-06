import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from shellshock_detector.training_data import (
    DEFAULT_BOX_SIZE_AT_REFERENCE,
    ObstacleDetection,
    PinkObstacleConfig,
    annotate_pink_obstacle_sample,
    append_yolo_obstacle_labels,
    detect_pink_obstacles,
    yolo_box_label_line,
    yolo_label_line,
    _draw_yolo_preview,
)


class TrainingDataTests(unittest.TestCase):
    def test_default_box_size_is_centralized(self):
        self.assertEqual(DEFAULT_BOX_SIZE_AT_REFERENCE, (38.4, 28.8))

    def test_yolo_label_line_uses_normalized_centered_box(self):
        line = yolo_label_line(2, (960, 540), 1920, 1080)

        self.assertEqual(line, "2 0.500000 0.500000 0.020000 0.026667")

    def test_yolo_label_line_clips_a_box_at_image_edge(self):
        line = yolo_label_line(0, (0, 0), 1920, 1080)

        self.assertEqual(line, "0 0.005000 0.006667 0.010000 0.013333")


class PinkObstacleDetectionTests(unittest.TestCase):
    @staticmethod
    def strict_pink() -> tuple[int, int, int]:
        return (255, 255, 255)

    def test_detects_a_strict_pink_circle_as_class_3(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(image, (180, 180), 70, self.strict_pink(), 5)

        detections = detect_pink_obstacles(image)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 3)
        x, y, width, height = detections[0].box
        self.assertLessEqual(x, 112)
        self.assertLessEqual(y, 112)
        self.assertGreaterEqual(width, 140)
        self.assertGreaterEqual(height, 140)

    def test_detects_a_strict_pink_diagonal_line_as_class_4(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.line(image, (80, 300), (360, 150), self.strict_pink(), 8)

        detections = detect_pink_obstacles(image)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 4)
        self.assertGreater(detections[0].box[2], 250)

    def test_rejects_pink_outside_the_configured_hue_range(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        red_pink = tuple(map(int, cv2.cvtColor(np.uint8([[[178, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
        cv2.circle(image, (180, 180), 70, red_pink, 5)

        self.assertEqual(detect_pink_obstacles(image), [])
        self.assertEqual(PinkObstacleConfig().hsv_lower, (0, 0, 230))
        self.assertEqual(PinkObstacleConfig().hsv_upper, (0, 0, 255))


class ObstacleLabelMergeTests(unittest.TestCase):
    def test_yolo_box_label_line_normalizes_an_axis_aligned_box(self):
        self.assertEqual(
            yolo_box_label_line(3, (100, 50, 200, 100), 1000, 500),
            "3 0.200000 0.200000 0.200000 0.200000",
        )

    def test_append_preserves_existing_bytes_and_is_idempotent(self):
        existing = "2 0.100000 0.200000 0.020000 0.030000\n0 0.700000 0.500000 0.020000 0.030000\n"
        detections = [ObstacleDetection(3, (100, 50, 200, 100)), ObstacleDetection(4, (500, 250, 300, 20))]

        merged, added = append_yolo_obstacle_labels(existing, detections, 1000, 500)
        again, added_again = append_yolo_obstacle_labels(merged, detections, 1000, 500)

        self.assertTrue(merged.startswith(existing))
        self.assertEqual(added, 2)
        self.assertEqual(added_again, 0)
        self.assertEqual(again, merged)


class PreviewRenderingTests(unittest.TestCase):
    def test_can_hide_geometric_classes_from_yolo_box_preview(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        labels = "2 0.100000 0.100000 0.100000 0.100000\n3 0.500000 0.500000 0.400000 0.400000\n"

        preview = _draw_yolo_preview(image, labels, hidden_class_ids={3})

        self.assertTrue(np.array_equal(preview[30, 30], image[30, 30]))
        self.assertFalse(np.array_equal(preview[5, 5], image[5, 5]))


class ObstacleSampleTests(unittest.TestCase):
    def test_sample_update_keeps_existing_labels_and_creates_preview(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            image = np.zeros((400, 600, 3), dtype=np.uint8)
            pink = (255, 255, 255)
            cv2.circle(image, (200, 200), 70, pink, 5)
            source = root / "sample.png"
            label = root / "sample.txt"
            preview = root / "preview.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            label.write_text("2 0.100000 0.200000 0.020000 0.030000\n", encoding="utf-8")

            added = annotate_pink_obstacle_sample(source, label, preview)

            self.assertEqual(added, [3])
            self.assertTrue(preview.exists())
            self.assertTrue(label.read_text(encoding="utf-8").startswith("2 0.100000 0.200000 0.020000 0.030000\n"))

import unittest

from shellshock_detector.models import Detection, DetectionResult, Wind


class ResultModelTests(unittest.TestCase):
    def test_result_serializes_image_tanks_and_wind(self):
        result = DetectionResult(
            image_width=2560,
            image_height=1600,
            self_tank=Detection(1672, 763, 0.95),
            enemies=[Detection(1145, 755, 0.93)],
            wind=Wind(value=20, direction="right", confidence=0.9),
            errors=[],
        )

        payload = result.to_dict()

        self.assertEqual(payload["image"], {"width": 2560, "height": 1600})
        self.assertEqual(payload["self"], {"x": 1672, "y": 763, "confidence": 0.95})
        self.assertEqual(payload["wind"]["direction"], "right")

    def test_result_serializes_ballistics(self):
        result = DetectionResult(
            image_width=1920,
            image_height=1080,
            self_tank=Detection(100, 400, 0.9),
            enemies=[],
            wind=Wind(0, "right", 0.9),
            errors=[],
            ballistics=[{"target_direction": "right", "power_100": {"status": "reachable"}}],
        )

        self.assertEqual(result.to_dict()["ballistics"][0]["target_direction"], "right")

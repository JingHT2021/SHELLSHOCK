import unittest

from shellshock_detector.resolution import resolution_warning


class ResolutionTests(unittest.TestCase):
    def test_auto_accepts_any_capture_size(self):
        self.assertIsNone(resolution_warning("auto", 3840, 2160))

    def test_4k_preset_accepts_4k_capture(self):
        self.assertIsNone(resolution_warning("3840x2160", 3840, 2160))

    def test_preset_reports_capture_mismatch(self):
        self.assertEqual(
            resolution_warning("3840x2160", 2560, 1600),
            "configured resolution 3840x2160 but captured 2560x1600",
        )

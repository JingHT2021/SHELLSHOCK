import unittest

from shellshock_detector.dpi import client_capture_box


class DpiTests(unittest.TestCase):
    def test_capture_box_preserves_physical_4k_client_size(self):
        self.assertEqual(
            client_capture_box((0, 0), (3840, 2160)),
            (0, 0, 3840, 2160),
        )

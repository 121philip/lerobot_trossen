import unittest
from types import SimpleNamespace

from important_code.utils import get_control_fps


class SharedRuntimeTimingTest(unittest.TestCase):
    def test_control_fps_prefers_new_argument_name(self):
        args = SimpleNamespace(control_fps=10, fps=30)

        self.assertEqual(get_control_fps(args), 10.0)

    def test_control_fps_falls_back_to_legacy_fps(self):
        args = SimpleNamespace(fps=12)

        self.assertEqual(get_control_fps(args), 12.0)


if __name__ == "__main__":
    unittest.main()

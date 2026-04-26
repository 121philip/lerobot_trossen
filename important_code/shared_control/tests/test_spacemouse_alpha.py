"""Deprecated prototype tests for the inactive alpha_sm SpaceMouse gate."""

import unittest

import numpy as np

from important_code.shared_control.shared_control_spacemouse import (
    AlphaController,
    SharedControlNode,
)


class SpaceMouseAlphaTest(unittest.TestCase):
    def test_high_confidence_and_idle_spacemouse_keeps_vla_authority(self):
        node = SharedControlNode(v_threshold=0.02, v_full=0.12)
        out = node.step(np.zeros(6), btn_l=False, btn_r=False, C_VLA=0.9)

        self.assertLess(out.alpha, 0.05)
        self.assertLess(out.alpha_sm, 0.01)

    def test_strong_spacemouse_input_overrides_high_cvla(self):
        node = SharedControlNode(v_threshold=0.02, v_full=0.12)
        out = node.step(np.array([0.2, 0, 0, 0, 0, 0]), btn_l=False, btn_r=False, C_VLA=0.9)

        self.assertGreater(out.alpha, 0.99)
        self.assertGreater(out.alpha_sm, out.alpha_cvla)

    def test_low_confidence_raises_alpha_when_spacemouse_idle(self):
        node = SharedControlNode(v_threshold=0.02, v_full=0.12)
        out = node.step(np.zeros(6), btn_l=False, btn_r=False, C_VLA=0.1)

        self.assertGreater(out.alpha, 0.9)
        self.assertAlmostEqual(out.alpha, out.alpha_cvla)

    def test_button_l_long_press_forces_human_authority(self):
        controller = AlphaController(v_dead=0.02, v_full=0.12)
        alpha = controller.update(C_VLA=0.9, v_norm=0.0, override_long=True, resume_long=False)

        self.assertEqual(alpha, 1.0)
        self.assertTrue(controller.is_locked)
        self.assertEqual(controller.lock_value, 1.0)

    def test_button_r_long_press_forces_vla_authority(self):
        controller = AlphaController(v_dead=0.02, v_full=0.12)
        controller.update(C_VLA=0.9, v_norm=0.0, override_long=True, resume_long=False)
        alpha = controller.update(C_VLA=0.1, v_norm=0.2, override_long=False, resume_long=True)

        self.assertEqual(alpha, 0.0)
        self.assertTrue(controller.is_locked)
        self.assertEqual(controller.lock_value, 0.0)


if __name__ == "__main__":
    unittest.main()

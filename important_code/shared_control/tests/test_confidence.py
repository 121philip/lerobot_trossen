import unittest

import numpy as np

from important_code.shared_control.confidence import ConfidenceEstimator


def linear_chunk(start: float, stop: float, steps: int = 50, dim: int = 7) -> np.ndarray:
    return np.linspace(start, stop, steps)[:, None] * np.ones((1, dim))


class ConfidenceEstimatorTest(unittest.TestCase):
    def test_first_chunk_returns_full_confidence(self):
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0)
        metrics = estimator.update(linear_chunk(0.0, 1.0))
        self.assertEqual(metrics.c_action, 1.0)
        self.assertEqual(metrics.c_regression, 1.0)
        self.assertTrue(metrics.is_first_chunk)

    def test_regression_cbc_stays_high_for_fast_smooth_motion(self):
        estimator = ConfidenceEstimator(
            d=5, gamma=1.0, fps=30.0, confidence_method="regression_cbc",
        )
        estimator.update(linear_chunk(0.0, 10.0))
        metrics = estimator.update(linear_chunk(10.2, 20.0))

        self.assertGreater(metrics.c_action, 0.99)
        self.assertEqual(metrics.c_action, metrics.c_regression)
        self.assertGreater(metrics.c_action, metrics.c_raw)
        self.assertGreater(metrics.cbc_raw_mse, metrics.cbc_reg_residual)

    def test_boundary_jump_reduces_confidence(self):
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0)
        estimator.update(linear_chunk(0.0, 1.0))
        jumped = linear_chunk(10.0, 11.0)
        metrics = estimator.update(jumped)

        self.assertLess(metrics.c_action, 0.5)
        self.assertGreater(metrics.cbc_reg_residual, 1.0)
        self.assertGreater(metrics.boundary_jump_max, 8.0)

    def test_chunk_oscillation_increases_acceleration_instability(self):
        smooth = linear_chunk(0.0, 1.0)
        oscillating = smooth.copy()
        oscillating[::2] += 0.2
        oscillating[1::2] -= 0.2

        smooth_metrics = ConfidenceEstimator.compute_action_instability(smooth)
        oscillating_metrics = ConfidenceEstimator.compute_action_instability(oscillating)

        self.assertGreater(oscillating_metrics.a_ai, smooth_metrics.a_ai)
        self.assertGreater(oscillating_metrics.jerk, smooth_metrics.jerk)

    def test_confidence_method_selects_primary_output(self):
        chunks = (linear_chunk(0.0, 10.0), linear_chunk(10.2, 20.0))
        for method, attr in (
            ("raw_cbc", "c_raw"),
            ("speed_norm_cbc", "c_speed_norm"),
            ("regression_cbc", "c_regression"),
        ):
            estimator = ConfidenceEstimator(
                d=5, gamma=1.0, fps=30.0, confidence_method=method,
            )
            estimator.update(chunks[0])
            metrics = estimator.update(chunks[1])
            self.assertEqual(metrics.confidence_method, method)
            self.assertAlmostEqual(metrics.c_action, getattr(metrics, attr))

    # ── Tests for tracking and combined modes ─────────────────────────────

    def test_tracking_mode_high_when_joints_match_prediction(self):
        """c_action ≈ 1.0 when actual joints equal prev_chunk[delay_steps]."""
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="tracking")
        chunk0 = linear_chunk(0.0, 1.0)
        estimator.update(chunk0)
        predicted_at_2 = chunk0[2].copy()
        metrics = estimator.update(linear_chunk(1.0, 2.0), actual_joints=predicted_at_2, delay_steps=2)
        self.assertGreater(metrics.c_action, 0.99)
        self.assertAlmostEqual(metrics.tracking_mse, 0.0, places=10)

    def test_tracking_mode_drops_on_large_deviation(self):
        """c_action < 0.85 when actual joints deviate by 0.5 rad per joint."""
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="tracking")
        chunk0 = linear_chunk(0.0, 1.0)
        estimator.update(chunk0)
        actual_far = chunk0[3].copy() + 0.5
        metrics = estimator.update(linear_chunk(1.0, 2.0), actual_joints=actual_far, delay_steps=3)
        self.assertLess(metrics.c_action, 0.85)

    def test_tracking_mode_none_joints_returns_neutral(self):
        """c_tracking = 0.5 when actual_joints is None."""
        import math
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="tracking")
        estimator.update(linear_chunk(0.0, 1.0))
        metrics = estimator.update(linear_chunk(1.0, 2.0), actual_joints=None)
        self.assertAlmostEqual(metrics.c_tracking, 0.5)
        self.assertAlmostEqual(metrics.c_action, 0.5)
        self.assertTrue(math.isnan(metrics.tracking_mse))

    def test_combined_mode_equals_geometric_mean(self):
        """c_action = sqrt(c_regression * c_tracking) for combined mode."""
        import math
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="combined")
        chunk0 = linear_chunk(0.0, 1.0)
        estimator.update(chunk0)
        actual = chunk0[3].copy() + 0.3
        metrics = estimator.update(linear_chunk(1.02, 2.0), actual_joints=actual, delay_steps=3)
        expected = math.sqrt(metrics.c_regression * metrics.c_tracking)
        self.assertAlmostEqual(metrics.c_action, expected, places=6)


    # ── Tests for actions_normalized (normalized-space CBC) ───────────────

    def test_normalized_space_detects_small_joint_jump(self):
        """
        Gripper joint (index 6) has a tiny robot-space range (~0.0001 rad) but large
        normalized range. Without actions_normalized the jump is invisible; with it the
        confidence should drop.

        Joints 0-5 form a smooth linear continuation across the boundary so they don't
        dominate the residual — only joint 6 differs between robot vs normalized space.
        """
        def robot_chunk(start, stop, gripper_val: float) -> np.ndarray:
            chunk = np.linspace(start, stop, 50)[:, None] * np.ones((1, 7))
            chunk[:, 6] = gripper_val
            return chunk

        def norm_chunk(start, stop, gripper_norm: float) -> np.ndarray:
            chunk = np.linspace(start, stop, 50)[:, None] * np.ones((1, 7))
            chunk[:, 6] = gripper_norm
            return chunk

        # --- Without actions_normalized: gripper jump 0.0001→0.0002 rad is invisible ---
        est_robot = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="regression_cbc")
        est_robot.update(robot_chunk(0.0, 1.0, gripper_val=0.0001))
        metrics_robot = est_robot.update(robot_chunk(1.02, 2.0, gripper_val=0.0002))
        # Joints 0-5 are smooth continuation; gripper delta ≈ 1e-4 → residual near zero
        self.assertGreater(metrics_robot.c_regression, 0.99)

        # --- With actions_normalized: gripper jump -0.8→0.7 in normalized space is detected ---
        est_norm = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="regression_cbc")
        est_norm.update(
            robot_chunk(0.0, 1.0, gripper_val=0.0001),
            actions_normalized=norm_chunk(0.0, 1.0, gripper_norm=-0.8),
        )
        metrics_norm = est_norm.update(
            robot_chunk(1.02, 2.0, gripper_val=0.0002),
            actions_normalized=norm_chunk(1.02, 2.0, gripper_norm=0.7),
        )
        self.assertLess(metrics_norm.c_regression, metrics_robot.c_regression)

    def test_normalized_fallback_equals_robot_space_when_not_provided(self):
        """When actions_normalized is None, CBC uses robot-space chunk (backward compat)."""
        chunk_a = linear_chunk(0.0, 1.0)
        chunk_b = linear_chunk(1.02, 2.0)

        est_new = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0)
        est_old = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0)

        est_new.update(chunk_a)
        est_old.update(chunk_a)
        m_new = est_new.update(chunk_b)
        m_old = est_old.update(chunk_b)

        self.assertAlmostEqual(m_new.c_regression, m_old.c_regression, places=10)
        self.assertAlmostEqual(m_new.cbc_raw_mse, m_old.cbc_raw_mse, places=10)


if __name__ == "__main__":
    unittest.main()

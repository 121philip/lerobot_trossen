import unittest

import numpy as np

from important_code.shared_control.confidence import (
    ConfidenceEstimator,
    compute_alpha,
    compute_alpha_from_confidence,
)


def linear_chunk(start: float, stop: float, steps: int = 50, dim: int = 7) -> np.ndarray:
    return np.linspace(start, stop, steps)[:, None] * np.ones((1, dim))


class ConfidenceEstimatorTest(unittest.TestCase):
    def test_first_chunk_returns_neutral_confidence(self):
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0)
        metrics = estimator.update(linear_chunk(0.0, 1.0))
        self.assertEqual(metrics.c_action, 0.5)
        self.assertEqual(metrics.c_regression, 0.5)
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

    def test_confidence_alpha_mapping(self):
        high_conf_alpha = compute_alpha_from_confidence(0.9, tau_c=0.4, k_c=8.0)
        low_conf_alpha = compute_alpha_from_confidence(0.1, tau_c=0.4, k_c=8.0)
        self.assertLess(high_conf_alpha, 0.05)
        self.assertGreater(low_conf_alpha, 0.9)

    def test_alpha_mode_constant_ignores_confidence(self):
        self.assertEqual(compute_alpha(0.1, alpha_mode="constant", alpha_const=0.5), 0.5)
        self.assertEqual(compute_alpha(0.9, alpha_mode="constant", alpha_const=1.5), 1.0)

    def test_alpha_mode_confidence_uses_selected_confidence(self):
        high_conf_alpha = compute_alpha(0.9, alpha_mode="confidence", tau_c=0.4, k_c=8.0)
        low_conf_alpha = compute_alpha(0.1, alpha_mode="confidence", tau_c=0.4, k_c=8.0)
        self.assertLess(high_conf_alpha, 0.05)
        self.assertGreater(low_conf_alpha, 0.9)

    # ── New tests for tracking and combined modes ──────────────────────────

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
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="tracking")
        estimator.update(linear_chunk(0.0, 1.0))
        metrics = estimator.update(linear_chunk(1.0, 2.0), actual_joints=None)
        self.assertAlmostEqual(metrics.c_tracking, 0.5)
        self.assertAlmostEqual(metrics.c_action, 0.5)

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


if __name__ == "__main__":
    unittest.main()

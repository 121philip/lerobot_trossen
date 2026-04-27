import unittest

import numpy as np

from important_code.shared_control.confidence import (
    CBCConfidenceEstimator,
    compute_alpha,
    compute_alpha_from_confidence,
)


def linear_chunk(start: float, stop: float, steps: int = 50, dim: int = 7) -> np.ndarray:
    return np.linspace(start, stop, steps)[:, None] * np.ones((1, dim))


class CBCConfidenceEstimatorTest(unittest.TestCase):
    def test_first_chunk_returns_neutral_confidence(self):
        estimator = CBCConfidenceEstimator(d=5, gamma=1.0, fps=30.0)
        metrics = estimator.update(linear_chunk(0.0, 1.0))
        self.assertEqual(metrics.c_cbc, 0.5)
        self.assertEqual(metrics.c_regression, 0.5)
        self.assertTrue(metrics.is_first_chunk)

    def test_regression_cbc_stays_high_for_fast_smooth_motion(self):
        estimator = CBCConfidenceEstimator(
            d=5,
            gamma=1.0,
            fps=30.0,
            confidence_method="regression_cbc",
        )
        estimator.update(linear_chunk(0.0, 10.0))
        metrics = estimator.update(linear_chunk(10.2, 20.0))

        self.assertGreater(metrics.c_cbc, 0.99)
        self.assertEqual(metrics.c_cbc, metrics.c_regression)
        self.assertGreater(metrics.c_cbc, metrics.c_raw)
        self.assertGreater(metrics.cbc_raw_mse, metrics.cbc_reg_residual)

    def test_boundary_jump_reduces_confidence(self):
        estimator = CBCConfidenceEstimator(d=5, gamma=1.0, fps=30.0)
        estimator.update(linear_chunk(0.0, 1.0))
        jumped = linear_chunk(10.0, 11.0)
        metrics = estimator.update(jumped)

        self.assertLess(metrics.c_cbc, 0.5)
        self.assertGreater(metrics.cbc_reg_residual, 1.0)
        self.assertGreater(metrics.boundary_jump_max, 8.0)

    def test_chunk_oscillation_increases_acceleration_instability(self):
        smooth = linear_chunk(0.0, 1.0)
        oscillating = smooth.copy()
        oscillating[::2] += 0.2
        oscillating[1::2] -= 0.2

        smooth_metrics = CBCConfidenceEstimator.compute_action_instability(smooth)
        oscillating_metrics = CBCConfidenceEstimator.compute_action_instability(oscillating)

        self.assertGreater(oscillating_metrics.a_ai, smooth_metrics.a_ai)
        self.assertGreater(oscillating_metrics.jerk, smooth_metrics.jerk)

    def test_confidence_method_selects_primary_output(self):
        chunks = (linear_chunk(0.0, 10.0), linear_chunk(10.2, 20.0))
        for method, attr in (
            ("raw_cbc", "c_raw"),
            ("speed_norm_cbc", "c_speed_norm"),
            ("regression_cbc", "c_regression"),
        ):
            estimator = CBCConfidenceEstimator(
                d=5,
                gamma=1.0,
                fps=30.0,
                confidence_method=method,
            )
            estimator.update(chunks[0])
            metrics = estimator.update(chunks[1])
            self.assertEqual(metrics.confidence_method, method)
            self.assertAlmostEqual(metrics.c_cbc, getattr(metrics, attr))

    def test_confidence_alpha_mapping(self):
        high_conf_alpha = compute_alpha_from_confidence(0.9, tau_c=0.4, k_c=8.0)
        low_conf_alpha = compute_alpha_from_confidence(0.1, tau_c=0.4, k_c=8.0)

        self.assertLess(high_conf_alpha, 0.05)
        self.assertGreater(low_conf_alpha, 0.9)

    def test_alpha_mode_constant_ignores_confidence(self):
        self.assertEqual(
            compute_alpha(0.1, alpha_mode="constant", alpha_const=0.5),
            0.5,
        )
        self.assertEqual(
            compute_alpha(0.9, alpha_mode="constant", alpha_const=1.5),
            1.0,
        )

    def test_alpha_mode_confidence_uses_selected_confidence(self):
        high_conf_alpha = compute_alpha(0.9, alpha_mode="confidence", tau_c=0.4, k_c=8.0)
        low_conf_alpha = compute_alpha(0.1, alpha_mode="confidence", tau_c=0.4, k_c=8.0)

        self.assertLess(high_conf_alpha, 0.05)
        self.assertGreater(low_conf_alpha, 0.9)


if __name__ == "__main__":
    unittest.main()

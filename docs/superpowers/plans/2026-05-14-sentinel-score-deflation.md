# Sentinel Score Deflation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Sentinel metric inflation by adding tracking-error c_action modes, a LocalMotionDetector stale fallback, a continuous VLM prompt, and DeepSeek provider support.

**Architecture:** Four files change. `confidence.py` gains tracking-error and combined modes; `sentinel.py` gains `LocalMotionDetector` and a capped stale fallback; `inference_thread.py` wires actual joint positions through; `run_inference.py` exposes three new CLI flags.

**Tech Stack:** Python 3.12, NumPy, urllib (no new dependencies)

**Test command:** `python -m pytest important_code/shared_control/tests/ -q`

---

## Task 1: confidence.py — Rename + tracking/combined modes (TDD)

**Files:**
- Modify: `important_code/shared_control/confidence.py`
- Modify: `important_code/shared_control/tests/test_confidence.py`

- [ ] **Step 1: Update test imports and rename `c_cbc` → `c_action` in test_confidence.py**

Replace the import block and `_metrics` helper, and rename the test class and all `c_cbc` field accesses:

```python
# test_confidence.py — full updated file
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
        predicted_at_2 = chunk0[2].copy()   # where VLA expected robot after 2 steps
        metrics = estimator.update(linear_chunk(1.0, 2.0), actual_joints=predicted_at_2, delay_steps=2)
        self.assertGreater(metrics.c_action, 0.99)
        self.assertAlmostEqual(metrics.tracking_mse, 0.0, places=10)

    def test_tracking_mode_drops_on_large_deviation(self):
        """c_action < 0.85 when actual joints deviate by 0.5 rad per joint."""
        estimator = ConfidenceEstimator(d=5, gamma=1.0, fps=30.0, confidence_method="tracking")
        chunk0 = linear_chunk(0.0, 1.0)
        estimator.update(chunk0)
        # 0.5 rad off → tracking_mse = 0.25 → c_tracking = exp(-0.25) ≈ 0.779
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
```

- [ ] **Step 2: Run tests to confirm failures**

```bash
cd /home/masterthesis/lerobot_trossen
python -m pytest important_code/shared_control/tests/test_confidence.py -q 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'ConfidenceEstimator'` (or similar).

- [ ] **Step 3: Implement confidence.py changes**

Replace `important_code/shared_control/confidence.py` with:

```python
"""
Runtime confidence metrics for SmolVLA action chunks.

Primary signals:
  Regression-CBC: fit a local affine trend across the chunk boundary, use residual as
                  speed-invariant discontinuity score.
  Tracking error: compare actual robot joint positions to VLA-predicted positions at
                  the chunk boundary, penalising task-space drift.
  Combined:       geometric mean of Regression-CBC and tracking confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CONFIDENCE_METHODS = (
    "raw_cbc", "speed_norm_cbc", "regression_cbc",
    "tracking",   # tracking error only
    "combined",   # sqrt(c_regression * c_tracking)
)


def smoothstep(x: float) -> float:
    """Cubic smoothstep on [0, 1]."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def compute_alpha_from_confidence(
    c_vla: float,
    tau_c: float = 0.4,
    k_c: float = 8.0,
) -> float:
    """Map VLA confidence to human authority alpha in [0, 1]."""
    z = float(k_c) * (float(c_vla) - float(tau_c))
    alpha = 1.0 - (1.0 / (1.0 + np.exp(-z)))
    return float(np.clip(alpha, 0.0, 1.0))


def compute_alpha(
    c_vla: float,
    alpha_mode: str = "constant",
    alpha_const: float = 0.5,
    tau_c: float = 0.4,
    k_c: float = 8.0,
) -> float:
    """Select the final alpha source for runtime inference."""
    if alpha_mode == "constant":
        return float(np.clip(alpha_const, 0.0, 1.0))
    if alpha_mode == "confidence":
        return compute_alpha_from_confidence(c_vla, tau_c=tau_c, k_c=k_c)
    raise ValueError(f"alpha_mode must be 'constant' or 'confidence', got {alpha_mode!r}")


def _to_numpy(actions: Any) -> np.ndarray:
    """Convert torch/numpy action chunks to a 2D float64 numpy array."""
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected action chunk with shape (H, D), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("Action chunk must contain at least one step")
    return arr


@dataclass(frozen=True)
class ActionInstabilityMetrics:
    """Velocity, acceleration, and jerk instability for one action chunk."""
    a_vi: float
    a_ai: float
    jerk: float
    vel_max: float
    accel_max: float
    jerk_max: float


@dataclass(frozen=True)
class ConfidenceMetrics:
    """All confidence diagnostics produced when a new action chunk arrives."""
    c_action: float          # selected confidence output (depends on confidence_method)
    c_raw: float
    c_speed_norm: float
    c_regression: float
    c_tracking: float        # tracking-error confidence; 0.5 when actual_joints unavailable
    confidence_method: str
    cbc_raw_mse: float
    cbc_reg_residual: float
    tracking_mse: float      # raw MSE between actual and predicted joint positions
    speed_norm: float
    a_vi: float
    a_ai: float
    jerk: float
    vel_max: float
    accel_max: float
    jerk_max: float
    boundary_jump_max: float
    is_first_chunk: bool


class ConfidenceEstimator:
    """
    Chunk Boundary Continuity + tracking-error confidence for synchronous action chunks.

    The first chunk returns neutral confidence (0.5) because no previous boundary
    exists. Subsequent calls compare the previous chunk tail with the current
    chunk head and update the internal previous-chunk buffer.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        d: int = 5,
        fps: float = 30.0,
        epsilon: float = 1e-6,
        confidence_method: str = "regression_cbc",
    ) -> None:
        if d < 2:
            raise ValueError("d must be at least 2 for regression CBC")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if confidence_method not in CONFIDENCE_METHODS:
            raise ValueError(
                f"confidence_method must be one of {CONFIDENCE_METHODS}, got {confidence_method!r}"
            )
        self.gamma = float(gamma)
        self.d = int(d)
        self.fps = float(fps)
        self.epsilon = float(epsilon)
        self.confidence_method = confidence_method
        self.prev_chunk: np.ndarray | None = None
        self._fit_matrix = self._make_fit_matrix(self.d)

    @staticmethod
    def _make_fit_matrix(d: int) -> np.ndarray:
        t = np.arange(2 * d, dtype=np.float64)
        t -= np.mean(t)
        x = np.column_stack([t, np.ones_like(t)])
        return np.linalg.pinv(x)

    def reset(self) -> None:
        self.prev_chunk = None

    def update(
        self,
        actions: Any,
        actual_joints: np.ndarray | None = None,
        delay_steps: int = 0,
    ) -> ConfidenceMetrics:
        chunk = _to_numpy(actions)
        instability = self.compute_action_instability(chunk, fps=self.fps)

        if self.prev_chunk is None:
            self.prev_chunk = chunk.copy()
            return ConfidenceMetrics(
                c_action=0.5,
                c_raw=0.5,
                c_speed_norm=0.5,
                c_regression=0.5,
                c_tracking=0.5,
                confidence_method=self.confidence_method,
                cbc_raw_mse=0.0,
                cbc_reg_residual=0.0,
                tracking_mse=0.0,
                speed_norm=0.0,
                a_vi=instability.a_vi,
                a_ai=instability.a_ai,
                jerk=instability.jerk,
                vel_max=instability.vel_max,
                accel_max=instability.accel_max,
                jerk_max=instability.jerk_max,
                boundary_jump_max=0.0,
                is_first_chunk=True,
            )

        tail = self.prev_chunk[-self.d :]
        head = chunk[: self.d]
        if tail.shape[0] != self.d or head.shape[0] != self.d:
            raise ValueError(
                f"Need at least d={self.d} steps in both chunks, got "
                f"prev={self.prev_chunk.shape}, curr={chunk.shape}"
            )

        raw_mse = float(np.mean((tail - head) ** 2))
        reg_residual = self.compute_regression_residual(tail, head)
        speed_norm = self.compute_speed_norm(chunk)
        boundary_jump_max = float(np.max(np.abs(chunk[0] - self.prev_chunk[-1])))

        # Tracking error: actual robot position vs VLA prediction after delay_steps
        if actual_joints is not None:
            idx = min(int(delay_steps), self.prev_chunk.shape[0] - 1)
            predicted = self.prev_chunk[idx]
            tracking_mse = float(
                np.mean((np.asarray(actual_joints, dtype=np.float64) - predicted) ** 2)
            )
            c_tracking = float(np.exp(-self.gamma * tracking_mse))
        else:
            tracking_mse = 0.0
            c_tracking = 0.5

        c_raw = float(np.exp(-self.gamma * raw_mse))
        c_speed_norm = float(np.exp(-self.gamma * raw_mse / (speed_norm + self.epsilon)))
        c_regression = float(np.exp(-self.gamma * reg_residual))
        c_action = self.select_confidence(c_raw, c_speed_norm, c_regression, c_tracking)

        self.prev_chunk = chunk.copy()
        return ConfidenceMetrics(
            c_action=float(np.clip(c_action, 0.0, 1.0)),
            c_raw=float(np.clip(c_raw, 0.0, 1.0)),
            c_speed_norm=float(np.clip(c_speed_norm, 0.0, 1.0)),
            c_regression=float(np.clip(c_regression, 0.0, 1.0)),
            c_tracking=float(np.clip(c_tracking, 0.0, 1.0)),
            confidence_method=self.confidence_method,
            cbc_raw_mse=raw_mse,
            cbc_reg_residual=reg_residual,
            tracking_mse=tracking_mse,
            speed_norm=speed_norm,
            a_vi=instability.a_vi,
            a_ai=instability.a_ai,
            jerk=instability.jerk,
            vel_max=instability.vel_max,
            accel_max=instability.accel_max,
            jerk_max=instability.jerk_max,
            boundary_jump_max=boundary_jump_max,
            is_first_chunk=False,
        )

    def select_confidence(
        self,
        c_raw: float,
        c_speed_norm: float,
        c_regression: float,
        c_tracking: float,
    ) -> float:
        if self.confidence_method == "raw_cbc":
            return c_raw
        if self.confidence_method == "speed_norm_cbc":
            return c_speed_norm
        if self.confidence_method == "regression_cbc":
            return c_regression
        if self.confidence_method == "tracking":
            return c_tracking
        if self.confidence_method == "combined":
            return float(np.sqrt(c_regression * c_tracking))
        raise ValueError(
            f"confidence_method must be one of {CONFIDENCE_METHODS}, got {self.confidence_method!r}"
        )

    def compute_regression_residual(self, tail: np.ndarray, head: np.ndarray) -> float:
        boundary = np.concatenate([tail, head], axis=0)
        beta = self._fit_matrix @ boundary
        t = np.arange(2 * self.d, dtype=np.float64)
        t -= np.mean(t)
        x = np.column_stack([t, np.ones_like(t)])
        fitted = x @ beta
        return float(np.mean((boundary - fitted) ** 2))

    @staticmethod
    def compute_speed_norm(chunk: np.ndarray) -> float:
        if chunk.shape[0] < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(chunk, axis=0))))

    @staticmethod
    def compute_action_instability(
        actions: Any,
        fps: float = 30.0,
    ) -> ActionInstabilityMetrics:
        chunk = _to_numpy(actions)
        if chunk.shape[0] > 1:
            vel = np.diff(chunk, n=1, axis=0) * fps
            a_vi = float(np.mean(vel**2))
            vel_max = float(np.max(np.abs(vel)))
        else:
            a_vi = vel_max = 0.0

        if chunk.shape[0] > 2:
            accel = np.diff(chunk, n=2, axis=0) * (fps**2)
            a_ai = float(np.mean(accel**2))
            accel_max = float(np.max(np.abs(accel)))
        else:
            a_ai = accel_max = 0.0

        if chunk.shape[0] > 3:
            jerk_arr = np.diff(chunk, n=3, axis=0) * (fps**3)
            jerk = float(np.mean(jerk_arr**2))
            jerk_max = float(np.max(np.abs(jerk_arr)))
        else:
            jerk = jerk_max = 0.0

        return ActionInstabilityMetrics(
            a_vi=a_vi, a_ai=a_ai, jerk=jerk,
            vel_max=vel_max, accel_max=accel_max, jerk_max=jerk_max,
        )


# Backwards-compat alias — remove after all callers are updated
CBCConfidenceEstimator = ConfidenceEstimator


if __name__ == "__main__":
    estimator = ConfidenceEstimator(gamma=1.0, d=5, fps=30.0)
    a0 = np.linspace(0.0, 1.0, 50)[:, None] * np.ones((1, 7))
    a1 = np.linspace(1.02, 2.0, 50)[:, None] * np.ones((1, 7))
    print(estimator.update(a0))
    print(estimator.update(a1))
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest important_code/shared_control/tests/test_confidence.py -v
```

Expected: all 11 tests pass (8 existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add important_code/shared_control/confidence.py \
        important_code/shared_control/tests/test_confidence.py
git commit -m "feat: add tracking-error and combined c_action modes to ConfidenceEstimator"
```

---

## Task 2: sentinel.py — LocalMotionDetector + stale fallback (TDD)

**Files:**
- Modify: `important_code/shared_control/sentinel.py`
- Modify: `important_code/shared_control/tests/test_sentinel.py`

- [ ] **Step 1: Add LocalMotionDetector import and tests to test_sentinel.py**

At the top of `test_sentinel.py`, add `LocalMotionDetector` to the import block and update `_metrics`:

```python
import time   # add at top if not present

from important_code.shared_control.sentinel import (
    LocalMotionDetector,     # new
    ProgressMonitorResult,
    SentinelFrameBuffer,
    SentinelRuntime,
)


def _metrics(c_action=0.5):   # rename param c_cbc → c_action
    return SimpleNamespace(c_action=c_action, jerk_max=0.0, boundary_jump_max=0.0)
```

All existing calls `_metrics(c_cbc=...)` → `_metrics(c_action=...)`.

Add the new test class at the end of `test_sentinel.py` (before `if __name__ == "__main__"`):

```python
class LocalMotionDetectorTest(unittest.TestCase):
    def test_insufficient_data_returns_neutral(self):
        det = LocalMotionDetector(window_s=3.0, stuck_threshold=0.02)
        det.push(np.zeros(7), time.time())
        self.assertAlmostEqual(det.c_progress_local(), 0.5)

    def test_stuck_returns_low_confidence(self):
        det = LocalMotionDetector(window_s=3.0, stuck_threshold=0.02)
        t = time.time()
        for i in range(8):
            det.push(np.zeros(7), t + i * 0.4)  # 3.2 s span, no movement
        self.assertAlmostEqual(det.c_progress_local(), 0.2)

    def test_moving_returns_mid_confidence(self):
        det = LocalMotionDetector(window_s=3.0, stuck_threshold=0.02)
        t = time.time()
        for i in range(8):
            det.push(np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + i * 0.4)
        self.assertAlmostEqual(det.c_progress_local(), 0.5)


class SentinelStaleCapTest(unittest.TestCase):
    def test_stale_fallback_caps_r_raw_when_moving(self):
        """r_raw ≤ 0.5 when c_progress is None (stale) and robot is moving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = time.time()
            for i in range(8):
                sentinel._motion_detector.push(
                    np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + i * 0.4
                )
            fast = sentinel._fast_action(_metrics(c_action=1.0))
            result = sentinel._arbitrate(fast, None)   # None → stale path
            sentinel.stop()

        self.assertLessEqual(result.r_raw, 0.5)

    def test_stale_fallback_lower_when_stuck(self):
        """r_raw ≤ 0.2 when stale and robot is stuck."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = time.time()
            for i in range(8):
                sentinel._motion_detector.push(np.zeros(7), t + i * 0.4)
            fast = sentinel._fast_action(_metrics(c_action=1.0))
            result = sentinel._arbitrate(fast, None)
            sentinel.stop()

        self.assertLessEqual(result.r_raw, 0.2)
```

- [ ] **Step 2: Run tests to confirm failures**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py -q 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'LocalMotionDetector'` or `AttributeError: c_cbc`.

- [ ] **Step 3: Add LocalMotionDetector class to sentinel.py**

Insert after the `SentinelFrameBuffer` class (before `class CloudVLMClient`):

```python
class LocalMotionDetector:
    """Estimates whether the robot is moving, used as stale c_progress fallback."""

    def __init__(self, window_s: float = 3.0, stuck_threshold: float = 0.02) -> None:
        self.window_s = float(window_s)
        self.stuck_threshold = float(stuck_threshold)
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()

    def push(self, joints: np.ndarray, t: float) -> None:
        joints = np.asarray(joints, dtype=np.float64).copy()
        with self._lock:
            self._history.append((float(t), joints))
            cutoff = t - self.window_s
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()

    def c_progress_local(self) -> float:
        with self._lock:
            if len(self._history) < 2:
                return 0.5
            t_oldest, j_oldest = self._history[0]
            t_newest, j_newest = self._history[-1]
            if t_newest - t_oldest < 1.0:
                return 0.5
            displacement = float(np.max(np.abs(j_newest - j_oldest)))
        return 0.2 if displacement < self.stuck_threshold else 0.5
```

- [ ] **Step 4: Update sentinel.py — `_fast_action`, `_arbitrate`, `update`, `__init__`**

**4a.** In `SentinelRuntime.__init__`, add after `self._r_smooth: float | None = None`:

```python
self._motion_detector = LocalMotionDetector()
```

**4b.** Replace `_fast_action` (line 480):

```python
def _fast_action(self, metrics: Any) -> FastActionResult:
    c_action = _clip01(metrics.c_action)   # was metrics.c_cbc
    reasons = []
    if c_action < self.tau_action:
        reasons.append(f"c_action<{self.tau_action:.3f}")
    if self.jerk_max is not None and metrics.jerk_max > self.jerk_max:
        reasons.append(f"jerk_max>{self.jerk_max:.3f}")
    if self.boundary_jump_max is not None and metrics.boundary_jump_max > self.boundary_jump_max:
        reasons.append(f"boundary_jump>{self.boundary_jump_max:.3f}")
    return FastActionResult(c_action, bool(reasons), ";".join(reasons) or "ok")
```

**4c.** Replace the `r_raw` line in `_arbitrate` (line 516):

```python
# Old: r_raw = _clip01(min(fast.c_action, c_progress) if c_progress is not None else fast.c_action)
if c_progress is not None:
    r_raw = _clip01(min(fast.c_action, c_progress))
else:
    c_fallback = self._motion_detector.c_progress_local()
    r_raw = _clip01(min(fast.c_action, c_fallback))
```

**4d.** Replace `update` signature (line 471):

```python
def update(
    self,
    confidence_metrics: Any,
    actual_joints: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> SentinelArbitrationResult:
    if actual_joints is not None:
        self._motion_detector.push(actual_joints, time.time())
    result = self._arbitrate(self._fast_action(confidence_metrics), self._fresh_progress())
    self._log(result, extra or {})
    return result
```

- [ ] **Step 5: Run all sentinel and confidence tests**

```bash
python -m pytest important_code/shared_control/tests/ -v
```

Expected: all tests pass (11 confidence + existing sentinel + 5 new sentinel).

- [ ] **Step 6: Commit**

```bash
git add important_code/shared_control/sentinel.py \
        important_code/shared_control/tests/test_sentinel.py
git commit -m "feat: add LocalMotionDetector stale fallback, cap r_raw <= 0.5 when VLM stale"
```

---

## Task 3: sentinel.py — Prompt fix + DeepSeek provider + EMA β default

**Files:**
- Modify: `important_code/shared_control/sentinel.py`
- Modify: `important_code/shared_control/tests/test_sentinel.py`

- [ ] **Step 1: Add DeepSeek and prompt tests**

Add to `test_sentinel.py` imports:

```python
from important_code.shared_control.sentinel import CloudVLMClient
```

Add new test class:

```python
class CloudVLMClientTest(unittest.TestCase):
    def test_deepseek_missing_key_returns_stale(self):
        client = CloudVLMClient(provider="deepseek", api_key="")
        result = client.classify_progress("test", "fake_b64", timeout_s=1.0)
        self.assertTrue(result.stale)
        self.assertIn("Missing API key", result.error)

    def test_openai_missing_key_returns_stale(self):
        client = CloudVLMClient(provider="openai", api_key="")
        result = client.classify_progress("test", "fake_b64", timeout_s=1.0)
        self.assertTrue(result.stale)
        self.assertIn("Missing API key", result.error)
```

- [ ] **Step 2: Run tests to confirm new tests fail for the right reason**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py::CloudVLMClientTest -v
```

Expected: `test_deepseek_missing_key_returns_stale` fails because "deepseek" is not a known provider yet. `test_openai_missing_key_returns_stale` should already pass.

- [ ] **Step 3: Update PROMPT_TEMPLATE**

In `sentinel.py`, replace the `failure_likelihood` line inside `PROMPT_TEMPLATE`:

Old:
```
  "failure_likelihood": 0.0-1.0,
```

New:
```
  "failure_likelihood": <float 0.00-1.00; use fine-grained values like 0.23 or 0.71, NOT just 0.1/0.5/0.9>,
```

- [ ] **Step 4: Update CloudVLMClient to support DeepSeek**

**4a.** Replace the `env_name` line in `__init__` (line 257):

```python
env_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}.get(
    self.provider, "OPENAI_API_KEY"
)
self.api_key = api_key or os.environ.get(env_name)
```

**4b.** Replace the `text = ...` block in `classify_progress` (lines 267-270):

```python
if self.provider == "gemini":
    text = self._call_gemini(prompt, image_b64, timeout_s)
elif self.provider == "deepseek":
    text = self._call_deepseek(prompt, image_b64, timeout_s)
else:
    text = self._call_openai(prompt, image_b64, timeout_s)
```

**4c.** Add `_call_deepseek` method after `_call_gemini`:

```python
def _call_deepseek(self, prompt: str, image_b64: str, timeout_s: float) -> str:
    # DeepSeek uses OpenAI-compatible Chat Completions API with vision.
    data = self._post_json(
        "https://api.deepseek.com/v1/chat/completions",
        {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        },
        {"Authorization": f"Bearer {self.api_key}"},
        timeout_s,
    )
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not text:
        raise ValueError("DeepSeek response did not contain text")
    return text
```

**4d.** Update `from_args` default model for deepseek (in `SentinelRuntime.from_args`):

```python
if not model:
    if provider == "openai":
        model = "gpt-4o"
    elif provider == "gemini":
        model = "gemini-3-flash-preview"
    else:
        model = "deepseek-chat"
```

- [ ] **Step 5: Run all sentinel tests**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py -v
```

Expected: all tests pass including the two new `CloudVLMClientTest` tests.

- [ ] **Step 6: Commit**

```bash
git add important_code/shared_control/sentinel.py \
        important_code/shared_control/tests/test_sentinel.py
git commit -m "feat: add DeepSeek VLM provider and continuous failure_likelihood prompt"
```

---

## Task 4: inference_thread.py — Wire actual_joints

**Files:**
- Modify: `important_code/inference/inference_thread.py`

No new tests (integration-level change); verify with `py_compile`.

- [ ] **Step 1: Update import (line 41–44)**

```python
from important_code.shared_control.confidence import (
    ConfidenceEstimator,
    compute_alpha,
)
```

- [ ] **Step 2: Prefer `sentinel_confidence_mode` for the estimator (line 142)**

```python
confidence_method = getattr(
    args, "sentinel_confidence_mode",
    getattr(args, "confidence_method", "combined"),
)
```

- [ ] **Step 3: Update estimator instantiation (line 147)**

```python
confidence_estimator = ConfidenceEstimator(
    d=5,
    gamma=1.0,
    fps=control_fps,
    confidence_method=confidence_method,
)
```

- [ ] **Step 4: Fetch actual_joints after CroSPI block (after line 213)**

Insert immediately after the CroSPI `if` block ends:

```python
actual_joints = rviz_publisher.get_latest_joints() if rviz_publisher is not None else None
```

- [ ] **Step 5: Pass actual_joints and delay_steps to confidence_estimator.update (line 266)**

```python
confidence_metrics = confidence_estimator.update(
    queued_robot_chunk,
    actual_joints=actual_joints,
    delay_steps=inference_delay,
)
```

- [ ] **Step 6: Update compute_alpha call (line 267)**

```python
alpha = compute_alpha(
    confidence_metrics.c_action,
    alpha_mode=alpha_mode,
    alpha_const=alpha_const,
    tau_c=alpha_tau_c,
    k_c=alpha_k_c,
)
```

- [ ] **Step 7: Pass actual_joints to sentinel_runtime.update (line 288)**

```python
sentinel_result = sentinel_runtime.update(
    confidence_metrics,
    actual_joints=actual_joints,
    extra={
        "chunk_latency_s": new_latency,
        "delay_steps": new_delay,
        "queue_size_before": queue_size_before,
        "confidence_method": confidence_method,
        "alpha": alpha,
    },
)
```

- [ ] **Step 8: Update all remaining `c_cbc` references**

In the `logger.info` format string (line ~397) and `rr.log` calls (line ~368):
- `confidence_metrics.c_cbc` → `confidence_metrics.c_action`
- `"inference/c_cbc"` → `"inference/c_action"`

Search and replace within the file:

```bash
grep -n "c_cbc" important_code/inference/inference_thread.py
```

Replace every occurrence of `confidence_metrics.c_cbc` with `confidence_metrics.c_action` and `"inference/c_cbc"` with `"inference/c_action"`.

- [ ] **Step 9: Verify syntax**

```bash
python -m py_compile important_code/inference/inference_thread.py && echo OK
```

Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add important_code/inference/inference_thread.py
git commit -m "feat: wire actual_joints tracking error through inference_thread"
```

---

## Task 5: run_inference.py — CLI args

**Files:**
- Modify: `important_code/inference/run_inference.py`

- [ ] **Step 1: Add DeepSeek to --sentinel-vlm-provider choices**

Find the existing `--sentinel-vlm-provider` argument and add `"deepseek"` to its choices:

```python
parser.add_argument(
    "--sentinel-vlm-provider", type=str, default="openai",
    choices=["openai", "gemini", "deepseek"],
    help="Cloud VLM provider for Sentinel slow monitor.",
)
```

- [ ] **Step 2: Change --sentinel-ema-beta default from 0.8 to 0.6**

```python
parser.add_argument(
    "--sentinel-ema-beta", type=float, default=0.6,
    help="EMA smoothing factor for Sentinel reliability (lower = more responsive).",
)
```

- [ ] **Step 3: Add --sentinel-confidence-mode**

Insert after the `--confidence-method` argument:

```python
parser.add_argument(
    "--sentinel-confidence-mode", type=str, default="combined",
    choices=["raw_cbc", "speed_norm_cbc", "regression_cbc", "tracking", "combined"],
    help="c_action computation: cbc modes measure boundary smoothness; "
         "tracking measures joint tracking error; combined = sqrt(regression * tracking).",
)
```

- [ ] **Step 4: Verify syntax**

```bash
python -m py_compile important_code/inference/run_inference.py && echo OK
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add important_code/inference/run_inference.py
git commit -m "feat: add --sentinel-confidence-mode, --sentinel-ema-beta default 0.6, deepseek provider"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
python -m pytest important_code/shared_control/tests/ important_code/inference/tests/ -q
```

Expected: all tests pass (no regressions).

- [ ] **Smoke-test CLI help**

```bash
python important_code/inference/run_inference.py --help | grep -E "sentinel-confidence|sentinel-ema|deepseek"
```

Expected: three lines showing the new args with their defaults.

- [ ] **Spot-check ablation commands**

```bash
# Verify these three commands parse without error:
python important_code/inference/run_inference.py --help > /dev/null
python -c "
import sys; sys.argv = ['x', '--sentinel', '--sentinel-confidence-mode', 'tracking']
# just confirm argparse accepts the new choice
"
```

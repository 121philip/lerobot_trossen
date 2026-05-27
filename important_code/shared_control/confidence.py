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

    The first chunk returns full confidence (1.0) because no previous boundary
    exists and there is no evidence of discontinuity. Subsequent calls compare
    the previous chunk tail with the current chunk head and update the internal
    previous-chunk buffer.
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
        self.prev_chunk_norm: np.ndarray | None = None
        self._fit_matrix, self._x_matrix = self._make_fit_matrix(self.d)

    @staticmethod
    def _make_fit_matrix(d: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (pinv_x, x) for a [slope, intercept] affine fit over 2d centred timesteps.

        x  : design matrix shape (2d, 2) — column 0 is centred time, column 1 is ones.
        pinv_x : Moore-Penrose pseudoinverse of x, shape (2, 2d).

        Both are cached in __init__ so compute_regression_residual avoids rebuilding
        the design matrix on every chunk boundary call.
        """
        t = np.arange(2 * d, dtype=np.float64)
        t -= np.mean(t)
        x = np.column_stack([t, np.ones_like(t)])
        return np.linalg.pinv(x), x

    def reset(self) -> None:
        self.prev_chunk = None
        self.prev_chunk_norm = None

    def update(
        self,
        actions: Any,
        actual_joints: np.ndarray | None = None,
        delay_steps: int = 0,
        actions_normalized: Any | None = None,
    ) -> ConfidenceMetrics:
        chunk = _to_numpy(actions)
        chunk_norm = _to_numpy(actions_normalized) if actions_normalized is not None else chunk
        instability = self.compute_action_instability(chunk, fps=self.fps)

        if self.prev_chunk is None:
            self.prev_chunk = chunk.copy()
            self.prev_chunk_norm = chunk_norm.copy()
            return ConfidenceMetrics(
                c_action=1.0,
                c_raw=1.0,
                c_speed_norm=1.0,
                c_regression=1.0,
                c_tracking=1.0,
                confidence_method=self.confidence_method,
                cbc_raw_mse=0.0,
                cbc_reg_residual=0.0,
                tracking_mse=float("nan"),
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

        tail_norm = self.prev_chunk_norm[-self.d :]
        head_norm = chunk_norm[: self.d]

        # CBC metrics use normalized space so all joints have equal scale weight.
        raw_mse = float(np.mean((tail_norm - head_norm) ** 2))
        reg_residual = self.compute_regression_residual(tail_norm, head_norm)
        print("reg_residual: ", reg_residual)
        speed_norm = self.compute_speed_norm(chunk_norm)
        boundary_jump_max = float(np.max(np.abs(chunk_norm[0] - self.prev_chunk_norm[-1])))

        if actual_joints is not None:
            idx = min(max(0, int(delay_steps)), self.prev_chunk.shape[0] - 1)
            predicted = self.prev_chunk[idx]
            tracking_mse = float(
                np.mean((np.asarray(actual_joints, dtype=np.float64) - predicted) ** 2)
            )
            c_tracking = float(np.exp(-self.gamma * tracking_mse))
        else:
            tracking_mse = float("nan")
            c_tracking = 0.5

        c_raw = float(np.exp(-self.gamma * raw_mse))
        c_speed_norm = float(np.exp(-self.gamma * raw_mse / (speed_norm + self.epsilon)))
        c_regression = float(np.exp(-self.gamma * reg_residual))
        # print("c_regression: ", c_regression)
        c_action = self.select_confidence(c_raw, c_speed_norm, c_regression, c_tracking)

        self.prev_chunk = chunk.copy()
        self.prev_chunk_norm = chunk_norm.copy()
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
        beta = self._fit_matrix @ boundary   # least-squares [slope, intercept] per joint
        fitted = self._x_matrix @ beta       # affine reconstruction over the 2d window
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
            a_vi=a_vi,
            a_ai=a_ai,
            jerk=jerk,
            vel_max=vel_max,
            accel_max=accel_max,
            jerk_max=jerk_max,
        )


if __name__ == "__main__":
    estimator = ConfidenceEstimator(gamma=1.0, d=5, fps=30.0)
    a0 = np.linspace(0.0, 1.0, 50)[:, None] * np.ones((1, 7))
    a1 = np.linspace(1.02, 2.0, 50)[:, None] * np.ones((1, 7))
    print(estimator.update(a0))
    print(estimator.update(a1))

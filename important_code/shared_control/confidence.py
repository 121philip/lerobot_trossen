"""
Runtime confidence metrics for SmolVLA action chunks.

The main signal is Regression-CBC: fit a local linear trend across the chunk
boundary and use the residual as a speed-invariant discontinuity score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CONFIDENCE_METHODS = ("raw_cbc", "speed_norm_cbc", "regression_cbc")


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
class CBCConfidenceMetrics:
    """All confidence diagnostics produced when a new action chunk arrives."""

    c_cbc: float
    c_raw: float
    c_speed_norm: float
    c_regression: float
    confidence_method: str
    cbc_raw_mse: float
    cbc_reg_residual: float
    speed_norm: float
    a_vi: float
    a_ai: float
    jerk: float
    vel_max: float
    accel_max: float
    jerk_max: float
    boundary_jump_max: float
    is_first_chunk: bool


class CBCConfidenceEstimator:
    """
    Chunk Boundary Continuity confidence for synchronous action chunk execution.

    The first chunk returns neutral confidence because no previous boundary
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
        # Centering the local time index improves conditioning and leaves
        # residuals unchanged for any local affine trend.
        t = np.arange(2 * d, dtype=np.float64)
        t -= np.mean(t)
        x = np.column_stack([t, np.ones_like(t)])
        return np.linalg.pinv(x)

    def reset(self) -> None:
        self.prev_chunk = None

    def update(self, actions: Any) -> CBCConfidenceMetrics:
        chunk = _to_numpy(actions)
        instability = self.compute_action_instability(chunk, fps=self.fps)

        if self.prev_chunk is None:
            self.prev_chunk = chunk.copy()
            return CBCConfidenceMetrics(
                c_cbc=0.5,
                c_raw=0.5,
                c_speed_norm=0.5,
                c_regression=0.5,
                confidence_method=self.confidence_method,
                cbc_raw_mse=0.0,
                cbc_reg_residual=0.0,
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

        c_raw = float(np.exp(-self.gamma * raw_mse))
        c_speed_norm = float(np.exp(-self.gamma * raw_mse / (speed_norm + self.epsilon)))
        c_regression = float(np.exp(-self.gamma * reg_residual))
        c_cbc = self.select_confidence(c_raw, c_speed_norm, c_regression)

        self.prev_chunk = chunk.copy()
        return CBCConfidenceMetrics(
            c_cbc=float(np.clip(c_cbc, 0.0, 1.0)),
            c_raw=float(np.clip(c_raw, 0.0, 1.0)),
            c_speed_norm=float(np.clip(c_speed_norm, 0.0, 1.0)),
            c_regression=float(np.clip(c_regression, 0.0, 1.0)),
            confidence_method=self.confidence_method,
            cbc_raw_mse=raw_mse,
            cbc_reg_residual=reg_residual,
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
    ) -> float:
        if self.confidence_method == "raw_cbc":
            return c_raw
        if self.confidence_method == "speed_norm_cbc":
            return c_speed_norm
        return c_regression

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
            a_vi=a_vi,
            a_ai=a_ai,
            jerk=jerk,
            vel_max=vel_max,
            accel_max=accel_max,
            jerk_max=jerk_max,
        )


if __name__ == "__main__":
    estimator = CBCConfidenceEstimator(gamma=1.0, d=5, fps=30.0)
    a0 = np.linspace(0.0, 1.0, 50)[:, None] * np.ones((1, 7))
    a1 = np.linspace(1.02, 2.0, 50)[:, None] * np.ones((1, 7))
    print(estimator.update(a0))
    print(estimator.update(a1))

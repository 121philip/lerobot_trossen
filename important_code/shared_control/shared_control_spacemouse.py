"""
Shared Control (SpaceMouse Edition) — α computation + button arbitration
=========================================================================

This module handles the *Python-side* logic for SpaceMouse-based shared
control.  The blending computation (Jacobian IK, velocity mixing, integration)
is handled **internally by eTaSL within CroSPI** via task specifications.

What this module does
---------------------
- Reads SpaceMouse 6-DOF velocity and button events.
- Computes blending factor α ∈ [0, 1] from C_VLA and button overrides.
- Outputs α, v_sm, and gripper commands for publishing to CroSPI.

What CroSPI/eTaSL handles internally
--------------------------------------
- Jacobian IK: v_sm (Cartesian) → q̇_human (joint space)
- Velocity blending: q̇_cmd = (1-α)·q̇_VLA + α·q̇_human
- Integration: q_d = q_m + q̇_cmd · dt
- Hard constraints: eTaSL WLN-QP

ROS2 topics published by this node
-----------------------------------
/spacemouse/twist        geometry_msgs/Twist     SpaceMouse 6-DOF velocity
/shared_control/alpha    std_msgs/Float32        blending factor α

ROS2 topics also published (VLA node, separate)
------------------------------------------------
/vla/joint_targets       crospi_interfaces/Input  q_VLA (7-DOF, rad)
/vla/confidence          std_msgs/Float32          C_VLA

Confirmed design decisions (2026-04-15 / 2026-04-24)
------------------------------------------------------
- There is one unified mode: Shared Control.
  α = 0 (VLA limit) and α = 1 (Human limit) are extreme cases.
- Btn_R long-press locks α = 0  (VLA limit).
- Btn_L long-press toggles α = 1 (Human limit) or unlocks back to C_VLA.
- Dead zone (||v_sm|| < v_threshold) zeroes human contribution in eTaSL,
  independent of α.
- q_m is fed back to VLA as part of the observation (joint states).

Open Questions
--------------
- v_threshold: calibrate on hardware.
- Jacobian source in eTaSL: KDL / Pinocchio / built-in.
- robot_driver_crospi for Trossen: confirm with Santiago.

References
----------
- HACTS (2025): arxiv 2503.24070
- Policy Blending Formalism (Dragan et al.)
- Hagenow et al. (2025): Shared Control survey — TechRxiv
"""

import time
import numpy as np
from typing import Optional


# =============================================================================
# Constants
# =============================================================================

DEFAULT_V_THRESHOLD = 0.02   # ||v_sm|| below this → SpaceMouse idle; calibrate on HW
DEFAULT_TAU_C       = 0.4    # sigmoid centre for α(C_VLA)
DEFAULT_K_C         = 8.0    # sigmoid steepness
LONG_PRESS_S        = 0.6    # seconds to distinguish long-press from short-press


# =============================================================================
# SpaceMouseArbiter — button detection and dead-zone
# =============================================================================

class SpaceMouseArbiter:
    """
    Reads SpaceMouse state and detects button events.

    Button mapping (SpaceMouse Compact, 2 physical keys):
        Btn_L short-press  → gripper close
        Btn_R short-press  → gripper open
        Btn_L long-press   → toggle α = 1 (Human limit) / unlock
        Btn_R long-press   → lock α = 0 (VLA limit)

    Parameters
    ----------
    v_threshold : float
        ||v_sm|| (3-sample smoothed) above this → SpaceMouse active.
    long_press_s : float
        Seconds to distinguish long from short press.
    """

    def __init__(
        self,
        v_threshold: float = DEFAULT_V_THRESHOLD,
        long_press_s: float = LONG_PRESS_S,
    ):
        self.v_threshold = v_threshold
        self.long_press_s = long_press_s

        self._btn_l_pressed_at: Optional[float] = None
        self._btn_r_pressed_at: Optional[float] = None
        self._btn_l_was_down = False
        self._btn_r_was_down = False
        self._v_norm_history: list = []

    def update(
        self,
        v_sm: np.ndarray,
        btn_l: bool,
        btn_r: bool,
        now: Optional[float] = None,
    ) -> "ArbiterResult":
        """
        Process one control cycle.

        Parameters
        ----------
        v_sm : np.ndarray, shape (6,)
            SpaceMouse 6-DOF velocity [vx, vy, vz, ωx, ωy, ωz].
        btn_l, btn_r : bool
            True while the respective button is physically held.
        now : float or None
            Current time (time.monotonic()). Auto-filled if None.
        """
        if now is None:
            now = time.monotonic()

        v_sm = np.asarray(v_sm, dtype=np.float64)

        # 3-sample sliding-average for noise rejection
        self._v_norm_history.append(float(np.linalg.norm(v_sm)))
        if len(self._v_norm_history) > 3:
            self._v_norm_history.pop(0)
        v_norm = float(np.mean(self._v_norm_history))

        gripper_close = False
        gripper_open  = False
        override_long = False   # Btn_L long → toggle Human limit
        resume_long   = False   # Btn_R long → VLA limit

        # Btn_L
        if btn_l and not self._btn_l_was_down:
            self._btn_l_pressed_at = now
        if not btn_l and self._btn_l_was_down:
            duration = now - (self._btn_l_pressed_at or now)
            if duration >= self.long_press_s:
                override_long = True
            else:
                gripper_close = True
            self._btn_l_pressed_at = None
        self._btn_l_was_down = btn_l

        # Btn_R
        if btn_r and not self._btn_r_was_down:
            self._btn_r_pressed_at = now
        if not btn_r and self._btn_r_was_down:
            duration = now - (self._btn_r_pressed_at or now)
            if duration >= self.long_press_s:
                resume_long = True
            else:
                gripper_open = True
            self._btn_r_pressed_at = None
        self._btn_r_was_down = btn_r

        return ArbiterResult(
            v_norm=v_norm,
            is_active=v_norm > self.v_threshold,
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            override_long=override_long,
            resume_long=resume_long,
        )


class ArbiterResult:
    """Output of SpaceMouseArbiter.update()."""
    __slots__ = ('v_norm', 'is_active',
                 'gripper_close', 'gripper_open',
                 'override_long', 'resume_long')

    def __init__(self, v_norm, is_active,
                 gripper_close, gripper_open, override_long, resume_long):
        self.v_norm        = v_norm
        self.is_active     = is_active
        self.gripper_close = gripper_close
        self.gripper_open  = gripper_open
        self.override_long = override_long
        self.resume_long   = resume_long


# =============================================================================
# AlphaController — computes blending factor α
# =============================================================================

class AlphaController:
    """
    Computes blending factor α ∈ [0, 1] (human authority).

    Default: α = 1 - σ(k_c · (C_VLA - τ_c))   — driven continuously by C_VLA.

    Button overrides (sticky until next button event):
        Btn_R long  →  α_lock = 0.0  (VLA limit; α stays 0 until Btn_L long)
        Btn_L long  →  toggle:
            if α_lock == 1.0  →  α_lock = None  (release back to C_VLA)
            otherwise         →  α_lock = 1.0   (Human limit)

    The dead zone in eTaSL zeroes human contribution when ||v_sm|| < threshold,
    so α can be non-zero even when SpaceMouse is idle — no extra logic needed here.

    Parameters
    ----------
    k_c : float
        Sigmoid steepness.
    tau_c : float
        Sigmoid centre (C_VLA value where α = 0.5).
    """

    def __init__(self, k_c: float = DEFAULT_K_C, tau_c: float = DEFAULT_TAU_C):
        self.k_c   = k_c
        self.tau_c = tau_c
        self._alpha_lock: Optional[float] = None  # None = C_VLA-driven

    def update(self, C_VLA: float, override_long: bool, resume_long: bool) -> float:
        """
        Advance one cycle and return current α.

        Parameters
        ----------
        C_VLA : float
            VLA confidence ∈ [0, 1].
        override_long : bool
            Btn_L long-press fired this cycle.
        resume_long : bool
            Btn_R long-press fired this cycle.
        """
        if resume_long:
            self._alpha_lock = 0.0                      # VLA limit
        elif override_long:
            self._alpha_lock = None if self._alpha_lock == 1.0 else 1.0   # toggle Human

        if self._alpha_lock is not None:
            return float(self._alpha_lock)

        # Continuous α from C_VLA
        C = float(np.clip(C_VLA, 0.0, 1.0))
        alpha = 1.0 - 1.0 / (1.0 + np.exp(-self.k_c * (C - self.tau_c)))
        return float(np.clip(alpha, 0.0, 1.0))

    @property
    def is_locked(self) -> bool:
        """True if α is currently overridden by a button."""
        return self._alpha_lock is not None

    @property
    def lock_value(self) -> Optional[float]:
        """Current override value, or None if C_VLA-driven."""
        return self._alpha_lock

    def reset(self):
        self._alpha_lock = None


# =============================================================================
# SharedControlNode — top-level orchestrator
# =============================================================================

class SharedControlNode:
    """
    Top-level node for SpaceMouse-based shared control.

    Inputs each cycle:
        v_sm   : (6,) SpaceMouse velocity
        btn_l  : bool left button
        btn_r  : bool right button
        C_VLA  : float VLA confidence from CBC

    Outputs each cycle (SharedControlOutput):
        v_sm          : (6,) — publish as geometry_msgs/Twist to /spacemouse/twist
        alpha         : float — publish as std_msgs/Float32 to /shared_control/alpha
        gripper_close : bool  — send gripper close command
        gripper_open  : bool  — send gripper open command
        alpha_locked  : bool  — UI feedback (is button override active?)

    Typical 30 Hz loop:

        node = SharedControlNode()

        while running:
            v_sm, bl, br = spacemouse.read()    # (6,), bool, bool
            C_VLA        = cbc.compute(...)

            out = node.step(v_sm, bl, br, C_VLA)

            ros_pub_twist.publish(out.v_sm)
            ros_pub_alpha.publish(out.alpha)
            if out.gripper_close: gripper.close()
            if out.gripper_open:  gripper.open()
    """

    def __init__(
        self,
        v_threshold: float = DEFAULT_V_THRESHOLD,
        k_c: float = DEFAULT_K_C,
        tau_c: float = DEFAULT_TAU_C,
        long_press_s: float = LONG_PRESS_S,
    ):
        self.arbiter    = SpaceMouseArbiter(v_threshold=v_threshold,
                                            long_press_s=long_press_s)
        self.alpha_ctrl = AlphaController(k_c=k_c, tau_c=tau_c)

    def step(
        self,
        v_sm: np.ndarray,
        btn_l: bool,
        btn_r: bool,
        C_VLA: float = 1.0,
    ) -> "SharedControlOutput":
        """Execute one control cycle."""
        arb   = self.arbiter.update(v_sm, btn_l, btn_r)
        alpha = self.alpha_ctrl.update(C_VLA, arb.override_long, arb.resume_long)

        return SharedControlOutput(
            v_sm=np.asarray(v_sm, dtype=np.float64),
            alpha=alpha,
            gripper_close=arb.gripper_close,
            gripper_open=arb.gripper_open,
            alpha_locked=self.alpha_ctrl.is_locked,
            lock_value=self.alpha_ctrl.lock_value,
        )

    def reset(self):
        """Call at episode start."""
        self.alpha_ctrl.reset()
        self.arbiter._v_norm_history.clear()


class SharedControlOutput:
    """Output of one SharedControlNode.step() call."""
    __slots__ = ('v_sm', 'alpha', 'gripper_close', 'gripper_open',
                 'alpha_locked', 'lock_value')

    def __init__(self, v_sm, alpha, gripper_close, gripper_open,
                 alpha_locked, lock_value):
        self.v_sm         = v_sm
        self.alpha        = alpha
        self.gripper_close = gripper_close
        self.gripper_open  = gripper_open
        self.alpha_locked  = alpha_locked
        self.lock_value    = lock_value

    def __repr__(self):
        lock_str = f"lock={self.lock_value:.1f}" if self.alpha_locked else "C_VLA-driven"
        return f"SharedControlOutput(α={self.alpha:.3f}, {lock_str})"


# =============================================================================
# Sanity check / demo
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SHARED CONTROL — ALPHA CONTROLLER DEMO")
    print("=" * 60)

    node = SharedControlNode()
    v_idle   = np.zeros(6)
    v_active = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    print("\n--- Phase 1: C_VLA-driven α (high confidence → VLA leads) ---")
    for cvla in [0.9, 0.6, 0.4, 0.1]:
        out = node.step(v_active, False, False, C_VLA=cvla)
        print(f"  C_VLA={cvla:.1f}  →  {out}")

    print("\n--- Phase 2: Btn_L long-press → Human limit (α = 1) ---")
    t0 = time.monotonic()
    while time.monotonic() - t0 < LONG_PRESS_S + 0.05:
        node.step(v_active, btn_l=True, btn_r=False, C_VLA=0.8)
        time.sleep(0.01)
    out = node.step(v_active, False, False, C_VLA=0.8)
    print(f"  After Btn_L release: {out}")

    print("\n--- Phase 3: Btn_L long-press again → unlock (back to C_VLA) ---")
    t0 = time.monotonic()
    while time.monotonic() - t0 < LONG_PRESS_S + 0.05:
        node.step(v_active, btn_l=True, btn_r=False, C_VLA=0.8)
        time.sleep(0.01)
    out = node.step(v_active, False, False, C_VLA=0.8)
    print(f"  After second Btn_L release: {out}")

    print("\n--- Phase 4: Btn_R long-press → VLA limit (α = 0) ---")
    t0 = time.monotonic()
    while time.monotonic() - t0 < LONG_PRESS_S + 0.05:
        node.step(v_active, btn_l=False, btn_r=True, C_VLA=0.3)
        time.sleep(0.01)
    out = node.step(v_idle, False, False, C_VLA=0.3)
    print(f"  After Btn_R release: {out}")

    print("=" * 60)

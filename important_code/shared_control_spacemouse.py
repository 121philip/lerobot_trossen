"""
Shared Control (SpaceMouse Edition) — Layer 2 Mode Arbitration
==============================================================

Replaces the Leader-arm-based shared_control.py with a SpaceMouse-based
design following the revised architecture agreed after the 2026-04-02
meeting with supervisors Erwin Aertbelijn, Maxim Vochten, and Santiago
Iregui Rincon.

Key Changes vs. shared_control.py
----------------------------------
- **Human input**: SpaceMouse (6-DOF task-space velocity v_sm)
  instead of Leader arm (q_leader, joint-space position).
- **Intervention detection**: SpaceMouse magnitude ||v_sm|| > threshold
  instead of position deviation ||q_leader - q_follower||.
- **Three FSM states**: VLA_AUTONOMOUS / SHARED_CONTROL / HUMAN_ONLY
  instead of the previous two (VLA / HUMAN).
- **Velocity domain blending**: unify VLA position output and SpaceMouse
  velocity to joint-space velocity before mixing.
- **Impedance control**: REMOVED for now (no leader arm hardware required).
- **Gripper**: SpaceMouse physical buttons (Btn_L = close, Btn_R = open).

Architecture
------------

    SpaceMouse (v_sm) ──► SpaceMouseArbiter ──► FSM state / α
                      ──► JacobianIK(q_m)  ──► q̇_human
    SmolVLA (chunk)   ──► q_VLA[t], q_VLA[t+1] ──► q̇_VLA (diff)
    Follower (q_m)    ──► both IK and integration baseline

    VelocityBlender:
        q̇_cmd  = (1 - α)·q̇_VLA + α·q̇_human
        q_d     = q_m + q̇_cmd · dt     ← sent to eTaSL / Follower

Confirmed Design Decisions
--------------------------
- SHARED → VLA_AUTONOMOUS requires explicit RESUME button press.
  SpaceMouse going idle only lowers α; state stays SHARED.  (Confirmed 2026-04-15)
- Gripper (joint 6): SpaceMouse Btn_L = close, Btn_R = open.
  Long-press Btn_L = OVERRIDE; Long-press Btn_R = RESUME.   (Confirmed 2026-04-15)

ROS2 Interface (confirmed from CroSPI docs 2026-04-16)
-------------------------------------------------------
SpaceMouse → CroSPI:
    Publish geometry_msgs/Twist to /spacemouse/twist
    CroSPI blackboard: TwistInputHandler { topic-name: /spacemouse/twist, varname: v_sm }

VLA joints → CroSPI:
    Publish crospi_interfaces/Input to /vla/joint_targets
    (names=["joint1",...,"joint7"], data=q_vla[0:7])

VLA confidence → CroSPI:
    Publish std_msgs/Float32 to /vla/confidence

Follower feedback → CroSPI:
    CroSPI robot driver reads /joint_states (sensor_msgs/JointState) internally

eTaSL is internal to CroSPI (not a separate node).
CroSPI robot driver bridges eTaSL output → ros2_control → arm_controller.
Topic names above are proposals; confirm exact names with Santiago.

Open Questions
--------------
- v_threshold: SpaceMouse dead-zone magnitude for SHARED entry (calibrate on hardware).
- Jacobian source: KDL vs. PyBullet vs. pin (Pinocchio).
- robot_driver_crospi: confirm Trossen/WidowX driver exists or needs to be written.

References
----------
- HACTS (2025): arxiv 2503.24070
- Policy Blending Formalism (Dragan et al.): personalrobotics.cs.washington.edu
- Argallab Blended Human-Robot Control: argallab.northwestern.edu
- Hagenow et al. (2025): Shared Control survey — TechRxiv
"""

import enum
import time
import numpy as np
from typing import Optional


# =============================================================================
# ControlMode FSM states
# =============================================================================

class ControlMode(enum.Enum):
    """Three-state FSM for SpaceMouse-based shared control."""
    VLA_AUTONOMOUS = "vla_autonomous"
    """VLA controls the Follower exclusively. α = 0.
    SpaceMouse is monitored but has zero weight.
    Gripper tracks VLA output (joint 6)."""

    SHARED_CONTROL = "shared_control"
    """VLA and SpaceMouse jointly control the Follower.
    α ∈ (0, 1) varies continuously with C_VLA:
        α = 1 - sigmoid(k_c * (C_VLA - τ_c))
    High C_VLA → VLA leads; Low C_VLA → Human leads.
    Gripper: SpaceMouse buttons."""

    HUMAN_ONLY = "human_only"
    """SpaceMouse controls the Follower exclusively. α = 1.
    VLA inference may continue in background for fresh CBC.
    Gripper: SpaceMouse buttons."""

    # Internal transition states (short-lived, used for smooth ramps)
    TRANSITION_TO_SHARED = "transition_to_shared"
    TRANSITION_TO_HUMAN  = "transition_to_human"
    TRANSITION_TO_VLA    = "transition_to_vla"


# =============================================================================
# Constants
# =============================================================================

# Default SpaceMouse dead-zone: total 6-DOF norm below this → "idle"
DEFAULT_V_THRESHOLD  = 0.02   # m/s equivalent; calibrate on hardware
DEFAULT_V_IDLE       = 0.005

# Default transition ramp duration (seconds)
DEFAULT_TRANSITION_S = 0.3

# Confidence thresholds (τ_low triggers SHARED → HUMAN escalation)
DEFAULT_TAU_LOW      = 0.15   # C_VLA below this sustained → escalate to HUMAN
DEFAULT_TAU_C        = 0.4    # Sigmoid centre for α computation
DEFAULT_K_C          = 8.0    # Sigmoid steepness for α

# Long-press duration to distinguish gripper click from mode switch
LONG_PRESS_S         = 0.6    # seconds


# =============================================================================
# SpaceMouseArbiter
# =============================================================================

class SpaceMouseArbiter:
    """
    Monitors SpaceMouse input and physical buttons to drive FSM transitions.

    SpaceMouse input v_sm = [vx, vy, vz, ωx, ωy, ωz] (6-DOF).

    Physical button mapping (confirmed 2026-04-15):
        Btn_L short-press  → gripper close
        Btn_R short-press  → gripper open
        Btn_L long-press   → OVERRIDE  (→ HUMAN_ONLY)
        Btn_R long-press   → RESUME    (→ VLA_AUTONOMOUS)

    Parameters
    ----------
    v_threshold : float
        ||v_sm|| above this triggers VLA_AUTO → SHARED.
    v_idle : float
        ||v_sm|| below this for hysteresis (idle detection within SHARED).
    long_press_s : float
        Seconds to distinguish long-press from short-press on buttons.
    """

    def __init__(
        self,
        v_threshold: float = DEFAULT_V_THRESHOLD,
        v_idle: float = DEFAULT_V_IDLE,
        long_press_s: float = LONG_PRESS_S,
    ):
        self.v_threshold = v_threshold
        self.v_idle = v_idle
        self.long_press_s = long_press_s

        # Button press tracking
        self._btn_l_pressed_at: Optional[float] = None
        self._btn_r_pressed_at: Optional[float] = None
        self._btn_l_was_down = False
        self._btn_r_was_down = False

        # Sliding-window norm for noise rejection (last 3 samples)
        self._v_norm_history: list = []

    def update(
        self,
        v_sm: np.ndarray,
        btn_l: bool,
        btn_r: bool,
        now: Optional[float] = None,
    ) -> "ArbiterResult":
        """
        Process one control cycle of SpaceMouse input.

        Parameters
        ----------
        v_sm : np.ndarray, shape (6,)
            SpaceMouse 6-DOF velocity [vx, vy, vz, ωx, ωy, ωz].
        btn_l : bool
            True while left button is physically pressed.
        btn_r : bool
            True while right button is physically pressed.
        now : float or None
            Current time (time.monotonic()). Defaults to time.monotonic().

        Returns
        -------
        ArbiterResult
            Contains: v_norm, is_active, gripper_close, gripper_open,
            override_pressed (long Btn_L), resume_pressed (long Btn_R).
        """
        if now is None:
            now = time.monotonic()

        v_sm = np.asarray(v_sm, dtype=np.float64)
        v_norm = float(np.linalg.norm(v_sm))

        # --- Sliding-window smoothing (simple 3-sample mean) ---
        self._v_norm_history.append(v_norm)
        if len(self._v_norm_history) > 3:
            self._v_norm_history.pop(0)
        v_norm_smooth = float(np.mean(self._v_norm_history))

        is_active = v_norm_smooth > self.v_threshold
        is_idle   = v_norm_smooth < self.v_idle

        # --- Button state machine ---
        gripper_close  = False
        gripper_open   = False
        override_long  = False  # OVERRIDE (→ HUMAN_ONLY)
        resume_long    = False  # RESUME   (→ VLA / SHARED)

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
            v_norm=v_norm_smooth,
            is_active=is_active,
            is_idle=is_idle,
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            override_long=override_long,
            resume_long=resume_long,
        )


class ArbiterResult:
    """Output of SpaceMouseArbiter.update()."""
    __slots__ = (
        'v_norm', 'is_active', 'is_idle',
        'gripper_close', 'gripper_open',
        'override_long', 'resume_long',
    )

    def __init__(self, v_norm, is_active, is_idle,
                 gripper_close, gripper_open, override_long, resume_long):
        self.v_norm       = v_norm
        self.is_active    = is_active
        self.is_idle      = is_idle
        self.gripper_close = gripper_close
        self.gripper_open  = gripper_open
        self.override_long = override_long
        self.resume_long   = resume_long


# =============================================================================
# JacobianIK — SpaceMouse task-space velocity → joint velocity
# =============================================================================

class JacobianIK:
    """
    Converts SpaceMouse 6-DOF Cartesian velocity to 7-DOF joint velocity
    using the robot Jacobian.

    Formula:
        q̇_human = J†(q_m) · v_sm
        J† = Jᵀ · (J·Jᵀ + λ·I)⁻¹   (damped least-squares, avoids singularities)

    NOTE: This class uses a stub Jacobian by default. In production, replace
    ``_compute_jacobian`` with a proper kinematic library call:
        - KDL (C++/Python)
        - Pinocchio (pin)
        - PyBullet forward kinematics

    Parameters
    ----------
    damping : float
        Damping factor λ for singularity robustness. Start at 1e-4,
        increase near singularities.
    q_dot_max : float
        Per-joint velocity saturation limit (rad/s).
    n_joints : int
        Number of arm joints to control via IK (default 6, excluding gripper).
    """

    def __init__(
        self,
        damping: float = 1e-4,
        q_dot_max: float = 1.5,
        n_joints: int = 6,
    ):
        self.damping   = damping
        self.q_dot_max = q_dot_max
        self.n_joints  = n_joints

    def compute(
        self,
        v_sm: np.ndarray,
        q_m: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        v_sm : np.ndarray, shape (6,)
            SpaceMouse velocity [vx, vy, vz, ωx, ωy, ωz].
        q_m : np.ndarray, shape (7,)
            Current Follower joint positions (for Jacobian evaluation).

        Returns
        -------
        np.ndarray, shape (6,)
            Joint velocities q̇_human for the first 6 joints.
            Gripper (joint 6) is handled separately via buttons.
        """
        v_sm = np.asarray(v_sm, dtype=np.float64)
        q_m  = np.asarray(q_m,  dtype=np.float64)

        J     = self._compute_jacobian(q_m[:self.n_joints])   # 6 × n_joints
        lam   = self.damping
        JJT   = J @ J.T                                        # 6 × 6
        J_dls = J.T @ np.linalg.inv(JJT + lam * np.eye(6))   # n_joints × 6

        q_dot = J_dls @ v_sm                                   # n_joints

        # Clip to safe joint velocity limits
        q_dot = np.clip(q_dot, -self.q_dot_max, self.q_dot_max)
        return q_dot

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        STUB: Returns a random Jacobian for offline testing only.

        REPLACE THIS with your kinematic library:
            import pinocchio as pin
            pin.computeJointJacobian(model, data, q, joint_id)

        Shape: (6, n_joints)
        """
        np.random.seed(int(np.sum(q * 1000)) % (2**31))
        J = np.random.randn(6, self.n_joints) * 0.1
        return J


# =============================================================================
# VelocityBlender — core mixing logic
# =============================================================================

class VelocityBlender:
    """
    Blends VLA joint velocity with SpaceMouse joint velocity to produce
    a desired joint position q_d for the Follower arm.

    Velocity domain blending (recommended over position blending):
        q̇_VLA   = (q_VLA[t+1] − q_VLA[t]) / dt
        q̇_human = JacobianIK(v_sm, q_m)
        q̇_cmd   = (1 − α) · q̇_VLA + α · q̇_human
        q_d      = q_m + q̇_cmd · dt        ← uses q_m as baseline (no drift)

    Gripper (joint index 6):
        - VLA_AUTONOMOUS / no button:  q_d[6] = q_VLA[6]
        - Btn_L pressed:               q_d[6] → GRIPPER_CLOSE
        - Btn_R pressed:               q_d[6] → GRIPPER_OPEN

    Parameters
    ----------
    dt : float
        Control cycle period in seconds (default 1/30).
    k_c : float
        Sigmoid steepness for α(C_VLA) in SHARED mode.
    tau_c : float
        Sigmoid centre for α(C_VLA) in SHARED mode.
    gripper_close_pos : float
        Joint-6 position for fully closed gripper.
    gripper_open_pos : float
        Joint-6 position for fully open gripper.
    conflict_override : bool
        If True, automatically increases α when VLA and human velocities
        conflict (angle > 90°). Recommended but can be disabled.
    """

    GRIPPER_JOINT = 6
    DEFAULT_GRIPPER_CLOSE =  0.57   # rad — calibrate per hardware
    DEFAULT_GRIPPER_OPEN  = -0.57

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        k_c: float = DEFAULT_K_C,
        tau_c: float = DEFAULT_TAU_C,
        gripper_close_pos: float = DEFAULT_GRIPPER_CLOSE,
        gripper_open_pos: float  = DEFAULT_GRIPPER_OPEN,
        conflict_override: bool  = True,
    ):
        self.dt               = dt
        self.k_c              = k_c
        self.tau_c            = tau_c
        self.gripper_close_pos = gripper_close_pos
        self.gripper_open_pos  = gripper_open_pos
        self.conflict_override = conflict_override

    def compute_alpha(self, mode: ControlMode, C_VLA: float) -> float:
        """
        α ∈ [0, 1]: human authority factor.
          α = 0  →  full VLA
          α = 1  →  full human
        """
        if mode in (ControlMode.VLA_AUTONOMOUS, ControlMode.TRANSITION_TO_VLA):
            return 0.0
        elif mode in (ControlMode.HUMAN_ONLY, ControlMode.TRANSITION_TO_HUMAN):
            return 1.0
        else:  # SHARED_CONTROL
            C = float(np.clip(C_VLA, 0.0, 1.0))
            # High confidence → low α (VLA leads); Low confidence → high α (human leads)
            alpha = 1.0 - 1.0 / (1.0 + np.exp(-self.k_c * (C - self.tau_c)))
            return float(np.clip(alpha, 0.0, 1.0))

    def blend(
        self,
        mode: ControlMode,
        q_m: np.ndarray,
        q_vla_t: np.ndarray,
        q_vla_t1: np.ndarray,
        q_dot_human: np.ndarray,
        C_VLA: float = 1.0,
        gripper_close: bool = False,
        gripper_open:  bool = False,
        transition_scale: float = 1.0,
    ) -> "BlendResult":
        """
        One blending cycle.

        Parameters
        ----------
        mode : ControlMode
        q_m : np.ndarray, shape (7,)
            Current Follower joint positions (integration baseline).
        q_vla_t : np.ndarray, shape (7,)
            VLA chunk position at current step t.
        q_vla_t1 : np.ndarray, shape (7,)
            VLA chunk position at next step t+1 (for differentiation).
        q_dot_human : np.ndarray, shape (6,)
            Joint velocity from JacobianIK (first 6 joints only).
        C_VLA : float
            VLA confidence in [0, 1].
        gripper_close : bool
            SpaceMouse Btn_L short-press this cycle.
        gripper_open : bool
            SpaceMouse Btn_R short-press this cycle.
        transition_scale : float
            Ramp scale during transitions [0, 1].

        Returns
        -------
        BlendResult
        """
        q_m      = np.asarray(q_m,      dtype=np.float64)
        q_vla_t  = np.asarray(q_vla_t,  dtype=np.float64)
        q_vla_t1 = np.asarray(q_vla_t1, dtype=np.float64)
        q_dh     = np.asarray(q_dot_human, dtype=np.float64)

        alpha = self.compute_alpha(mode, C_VLA) * transition_scale

        # --- VLA joint velocity (first 6 joints via differentiation) ---
        q_dot_vla = (q_vla_t1[:self.GRIPPER_JOINT] - q_vla_t[:self.GRIPPER_JOINT]) / self.dt

        # --- Conflict override (optional) ---
        if self.conflict_override and alpha > 0.0 and alpha < 1.0:
            dot_product = float(np.dot(q_dot_vla, q_dh[:self.GRIPPER_JOINT]))
            norm_product = (np.linalg.norm(q_dot_vla) *
                            np.linalg.norm(q_dh[:self.GRIPPER_JOINT]) + 1e-9)
            cos_angle = dot_product / norm_product
            if cos_angle < 0:   # angle > 90° → conflict → human overrides
                conflict_boost = (-cos_angle) * (1.0 - alpha)  # boost α proportional to conflict
                alpha = float(np.clip(alpha + conflict_boost, 0.0, 1.0))

        # --- Blend first 6 joints in velocity domain ---
        q_dot_cmd = (1.0 - alpha) * q_dot_vla + alpha * q_dh[:self.GRIPPER_JOINT]

        # --- Integrate → desired position (q_m baseline prevents drift) ---
        q_d = q_m.copy()
        q_d[:self.GRIPPER_JOINT] = q_m[:self.GRIPPER_JOINT] + q_dot_cmd * self.dt

        # --- Gripper (joint 6) ---
        if gripper_close:
            q_d[self.GRIPPER_JOINT] = self.gripper_close_pos
        elif gripper_open:
            q_d[self.GRIPPER_JOINT] = self.gripper_open_pos
        else:
            # Default: follow VLA gripper target regardless of mode
            q_d[self.GRIPPER_JOINT] = q_vla_t[self.GRIPPER_JOINT]

        # --- Feed-forward velocity for eTaSL ---
        q_dot_d = np.zeros(7)
        q_dot_d[:self.GRIPPER_JOINT] = q_dot_cmd

        return BlendResult(
            q_d=q_d,
            q_dot_d=q_dot_d,
            alpha=alpha,
        )


class BlendResult:
    """Output of VelocityBlender.blend()."""
    __slots__ = ('q_d', 'q_dot_d', 'alpha')

    def __init__(self, q_d, q_dot_d, alpha):
        self.q_d     = q_d
        self.q_dot_d = q_dot_d
        self.alpha   = alpha

    def __repr__(self):
        return (f"BlendResult(α={self.alpha:.3f}, "
                f"q_d={np.round(self.q_d, 3).tolist()})")


# =============================================================================
# ModeSwitcher (3-state FSM)
# =============================================================================

class ModeSwitcher:
    """
    Manages transitions between three control modes with smooth ramps.

    Transition ramps prevent jerky motion when switching modes.
    During a transition, ``transition_scale`` goes from 0→1 or 1→0
    and is used by VelocityBlender to ramp α.

    FSM transitions (confirmed 2026-04-15):
        VLA_AUTO   → SHARED      : SpaceMouse active (||v_sm|| > threshold)
        VLA_AUTO   → HUMAN       : OVERRIDE long-press
        SHARED     → VLA_AUTO    : RESUME long-press ONLY (not automatic)
        SHARED     → HUMAN       : C_VLA low sustained, or OVERRIDE long-press
        HUMAN      → SHARED      : SHARED button (long-press Btn_L from HUMAN)
        HUMAN      → VLA_AUTO    : RESUME long-press

    Parameters
    ----------
    transition_time : float
        Ramp duration in seconds (default 0.3).
    tau_low : float
        C_VLA threshold below which SHARED escalates to HUMAN.
    tau_low_duration : float
        Seconds C_VLA must stay below tau_low before escalation.
    """

    def __init__(
        self,
        transition_time: float = DEFAULT_TRANSITION_S,
        tau_low: float = DEFAULT_TAU_LOW,
        tau_low_duration: float = 2.0,
    ):
        self.transition_time   = transition_time
        self.tau_low           = tau_low
        self.tau_low_duration  = tau_low_duration

        self._mode = ControlMode.VLA_AUTONOMOUS
        self._transition_start: Optional[float] = None
        self._low_confidence_since: Optional[float] = None

    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def transition_scale(self) -> float:
        """
        Progress of current transition ramp ∈ [0, 1].
        1.0 when not in transition (stable state).
        """
        if self._mode not in (
            ControlMode.TRANSITION_TO_SHARED,
            ControlMode.TRANSITION_TO_HUMAN,
            ControlMode.TRANSITION_TO_VLA,
        ):
            return 1.0
        if self._transition_start is None:
            return 1.0
        elapsed = time.monotonic() - self._transition_start
        return float(min(elapsed / self.transition_time, 1.0))

    def update(
        self,
        arbiter: ArbiterResult,
        C_VLA: float,
        now: Optional[float] = None,
    ) -> ControlMode:
        """
        Advance FSM by one cycle.

        Parameters
        ----------
        arbiter : ArbiterResult
            Output of SpaceMouseArbiter.update().
        C_VLA : float
            Current VLA confidence [0, 1].
        now : float or None
            Current time.

        Returns
        -------
        ControlMode
            Current (possibly updated) mode.
        """
        if now is None:
            now = time.monotonic()

        m = self._mode

        # ── Transition completions ──────────────────────────────────────
        if m == ControlMode.TRANSITION_TO_SHARED and self.transition_scale >= 1.0:
            self._mode = ControlMode.SHARED_CONTROL
            self._transition_start = None
            m = self._mode

        elif m == ControlMode.TRANSITION_TO_HUMAN and self.transition_scale >= 1.0:
            self._mode = ControlMode.HUMAN_ONLY
            self._transition_start = None
            m = self._mode

        elif m == ControlMode.TRANSITION_TO_VLA and self.transition_scale >= 1.0:
            self._mode = ControlMode.VLA_AUTONOMOUS
            self._transition_start = None
            self._low_confidence_since = None
            m = self._mode

        # ── VLA_AUTONOMOUS ──────────────────────────────────────────────
        if m == ControlMode.VLA_AUTONOMOUS:
            if arbiter.override_long:
                self._start_transition(ControlMode.TRANSITION_TO_HUMAN, now)
            elif arbiter.is_active:
                self._start_transition(ControlMode.TRANSITION_TO_SHARED, now)

        # ── SHARED_CONTROL ──────────────────────────────────────────────
        elif m == ControlMode.SHARED_CONTROL:
            # Track low-confidence duration for HUMAN escalation
            if C_VLA < self.tau_low:
                if self._low_confidence_since is None:
                    self._low_confidence_since = now
                elif (now - self._low_confidence_since) >= self.tau_low_duration:
                    self._start_transition(ControlMode.TRANSITION_TO_HUMAN, now)
                    self._low_confidence_since = None
            else:
                self._low_confidence_since = None

            if arbiter.override_long:
                self._start_transition(ControlMode.TRANSITION_TO_HUMAN, now)
                self._low_confidence_since = None
            elif arbiter.resume_long:
                # CONFIRMED: only button press returns to VLA (not automatic)
                self._start_transition(ControlMode.TRANSITION_TO_VLA, now)
                self._low_confidence_since = None

        # ── HUMAN_ONLY ──────────────────────────────────────────────────
        elif m == ControlMode.HUMAN_ONLY:
            if arbiter.resume_long:
                self._start_transition(ControlMode.TRANSITION_TO_VLA, now)
            elif arbiter.override_long:
                # Long Btn_L from HUMAN_ONLY → back to SHARED
                self._start_transition(ControlMode.TRANSITION_TO_SHARED, now)

        return self._mode

    def _start_transition(self, target: ControlMode, now: float):
        if self._mode not in (
            ControlMode.TRANSITION_TO_SHARED,
            ControlMode.TRANSITION_TO_HUMAN,
            ControlMode.TRANSITION_TO_VLA,
        ):
            self._mode = target
            self._transition_start = now

    def reset(self):
        """Reset to VLA_AUTONOMOUS (call at episode start)."""
        self._mode = ControlMode.VLA_AUTONOMOUS
        self._transition_start = None
        self._low_confidence_since = None


# =============================================================================
# SharedControlSystemSM — unified orchestrator
# =============================================================================

class SharedControlSystemSM:
    """
    Top-level orchestrator for SpaceMouse-based shared control.

    Typical usage in a 30 Hz control loop:

        system = SharedControlSystemSM()

        while running:
            q_m          = follower.get_all_positions()        # (7,)
            v_sm, bl, br = spacemouse.read()                   # (6,), bool, bool
            q_vla_t      = chunk[step_idx]                     # (7,)
            q_vla_t1     = chunk[step_idx + 1]                 # (7,)
            C_VLA        = cbc.compute(prev_chunk, curr_chunk) # float

            result = system.step(
                q_m=q_m, v_sm=v_sm, btn_l=bl, btn_r=br,
                q_vla_t=q_vla_t, q_vla_t1=q_vla_t1, C_VLA=C_VLA,
            )

            # Send to eTaSL / CroSPI
            send_to_etasl(result.q_d, result.q_dot_d)

            print(f"Mode: {result.mode.value}  α={result.alpha:.2f}")
    """

    def __init__(
        self,
        arbiter:  Optional[SpaceMouseArbiter] = None,
        ik:       Optional[JacobianIK]        = None,
        blender:  Optional[VelocityBlender]   = None,
        switcher: Optional[ModeSwitcher]      = None,
    ):
        self.arbiter  = arbiter  or SpaceMouseArbiter()
        self.ik       = ik       or JacobianIK()
        self.blender  = blender  or VelocityBlender()
        self.switcher = switcher or ModeSwitcher()

    def step(
        self,
        q_m: np.ndarray,
        v_sm: np.ndarray,
        btn_l: bool,
        btn_r: bool,
        q_vla_t: np.ndarray,
        q_vla_t1: np.ndarray,
        C_VLA: float = 1.0,
    ) -> "SharedControlResultSM":
        """
        Execute one 30 Hz control cycle.

        Returns
        -------
        SharedControlResultSM
            q_d, q_dot_d, mode, alpha.
        """
        # 1. SpaceMouse arbitration (button detection + magnitude)
        arb = self.arbiter.update(v_sm, btn_l, btn_r)

        # 2. FSM state transition
        self.switcher.update(arb, C_VLA)
        mode  = self.switcher.mode
        scale = self.switcher.transition_scale

        # 3. Jacobian IK: v_sm (task space) → q̇_human (joint space, 6-DOF)
        q_dot_human = self.ik.compute(v_sm, q_m)

        # 4. Velocity blending → q_d, q̇_d
        blend = self.blender.blend(
            mode=mode,
            q_m=q_m,
            q_vla_t=q_vla_t,
            q_vla_t1=q_vla_t1,
            q_dot_human=q_dot_human,
            C_VLA=C_VLA,
            gripper_close=arb.gripper_close,
            gripper_open=arb.gripper_open,
            transition_scale=scale,
        )

        return SharedControlResultSM(
            q_d=blend.q_d,
            q_dot_d=blend.q_dot_d,
            mode=mode,
            alpha=blend.alpha,
        )

    def reset(self):
        """Reset all state. Call at episode start."""
        self.switcher.reset()
        self.arbiter._v_norm_history.clear()


class SharedControlResultSM:
    """Output of one SharedControlSystemSM.step() call."""
    __slots__ = ('q_d', 'q_dot_d', 'mode', 'alpha')

    def __init__(self, q_d, q_dot_d, mode, alpha):
        self.q_d     = q_d
        self.q_dot_d = q_dot_d
        self.mode    = mode
        self.alpha   = alpha

    def __repr__(self):
        return (f"SharedControlResultSM(mode={self.mode.value}, "
                f"α={self.alpha:.3f})")


# =============================================================================
# Sanity check / demo
# =============================================================================

if __name__ == "__main__":
    """
    Simulate: VLA → SpaceMouse nudge → Shared → full override → VLA resume.
    """
    print("\n" + "=" * 72)
    print("  SHARED CONTROL (SpaceMouse) SANITY CHECK")
    print("=" * 72)

    system = SharedControlSystemSM()

    q_m     = np.array([0.0,  1.0, 0.5, 0.0, 0.0, 0.0, 0.02])
    q_vla_t = np.array([0.05, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02])
    q_vla_t1= np.array([0.10, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02])
    v_sm_idle   = np.zeros(6)
    v_sm_active = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Phase 1: VLA autonomous (SpaceMouse idle)
    print("\n--- Phase 1: VLA autonomous ---")
    for i in range(3):
        r = system.step(q_m, v_sm_idle, False, False, q_vla_t, q_vla_t1, C_VLA=0.9)
        print(f"  [{i}] {r}")

    # Phase 2: SpaceMouse becomes active → auto-enter Shared
    print("\n--- Phase 2: SpaceMouse active → SHARED ---")
    for i in range(5):
        r = system.step(q_m, v_sm_active, False, False, q_vla_t, q_vla_t1, C_VLA=0.7)
        print(f"  [{i}] {r}")

    # Phase 3: Long-press Btn_L → HUMAN_ONLY
    print("\n--- Phase 3: Long-press Btn_L (OVERRIDE) → HUMAN_ONLY ---")
    # Simulate long press: hold btn_l for LONG_PRESS_S then release
    t0 = time.monotonic()
    while time.monotonic() - t0 < LONG_PRESS_S + 0.05:
        r = system.step(q_m, v_sm_active, btn_l=True, btn_r=False,
                        q_vla_t=q_vla_t, q_vla_t1=q_vla_t1, C_VLA=0.3)
        time.sleep(0.01)
    # Release
    r = system.step(q_m, v_sm_active, False, False, q_vla_t, q_vla_t1, C_VLA=0.3)
    print(f"  [after release] {r}")

    # Phase 4: Long-press Btn_R → back to VLA
    print("\n--- Phase 4: Long-press Btn_R (RESUME) → VLA ---")
    t0 = time.monotonic()
    while time.monotonic() - t0 < LONG_PRESS_S + 0.05:
        r = system.step(q_m, v_sm_idle, btn_l=False, btn_r=True,
                        q_vla_t=q_vla_t, q_vla_t1=q_vla_t1, C_VLA=0.8)
        time.sleep(0.01)
    r = system.step(q_m, v_sm_idle, False, False, q_vla_t, q_vla_t1, C_VLA=0.8)
    print(f"  [after release] {r}")

    print(f"\n  Final mode: {system.switcher.mode.value}")
    print("=" * 72)

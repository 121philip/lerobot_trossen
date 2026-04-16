"""
Shared Control — Layer 2 Mode Arbitration (Impedance-Based)
=============================================================

Purpose
-------
Replaces the original Conflict Gate with a system designed for the actual
WidowX AI Leader/Follower hardware setup. The key insight: the operator
controls the Follower arm by physically moving a Leader arm (not a joystick),
so the Leader must stay synchronized with the Follower during VLA autonomous
mode to enable safe control handover.

Architecture
------------
Three modules work together:

1. **LeaderImpedanceController** — During VLA mode, applies a virtual
   spring-damper to make the Leader arm gently track the Follower position.
   The operator feels the arm moving and can override it by applying force.

2. **InterventionDetector** — Monitors the deviation between Leader and
   Follower positions. When the operator applies force against the impedance
   spring, the deviation grows; exceeding a threshold triggers handover.

3. **ModeSwitcher** — Manages transitions between VLA mode (Mode A) and
   Human mode (Mode B) with a smooth ramp to avoid jerky motion.


Design Rationale (vs. Original Conflict Gate)
----------------------------------------------
The original Conflict Gate (conflict_gate.py) was designed for a joystick
input model: compare ∠(u_human, a_vla) to detect directional conflict.

Problem: with a Leader-Follower setup, during VLA autonomous mode the Leader
arm sits idle in gravity-comp mode while the Follower moves. If the operator
suddenly grabs the Leader to intervene, the massive position mismatch between
Leader (still at old position) and Follower (at VLA-driven position) would
cause the Follower to violently jump back — catastrophic in EOD scenarios.

Solution: keep the Leader synchronized via impedance control (virtual spring).
The intervention signal becomes position deviation (how much the operator
pushes the Leader away from the Follower), not angular disagreement.


Hardware Support
----------------
Trossen SDK natively supports this approach:
  - `external_effort` mode with `set_all_external_efforts()` for impedance
  - `get_all_positions()` / `get_all_velocities()` for state reading
  - Official demo `cartesian_external_effort.py` implements impedance control
  - Official demo `teleoperation.py` implements force feedback


References
----------
- HACTS (2025): Human-As-Copilot Teleoperation System — arxiv 2503.24070
  Uses bilateral position sync with foot pedal for mode switching.
- IGBT (2025): Input-Gated Bilateral Teleoperation — arxiv 2509.08226
  Force feedback without force sensors on low-cost hardware.
- Hagenow et al. (2025): Shared Control/Autonomy survey — TechRxiv
  Classifies our approach as "mode-switching" arbitration.
- De Winter et al. (2023): Shared vs traded control — Zotero R7XC9JRA
- Trossen cartesian_external_effort.py demo (impedance control)
- Trossen teleoperation.py demo (force feedback)
"""

import enum
import time
import numpy as np
from typing import Optional


# =============================================================================
# Constants
# =============================================================================

# WidowX AI joint weights — same rationale as conflict_gate.py.
# Used for computing weighted deviation norm.
DEFAULT_JOINT_WEIGHTS = np.array([
    3.0,   # joint_0: base rotation
    3.0,   # joint_1: shoulder pitch
    2.0,   # joint_2: elbow pitch
    1.0,   # joint_3: forearm roll
    1.0,   # joint_4: wrist pitch
    0.5,   # joint_5: wrist roll
    0.5,   # left_carriage_joint: gripper (included with low weight)
])


# =============================================================================
# ControlMode enum
# =============================================================================

class ControlMode(enum.Enum):
    """Active control mode for the shared control system."""
    VLA = "vla"              # VLA autonomous: follower executes VLA, leader tracks follower
    TRANSITION_TO_HUMAN = "transition_to_human"  # Smooth ramp from VLA to human
    HUMAN = "human"          # Human controls: follower tracks leader
    TRANSITION_TO_VLA = "transition_to_vla"      # Smooth ramp from human to VLA


# =============================================================================
# LeaderImpedanceController
# =============================================================================

class LeaderImpedanceController:
    """
    Virtual spring-damper that makes the Leader arm track the Follower.

    During VLA autonomous mode, computes external efforts to gently pull the
    Leader toward the Follower's current position:

        effort[i] = K[i] * (q_follower[i] - q_leader[i]) - D[i] * v_leader[i]

    The operator can still backdrive the Leader — the spring stiffness K is
    low enough for comfortable override but high enough for tracking at 30 Hz.

    When C_VLA is low (VLA is uncertain), K is automatically reduced via:

        K_effective = K_base * sigmoid(k_c * (C_VLA - tau_c))

    This gives the operator a haptic cue: the Leader arm feels "looser" when
    the VLA is uncertain, inviting human intervention.

    Parameters
    ----------
    K_base : np.ndarray, shape (7,)
        Base stiffness per joint (Nm/rad or N/m for gripper).
    D : np.ndarray, shape (7,)
        Damping per joint (Nm·s/rad or N·s/m for gripper).
    k_c : float
        Sigmoid steepness for confidence-based stiffness modulation.
    tau_c : float
        Sigmoid center for confidence modulation. At C_VLA = tau_c,
        stiffness is 50% of K_base.
    """

    # Reasonable defaults for WidowX AI Leader arm.
    # These MUST be tuned on actual hardware — start low and increase.
    DEFAULT_K = np.array([3.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.5])
    DEFAULT_D = np.array([0.3, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05])

    def __init__(
        self,
        K_base: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        k_c: float = 8.0,
        tau_c: float = 0.4,
    ):
        self.K_base = np.asarray(K_base if K_base is not None else self.DEFAULT_K, dtype=np.float64)
        self.D = np.asarray(D if D is not None else self.DEFAULT_D, dtype=np.float64)
        self.k_c = k_c
        self.tau_c = tau_c

    def compute_efforts(
        self,
        q_follower: np.ndarray,
        q_leader: np.ndarray,
        v_leader: np.ndarray,
        C_VLA: float = 1.0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Compute external efforts to apply to the Leader arm.

        Parameters
        ----------
        q_follower : np.ndarray, shape (7,)
            Current Follower joint positions (tracking target).
        q_leader : np.ndarray, shape (7,)
            Current Leader joint positions.
        v_leader : np.ndarray, shape (7,)
            Current Leader joint velocities.
        C_VLA : float
            VLA confidence in [0, 1]. Lower confidence → lower stiffness.
        scale : float
            Additional scale factor in [0, 1]. Used during transitions
            to ramp stiffness down to 0.

        Returns
        -------
        np.ndarray, shape (7,)
            External efforts to send to Leader via set_all_external_efforts().
        """
        q_follower = np.asarray(q_follower, dtype=np.float64)
        q_leader = np.asarray(q_leader, dtype=np.float64)
        v_leader = np.asarray(v_leader, dtype=np.float64)

        # Confidence-modulated stiffness
        C_VLA = float(np.clip(C_VLA, 0.0, 1.0))
        confidence_factor = 1.0 / (1.0 + np.exp(-self.k_c * (C_VLA - self.tau_c)))
        K_effective = self.K_base * confidence_factor * scale

        # Virtual spring-damper: pull leader toward follower
        position_error = q_follower - q_leader
        efforts = K_effective * position_error - self.D * scale * v_leader

        return efforts


# =============================================================================
# InterventionDetector
# =============================================================================

class InterventionDetector:
    """
    Detects when the operator is actively intervening by monitoring the
    position deviation between Leader and Follower arms.

    During VLA mode, the Leader tracks the Follower via impedance control.
    If the operator does NOT touch the Leader, deviation ≈ 0.
    If the operator grabs and pushes the Leader, deviation grows.
    When deviation exceeds `threshold_enter`, intervention is detected.

    Uses hysteresis to prevent oscillation:
      - Enter intervention: deviation > threshold_enter
      - Exit intervention:  deviation < threshold_exit (< threshold_enter)

    This replaces the angular Conflict Gate from the original design.
    Conceptually, it is a simplified Magnitude Guard (α) operating on
    Leader-Follower position error instead of human-vs-VLA direction.

    Parameters
    ----------
    threshold_enter : float
        Weighted deviation norm to trigger intervention (default 0.15 rad).
    threshold_exit : float
        Weighted deviation norm to clear intervention (default 0.05 rad).
        Must be < threshold_enter for hysteresis.
    joint_weights : np.ndarray or None
        Per-joint importance weights for deviation norm.
    """

    def __init__(
        self,
        threshold_enter: float = 0.15,
        threshold_exit: float = 0.05,
        joint_weights: Optional[np.ndarray] = None,
    ):
        if threshold_exit >= threshold_enter:
            raise ValueError(
                f"threshold_exit ({threshold_exit}) must be < "
                f"threshold_enter ({threshold_enter}) for hysteresis"
            )
        self.threshold_enter = threshold_enter
        self.threshold_exit = threshold_exit
        self.W = np.asarray(
            joint_weights if joint_weights is not None else DEFAULT_JOINT_WEIGHTS,
            dtype=np.float64,
        )
        self._intervening = False

    def check(
        self,
        q_leader: np.ndarray,
        q_follower: np.ndarray,
    ) -> bool:
        """
        Check whether the operator is intervening.

        Parameters
        ----------
        q_leader : np.ndarray, shape (7,)
            Current Leader arm positions.
        q_follower : np.ndarray, shape (7,)
            Current Follower arm positions.

        Returns
        -------
        bool
            True if operator is intervening (deviation above threshold).
        """
        deviation = np.asarray(q_leader, dtype=np.float64) - np.asarray(q_follower, dtype=np.float64)
        weighted_norm = float(np.sqrt(np.dot(deviation, self.W * deviation)))

        if self._intervening:
            # Currently in intervention → require low deviation to exit
            if weighted_norm < self.threshold_exit:
                self._intervening = False
        else:
            # Currently not intervening → require high deviation to enter
            if weighted_norm > self.threshold_enter:
                self._intervening = True

        return self._intervening

    @property
    def deviation_norm(self) -> float:
        """Last computed deviation norm (for logging/visualization)."""
        return getattr(self, '_last_deviation_norm', 0.0)

    def reset(self):
        """Reset to non-intervening state."""
        self._intervening = False


# =============================================================================
# ModeSwitcher
# =============================================================================

class ModeSwitcher:
    """
    Manages control mode transitions with smooth ramps.

    Transitions use a linear ramp over `transition_time` seconds to avoid
    jerky motion. During transition, `alpha` goes from 0→1 (to human)
    or 1→0 (to VLA).

    The Follower target is blended during transition:
        q_target = (1 - alpha) * q_vla + alpha * q_leader

    The Leader impedance stiffness is scaled by (1 - alpha) during
    transition to human (releasing the spring), and by alpha during
    transition to VLA (re-engaging the spring).

    Parameters
    ----------
    transition_time : float
        Duration of transition ramp in seconds (default 0.4).
    """

    def __init__(self, transition_time: float = 0.4):
        self.transition_time = transition_time
        self._mode = ControlMode.VLA
        self._transition_start: Optional[float] = None

    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def alpha(self) -> float:
        """
        Human authority factor in [0, 1].
          0.0 = full VLA control
          1.0 = full human control
        """
        if self._mode == ControlMode.VLA:
            return 0.0
        elif self._mode == ControlMode.HUMAN:
            return 1.0
        elif self._mode in (ControlMode.TRANSITION_TO_HUMAN, ControlMode.TRANSITION_TO_VLA):
            if self._transition_start is None:
                return 0.5  # should not happen
            elapsed = time.monotonic() - self._transition_start
            progress = min(elapsed / self.transition_time, 1.0)
            if self._mode == ControlMode.TRANSITION_TO_HUMAN:
                return progress        # 0 → 1
            else:
                return 1.0 - progress  # 1 → 0
        return 0.0

    def request_human_takeover(self):
        """Trigger transition from VLA to human control."""
        if self._mode == ControlMode.VLA:
            self._mode = ControlMode.TRANSITION_TO_HUMAN
            self._transition_start = time.monotonic()

    def request_vla_resume(self):
        """Trigger transition from human back to VLA control (button press)."""
        if self._mode == ControlMode.HUMAN:
            self._mode = ControlMode.TRANSITION_TO_VLA
            self._transition_start = time.monotonic()

    def update(self):
        """
        Call each control cycle to check if transitions have completed.
        """
        if self._mode == ControlMode.TRANSITION_TO_HUMAN:
            if self.alpha >= 1.0:
                self._mode = ControlMode.HUMAN
                self._transition_start = None

        elif self._mode == ControlMode.TRANSITION_TO_VLA:
            if self.alpha <= 0.0:
                self._mode = ControlMode.VLA
                self._transition_start = None

    def reset(self):
        """Reset to VLA mode."""
        self._mode = ControlMode.VLA
        self._transition_start = None


# =============================================================================
# SharedControlSystem — Unified orchestrator
# =============================================================================

class SharedControlSystem:
    """
    Top-level orchestrator integrating all three modules.

    Typical usage in a control loop:

        sc = SharedControlSystem()

        while running:
            q_follower = follower.get_all_positions()
            q_leader   = leader.get_all_positions()
            v_leader   = leader.get_all_velocities()

            result = sc.step(
                q_leader=q_leader,
                q_follower=q_follower,
                v_leader=v_leader,
                q_vla_target=q_vla_target,
                C_VLA=c_vla,
                button_pressed=button_state,
            )

            # Apply leader efforts (impedance tracking)
            leader.set_all_external_efforts(result.leader_efforts, 0.0, False)

            # Send follower target
            follower.set_all_positions(result.follower_target, 0.0, False)
    """

    def __init__(
        self,
        impedance: Optional[LeaderImpedanceController] = None,
        detector: Optional[InterventionDetector] = None,
        switcher: Optional[ModeSwitcher] = None,
    ):
        self.impedance = impedance or LeaderImpedanceController()
        self.detector = detector or InterventionDetector()
        self.switcher = switcher or ModeSwitcher()

    def step(
        self,
        q_leader: np.ndarray,
        q_follower: np.ndarray,
        v_leader: np.ndarray,
        q_vla_target: np.ndarray,
        C_VLA: float = 1.0,
        button_pressed: bool = False,
    ) -> "SharedControlResult":
        """
        Execute one control cycle.

        Parameters
        ----------
        q_leader : np.ndarray, shape (7,)
            Current Leader arm positions.
        q_follower : np.ndarray, shape (7,)
            Current Follower arm positions.
        v_leader : np.ndarray, shape (7,)
            Current Leader arm velocities.
        q_vla_target : np.ndarray, shape (7,)
            VLA's desired Follower position (from policy output).
        C_VLA : float
            VLA confidence [0, 1].
        button_pressed : bool
            True if the operator pressed the "resume VLA" button this cycle.

        Returns
        -------
        SharedControlResult
            Contains leader_efforts, follower_target, mode, alpha.
        """
        q_leader = np.asarray(q_leader, dtype=np.float64)
        q_follower = np.asarray(q_follower, dtype=np.float64)
        v_leader = np.asarray(v_leader, dtype=np.float64)
        q_vla_target = np.asarray(q_vla_target, dtype=np.float64)

        mode = self.switcher.mode
        alpha = self.switcher.alpha

        # --- Intervention detection (only meaningful in VLA mode) ---
        if mode == ControlMode.VLA:
            if self.detector.check(q_leader, q_follower):
                self.switcher.request_human_takeover()

        # --- Button: resume VLA (only in human mode) ---
        if button_pressed and mode == ControlMode.HUMAN:
            self.switcher.request_vla_resume()

        # --- Update transition progress ---
        self.switcher.update()
        mode = self.switcher.mode
        alpha = self.switcher.alpha

        # --- Compute Leader efforts ---
        if mode == ControlMode.HUMAN:
            # Pure gravity compensation — no spring, operator has full freedom
            leader_efforts = np.zeros(7)
        elif mode == ControlMode.VLA:
            # Full impedance tracking
            leader_efforts = self.impedance.compute_efforts(
                q_follower=q_follower,
                q_leader=q_leader,
                v_leader=v_leader,
                C_VLA=C_VLA,
                scale=1.0,
            )
        else:
            # Transition: scale impedance with (1 - alpha)
            # As alpha → 1 (human takes over), spring releases
            # As alpha → 0 (VLA resumes), spring engages
            leader_efforts = self.impedance.compute_efforts(
                q_follower=q_follower,
                q_leader=q_leader,
                v_leader=v_leader,
                C_VLA=C_VLA,
                scale=1.0 - alpha,
            )

        # --- Compute Follower target ---
        if mode == ControlMode.VLA:
            follower_target = q_vla_target
        elif mode == ControlMode.HUMAN:
            follower_target = q_leader.copy()
        else:
            # Transition: blend between VLA and human targets
            follower_target = (1.0 - alpha) * q_vla_target + alpha * q_leader

        return SharedControlResult(
            leader_efforts=leader_efforts,
            follower_target=follower_target,
            mode=mode,
            alpha=alpha,
        )

    def reset(self):
        """Reset all state. Call at episode start."""
        self.detector.reset()
        self.switcher.reset()


class SharedControlResult:
    """Output of one SharedControlSystem.step() call."""

    __slots__ = ('leader_efforts', 'follower_target', 'mode', 'alpha')

    def __init__(
        self,
        leader_efforts: np.ndarray,
        follower_target: np.ndarray,
        mode: ControlMode,
        alpha: float,
    ):
        self.leader_efforts = leader_efforts
        self.follower_target = follower_target
        self.mode = mode
        self.alpha = alpha

    def __repr__(self) -> str:
        return (
            f"SharedControlResult(mode={self.mode.value}, alpha={self.alpha:.3f}, "
            f"leader_efforts_norm={np.linalg.norm(self.leader_efforts):.4f})"
        )


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    """
    Simulate a VLA autonomous → human intervention → VLA resume scenario.
    """
    import time as _time

    sc = SharedControlSystem(
        impedance=LeaderImpedanceController(),
        detector=InterventionDetector(threshold_enter=0.15, threshold_exit=0.05),
        switcher=ModeSwitcher(transition_time=0.4),
    )

    print("\n" + "=" * 70)
    print("  SHARED CONTROL SANITY CHECK")
    print("=" * 70)

    # Initial state: both arms at same position
    q_follower = np.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02])
    q_leader   = np.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02])
    v_leader   = np.zeros(7)
    q_vla      = np.array([0.1, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02])

    # Phase 1: VLA mode — leader should track follower (which follows VLA)
    print("\n--- Phase 1: VLA autonomous mode ---")
    for i in range(5):
        # Simulate follower moving toward VLA target
        q_follower = q_follower + 0.2 * (q_vla - q_follower)
        # Leader tracks follower via impedance (simplified — in reality the
        # efforts would be sent to hardware and physics would integrate)
        q_leader = q_leader + 0.15 * (q_follower - q_leader)  # imperfect tracking

        result = sc.step(q_leader, q_follower, v_leader, q_vla, C_VLA=0.9)
        print(f"  step {i}: {result}")

    # Phase 2: Operator intervenes — push leader away
    print("\n--- Phase 2: Operator pushes Leader arm ---")
    q_leader[0] += 0.3  # Large push on base joint
    for i in range(5):
        result = sc.step(q_leader, q_follower, v_leader, q_vla, C_VLA=0.9)
        print(f"  step {i}: {result}")
        if result.mode in (ControlMode.HUMAN, ControlMode.TRANSITION_TO_HUMAN):
            # In human mode, follower tracks leader
            q_follower = q_follower + 0.3 * (result.follower_target - q_follower)

    # Phase 3: Operator presses button to resume VLA
    print("\n--- Phase 3: Button press → resume VLA ---")
    for i in range(5):
        result = sc.step(q_leader, q_follower, v_leader, q_vla,
                         C_VLA=0.9, button_pressed=(i == 0))
        print(f"  step {i}: {result}")

    print("\n  Final mode:", sc.switcher.mode.value)
    print("=" * 70)

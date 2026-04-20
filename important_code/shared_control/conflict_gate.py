"""
Conflict Gate — Layer 2 Confidence Arbitration
===============================================

Purpose
-------
The Conflict Gate sits between the VLA inference output and the eTaSL execution
layer. It answers one question: "Is the human operator actively trying to
override the VLA?"

When the human pushes the joystick in a direction that **disagrees** with what
the VLA wants to do, C_eff (effective confidence) is driven toward 0, causing
the weight calculator to hand control back to the human.

When the human does NOT touch the joystick (zero input), the gate passes
C_VLA through unchanged — the VLA operates autonomously.


Design
------
Old (binary, had problems):
    C_eff = C_VLA * 1[angle(u_human, a_vla) < theta]

Problems with binary gate (from Meeting 8, Erwin Aertbeliën):
  1. Zero-input undefined:  angle(0, a_vla) is mathematically undefined
  2. Binary oscillation:    gate flips 0/1 when angle hovers near theta
  3. Semantic oversimplification: conflict is about goal disagreement over time,
     not just momentary direction difference

New (continuous, hysteresis):
    C_eff = C_VLA * G(u_human, a_vla)
    G     = 1 - alpha * (1 - S_theta(phi))

Where:
    alpha   = sigmoid(k_alpha * (||delta_human|| - epsilon))  # magnitude guard
    phi     = angle(delta_human, delta_vla)                    # angular disagreement
    S_theta = sigmoid(-k_phi * (phi_deg - theta_eff))          # smooth agreement
    theta_eff switches between (theta + delta) / (theta - delta) for hysteresis


CRITICAL: Angle Computation — Absolute Positions vs. Deltas
------------------------------------------------------------
The entire WidowX / LeRobot pipeline works in ABSOLUTE JOINT POSITIONS:
  - Leader arm get_action()  → absolute positions {joint_0.pos: 0.785, ...}
  - VLA policy output        → absolute positions (after postprocessor denormalization)
  - robot.send_action()      → expects absolute target positions

We CANNOT compute ∠(q_human_abs, q_vla_abs) directly — that would compare
"where they are" not "where they want to go".

Instead, we must first compute DELTAS from the current robot state:

    delta_human = q_human_target - q_current     ← "human wants to move HERE from NOW"
    delta_vla   = q_vla_target   - q_current     ← "VLA   wants to move HERE from NOW"
    phi         = ∠(delta_human, delta_vla)       ← "do they agree on the direction?"

This is why compute() takes q_current as a required parameter.


Joint Weighting — Why Not All Joints Are Equal
-----------------------------------------------
WidowX has 7 DOFs:
  joint_0 .. joint_5  : revolute joints (radians)
  left_carriage_joint  : prismatic gripper (meters)

Problem 1: Mixed units.
  A 0.1 rad change in joint_0 (base rotation) moves the end-effector ~15 cm.
  A 0.1 m change in the gripper is physically huge.
  Naive dot product treats them identically → nonsensical angle.

Problem 2: Proximal vs. distal joints.
  Joint_0 (base) rotates the entire arm. Joint_5 (wrist roll) barely moves
  the end-effector. If human and VLA disagree only on wrist roll, that's a
  minor nuance — not a "conflict" worth triggering human takeover.

Solution: Weighted dot product with joint_weights W = diag(w0, ..., w6).

  phi = arccos( delta_h^T W delta_v / sqrt((delta_h^T W delta_h)(delta_v^T W delta_v)) )

Default weights emphasize proximal joints and separate gripper:
  joint_0 (base):    3.0   ← most impactful on end-effector position
  joint_1 (shoulder): 3.0
  joint_2 (elbow):    2.0
  joint_3 (forearm):  1.0
  joint_4 (wrist):    1.0
  joint_5 (wrist):    0.5   ← least impactful (roll)
  gripper:            0.0   ← excluded by default (open/close is separate intent)

These approximate the squared Jacobian column norms at a typical configuration.
For a more principled choice: compute J(q)^T J(q) diagonal at the current q
and use that as W. But the static defaults work for initial experiments.


Integration into run_shared_control.py (future entry point)
-------------------------------------------------------------
    from conflict_gate import ConflictGate, WeightCalculator

    gate = ConflictGate()
    wc   = WeightCalculator()

    # In the control loop (30 Hz):
    q_current      = robot_wrapper.get_observation()       # (7,) absolute positions
    q_human_target = teleop_leader.get_action()            # (7,) absolute positions
    q_vla_target   = action_queue.get().cpu().numpy()      # (7,) absolute positions

    G      = gate.compute(q_human_target, q_vla_target, q_current)
    C_eff  = C_vla * G
    w_vla, w_human = wc.compute(C_eff)

    # Blend in ABSOLUTE POSITION space (NOT delta space!)
    q_blended = w_vla * q_vla_target + w_human * q_human_target
    robot_wrapper.send_action(to_action_dict(q_blended))


References
----------
- Dragan & Srinivasa (2013). A policy-blending formalism for shared control.
  IJRR 32(7):790–805. [Zotero: 42X25GVP]
- Javdani et al. (2018). Shared autonomy via hindsight optimization.
  IJRR 37(7). [Zotero: CPC2CPXI]
- Meeting 8 notes (2026-03-16), feedback items #5 and #6 from Erwin Aertbeliën.
"""

import numpy as np
from typing import Optional


# =============================================================================
# ConflictGate
# =============================================================================

class ConflictGate:
    """
    Smooth, continuous conflict gate with magnitude guard and hysteresis.

    Formula
    -------
        G = 1 - alpha * (1 - S_theta(phi))

    This decomposes into three intuitive sub-components:
      - alpha:    "Is the human actively moving the leader arm?"
      - phi:      "How much do human and VLA disagree on WHERE TO GO NEXT?"
      - S_theta:  "Smooth threshold on phi, with hysteresis to prevent oscillation"

    Parameters
    ----------
    theta_deg : float
        Central conflict threshold in degrees (default 90°).
        Below this angle → agreement; above → conflict.
    delta_theta_deg : float
        Half-width of the hysteresis band in degrees (default 15°).
        Switching from "agreement" to "conflict" requires phi > theta + delta.
        Recovering back to "agreement" requires phi < theta - delta.
        Wider band = less oscillation, but slower response.
    k_alpha : float
        Steepness of the magnitude guard sigmoid (default 50).
        Higher value = sharper transition between "idle" and "active" human.
    epsilon : float
        Dead-zone threshold for human delta magnitude (default 0.03 rad).
        Below this value, human is considered to not be commanding movement.
        Set to ~99th percentile of resting leader arm noise.
    k_phi : float
        Steepness of the angular agreement sigmoid, per degree (default 0.05).
        Lower value = smoother gate, but VLA retains more influence during conflict.
    joint_weights : np.ndarray or None
        Per-joint importance weights for the weighted dot product.
        Shape (7,). Higher weight → this joint's disagreement matters more.
        Default: [3, 3, 2, 1, 1, 0.5, 0] (proximal joints dominate, gripper excluded).
        Set to None for uniform weighting across all 7 joints.
    """

    # WidowX default joint weights (see module docstring for rationale)
    DEFAULT_JOINT_WEIGHTS = np.array([
        3.0,   # joint_0: base rotation — moves entire arm, huge EE impact
        3.0,   # joint_1: shoulder pitch — lifts/lowers arm, huge EE impact
        2.0,   # joint_2: elbow pitch — extends/retracts forearm
        1.0,   # joint_3: forearm roll — rotates forearm, moderate EE impact
        1.0,   # joint_4: wrist pitch — tilts end-effector
        0.5,   # joint_5: wrist roll — rolls gripper, minimal EE position change
        0.0,   # left_carriage_joint: gripper open/close — EXCLUDED from angle
               #   Reason: gripper is a discrete-ish intent (grasp vs release),
               #   not a directional motion. A human opening the gripper while
               #   VLA closes it is NOT a "directional conflict" — it's a
               #   task-level disagreement that should be handled separately.
    ])

    def __init__(
        self,
        theta_deg: float = 90.0,
        delta_theta_deg: float = 15.0,
        k_alpha: float = 50.0,
        epsilon: float = 0.03,
        k_phi: float = 0.05,
        joint_weights: Optional[np.ndarray] = None,
    ):
        # Hysteresis thresholds (in radians for internal use)
        self.theta_high = np.radians(theta_deg + delta_theta_deg)  # agreement → conflict
        self.theta_low  = np.radians(theta_deg - delta_theta_deg)  # conflict → agreement

        self.k_alpha = k_alpha
        self.epsilon = epsilon
        self.k_phi   = k_phi       # operates on degrees (phi converted before use)

        # Joint weighting matrix (diagonal)
        if joint_weights is not None:
            self.W = np.asarray(joint_weights, dtype=np.float64)
        else:
            self.W = self.DEFAULT_JOINT_WEIGHTS.copy()

        # Hysteresis state: True = "in agreement", False = "in conflict"
        # Starts in agreement — VLA begins in full control.
        self._in_agreement: bool = True

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    def _weighted_angle(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute angle between two vectors using the weighted inner product.

            phi = arccos( a^T W b / sqrt((a^T W a)(b^T W b)) )

        where W = diag(self.W).

        This is mathematically equivalent to computing the standard angle
        between sqrt(W)*a and sqrt(W)*b, but avoids explicit matrix construction.

        Parameters
        ----------
        a, b : np.ndarray, shape (7,)
            The two delta vectors (NOT absolute positions).

        Returns
        -------
        float
            Angle in radians, range [0, π].
        """
        # Element-wise weighting: W is diagonal, so a^T W b = sum(W * a * b)
        aw = self.W * a     # shape (7,): element-wise W[i] * a[i]
        bw = self.W * b     # shape (7,): element-wise W[i] * b[i]

        numerator   = np.dot(a, bw)                  # a^T W b
        denom_left  = np.sqrt(np.dot(a, aw))         # sqrt(a^T W a)
        denom_right = np.sqrt(np.dot(b, bw))         # sqrt(b^T W b)
        denominator = denom_left * denom_right

        if denominator < 1e-12:
            return 0.0   # degenerate case — both near zero after weighting

        cos_phi = numerator / denominator
        cos_phi = float(np.clip(cos_phi, -1.0, 1.0))   # numerical safety
        return float(np.arccos(cos_phi))

    def compute(
        self,
        q_human_target: np.ndarray,
        q_vla_target: np.ndarray,
        q_current: np.ndarray,
    ) -> float:
        """
        Compute the gate value G ∈ [0, 1].

        IMPORTANT — all three inputs are ABSOLUTE joint positions (rad / m),
        which is what both the leader arm and the VLA policy produce.
        This method internally computes the deltas before comparing directions.

        Parameters
        ----------
        q_human_target : np.ndarray, shape (7,)
            Where the human wants the robot to go (absolute joint positions).
            Read from leader arm via teleop.get_action().
            If human is NOT touching the leader arm, this ≈ q_current
            (leader arm in gravity-comp mode returns its resting position,
             which should be close to follower's current position when idle).

        q_vla_target : np.ndarray, shape (7,)
            Where the VLA wants the robot to go (absolute joint positions).
            Read from action_queue.get() after postprocessor denormalization.

        q_current : np.ndarray, shape (7,)
            Where the robot IS right now (absolute joint positions).
            Read from robot.get_observation()["observation.state"].

        Returns
        -------
        float
            G ∈ [0, 1] where:
              G ≈ 1 → no conflict, C_eff ≈ C_VLA  (VLA trusted as-is)
              G ≈ 0 → strong conflict, C_eff ≈ 0   (human takes over)
        """
        # ==================================================================
        # Step 0: Convert absolute positions → deltas (movement intentions)
        # ==================================================================
        # This is the CRITICAL step. Without it, we'd be comparing "where
        # they ARE" instead of "where they WANT TO GO".
        #
        # Example:
        #   q_current       = [0.0, 1.0, 0.5, ...]
        #   q_human_target  = [0.1, 1.0, 0.5, ...]  ← human moves joint_0 right
        #   q_vla_target    = [-0.1, 1.0, 0.5, ...]  ← VLA moves joint_0 left
        #
        #   delta_human = [+0.1, 0, 0, ...]   delta_vla = [-0.1, 0, 0, ...]
        #   → phi = 180° → CONFLICT. Correct!
        #
        #   Without deltas: ∠([0.1, 1.0, 0.5], [-0.1, 1.0, 0.5]) ≈ 10°
        #   → No conflict detected. WRONG — they're moving opposite directions!
        #
        delta_human = np.asarray(q_human_target, dtype=np.float64) - np.asarray(q_current, dtype=np.float64)
        delta_vla   = np.asarray(q_vla_target, dtype=np.float64)   - np.asarray(q_current, dtype=np.float64)

        # ==================================================================
        # Step 1: Magnitude Guard — α
        # "Is the human actually commanding movement?"
        # ==================================================================
        # We use the WEIGHTED norm of delta_human so that the magnitude
        # threshold ε is also in the importance-weighted space.
        # This means: a tiny wrist-roll movement (low weight) won't trigger
        # the gate, but a small base rotation (high weight) will.
        mag_human_weighted = float(np.sqrt(np.dot(delta_human, self.W * delta_human)))

        # alpha ≈ 0 when human is idle, ≈ 1 when actively commanding
        alpha = self._sigmoid(self.k_alpha * (mag_human_weighted - self.epsilon))

        # ------------------------------------------------------------------
        # Short-circuit: if human delta is essentially zero, no conflict.
        # G = 1 regardless of angle (angle undefined for zero vectors).
        # This is the fix for Meeting 8 feedback item #5 (zero-input case).
        # ------------------------------------------------------------------
        mag_vla_weighted = float(np.sqrt(np.dot(delta_vla, self.W * delta_vla)))
        if mag_human_weighted < 1e-9 or mag_vla_weighted < 1e-9:
            self._in_agreement = True
            return 1.0

        # ==================================================================
        # Step 2: Angular Disagreement — φ (weighted)
        # "How different are the human and VLA movement DIRECTIONS?"
        # ==================================================================
        # Uses weighted dot product so that disagreement on important joints
        # (base, shoulder) counts more than disagreement on wrist roll.
        # Gripper (weight=0) is fully excluded.
        phi = self._weighted_angle(delta_human, delta_vla)

        # ==================================================================
        # Step 3: Hysteresis Threshold — θ_eff
        # "Which threshold to use, based on last step's state?"
        # ==================================================================
        # This is the 'thermostat' recommended by Erwin in Meeting 8:
        #   - In agreement state → use HIGH threshold (harder to enter conflict)
        #   - In conflict state  → use LOW threshold (harder to exit conflict)
        # Prevents rapid oscillation when phi ≈ theta.
        theta_eff = self.theta_high if self._in_agreement else self.theta_low

        # ==================================================================
        # Step 4: Smooth Angular Agreement — S_θ
        # S_theta ≈ 1 when phi << theta_eff  (directions agree)
        # S_theta ≈ 0 when phi >> theta_eff  (directions conflict)
        # ==================================================================
        phi_deg = np.degrees(phi)
        theta_eff_deg = np.degrees(theta_eff)
        S_theta = self._sigmoid(-self.k_phi * (phi_deg - theta_eff_deg))

        # ==================================================================
        # Step 5: Combine — G
        # G = 1 - alpha * (1 - S_theta)
        # ==================================================================
        # Truth table:
        #   alpha ≈ 0 (idle human):    G ≈ 1       (VLA autonomous)
        #   alpha ≈ 1, S ≈ 1 (agree):  G ≈ 1       (cooperative)
        #   alpha ≈ 1, S ≈ 0 (clash):  G ≈ 0       (human takeover)
        G = 1.0 - alpha * (1.0 - S_theta)

        # Update hysteresis state
        self._in_agreement = (G > 0.5)

        return float(G)

    def reset(self):
        """Reset hysteresis state to 'agreement'. Call at episode start."""
        self._in_agreement = True


# =============================================================================
# WeightCalculator
# =============================================================================

class WeightCalculator:
    """
    Converts C_eff into blending weights for VLA and human actions.

    Formula
    -------
        w_vla   = W_max * sigmoid(k * (C_eff - tau))
        w_human = W_max - w_vla

    Both weights always sum to W_max, so the total "force" sent to eTaSL
    is constant regardless of who is in control.

    Parameters
    ----------
    W_max : float
        Maximum total weight (default 1.0 for normalized blend).
        In eTaSL integration this would be the max task weight.
    tau : float
        Sigmoid center — C_eff at which weights are equal (default 0.5).
    k : float
        Sigmoid steepness (default 10.0). Higher = sharper handover.
    """

    def __init__(self, W_max: float = 1.0, tau: float = 0.5, k: float = 10.0):
        self.W_max = W_max
        self.tau   = tau
        self.k     = k

    def compute(self, C_eff: float) -> tuple[float, float]:
        """
        Compute VLA and human blending weights.

        Parameters
        ----------
        C_eff : float
            Effective confidence in [0, 1].

        Returns
        -------
        w_vla : float
            Weight for VLA action.
        w_human : float
            Weight for human action. Always: w_vla + w_human = W_max.
        """
        # Clamp to valid range
        C_eff = float(np.clip(C_eff, 0.0, 1.0))

        w_vla   = self.W_max / (1.0 + np.exp(-self.k * (C_eff - self.tau)))
        w_human = self.W_max - w_vla

        return w_vla, w_human


# =============================================================================
# Quick sanity-check (run this file directly: python conflict_gate.py)
# =============================================================================

if __name__ == "__main__":
    """
    Sanity-check demonstrating how the angle is computed from absolute positions.

    Setup:
      Robot is currently at q_current = [0, 1, 0.5, 0, 0, 0, 0.02].
      VLA wants to move joint_0 to +0.1 (rightward base rotation).
      We test four human behaviours:

    Expected outcomes:
      Scenario A (idle — human target ≈ q_current):  G ≈ 1.0
      Scenario B (aligned — human also moves right):  G ≈ 1.0
      Scenario C (opposing — human moves left):       G ≈ 0.0
      Scenario D (tiny drift — below noise floor):    G ≈ 1.0
      Scenario E (gripper-only disagreement):          G ≈ 1.0 (gripper weight = 0)
      Scenario F (wrist-only disagreement):            G high (wrist has low weight)
    """

    gate   = ConflictGate(theta_deg=90.0, delta_theta_deg=15.0)
    wc     = WeightCalculator(W_max=1.0, tau=0.5, k=10.0)

    # Robot's current state (absolute positions)
    q_current = np.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02])

    # VLA target: move base joint rightward by 0.1 rad
    q_vla = np.array([0.1, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02])
    # → delta_vla = [+0.1, 0, 0, 0, 0, 0, 0]

    scenarios = {
        "A — Human idle (target ≈ current, not touching leader arm)": {
            "q_human": np.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02]),
            # delta_human = [0, 0, 0, 0, 0, 0, 0] → magnitude ≈ 0 → G ≈ 1
        },
        "B — Human agrees (also moves base rightward)": {
            "q_human": np.array([0.08, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02]),
            # delta_human = [+0.08, 0, 0, 0, 0, 0, 0] → same direction as VLA → G ≈ 1
        },
        "C — Human opposes (moves base leftward)": {
            "q_human": np.array([-0.1, 1.0, 0.5, 0.0, 0.0, 0.0, 0.02]),
            # delta_human = [-0.1, 0, 0, 0, 0, 0, 0] → opposite direction! → G ≈ 0
        },
        "D — Tiny drift (below dead-zone ε=0.03)": {
            "q_human": np.array([0.005, 1.001, 0.5, 0.0, 0.0, 0.0, 0.02]),
            # delta_human = [0.005, 0.001, 0, 0, 0, 0, 0] → weighted norm < ε → G ≈ 1
        },
        "E — Gripper-only disagreement (human opens, VLA closes)": {
            "q_human": np.array([0.1, 1.0, 0.5, 0.0, 0.0, 0.0, 0.06]),
            # delta_human = [0.1, 0, 0, 0, 0, 0, +0.04] → gripper differs but weight=0
            # delta_vla   = [0.1, 0, 0, 0, 0, 0,  0   ] → arm part agrees → G ≈ 1
        },
        "F — Wrist-roll-only disagreement (low-weight joint)": {
            "q_human": np.array([0.1, 1.0, 0.5, 0.0, 0.0, -0.3, 0.02]),
            # delta_human = [0.1, 0, 0, 0, 0, -0.3, 0] → wrist roll differs
            # But joint_5 weight is only 0.5 vs base weight 3.0 → moderate G
        },
    }

    print("\n" + "=" * 70)
    print("  CONFLICT GATE SANITY CHECK — Absolute Position API")
    print("=" * 70)
    print(f"\n  q_current = {q_current}")
    print(f"  q_vla     = {q_vla}  (delta_vla = {q_vla - q_current})")
    print(f"  weights   = {gate.W}")

    C_vla = 0.8

    for name, data in scenarios.items():
        gate.reset()
        q_human = data["q_human"]
        delta_h = q_human - q_current

        G     = gate.compute(q_human, q_vla, q_current)
        C_eff = C_vla * G
        w_vla, w_human = wc.compute(C_eff)

        print(f"\n  {name}")
        print(f"    q_human     = {q_human}")
        print(f"    delta_human = {delta_h}")
        print(f"    ||delta_h|| (weighted) = {np.sqrt(np.dot(delta_h, gate.W * delta_h)):.4f}")
        print(f"    G       = {G:.4f}")
        print(f"    C_eff   = {C_eff:.4f}  (C_VLA={C_vla})")
        print(f"    w_vla   = {w_vla:.4f}   w_human = {w_human:.4f}")

    # === Hysteresis demo ===
    print("\n" + "=" * 70)
    print("  HYSTERESIS DEMO — angle sweeping 60° → 120° → 60°")
    print("  (human moves at angle φ from VLA direction in joint_0/joint_1 plane)")
    print("-" * 70)
    gate.reset()
    q_current_demo = np.zeros(7)
    q_vla_demo     = np.array([0.1, 0.0, 0, 0, 0, 0, 0])  # VLA: +x direction

    angles = list(range(60, 125, 5)) + list(range(120, 55, -5))
    for deg in angles:
        rad = np.radians(deg)
        # Human moves at angle 'deg' relative to +x (VLA direction) in joint_0/joint_1 plane
        q_human_demo = np.array([0.1 * np.cos(rad), 0.1 * np.sin(rad), 0, 0, 0, 0, 0])
        G = gate.compute(q_human_demo, q_vla_demo, q_current_demo)
        state_str = "agree " if gate._in_agreement else "CONFLICT"
        print(f"    phi={deg:3d}° → G={G:.3f}  state={state_str}")

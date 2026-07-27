"""UR5e kinematics in pure numpy (nominal DH) for **resolved-rate / damped-least-squares** servoing.

Why this exists: the UR controller's `get_inverse_kin` (used by `servoL` and ur_rtde's
`getInverseKinematics`) throws "the robot cannot reach the requested pose" and faults the RTDE control
script on near-singular targets — which a downward-pointing pick-place gripper sits right on (wrist
singularity). Instead we convert the EE-pose target to a JOINT target here, in Python, with one DLS
step, and command `servoJ`. DLS moves *through* singularities gracefully (damps the degenerate
direction) instead of faulting, and never touches the controller's IK.

Standard resolved-rate control:  dq = Jᵀ (J Jᵀ + λ²I)⁻¹ · dx ,  dx = pose error (target − current).
For the small per-command increments we stream (≤5 cm / ≤8.6°), one step per control cycle tracks the
target; over cycles it closes the loop on the ACTUAL measured pose (so nominal-vs-calibrated DH error
is corrected, not accumulated).
"""
from __future__ import annotations

import numpy as np

# Nominal UR5e DH (a, d, alpha) — matches the FK validated against UR's canonical zero-config flange.
_DH = ((0.0, 0.1625, np.pi / 2), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0),
       (0.0, 0.1333, np.pi / 2), (0.0, 0.0997, -np.pi / 2), (0.0, 0.0996, 0.0))


def _frames(q):
    """Cumulative base->frame transforms for joints 1..6; returns [T0(=I) .. T6(=flange)]."""
    T = np.eye(4)
    out = [T.copy()]
    for theta, (a, d, alpha) in zip(q, _DH):
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0.0, sa, ca, d],
                          [0.0, 0.0, 0.0, 1.0]])
        out.append(T.copy())
    return out


def fk_flange(q):
    """Forward kinematics -> (position[3], rotation[3x3]) of the flange (frame 6)."""
    T = _frames(q)[-1]
    return T[:3, 3], T[:3, :3]


def jacobian_at(q, p_end):
    """6x6 geometric Jacobian for the point `p_end` (base frame) — columns [z_i x (p_end - o_i); z_i].
    Passing the ACTUAL TCP point as p_end gives the TCP Jacobian without needing the tool offset."""
    F = _frames(q)
    J = np.zeros((6, 6))
    for i in range(6):
        z = F[i][:3, 2]
        o = F[i][:3, 3]
        J[:3, i] = np.cross(z, np.asarray(p_end) - o)
        J[3:, i] = z
    return J


def dls_ik_step(q, cur_pose, target_pose, lam: float = 0.08, dq_max: float = 0.2):
    """One damped-least-squares step from joints `q` toward `target_pose`, given the CURRENT measured
    pose `cur_pose`. Poses are [x, y, z, rx, ry, rz] (rotvec), base frame. Returns the joint target
    (list) — feed to servoJ. `lam` damps singular directions; `dq_max` caps per-joint motion."""
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(q, dtype=float)
    cur = np.asarray(cur_pose, dtype=float)
    tgt = np.asarray(target_pose, dtype=float)

    J = jacobian_at(q, cur[:3])                      # TCP Jacobian at the actual TCP point
    dp = tgt[:3] - cur[:3]                            # position error (base frame)
    dw = (R.from_rotvec(tgt[3:6]) * R.from_rotvec(cur[3:6]).inv()).as_rotvec()  # orientation error
    dx = np.concatenate([dp, dw])

    dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), dx)
    dq = np.clip(dq, -dq_max, dq_max)
    return (q + dq).tolist()

"""Workspace guard — keep the UR end-effector out of a no-go rectangle (the mobile base's body).

On the REAL robot the UR is mounted so its body/electronics sit at **+y** and the open workspace is
in front at **-y** (verified on hardware: at zero-ish config the TCP hovers over the base box at +y).
This clamps any commanded EE target's (x, y) out of the body footprint so the arm can never be driven
back over itself. Applied in BOTH paths: the sim cell (`MujocoCell.set_ee_pose`) and hardware
(`RtdeArmInterface.send_ee_pose`). The clamp projects an in-zone target to the nearest boundary, so
teleop slides along the edge instead of stopping dead.

NOTE: the MuJoCo sim mounts the UR 180° from this (its body is at -y); align the sim model to this
convention before relying on the sim's no-go box again.

Coordinates are in the UR base frame (same frame as the EE pose / RTDE `getActualTCPPose`).
"""
from __future__ import annotations

from math import pi

# No-go rectangle (xmin, xmax, ymin, ymax), in metres, in the UR base frame — the robot's body (+y).
NO_GO_XY = (-0.27, 0.27, 0.0, 0.88)
NO_GO_MARGIN = 0.03  # keep-out buffer (m) added around the rectangle
LINK_MARGIN = 0.06   # larger buffer for the elbow/wrist LINK check (links have radius; joints are points)


def clamp_out_of_nogo(pose, zone=NO_GO_XY, margin: float = NO_GO_MARGIN):
    """Project the EE target's (x, y) out of the no-go rectangle (to the nearest edge).

    Returns (clamped_pose: list, was_clamped: bool). Orientation / z are untouched.
    """
    out = [float(v) for v in pose]
    x, y = out[0], out[1]
    xmin, xmax = zone[0] - margin, zone[1] + margin
    ymin, ymax = zone[2] - margin, zone[3] + margin
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return out, False
    # inside the keep-out -> push to the nearest edge
    dist = {"xmin": x - xmin, "xmax": xmax - x, "ymin": y - ymin, "ymax": ymax - y}
    edge = min(dist, key=dist.get)
    if edge == "xmin":
        out[0] = xmin
    elif edge == "xmax":
        out[0] = xmax
    elif edge == "ymin":
        out[1] = ymin
    else:
        out[1] = ymax
    return out, True


def point_in_nogo(x: float, y: float, zone=NO_GO_XY, margin: float = NO_GO_MARGIN) -> bool:
    """True if (x, y) is inside the no-go rectangle plus `margin`."""
    return (zone[0] - margin) <= x <= (zone[1] + margin) and (zone[2] - margin) <= y <= (zone[3] + margin)


# UR5e nominal DH (a, d, alpha), metres/rad. Good enough for a keep-out check — per-robot calibration
# differs by ~mm, absorbed by LINK_MARGIN. Joint angles from RTDE getActualQ are the DH thetas directly.
_UR5E_DH = ((0.0, 0.1625, pi / 2), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0),
            (0.0, 0.1333, pi / 2), (0.0, 0.0997, -pi / 2), (0.0, 0.0996, 0.0))


def arm_link_xy(q):
    """(x, y) of the elbow + wrist joint origins (DH frames 3..6) for joint config q, via nominal
    UR5e forward kinematics — so we can keep the arm's *links*, not just the TCP, out of the base zone."""
    import numpy as np

    T = np.eye(4)
    out = []
    for i, (theta, (a, d, alpha)) in enumerate(zip(q, _UR5E_DH)):
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0.0, sa, ca, d],
                          [0.0, 0.0, 0.0, 1.0]])
        if i >= 2:  # elbow (frame 3) + wrists (4,5,6) — the segments that can swing back over the base
            out.append((float(T[0, 3]), float(T[1, 3])))
    return out


def links_in_nogo(q, zone=NO_GO_XY, margin: float = LINK_MARGIN) -> bool:
    """True if any elbow/wrist joint of config q lies inside the no-go rectangle (+ link margin).
    A crude self-collision guard for the arm folding a LINK over the base while the TCP stays legal."""
    return any(point_in_nogo(x, y, zone, margin) for x, y in arm_link_xy(q))

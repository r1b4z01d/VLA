"""Kinematic sim arm — an ArmInterface that just tracks the commanded EE pose.

No dynamics or IK: `send_ee_pose` stores the target and `get_ee_pose` echoes it. Enough
to exercise the LeRobot data pipeline on the Mac before the real UR5e / Gazebo bridge
(Phase 2) exists. Swap for a MuJoCo/Isaac arm or Ros2ArmInterface later.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from ..robot.arm_interface import ArmInterface

HOME_POSE = (0.4, 0.0, 0.3, 0.0, math.pi, 0.0)  # [x,y,z,rx,ry,rz] base frame


class SimArm(ArmInterface):
    def __init__(self, home_pose: Sequence[float] = HOME_POSE):
        self._home = [float(v) for v in home_pose]
        self._pose = list(self._home)
        self._connected = False

    def connect(self) -> None:
        self._pose = list(self._home)  # reset to home so each episode starts the same
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_ee_pose(self) -> list[float]:
        return list(self._pose)

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 6  # kinematic stub — no IK yet

    def send_ee_pose(self, pose: Sequence[float]) -> list[float]:
        self._pose = [float(v) for v in pose]
        return list(self._pose)

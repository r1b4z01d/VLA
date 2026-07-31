"""Transport-agnostic UR5e arm interface used by the LeRobot adapter.

EE pose is ``[x, y, z, rx, ry, rz]`` (meters + axis-angle rad) in the UR base frame,
matching ``ur5e_lerobot.schema``. Keeping the arm behind this interface lets the same
adapter drive the real robot, a Gazebo/Isaac sim, or a test double — and keeps the
rclpy node isolated from the lerobot Python env (decision D5).
"""
from __future__ import annotations

import abc
from collections.abc import Sequence


class ArmInterface(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...

    @abc.abstractmethod
    def get_ee_pose(self) -> list[float]:
        """Return current TCP pose [x,y,z,rx,ry,rz] in the base frame."""

    @abc.abstractmethod
    def get_joint_positions(self) -> list[float]:
        """Return the 6 UR joint positions (rad)."""

    @abc.abstractmethod
    def send_ee_pose(self, pose: Sequence[float]) -> list[float]:
        """Command a TCP pose target; return the pose actually commanded (may be clipped)."""

    # --- optional: kinesthetic freedrive (teach mode). Default: unsupported (no-op). ---
    def start_freedrive(self) -> bool:
        """Put the arm in gravity-comp freedrive so it can be hand-guided. Return True if enabled.

        Only the real arm (RtdeArmInterface) supports this; sim/stub arms return False.
        """
        return False

    def stop_freedrive(self) -> None:
        """Leave freedrive (re-lock the arm). No-op if freedrive isn't active/supported."""

    def reconnect(self) -> str:
        """Re-establish the connection after a fault / stopped control script, without restarting
        the app. Returns a short status message. The real arm overrides this to also clear a
        protective stop; the generic version just disconnects + reconnects."""
        self.disconnect()
        self.connect()
        return "reconnected"


class Ros2ArmInterface(ArmInterface):
    """UR5e over ROS 2 Jazzy (ur_robot_driver + MoveIt Servo).  TODO(Phase 2).

    Plan: run an rclpy node that
      - subscribes ``/joint_states`` and the TCP pose,
      - publishes Cartesian/twist targets to MoveIt Servo for EE-pose control.
    Runs in its own process/env (decision D5). Reference: github.com/ycheng517/lerobot-ros.
    """

    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self._connected = False

    def connect(self) -> None:
        raise NotImplementedError(
            "Phase 2: start rclpy node and connect to ur_robot_driver / MoveIt Servo"
        )

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_ee_pose(self) -> list[float]:
        raise NotImplementedError("Phase 2: read TCP pose from ROS 2")

    def get_joint_positions(self) -> list[float]:
        raise NotImplementedError("Phase 2: read /joint_states from ROS 2")

    def send_ee_pose(self, pose: Sequence[float]) -> list[float]:
        raise NotImplementedError("Phase 2: publish EE target via MoveIt Servo")

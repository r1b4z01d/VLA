"""LeRobot `Robot` adapter for the UR5e + AmazingHand.

Composes:
  * an ArmInterface (UR5e EE-pose control via ROS 2 — Phase 2 stub for now), and
  * an AmazingHandClient (TCP, fully implemented),
behind LeRobot's standard Robot API so `record`/`train`/`eval` work unchanged.

Action / observation follow ur5e_lerobot.schema (10-D action, 16-D state + cameras).
"""
from __future__ import annotations

from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

try:
    from lerobot.processor import RobotAction, RobotObservation
except Exception:  # lerobot.processor eagerly imports transformers (heavy, unneeded on the record
    RobotAction = RobotObservation = dict  # path); these names are only used as (stringized) hints

from ..hand import AmazingHandClient
from ..schema import ACTION_NAMES, ARM_POSE_NAMES, HAND_CURL_DIM, HAND_CURL_NAMES, STATE_NAMES
from .arm_interface import ArmInterface, Ros2ArmInterface
from .config import URAmazingHandConfig


class URAmazingHand(Robot):
    config_class = URAmazingHandConfig
    name = "ur_amazinghand"

    def __init__(
        self,
        config: URAmazingHandConfig,
        arm: ArmInterface | None = None,
        hand: AmazingHandClient | None = None,
    ):
        super().__init__(config)
        self.config = config
        # `arm`/`hand` are injectable for sim and testing; default to real backends.
        self.arm = arm or Ros2ArmInterface(config.robot_ip)
        self.hand = hand or AmazingHandClient(
            config.hand_host, config.hand_port, speed=config.hand_speed
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        # Last commanded curls — used for proprioception until hand state read-back exists.
        self._hand_curls = [0.0] * HAND_CURL_DIM

    # --- features ---------------------------------------------------------
    @property
    def _state_ft(self) -> dict[str, type]:
        return {name: float for name in STATE_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return {name: float for name in ACTION_NAMES}

    # --- lifecycle --------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        self.arm.connect()
        self.hand.connect()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        # The UR5e is calibrated at its controller; no per-session calibration here yet.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        # Placeholder: servo gains / MoveIt Servo params (Phase 2).
        pass

    def disconnect(self) -> None:
        self.arm.disconnect()
        self.hand.close()
        for cam in self.cameras.values():
            cam.disconnect()

    # --- io ---------------------------------------------------------------
    def get_observation(self) -> RobotObservation:
        self._require_connected()
        state_vals = (
            list(self.arm.get_ee_pose())
            + list(self.arm.get_joint_positions())
            + list(self._hand_curls)
        )
        obs: RobotObservation = {name: float(v) for name, v in zip(STATE_NAMES, state_vals)}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_connected()
        pose = [float(action[k]) for k in ARM_POSE_NAMES]
        curls = [float(action[k]) for k in HAND_CURL_NAMES]

        sent_pose = self.arm.send_ee_pose(pose)
        self.hand.send_curls(curls)
        self._hand_curls = curls

        sent: RobotAction = {k: float(v) for k, v in zip(ARM_POSE_NAMES, sent_pose)}
        sent.update({k: float(v) for k, v in zip(HAND_CURL_NAMES, curls)})
        return sent

    # --- helpers ----------------------------------------------------------
    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected — call connect() first")

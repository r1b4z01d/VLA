"""LeRobot config for the UR5e + AmazingHand robot (decisions D4/D6, data_schema.md)."""
from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("ur_amazinghand")
@dataclass(kw_only=True)
class URAmazingHandConfig(RobotConfig):
    # --- Arm: UR5e via ROS 2 (connection params finalized in Phase 2) ---
    robot_ip: str = "192.168.1.100"  # UR controller IP (placeholder)

    # --- Hand: AmazingHand TCP service (protocol confirmed, decision D4) ---
    hand_host: str = "192.168.1.194"
    hand_port: int = 8765
    hand_speed: int = 200  # placeholder until firmware speed range is confirmed

    # --- Cameras: key -> CameraConfig (RealSenseCameraConfig / OpenCVCameraConfig).
    #     Keys should match ur5e_lerobot.schema.CAMERA_KEYS (scene/wrist/side). ---
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # --- Safety: clip EE-target deltas if set (Phase 2). ---
    max_relative_target: float | None = None
